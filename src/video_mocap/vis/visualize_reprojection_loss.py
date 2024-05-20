import argparse
import os
import time

import cv2
import joblib
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch3d.loss import chamfer_distance
import pyrender
import seaborn as sns
import torch
import torch.nn.functional as F
import trimesh

from video_mocap.markers.markers import Markers
from video_mocap.markers.markers_utils import cleanup_markers
from video_mocap.utils.config import load_config
from video_mocap.utils.hmr_utils import convert_hmr_pos_to_mocap_pos, optim_reprojection
from video_mocap.utils.smpl import SmplInference
from video_mocap.vis.scene import VideoMocapScene, SMPL_COLORS
from video_mocap.vis.renderer import VideoMocapRenderer


def visualize(args):
    device = torch.device("cuda:0")

    config = load_config(args.config)

    camera = {
        "umpm": "l",
        "cmu_kitchen_pilot": "7151062",
        "cmu_kitchen_pilot_rb": "7151062",
        "moyo_train": None,
        "moyo_val": None,
        "bmlmovi_train": None,
        "bmlmovi_val": None,
    }[args.dataset]

    video_filename = os.path.join(
        args.input_dir,
        args.dataset,
        "videos",
        args.subject,
        args.sequence + "." + camera + ".avi",
    )

    hmr_pkl_filename = os.path.join(
        args.input_dir,
        args.dataset,
        "comparisons",
        "4d_humans",
        args.subject,
        args.sequence + "." + camera,
        "results",
        "demo_" + args.sequence + ".pkl",
    )

    mocap_filename = os.path.join(
        args.input_dir,
        args.dataset,
        "mocap",
        args.subject,
        args.sequence + ".c3d",
    )
    markers = Markers(mocap_filename)
    points = markers.get_points()
    points = np.nan_to_num(points, 0)
    points = cleanup_markers(points)
    points = torch.from_numpy(points).float().to(device)

    hmr_data = joblib.load(hmr_pkl_filename)
    hmr_data = stack_hmr_data(hmr_data)
    num_frames = hmr_data["camera_bbox"].shape[0]

    reproject_2d_joints = []

    # extract features
    smpl_inference = SmplInference(device=device)

    hmr_betas = torch.from_numpy(hmr_data["smpl"]["betas"]).float().to(device)
    betas = torch.mean(hmr_betas, dim=0, keepdim=True).detach()
    pose_body = torch.from_numpy(hmr_data["smpl"]["body_pose"]).float().to(device)
    root_orient = torch.from_numpy(hmr_data["smpl"]["global_orient"]).float().to(device)

    trans = torch.zeros((betas.shape[0], 3)).float().to(device)
    pred_cam = torch.from_numpy(hmr_data["camera_bbox"]).float().to(device)

    cam_center = torch.from_numpy(hmr_data["center"]).float().to(device)
    cam_size = torch.from_numpy(hmr_data["size"]).float().to(device)
    cam_scale = torch.tensor(hmr_data["scale"]).float().to(device)

    num_angles = config["stages"]["reprojection"]["num_angles"]  # A
    angles = torch.arange(0, 2*np.pi, (2*np.pi)/num_angles)

    reproject_output = {}
    reproject_output["pose_body"] = []
    reproject_output["betas"] = []
    reproject_output["root_orient"] = []
    reproject_output["trans"] = []
    reproject_output["joints_2d"] = []
    reproject_output["joints_2d_gt"] = []
    reproject_output["cam_trans"] = []
    reproject_output["input_angle"] = []
    reproject_output["output_angle"] = []
    reproject_output["metrics"] = []

    for angle_index in range(num_angles):
        angle = angles[angle_index]
        reproject_output_single = optim_reprojection(
            markers=points,
            pose_body=pose_body,
            betas=betas,
            hmr_betas=betas,
            root_orient=root_orient,
            trans=trans,
            pred_cam=pred_cam,
            cam_center=cam_center,
            cam_size=cam_size,
            cam_scale=cam_scale,
            smpl_inference=smpl_inference,
            angle=angle,
            config=config,
            verbose=True,
        )
        reproject_output["pose_body"].append(reproject_output_single["pose_body"])
        reproject_output["betas"].append(reproject_output_single["betas"])
        reproject_output["root_orient"].append(reproject_output_single["root_orient"])
        reproject_output["trans"].append(reproject_output_single["trans"])
        reproject_output["joints_2d"].append(reproject_output_single["joints_2d"])
        reproject_output["joints_2d_gt"].append(reproject_output_single["joints_2d_gt"])
        reproject_output["cam_trans"].append(reproject_output_single["cam_trans"])
        reproject_output["input_angle"].append(reproject_output_single["input_angle"])
        reproject_output["output_angle"].append(reproject_output_single["output_angle"])
        reproject_output["metrics"].append(reproject_output_single["metrics"])

    os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
    _, video_path_extension = os.path.splitext(args.video_path)

    input_angle = [np.rad2deg(x) % 360 for x in reproject_output["input_angle"]]
    output_angle = [np.rad2deg(x) % 360 for x in reproject_output["output_angle"]]

    x_axis = []
    y_axis = []
    labels = []
    reprojection_error = []
    for i in range(len(input_angle)):
        x_axis.append("input angle")
        x_axis.append("output angle")
        y_axis.append(input_angle[i])
        y_axis.append(output_angle[i])
        labels.append(input_angle[i])
        labels.append(input_angle[i])
        reprojection_error.append(reproject_output["metrics"][i]["reproject"])
        reprojection_error.append(reproject_output["metrics"][i]["reproject"])

    data_frame = {
        "x": x_axis,
        "y": y_axis,
        "labels": labels,
        "reprojection_error": reprojection_error,
    }
    data_frame = pd.DataFrame.from_dict(data_frame, orient="index").transpose()

    palette="flare"
    cmap = plt.get_cmap(palette, 256)
    cmap_norm = matplotlib.colors.Normalize(
        vmin=np.min(np.array(reprojection_error)),
        vmax=np.max(np.array(reprojection_error)),
    )
    plot = sns.lineplot(
        data=data_frame,
        x="x",
        y="y",
        hue="reprojection_error",
        style="labels",
        dashes=False,
        palette=palette,
    )
    fig = plot.get_figure()
    plt.xlabel("")
    plt.ylabel("angle (degrees)")
    plt.legend([], [], frameon=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cmap_norm)
    plt.colorbar(sm, ax=plt.gca())
    fig.savefig(args.video_path.replace(video_path_extension, ".angles.png"))

    if args.render_mode == "all":
        angle_indices = range(config["stages"]["reprojection"]["num_angles"])
    elif args.render_mode == "extreme":
        angle_indices = [
            np.argmin([x["reproject"] for x in reproject_output["metrics"]]),
            np.argmax([x["reproject"] for x in reproject_output["metrics"]]),
        ]
    elif args.render_mode == "none":
        angle_indices = []

    for angle_index in angle_indices:
        index = 0
        frames = []
        radius = 2 * args.target_scale
        angle = angles[angle_index].item()

        # get source video data
        input_video = cv2.VideoCapture(video_filename)
        res = (
            int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        freq = input_video.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = input_video.read()

            if ret:
                try:
                    pred_joints = reproject_output["joints_2d"][angle_index][0, index]
                    gt_joints = reproject_output["joints_2d_gt"][angle_index][0, index]
                    color_index = 0
                    for joints in [gt_joints, pred_joints]:
                        for i in range(45):
                            diff = (res[0] - res[1]) / 2
                            j_x = joints[i, 0] * res[0]
                            j_y = (joints[i, 1] * res[0]) - diff

                            color = [
                                (0, 0, 255),
                                (0, 128, 255),
                                (0, 255, 255),
                            ][color_index]

                            target_res = [res[0] * args.target_scale, res[1] * args.target_scale]
                            frame = cv2.resize(frame, target_res, cv2.INTER_CUBIC)
                            frame = cv2.circle(
                                img=frame,
                                center=(int(j_x * args.target_scale), int(j_y * args.target_scale)),
                                radius=radius,
                                color=color,
                                thickness=-10,
                            )
                        color_index += 1
                except Exception as e:
                    print(e)
                frames.append(frame)
            else:
                break
            index += 1

        input_video.release()

        print(
            np.rad2deg(angle),
            reproject_output["metrics"][angle_index]["chamfer"],
            reproject_output["metrics"][angle_index]["reproject"],
        )

        # write out video animation
        reprojection_video_name = args.video_path.replace(video_path_extension, ".reprojection." + str(angle_index) + video_path_extension)
        if video_path_extension == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            output_video = cv2.VideoWriter(
                reprojection_video_name,
                fourcc,
                freq,
                res,
            )
            for frame in frames:
                output_video.write(frame)
            output_video.release()
        elif video_path_extension == ".gif":
            with imageio.get_writer(reprojection_video_name, mode="I", duration=(1000.0 * (1.0 / freq))) as writer:
                for frame in frames:
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # render camera and SMPL
        mocap_freq = 30
        scene = VideoMocapScene()
        meshes_0 = []

        cam_translation = reproject_output["cam_trans"][angle_index]

        smpl_output = smpl_inference(
            poses=reproject_output["pose_body"][angle_index][0],
            betas=reproject_output["betas"][angle_index][0],
            root_orient=reproject_output["root_orient"][angle_index][0],
            trans=reproject_output["trans"][angle_index][0],
        )

        # camera meshes
        camera_mesh = trimesh.creation.box(extents=[1.0]*3, transform=np.eye(4))
        camera_node = pyrender.Mesh.from_trimesh(camera_mesh)
        camera_node = scene.add(camera_node)

        # load SMPL meshes
        for s_i in range(num_frames):
            sm = trimesh.Trimesh(
                vertices=smpl_output["vertices"].detach().cpu().numpy()[s_i],
                faces=smpl_inference.smpl.faces,
            )
            smpl_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[x/255.0 for x in SMPL_COLORS[0]] + [1.0], alphaMode="OPAQUE")
            meshes_0.append(pyrender.Mesh.from_trimesh(sm, material=smpl_material))

        nodes_0 = scene.add(meshes_0[0])

        # marker meshes
        mm = trimesh.creation.uv_sphere(radius=0.02)
        color = [0, 0, 0]
        marker_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.0, 0.0, 0.0, 1.0), alphaMode="OPAQUE")
        tfs = np.tile(np.eye(4), (points.shape[1], 1, 1))
        tfs[:, :3, 3] = points.detach().cpu().numpy()[0]
        m_mesh = pyrender.Mesh.from_trimesh(mm, poses=tfs, material=marker_material)

        m_node = scene.add(m_mesh)

        def render_frame(frame):
            # update markers
            marker_frame = frame
            nodes_0.mesh = meshes_0[marker_frame]
            camera_node.translation = cam_translation[0, marker_frame].detach().cpu().numpy()

            tfs[:, :3, 3] = points.detach().cpu().numpy()[marker_frame]
            m_node.mesh = pyrender.Mesh.from_trimesh(mm, poses=tfs, material=marker_material)

        renderer = VideoMocapRenderer(
            scene=scene,
            render_frame_fn=render_frame,
            num_frames=num_frames,
            data_freq=mocap_freq,
            video_fps=30,
            video_res=args.video_res,
            video_path=args.video_path.replace(video_path_extension, ".mocap." + str(angle_index) + video_path_extension),
        )


def stack_hmr_data(hmr_data):
    num_frames = len(hmr_data.keys())
    output = {
        "2d_joints": np.zeros((num_frames, 90)),
        "3d_joints": np.zeros((num_frames, 45, 3)),
        "annotations": [],
        "appe": np.zeros((num_frames, 4096)),
        "bbox": np.zeros((num_frames, 4)),
        "camera": np.zeros((num_frames, 3)),
        "camera_bbox": np.zeros((num_frames, 3)),
        "center": np.zeros((num_frames, 2)),
        "class_name": np.zeros((num_frames), dtype=int),
        "conf": np.zeros((num_frames)),
        "extra_data": [],
        "frame_path": [],
        "img_name": [],
        "img_path": [],
        "loca": np.zeros((num_frames, 99)),
        "mask": [],
        "pose": np.zeros((num_frames, 229)),
        "scale": [],
        "shot": np.zeros((num_frames)),
        "size": np.zeros((num_frames, 2)),
        "smpl": {
            "global_orient": np.zeros((num_frames, 1, 3, 3)),
            "betas": np.zeros((num_frames, 10)),
            "body_pose": np.zeros((num_frames, 23, 3, 3)),
        },
        "tid": np.zeros((num_frames)),
        "time": np.zeros((num_frames)),
        "tracked_bbox": np.zeros((num_frames, 4)),
        "tracked_ids": np.zeros((num_frames)),
        "tracked_time": np.zeros((num_frames)),
    }

    frame = 0
    for frame_key in sorted(hmr_data.keys()):
        for key in hmr_data[frame_key].keys():
            if key == "smpl":
                if len(hmr_data[frame_key][key][0]) > 0:
                    output[key]["betas"][frame] = hmr_data[frame_key][key][0]["betas"]
                    output[key]["body_pose"][frame] = hmr_data[frame_key][key][0]["body_pose"]
                    output[key]["global_orient"][frame] = hmr_data[frame_key][key][0]["global_orient"]
            elif isinstance(output[key], list):
                output[key].append(hmr_data[frame_key][key])
            else:
                if isinstance(hmr_data[frame_key][key], int):
                    output[key][frame] = hmr_data[frame_key][key]
                elif len(hmr_data[frame_key][key]) > 0:
                    output[key][frame] = hmr_data[frame_key][key][0]
        frame += 1

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file", required=True)
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.01)
    parser.add_argument("--render_mode", type=str, help="[\"none\", \"extreme\", \"all\"]", default="none")
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--target_scale", type=int, help="target video scale", default=1)
    parser.add_argument("--video_path", type=str, help="video path", required=True)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    parser.add_argument("--zoom_level", type=float, help="zoom path", default=1.0)
    args = parser.parse_args()

    visualize(args)
