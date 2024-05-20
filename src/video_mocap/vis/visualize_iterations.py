"""
This script is mostly for visualizing intermediate outputs. This is useful for
creating diagrams and instructional videos.
"""
import argparse
import imageio
import joblib
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyrender
import seaborn as sns
import trimesh
import torch

from video_mocap.datasets.dataset_utils import get_overlay_font
from video_mocap.utils.smpl import SmplInference
from video_mocap.vis.renderer import VideoMocapRenderer
from video_mocap.vis.scene import VideoMocapScene, create_smpl_meshes, extract_part_vertices


DATA_FREQ = 30
FONT_FACE = cv2.FONT_HERSHEY_PLAIN
FONT_SIZE = 16
FONT_SCALE = 2
FONT_COLOR = (0, 0, 0)
FONT_POS = (10, 25)
FONT_SCALE_VIDEO = 3
FONT_COLOR_VIDEO = (255, 255, 255)
FONT_POS_VIDEO = (20, 50)
VIDEO_RES = (2560, 1920)


def vis_iterations(
    filename,
    iteration_time,
    extension,
    show_text=False,
    angle=0,
):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))

    data = joblib.load(filename)

    dirname = os.path.splitext(filename)[0] + "___" + str(iteration_time).replace(".", "_")
    os.makedirs(dirname, exist_ok=True)

    # render input
    render_video(
        os.path.join(dirname, "video." + extension),
        data["input"]["video"],
    )

    # render HMR output
    render_video(
        os.path.join(dirname, "video_hmr." + extension),
        data["input"]["video_hmr"],
    )

    render_markers(
        os.path.join(dirname, "markers." + extension),
        data["input"]["markers"],
        iteration_time,
        show_text=show_text,
        device=device,
    )

    # render reprojection
    if "reprojection" in data:
        os.makedirs(os.path.join(dirname, "reprojection"), exist_ok=True)
        render_reprojection_plot(
            os.path.join(dirname, "reprojection", "plot.png"),
            data["reprojection_output"],
        )

        font = get_overlay_font(data["meta"]["dataset"])
        for angle in data["reprojection"]:
            render_reprojection_video(
                os.path.join(dirname, "reprojection", "video_" + str(angle) + "." + extension),
                data["input"]["video"],
                data["reprojection"][angle],
                iteration_time,
                show_text=show_text,
                font_color=font["color"],
                font_scale=font["scale"],
                font_pos=font["pos"],
            )
            render_stage(
                os.path.join(dirname, "reprojection", "smpl_" + str(angle) + "." + extension),
                data["input"]["markers"],
                data["reprojection"][angle],
                iteration_time,
                show_text=show_text,
                device=device,
            )

    # render parts
    os.makedirs(os.path.join(dirname, "parts"), exist_ok=True)
    for part in data["part"]:
        render_part(
            os.path.join(dirname, "parts", part + "." + extension),
            data["part"][part][0]["markers"],
            data["part"][part],
            data["part"][part][0]["part_joints"],
            iteration_time,
            show_text=show_text,
            device=device,
        )

    # render stages
    for rotation in data["chamfer_0"].keys():
        rotation_str = str(round(np.rad2deg(rotation)))
        render_stage(
            os.path.join(dirname, "chamfer_0_" + rotation_str + "." + extension),
            data["input"]["markers"],
            data["chamfer_0"][rotation],
            iteration_time,
            show_text=show_text,
            device=device,
        )
        render_stage(
            os.path.join(dirname, "marker_0_" + rotation_str + "." + extension),
            data["input"]["markers"],
            data["marker_0"][rotation],
            iteration_time,
            show_text=show_text,
            device=device,
        )

    render_stage(
        os.path.join(dirname, "marker_1." + extension),
        data["input"]["markers"],
        data["marker_1"][0],
        iteration_time,
        show_text=show_text,
        device=device,
    )



def render_part(
    filename: str,
    markers: np.ndarray,
    smpl_data: Dict,
    part_joints: np.ndarray,
    iteration_time: float,
    show_text: bool=False,
    device=torch.device("cpu"),
):
    scene = VideoMocapScene()
    smpl_inference = SmplInference(device)

    # create markers
    num_markers = markers.shape[1]

    marker_trimesh = trimesh.creation.uv_sphere(radius=0.02)
    marker_trimesh.visual.vertex_colors = [0.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (num_markers, 1, 1))
    tfs[:, :3, 3] = markers[0]
    marker_mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)
    marker_node = scene.add(marker_mesh)

    # create SMPL meshes
    iteration_indices = [len(smpl_data.keys())-1]
    if iteration_time > 0.0:
        iteration_indices = np.linspace(0, 1.0, round(DATA_FREQ * iteration_time)) ** 3
        iteration_indices = np.round(iteration_indices * (len(smpl_data.keys())-1)).tolist()

    smpl_points_iterations = []
    for iteration in iteration_indices:
        smpl_points_iterations.append(extract_part_vertices(
            pose_body=smpl_data[iteration]["pose_body"],
            root_orient=smpl_data[iteration]["root_orient"],
            trans=smpl_data[iteration]["trans"],
            betas=smpl_data[iteration]["betas"],
            smpl_inference=smpl_inference,
            part_joints=part_joints,
        ))

    num_points = smpl_points_iterations[0].shape[1]
    point_trimesh = trimesh.creation.uv_sphere(radius=0.01)
    point_trimesh.visual.vertex_colors = [0.0, 0.7, 0.0]
    tfs_points = np.tile(np.eye(4), (num_points, 1, 1))
    tfs_points[:, :3, 3] = smpl_points_iterations[0][0]
    points_mesh = pyrender.Mesh.from_trimesh(point_trimesh, poses=tfs_points)
    points_node = scene.add(points_mesh)

    num_frames = smpl_data[0]["trans"].shape[0]

    def render_part_fn(frame):
        nonlocal smpl_points_iterations
        nonlocal points_node
        nonlocal marker_node

        # render markers
        tfs = np.tile(np.eye(4), (markers.shape[1], 1, 1))
        tfs[:, :3, 3] = markers[frame]
        marker_node.mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)

        # render SMPL
        if frame < DATA_FREQ * iteration_time:
            tfs_points = np.tile(np.eye(4), (num_points, 1, 1))
            tfs_points[:, :3, 3] = smpl_points_iterations[frame][frame]
            points_node.mesh = pyrender.Mesh.from_trimesh(point_trimesh, poses=tfs_points)
        else:
            tfs_points = np.tile(np.eye(4), (num_points, 1, 1))
            tfs_points[:, :3, 3] = smpl_points_iterations[-1][frame]
            points_node.mesh = pyrender.Mesh.from_trimesh(point_trimesh, poses=tfs_points)

    def image_postprocess_fn(frame, image):
        iteration = iteration_indices[-1]
        if frame < DATA_FREQ * iteration_time:
            iteration = iteration_indices[frame]

        image = np.ascontiguousarray(image)
        image = cv2.putText(
            img=image,
            text="Iter: " + str(round(iteration+1)),
            org=FONT_POS,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=FONT_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return image

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_part_fn,
        num_frames=num_frames,
        data_freq=DATA_FREQ,
        video_fps=DATA_FREQ,
        video_res=VIDEO_RES,
        video_path=filename,
        image_postprocess_fn=image_postprocess_fn if show_text else None,
    )


def render_stage(
    filename: str,
    markers: np.ndarray,
    smpl_data: Dict,
    iteration_time: float,
    show_text: bool=False,
    device=torch.device("cpu"),
):
    scene = VideoMocapScene()

    smpl_meshes_iterations = []
    smpl_inference = SmplInference()

    # create markers
    num_markers = markers.shape[1]

    marker_trimesh = trimesh.creation.uv_sphere(radius=0.02)
    marker_trimesh.visual.vertex_colors = [0.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (num_markers, 1, 1))
    tfs[:, :3, 3] = markers[0]
    marker_mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)
    marker_node = scene.add(marker_mesh)

    # create SMPL meshes
    iteration_indices = [len(smpl_data.keys())-1]
    if iteration_time > 0.0:
        iteration_indices = np.linspace(0, 1.0, round(DATA_FREQ * iteration_time)) ** 3
        iteration_indices = np.round(iteration_indices * (len(smpl_data.keys())-1)).tolist()

    for iteration in iteration_indices:
        smpl_meshes_iterations.append(create_smpl_meshes(
            pose_body=smpl_data[iteration]["pose_body"],
            root_orient=smpl_data[iteration]["root_orient"],
            trans=smpl_data[iteration]["trans"],
            betas=smpl_data[iteration]["betas"],
            smpl_inference=smpl_inference,
        ))
    mesh_node = scene.add(smpl_meshes_iterations[0][0])

    num_frames = smpl_data[0]["trans"].shape[0]

    def render_stage_fn(frame):
        nonlocal smpl_meshes_iterations
        nonlocal mesh_node

        # render markers
        tfs = np.tile(np.eye(4), (markers.shape[1], 1, 1))
        tfs[:, :3, 3] = markers[frame]
        marker_node.mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)

        # render SMPL
        if frame < DATA_FREQ * iteration_time:
            mesh_node.mesh = smpl_meshes_iterations[frame][frame]
        else:
            mesh_node.mesh = smpl_meshes_iterations[-1][frame]

    def image_postprocess_fn(frame, image):
        iteration = iteration_indices[-1]
        if frame < DATA_FREQ * iteration_time:
            iteration = iteration_indices[frame]

        image = np.ascontiguousarray(image)
        image = cv2.putText(
            img=image,
            text="Iter: " + str(round(iteration+1)),
            org=FONT_POS,
            fontFace=FONT_FACE,
            fontScale=FONT_SCALE,
            color=FONT_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        return image

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_stage_fn,
        num_frames=num_frames,
        data_freq=DATA_FREQ,
        video_fps=DATA_FREQ,
        video_res=VIDEO_RES,
        video_path=filename,
        image_postprocess_fn=image_postprocess_fn if show_text else None,
    )



def render_markers(
    filename: str,
    markers: np.ndarray,
    iteration_time: float,
    show_text: bool=False,
    device=torch.device("cpu"),
):
    scene = VideoMocapScene()

    # create markers
    num_markers = markers.shape[1]

    marker_trimesh = trimesh.creation.uv_sphere(radius=0.02)
    marker_trimesh.visual.vertex_colors = [0.0, 0.0, 0.0]
    tfs = np.tile(np.eye(4), (num_markers, 1, 1))
    tfs[:, :3, 3] = markers[0]
    marker_mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)
    marker_node = scene.add(marker_mesh)

    num_frames = markers.shape[0]

    def render_marker_fn(frame):
        # render markers
        tfs = np.tile(np.eye(4), (markers.shape[1], 1, 1))
        tfs[:, :3, 3] = markers[frame]
        marker_node.mesh = pyrender.Mesh.from_trimesh(marker_trimesh, poses=tfs)

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_marker_fn,
        num_frames=num_frames,
        data_freq=DATA_FREQ,
        video_fps=DATA_FREQ,
        video_res=VIDEO_RES,
        video_path=filename,
        image_postprocess_fn=None,
    )


def render_reprojection_video(
    filename: str,
    video: List,
    smpl_data: Dict,
    iteration_time: float,
    show_text: bool=False,
    font_color: Tuple=FONT_COLOR_VIDEO,
    font_scale: Tuple=FONT_SCALE_VIDEO,
    font_pos: Tuple=FONT_POS_VIDEO,
):
    height, width, _ = video[0].shape
    res = (width, height)
    freq = DATA_FREQ

    iteration_indices = [len(smpl_data.keys())-1]
    if iteration_time > 0.0:
        iteration_indices = np.linspace(0, 1.0, round(DATA_FREQ * iteration_time)) ** 3
        iteration_indices = np.round(iteration_indices * (len(smpl_data.keys())-1)).tolist()

    frames_video = []
    for frame in range(len(video)):
        iteration = iteration_indices[-1]
        if frame < DATA_FREQ * iteration_time:
            iteration = iteration_indices[frame]

        pred_joints = smpl_data[iteration]["pred_2d_joints"][frame]
        gt_joints = smpl_data[iteration]["gt_2d_joints"][frame]

        target_scale = 1.0
        target_res = [res[0] * target_scale, res[1] * target_scale]
        target_res = [int(x) for x in target_res]
        frame_video = np.ascontiguousarray(video[frame])
        frame_video = cv2.resize(frame_video, target_res, cv2.INTER_CUBIC)

        color_index = 0
        for joints in [gt_joints, pred_joints]:
            for i in range(45):
                diff = (res[0] - res[1]) / 2
                j_x = joints[i, 0] * res[0]
                j_y = (joints[i, 1] * res[0]) - diff

                color = [
                    (0, 0, 255),
                    (0, 128, 255),
                ][color_index]

                frame_video = cv2.circle(
                    img=frame_video,
                    center=(int(j_x * target_scale), int(j_y * target_scale)),
                    radius=3,
                    color=color,
                    thickness=-10,
                )
            color_index += 1

        if show_text:
            frame_video = cv2.putText(
                img=frame_video,
                text="Iter: " + str(round(iteration+1)),
                org=font_pos,
                fontFace=FONT_FACE,
                fontScale=font_scale,
                color=font_color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        frames_video.append(cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB))

    extension = os.path.splitext(filename)[1][1:]
    video_sequence = os.path.basename(os.path.splitext(filename)[0])

    num_frames = len(video)
    if extension == "mp4":
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter(filename, fourcc, freq, res)
        for frame_video in frames_video:
            frame_video = np.flip(frame_video, axis=2)
            output_video.write(frame_video)
        cap.release()
        output_video.release()
    elif extension == "gif":
        # write out video
        with imageio.get_writer(filename, mode="I", duration=(1000.0 * (1.0 / DATA_FREQ))) as writer:
            for frame in range(num_frames):
                writer.append_data(frames_video[frame])
    elif extension == "png":
        output_dir = os.path.join(os.path.dirname(filename), video_sequence)
        os.makedirs(output_dir, exist_ok=True)
        for frame in range(num_frames):
            frame_video = frames_video[frame]
            cv2.imwrite(os.path.join(output_dir, str(frame).zfill(8) + ".png"), cv2.cvtColor(frame_video, cv2.COLOR_RGB2BGR))


def render_video(
    filename: str,
    video: List,
):
    height, width, _ = video[0].shape
    res = (width, height)
    freq = DATA_FREQ

    frames_video = []
    for frame in range(len(video)):
        target_scale = 1.0
        target_res = [res[0] * target_scale, res[1] * target_scale]
        target_res = [int(x) for x in target_res]
        frame_video = np.ascontiguousarray(video[frame])
        frame_video = cv2.resize(frame_video, target_res, cv2.INTER_CUBIC)
        frames_video.append(cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB))

    extension = os.path.splitext(filename)[1][1:]
    video_sequence = os.path.basename(os.path.splitext(filename)[0])

    num_frames = len(video)
    if extension == "mp4":
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter(filename, fourcc, freq, res)
        for frame_video in frames_video:
            frame_video = np.flip(frame_video, axis=2)
            output_video.write(frame_video)
        cap.release()
        output_video.release()
    elif extension == "gif":
        # write out video
        with imageio.get_writer(filename, mode="I", duration=(1000.0 * (1.0 / DATA_FREQ))) as writer:
            for frame in range(num_frames):
                writer.append_data(frames_video[frame])
    elif extension == "png":
        output_dir = os.path.join(os.path.dirname(filename), video_sequence)
        os.makedirs(output_dir, exist_ok=True)
        for frame in range(num_frames):
            frame_video = frames_video[frame]
            cv2.imwrite(os.path.join(output_dir, str(frame).zfill(8) + ".png"), cv2.cvtColor(frame_video, cv2.COLOR_RGB2BGR))


def render_reprojection_plot(
    filename: str,
    smpl_data_reprojection: Dict,
):
    x_axis = []
    y_axis = []
    labels = []
    reprojection_error = []

    for angle_index in range(len(smpl_data_reprojection["input_angle"])):
        angle = smpl_data_reprojection["input_angle"][angle_index]
        pred_angle = smpl_data_reprojection["output_angle"][angle_index]

        angle_deg = np.rad2deg(angle) % 360
        x_axis.append("initial angle")
        y_axis.append(angle_deg)
        labels.append(angle_deg)
        reprojection_error.append(smpl_data_reprojection["metrics"][angle_index]["reproject"])

        pred_angle_deg = np.rad2deg(pred_angle) % 360
        x_axis.append("optimized angle")
        y_axis.append(pred_angle_deg)
        labels.append(angle_deg)
        reprojection_error.append(smpl_data_reprojection["metrics"][angle_index]["reproject"])

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
    fig.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0.0)
    parser.add_argument("--extension", type=str, help="extension", default="gif")
    parser.add_argument("--filename", type=str, help="input .pkl filename", required=True)
    parser.add_argument("--gpu", type=int, help="GPU ID", default=0)
    parser.add_argument("--iteration_time", type=float, help="seconds until convergence", default=0.0)
    parser.add_argument("--show_text", action="store_true", help="show text on top of video")
    args = parser.parse_args()

    vis_iterations(
        args.filename,
        args.iteration_time,
        args.extension,
        args.show_text,
        args.angle,
    )
