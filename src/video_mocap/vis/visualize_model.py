import argparse
import os

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyrender
from scipy.spatial.transform import Rotation
import torch
import trimesh

from video_mocap.datasets.dataset_utils import get_camera_name
from video_mocap.img_smpl.img_smpl import ImgSmpl
from video_mocap.markers.markers import Markers
from video_mocap.markers.markers_noise import markers_tracking_loss, markers_swap
from video_mocap.markers.markers_utils import cleanup_markers
from video_mocap.multimodal import multimodal_video_mocap
from video_mocap.utils.colors import colors_perceptually_distinct_24
from video_mocap.utils.config import load_config
from video_mocap.utils.mesh import cull_parts
from video_mocap.utils.smpl import SmplInference
from video_mocap.utils.smpl_utils import get_joint_colors_vertices, get_joint_name
from video_mocap.vis.renderer import VideoMocapRenderer
from video_mocap.vis.scene import SMPL_COLORS, VideoMocapScene


def visualize(args):
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda:" + str(args.gpu))

    config = load_config(args.config)

    camera = get_camera_name(args.dataset)

    mocap_dir = "mocap"
    if args.synthetic:
        mocap_dir = "mocap_synthetic"
    elif args.description is not None:
        mocap_dir = mocap_dir + "_parts___" + args.description

    markers_filename = args.sequence + ".c3d"

    markers_filename = os.path.join(
        args.input_dir,
        args.dataset,
        mocap_dir,
        args.subject,
        markers_filename,
    )

    file_extensions = [".avi", ".mp4", ".mpg"]

    video_sequence_name = args.sequence
    if camera is not None:
        video_sequence_name = video_sequence_name + "." + camera

    for file_extension in file_extensions:
        smpl_video_filename = os.path.join(
            args.input_dir,
            args.dataset,
            "videos",
            args.subject,
            video_sequence_name + file_extension,
        )
        if os.path.exists(smpl_video_filename):
            break

    smpl_video_hmr_filename = os.path.join(
        args.input_dir,
        args.dataset,
        "comparisons",
        "4d_humans",
        args.subject,
        video_sequence_name,
        "PHALP_" + args.sequence + ".mp4",
    )

    smpl_img_filename = os.path.join(
        args.input_dir,
        args.dataset,
        "comparisons",
        "4d_humans",
        args.subject,
        video_sequence_name,
        "results",
        "demo_" + args.sequence + ".pkl",
    )

    # get source video data
    video = cv2.VideoCapture(smpl_video_filename)
    video_freq = video.get(cv2.CAP_PROP_FPS)

    video_frames = []
    while True:
        ret, video_frame = video.read()
        if ret:
            video_frames.append(video_frame)
        else:
            break
    video.release()

    # get HMR source video data
    video_hmr = cv2.VideoCapture(smpl_video_hmr_filename)

    video_frames_hmr = []
    while True:
        ret, video_frame = video_hmr.read()
        if ret:
            video_frames_hmr.append(video_frame)
        else:
            break
    video_hmr.release()

    # SMPL data from monocular
    smpl_img_data = joblib.load(smpl_img_filename)
    img_smpl = ImgSmpl(smpl_img_data, video_freq)
    smpl_inference = SmplInference()

    markers = Markers(markers_filename)

    points = markers.get_points()
    points = np.nan_to_num(points, 0)
    points = cleanup_markers(points)
    
    if args.marker_swap:
        points = markers_swap(points, min_frames=10, max_frames=10, distance_threshold=0.1, p=0.1)

    if args.marker_tracking_loss:
        if args.seed is not None:
            np.random.seed(args.seed)
        points = markers_tracking_loss(points, min_frames=10, max_frames=10, p=args.tracking_loss_prob)

    markers.set_points(points)
    num_markers = points.shape[1]
    mocap_freq = markers.get_frequency()

    lbs_weights = smpl_inference.get_lbs_weights()
    vertex_ids = torch.argmax(lbs_weights, dim=-1)
    vertex_colors = get_joint_colors_vertices(vertex_ids)

    def get_marker_colors(i):
        num_colors = len(colors_perceptually_distinct_24)
        return colors_perceptually_distinct_24[i % num_colors]

    # create img smpl meshes
    with torch.no_grad():
        output_frame = smpl_inference(
            poses = img_smpl.pose_body,
            root_orient = img_smpl.root_orient,
            trans = img_smpl.trans,
            betas = img_smpl.betas,
        )
        img_verts = output_frame["vertices"].detach().cpu().numpy()

    if args.show_img_smpl and not args.hide_smpl:
        smpl_img_meshes = []
        for i in range(len(smpl_img_data)):
            tm = trimesh.Trimesh(
                vertices=img_verts[i],
                faces=smpl_inference.smpl.faces,
            )
            tm.visual.vertex_colors = vertex_colors
            smpl_img_meshes.append(pyrender.Mesh.from_trimesh(tm))
        smpl_img_node = smpl_img_meshes[0]

    # create multimodal smpl meshes
    multimodal_smpl = multimodal_video_mocap(
        img_smpl,
        markers,
        device,
        config=config,
        offset=args.offset,
        print_options=args.print_options,
        save_iterations=args.save_iterations,
        visualize_fits=args.visualize_fits,
    )
    markers = multimodal_smpl["mocap_markers"]
    num_frames = points.shape[0]

    if args.save_iterations:
        multimodal_smpl["iterations"]["input"]["video"] = video_frames
        multimodal_smpl["iterations"]["input"]["video_hmr"] = video_frames_hmr
        multimodal_smpl["iterations"]["meta"] = {
            "dataset": args.dataset,
            "subject": args.subject,
            "sequence": args.sequence,
        }

        joblib.dump(
            multimodal_smpl["iterations"],
            "results/visualize_model___" + args.subject + "___" + args.sequence + ".pkl",
        )

    # plot markers
    #plot_marker_classification(multimodal_smpl["markers_labels"][0])
    if args.plot_classification:
        joint_ids, joint_counts = np.unique(multimodal_smpl["markers_labels"], return_counts=True)
        indices = (-joint_counts).argsort()  # sort in descending order
        joint_ids = joint_ids[indices]
        joint_counts = joint_counts[indices]
        joint_names = [get_joint_name(joint_id) for joint_id in joint_ids]
        joint_percentages = [(joint_count / sum(joint_counts)) * 100 for joint_count in joint_counts]
        print("Markers labels:")
        for i in range(joint_ids.shape[0]):
            print_string = [get_joint_name(joint_ids[i]), str(joint_counts[i])]
            print(f"\t{print_string[0]:<20}{print_string[1]:<20}")

        if args.video_path is not None:
            bar_graph = plt.bar(joint_names, joint_percentages)
            plt.title("Percentage of predicted labels")
            index = 0
            for bar in bar_graph:
                plt.text(
                    bar.get_xy()[0] + bar.get_width() / 2,
                    bar.get_xy()[1] + bar.get_height() * 1.01,
                    str(round(joint_percentages[index], 1)) + "%",
                    ha="center",
                )
                index += 1
            plt.savefig(os.path.splitext(args.video_path)[0] + "_joint_labels.png")

    # create marker meshes
    if args.marker_segment:
        markers_labels = multimodal_smpl["markers_labels"]  # [F, M]
        m_meshes = []
        for m_i in range(num_markers):
            sm = trimesh.creation.uv_sphere(radius=args.marker_size)
            sm.visual.vertex_colors = get_marker_colors(m_i)

            tfs = np.expand_dims(np.eye(4), axis=0)
            m_meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))
    else:
        sm = trimesh.creation.uv_sphere(radius=args.marker_size)
        sm.visual.vertex_colors = [0.0, 0.0, 0.0]

        tfs = np.tile(np.eye(4), (num_markers, 1, 1))
        tfs[:, :3, 3] = points[0]
        m_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)


    with torch.no_grad():
        output_frame = smpl_inference(
            poses = multimodal_smpl["pose_body"],
            root_orient = multimodal_smpl["root_orient"],
            trans = multimodal_smpl["trans"],
            betas = multimodal_smpl["betas"],
        )
        multimodal_verts = output_frame["vertices"].detach().cpu().numpy()

    # create multimodal meshes
    if not args.hide_smpl:
        multimodal_meshes = []
        for i in range(multimodal_smpl["trans"].shape[0]):
            tm = trimesh.Trimesh(
                vertices=multimodal_verts[i],
                faces=smpl_inference.smpl.faces,
            )
            # show part colors:
            if args.show_parts_colors:
                tm.visual.vertex_colors = vertex_colors
            else:
                if args.translucent:
                    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(SMPL_COLORS[0][0]/255.0, SMPL_COLORS[0][1]/255.0, SMPL_COLORS[0][2]/255.0, 0.8), alphaMode="BLEND")
                else:
                    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(SMPL_COLORS[0][0]/255.0, SMPL_COLORS[0][1]/255.0, SMPL_COLORS[0][2]/255.0, 1.0), alphaMode="OPAQUE")

            # cull parts
            if args.show_parts_only:
                tm = cull_parts(
                    tm,
                    multimodal_smpl["chain"],
                    vertex_ids.detach().cpu().numpy(),
                )

            if args.show_parts_colors:
                multimodal_meshes.append(pyrender.Mesh.from_trimesh(tm))
            else:
                multimodal_meshes.append(pyrender.Mesh.from_trimesh(tm, material=material))

        multimodal_node = multimodal_meshes[0]

    # visualize various features
    if args.visualize_root_paths:
        # plot img_smpl
        img_trans_xy = img_smpl.trans
        plt.scatter(
            img_trans_xy[:, 0],
            img_trans_xy[:, 1],
            c=np.arange(img_trans_xy.shape[0]),
        )
        plt.title("SMPL root translation from images")
        plt.xlabel("x-axis (m)")
        plt.ylabel("y-axis (m)")
        plt.savefig("root_trans_img_smpl.png")
        plt.close()

        # plot multimodal_smpl
        multimodal_trans_xy = multimodal_smpl["trans"]
        plt.scatter(
            multimodal_trans_xy[:, 0],
            multimodal_trans_xy[:, 1],
            c=np.arange(multimodal_trans_xy.shape[0]),
        )
        plt.title("SMPL root translation from markers")
        plt.xlabel("x-axis (m)")
        plt.ylabel("y-axis (m)")
        plt.savefig("root_trans_img_multimodal.png")
        plt.close()

    scene = VideoMocapScene()

    if args.marker_segment:
        m_nodes = []
        for m_mesh in m_meshes:
            m_nodes.append(scene.add(m_mesh))
    else:
        m_node = scene.add(m_mesh)

    if not args.hide_smpl:
        multimodal_node = scene.add(multimodal_node)
    if args.show_img_smpl and not args.hide_smpl:
        smpl_img_node = scene.add(smpl_img_node)

    def render_frame(frame):
        # update markers
        marker_frame = frame

        nonlocal m_node

        if args.marker_segment:
            m_i = 0
            for m_node in m_nodes:
                m_node.mesh.primitives[0].color_0 = get_marker_colors(m_i)

                m_node.translation = points[marker_frame, m_i]
                m_i += 1
        else:
            tfs[:, :3, 3] = points[marker_frame]
            m_node.mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        # smpl img
        if args.show_img_smpl and not args.hide_smpl:
            smpl_img_frame = min(len(smpl_img_meshes)-1, int(frame * (video_freq / mocap_freq)))
            smpl_img_node.mesh = smpl_img_meshes[smpl_img_frame]

        # multimodal
        if not args.hide_smpl:
            multimodal_frame = min(len(multimodal_meshes)-1, int(frame * (video_freq / mocap_freq)))
            multimodal_node.mesh = multimodal_meshes[frame]

    scene.directional_light.matrix = np.array([
        [0.75, 0.5, 0.433, 0],
        [-0.433, 0.866, -0.2499, 0],
        [-0.5, 0.0, 0.866, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    camera_matrix = [
        [0.750, -0.252, 0.611, 3.4],
        [0.661, 0.291, -0.691, -2.0],
        [-0.004, 0.923, 0.385, 2.3],
        [0, 0, 0, 1],
    ]

    angle = 0.0
    if angle != 0.0:
        rot = Rotation.from_rotvec(np.array([0, 0, -angle]), degrees=True).as_matrix()
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot
        camera_matrix = rot_4x4 @ camera_matrix
        light_matrix = scene.directional_light.matrix
        light_matrix = rot_4x4 @ light_matrix
        scene.directional_light.matrix = light_matrix

    quality_mode = "normal"
    if args.video_path and (args.video_path.endswith(".png") or args.video_path.endswith(".mp4")):
        quality_mode = "ultra"

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_frame,
        num_frames=num_frames,
        data_freq=mocap_freq,
        video_fps=args.video_fps,
        video_res=args.video_res,
        video_path=args.video_path,
        quality_mode=quality_mode,
        camera_matrix=camera_matrix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="configuration file", required=True)
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--description", type=str, help="layout for .c3d file name", default=None)
    parser.add_argument("--export_smpl", type=str, help="export SMPL filename", default=None)
    parser.add_argument("--gpu", type=int, help="GPU ID")
    parser.add_argument("--hide_smpl", action="store_true", help="hide smpl meshes")
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--marker_segment", action="store_true", help="marker segmentation")
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.01)
    parser.add_argument("--marker_swap", action="store_true", help="swap markers")
    parser.add_argument("--marker_tracking_loss", action="store_true", help="tracking loss for markers")
    parser.add_argument("--num_markers", type=int, help="number of markers", default=43)
    parser.add_argument("--offset", type=int, help="video offset", default=None)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--plot_classification", action="store_true", help="plot marker classification")
    parser.add_argument("--print_options", type=str, nargs="*", default=["loss", "progress"])
    parser.add_argument("--save_iterations", action="store_true", help="save iterations for optimization")
    parser.add_argument("--seed", type=int, help="seed", default=None)
    parser.add_argument("--show_img_smpl", action="store_true", help="show image SMPL")
    parser.add_argument("--show_parts_colors", action="store_true", help="show part colors on SMPL mesh")
    parser.add_argument("--show_parts_only", action="store_true", help="show only parts")
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data")
    parser.add_argument("--tracking_loss_prob", type=float, help="tracking loss probability", default=0.1)
    parser.add_argument("--translucent", action="store_true", help="translucent")
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    parser.add_argument("--visualize_fits", action="store_true", help="visualize part fits")
    parser.add_argument("--visualize_root_paths", action="store_true", help="visualize paths")
    args = parser.parse_args()

    markers = visualize(args)
