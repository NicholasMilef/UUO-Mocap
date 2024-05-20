import argparse
import os
import time

import cv2
import imageio
import numpy as np
import pyrender
from pytorch3d.transforms import axis_angle_to_matrix
import torch
from torch.utils.data import DataLoader
import trimesh

from video_mocap.datasets.dataset_mocap import DatasetMocap, apply_random_rotation_to_pos, apply_random_rotation_to_rot, apply_random_translation_to_pos
from video_mocap.markers.markers_utils import find_best_part_fits
from video_mocap.models.marker_segmenter_multimodal import MarkerSegmenterMultimodal
from video_mocap.train.train_marker_segmenter_multimodal import add_noise_shape
from video_mocap.utils.random_utils import set_random_seed
from video_mocap.utils.smpl import SmplInference, SmplInferenceGender
from video_mocap.utils.smpl_utils import get_joint_colors, get_joint_name, smpl_limbs, get_all_joint_ids
from video_mocap.utils.tensor import dict2device
from video_mocap.vis.visualize_part import visualize_part


def visualize(args):
    if args.seed is not None:
        set_random_seed(args.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda:0")

    # dataset setup
    if args.limb is None:
        parts_set = [[get_all_joint_ids(), 1.0]]
    else:
        parts_set = [[smpl_limbs[args.limb], 1.0]]
    filename = os.path.join("data/processed/SMPL_H_G", args.dataset, args.subject, args.sequence + "_poses.npz")
    dataset = DatasetMocap(
        batch_size=1,
        num_markers=args.num_markers,
        sequence_length=-1,
        stride=4,
        parts_set=parts_set,
        filename=filename,
        seed=args.seed,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    # model segmentation
    smpl_inference = SmplInferenceGender(device)

    num_joints = 24

    model = MarkerSegmenterMultimodal(
        num_parts = num_joints,
        latent_dim = 64,
        sequence_length = 32,
        modalities = ["markers"],
    ).to(device)
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_15_s4_40m_rot_fast_new/model155.pth"
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_14_s4_40m_bn_rot_fast/model.pth"
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_15_s4_40m_rot_fast_l32/model10.pth"
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_16_s4_40m_rot_fast_l256/model.pth"
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_16_s1_40m_rot_fast_l32/model600.pth"
    #marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_17_s4_40m_rot_fast_l128_hard/model.pth"
    marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_26_s4_40m_rot_fast_l64_hard_mp_full/model.pth"
    model.load_state_dict(
        torch.load(marker_segmenter_filename, map_location=device),
    )

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            model.eval()

            data = dict2device(data, device)

            angle = 0*torch.from_numpy((1.0 - np.random.rand(data["trans"].shape[0])) * np.pi * 2).to(device)
            root_orient = apply_random_rotation_to_rot(data["poses"][:, :, :3], angle)
            trans = apply_random_rotation_to_pos(data["trans"][:, :, :3], angle)
            trans = apply_random_translation_to_pos(trans, std=0.0, center=True)
            betas = add_noise_shape(data["betas"], 0)

            smpl_markers_output = smpl_inference(
                poses=data["poses"][:, :, 3:72],
                betas=data["betas"],
                root_orient=root_orient,
                trans=trans,
                gender_one_hot=data["gender_one_hot"],
                pose2rot=True,
                compute_part_labels=True,
            )

            smpl_img_output = smpl_inference(
                poses=data["poses"][:, :, 3:72],
                betas=betas,
                root_orient=data["poses"][:, :, :3],
                trans=data["trans"][:, :, :3] * 0,  # zero out translation since it's unreliable in monocular video
                gender_one_hot=data["gender_one_hot"],
                pose2rot=True,
                compute_part_labels=True,
            )

            markers = dataset.compute_markers(
                smpl_markers_output["vertices"],
                smpl_markers_output["vertex_part_labels"],
            )

            segmented_markers = model.forward_sequence(
                markers["marker_pos"],
                smpl_img_output["joints"][:, :, :24, :],
                stride=1,
                center=True
            )  # [1, M, P]

    segmented_markers = torch.topk(segmented_markers, dim=-1, k=2, sorted=True)[1][0]  # [F, M, 2]
    marker_labels = segmented_markers[..., 0].detach().cpu().numpy()
    """
    filter_output = filter_rigid_with_video(
        markers=markers["marker_pos"][0].detach().cpu().numpy(),
        labels=marker_labels,
        joints_video=smpl_img_output["joints"][0, :, :24, :].detach().cpu().numpy(),
        hierarchy=smpl_inference.smpls["male"].parents.detach().cpu().numpy(),
        device=device,
    )
    """
    smpl_inference_optim = SmplInference(device)

    num_frames = data["poses"].shape[1]
    video_pose_body = axis_angle_to_matrix(torch.reshape(data["poses"][0, :, 3:72], (num_frames, -1, 3)))
    video_betas = torch.repeat_interleave(data["betas"], repeats=num_frames, dim=0)
    video_root_orient = axis_angle_to_matrix(torch.reshape(data["poses"][0, :, 0:3], (num_frames, -1, 3)))

    def visualize_parts(subtree, markers_subset, vertices_subset):
        visualize_part(
            os.path.join("render_output", "_".join([str(x) for x in subtree]) + ".gif"),
            markers_subset.detach().cpu().numpy(),
            vertices_subset.detach().cpu().numpy(),
        )

    filter_output = {}
    if args.apply_fitting:
        filter_output = find_best_part_fits(
            markers=markers["marker_pos"][0],
            pose_body=video_pose_body,
            betas=video_betas,
            root_orient=video_root_orient,
            marker_labels=torch.from_numpy(marker_labels).to(device),
            smpl_inference=smpl_inference_optim,
            hierarchy=smpl_inference_optim.smpl.parents,
            visualize_fn=visualize_parts if args.visualize_parts else None,
        )
    else:
        filter_output["labels"] = torch.from_numpy(marker_labels).to(device)

    segments = None
    endpoints = None
    
    if "segments" in filter_output.keys():
        segments = filter_output["segments"]
    if "marker_labels" in filter_output.keys():
        marker_labels = filter_output["marker_labels"]
    if "endpoints" in filter_output.keys():
        endpoints = filter_output["endpoints"]

    points = markers["marker_pos"].detach().cpu().numpy()[0]
    num_markers = points.shape[1]
    num_frames = points.shape[0]

    for i in range(segmented_markers.shape[-1]):
        label_names, label_counts = torch.unique(segmented_markers[..., i], return_counts=True)
        label_counts = [(get_joint_name(int(label_names[i])), int(label_counts[i])) for i in range(len(label_names))]
        label_counts = sorted(label_counts, key=lambda x: -x[1])
        print("Top", str(i + 1))
        for name, count in label_counts:
            print("\t", name, count)

    def get_color(i):
        return get_joint_colors(i)

    m_meshes = []
    for m_i in range(num_markers):
        sm = trimesh.creation.uv_sphere(radius=args.marker_size)
        color = get_color(marker_labels[0, m_i])
        sm.visual.vertex_colors = color

        tfs = np.expand_dims(np.eye(4), axis=0)
        m_meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    if segments is not None:
        bone_meshes = []
        for f in range(num_frames):
            primitives = []
            for s_i in range(segments.shape[1]):
                primitives.append(pyrender.Primitive(segments[f, s_i, :, :], mode=1, color_0=[0, 0, 0, 1]))
            bone_meshes.append(pyrender.Mesh(primitives))

    if endpoints is not None:
        joint_meshes = []
        for s_i in range(endpoints.shape[1]):
            sm = trimesh.creation.uv_sphere(radius=args.marker_size)
            sm.visual.vertex_colors = np.array([0, 0, 0])
            tfs = np.expand_dims(np.eye(4), axis=0)
            joint_meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])

    # create floor mesh
    floor_mesh = trimesh.Trimesh(
        vertices=[[-25, -25, 0], [-25, 25, 0], [25, -25, 0], [25, 25, 0]],
        faces=[[0, 2, 1], [1, 2, 3]],
    )
    f_node = pyrender.Mesh.from_trimesh(floor_mesh)
    f_node = scene.add(f_node)

    # setup lighting
    directional_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(directional_light)

    # render mode
    render_mode = "online"
    if args.video_path is not None:
        render_mode = "offline"
        video_format = os.path.splitext(args.video_path)[1][1:]

    # setup scene
    run_in_thread = True
    if render_mode == "offline":
        run_in_thread = False

    camera_matrix = [
        [0.750, -0.252, 0.611, 3.4],
        [0.661, 0.291, -0.691, -2.0],
        [-0.004, 0.923, 0.385, 2.3],
        [0, 0, 0, 1],
    ]

    # setup nodes
    m_nodes = []
    for m_mesh in m_meshes:
        m_nodes.append(scene.add(m_mesh))

    if segments is not None:
        bone_node = scene.add(bone_meshes[0])

    if endpoints is not None:
        joint_nodes = []
        for joint_mesh in joint_meshes:
            joint_nodes.append(scene.add(joint_mesh))

    def render_frame(frame):
        # update markers
        marker_frame = frame

        m_i = 0
        for m_node in m_nodes:
            color = get_color(marker_labels[marker_frame, m_i])
            
            m_node.mesh.primitives[0].color_0 = color

            m_node.translation = points[marker_frame, m_i]
            m_i += 1

        if segments is not None:
            bone_node.mesh = bone_meshes[marker_frame]

        if endpoints is not None:
            j_i = 0
            for joint_node in joint_nodes:
                joint_node.translation = endpoints[marker_frame, j_i]
                j_i += 1

    if render_mode == "online":
        viewer = pyrender.Viewer(
            scene,
            use_raymond_lighting=False,
            run_in_thread=run_in_thread,
            shadows=True,
        )

        frame = 0
        while viewer.is_active:
            viewer.render_lock.acquire()
            render_frame(frame)
            viewer.render_lock.release()

            frame = (frame + 1) % num_frames
            time.sleep(1.0 / 30.0)

    elif render_mode == "offline":
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=camera_matrix)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=640,
            viewport_height=480,
            point_size=1.0,
        )

        flags = pyrender.constants.RenderFlags.SHADOWS_ALL

        if video_format == "mp4":
            cap = cv2.VideoCapture(0)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.video_path, fourcc, args.video_fps, args.video_res)

            # write out video
            for frame in range(0, num_frames, 30 // args.video_fps):
                render_frame(frame)
                color, _ = renderer.render(scene, flags=flags)
                out.write(np.flip(color, axis=2))

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        elif video_format == "gif":
            # write out video
            with imageio.get_writer(args.video_path, mode="I", duration=(1000.0 * (1.0 / args.video_fps))) as writer:
                for frame in range(0, num_frames, 30 // args.video_fps):
                    render_frame(frame)
                    color, _ = renderer.render(scene, flags=flags)
                    writer.append_data(color)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_fitting", action="store_true", help="apply fitting")
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--description", type=str, help="layout for .c3d file name", default=None)
    parser.add_argument("--limb", type=str, help="limb name", default=None)
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.03)
    parser.add_argument("--num_markers", type=int, help="number of markers", default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data")
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    parser.add_argument("--visualize_parts", action="store_true", help="visualize parts")
    args = parser.parse_args()

    visualize(args)