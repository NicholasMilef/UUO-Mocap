import argparse
import os
import time

import cv2
import imageio
import numpy as np
import pyrender
import torch
from torch.utils.data import DataLoader
import trimesh

from video_mocap.datasets.dataset_mocap import DatasetMocap, apply_random_rotation_to_pos, apply_random_rotation_to_rot, apply_random_translation_to_pos
from video_mocap.utils.smpl import SmplInferenceGender
from video_mocap.utils.smpl_utils import get_joint_id


def visualize(args):
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda:0")

    num_markers = args.num_markers
    part_ids = None
    if args.parts is not None:
        part_ids = [[get_joint_id(part) for part in args.parts]]

    dataset = DatasetMocap(batch_size=1, num_markers=num_markers, sequence_length=-1, parts_set=part_ids)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    index = 0
    s_index = 20
    for _, data in enumerate(dataloader):
        if index == s_index:
            data = data
            break
        index += 1

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])

    smpl = SmplInferenceGender(torch.device("cpu"))

    root_orient = data["poses"][:, :, :3]
    trans = data["trans"][:, :, :3]
    if args.apply_augmentation:
        angle = torch.from_numpy((1.0 - np.random.rand(data["trans"].shape[0])) * np.pi * 2).to(device)
        root_orient = apply_random_rotation_to_rot(root_orient, angle)
        trans = apply_random_rotation_to_pos(trans, angle)
        trans = apply_random_translation_to_pos(trans, std=2.0, center=True)

    smpl_output = smpl(
        poses=data["poses"][:, :, 3:72],
        betas=data["betas"],
        root_orient=root_orient,
        trans=trans,
        gender_one_hot=data["gender_one_hot"],
        pose2rot=True,
        compute_part_labels=False,
    )

    # render markers
    markers = dataset.compute_markers(
        smpl_output["vertices"],
    )
    
    points = markers["marker_pos"][0]
    num_frames = points.shape[0]
    mocap_freq = 30

    sm = trimesh.creation.uv_sphere(radius=args.marker_size)
    sm.visual.vertex_colors = [1.0, 0.0, 0.0]

    # render SMPL mesh
    smpl_meshes = []
    for i in range(data["trans"].shape[1]):
        tm = trimesh.Trimesh(
            vertices=smpl_output["vertices"][0, i].detach().cpu().numpy(),
            faces=smpl.smpls["male"].faces,
        )
        smpl_meshes.append(pyrender.Mesh.from_trimesh(tm))
    smpl_node = scene.add(smpl_meshes[0])

    smpl = SmplInferenceGender(device)

    tfs = np.tile(np.eye(4), (num_markers, 1, 1))
    tfs[:, :3, 3] = points[0]
    m_mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)


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

    m_node = scene.add(m_mesh)

    def render_frame(frame):
        # update markers
        marker_frame = frame

        nonlocal m_node

        # render markers
        tfs[:, :3, 3] = points[marker_frame]
        m_node.mesh = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        # render markers
        smpl_node.mesh = smpl_meshes[marker_frame]

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
            time.sleep(1.0 / mocap_freq)

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
            for frame in range(0, num_frames, mocap_freq // args.video_fps):
                render_frame(frame)
                color, _ = renderer.render(scene, flags=flags)
                out.write(np.flip(color, axis=2))

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        elif video_format == "gif":
            # write out video
            with imageio.get_writer(args.video_path, mode="I", duration=(1000.0 * (1.0 / args.video_fps))) as writer:
                for frame in range(0, num_frames, mocap_freq // args.video_fps):
                    render_frame(frame)
                    color, _ = renderer.render(scene, flags=flags)
                    writer.append_data(color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_augmentation", action="store_true", help="apply data augmentation")
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.01)
    parser.add_argument("--num_drop", type=int, help="number of markers to drop", default=0)
    parser.add_argument("--num_markers", type=int, help="number of markers", default=43)
    parser.add_argument("--rand_order", action="store_true", help="randomize parker order (simulates passive markers)")
    parser.add_argument("--rand_rotation", action="store_true", help="randomize root rotation")
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--parts", nargs="+", type=str, help="parts", default=None)
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    args = parser.parse_args()

    visualize(args)