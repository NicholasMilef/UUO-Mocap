import argparse
import os
import time

import cv2
import imageio
import matplotlib
import numpy as np
import pyrender
import torch
import trimesh

from video_mocap.datasets.dataset_mocap import DatasetMocap


def visualize(args):
    device = torch.device("cpu")
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda:0")

    num_markers = 43
    stride = 1
    dataset = DatasetMocap(\
        split="train",
        num_markers=num_markers,
        sequence_length=-1,
        batch_size=1,
        stride=stride,
    )

    import pdb; pdb.set_trace()
    #output = smpl(
        #poses = ,
    #)
    #multimodal_verts = output[]

    points = shuffle_markers(points)
    points = id_markers(points)
    mocap_freq = markers.get_frequency()
    num_frames = points.shape[0]

    def get_color(i):
        return matplotlib.cm.Paired(i  % 10)

    for i in range(num_markers):
        m_meshes = []
        for m_i in range(num_markers):
            sm = trimesh.creation.uv_sphere(radius=args.marker_size)
            sm.visual.vertex_colors = get_color(i)

            tfs = np.expand_dims(np.eye(4), axis=0)
            m_meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))

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

    m_nodes = []
    for m_mesh in m_meshes:
        m_nodes.append(scene.add(m_mesh))

    def render_frame(frame):
        # update markers
        marker_frame = frame

        m_i = 0
        for m_node in m_nodes:
            color = get_color(m_i)
            m_node.mesh.primitives[0].color_0 = color

            m_node.translation = points[marker_frame, m_i]
            m_i += 1


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
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.03)
    parser.add_argument("--num_drop", type=int, help="number of markers to drop", default=0)
    parser.add_argument("--rand_order", action="store_true", help="randomize parker order (simulates passive markers)")
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    args = parser.parse_args()

    visualize(args)