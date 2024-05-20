import argparse
import os

import numpy as np
import pyrender
import trimesh

from video_mocap.markers.markers import Markers
from video_mocap.markers.markers_utils import cleanup_markers, id_markers, randomly_drop_markers, shuffle_markers, segment_rigid
from video_mocap.utils.colors import colors_perceptually_distinct_24
from video_mocap.vis.renderer import VideoMocapRenderer
from video_mocap.vis.scene import VideoMocapScene
from video_mocap.vis.utils.line_3d import Line3D


def visualize(args):
    mocap_dir = "mocap"
    if args.description is not None:
        mocap_dir = mocap_dir + "_parts___" + args.description

    markers_filename = os.path.join(
        args.input_dir,
        args.dataset,
        mocap_dir,
        args.subject,
        args.sequence + ".c3d",
    )

    markers = Markers(markers_filename, shuffle=False)

    points = markers.get_points()
    points = np.nan_to_num(points, 0)
    points = cleanup_markers(points)
    markers.set_points(points)

    points = randomly_drop_markers(
        points,
        markers.get_frequency(),
        marker_radius=args.marker_size,
        num_drop=args.num_drop,
    )
    if args.id_markers:
        points = shuffle_markers(points)
        points = id_markers(points)
    if args.segment_rigid:
        rigid_groups = segment_rigid(points)
        rigid_labels = {}
        index = 0
        for rigid_group in rigid_groups:
            for value in rigid_group:
                rigid_labels[value] = index
            index += 1


    mocap_freq = markers.get_frequency()
    num_frames = points.shape[0]

    def get_color(i):
        return colors_perceptually_distinct_24[i % 24]

    m_meshes = []
    for m_i in range(markers.get_num_markers()):
        sm = trimesh.creation.uv_sphere(radius=args.marker_size)
        if args.segment_rigid:
            color = get_color(rigid_labels[m_i])
        else:
            color = np.array([0, 0, 0])
        sm.visual.vertex_colors = color

        tfs = np.expand_dims(np.eye(4), axis=0)
        m_meshes.append(pyrender.Mesh.from_trimesh(sm, poses=tfs))

    # setup scene
    scene = VideoMocapScene()

    if args.segment_rigid:
        bone_positions = []
        bone_meshes = []
        bone_nodes = []
        bone_materials = []
        rigid_group_index = 0
        for rigid_group in rigid_groups:
            for i in range(len(rigid_group)):
                for j in range(i+1, len(rigid_group)):
                    bone_i, bone_j = rigid_group[i], rigid_group[j]
                    bone = np.stack((points[:, bone_i], points[:, bone_j]), axis=1)
                    bone_positions.append(bone)
                    bone_meshes.append(Line3D(radius=args.marker_size / 4))
                    bone_materials.append(pyrender.MetallicRoughnessMaterial(baseColorFactor=(0, 0, 0), smooth=False))
                    bone_nodes.append(scene.add(pyrender.Mesh.from_trimesh(bone_meshes[-1].get_mesh())))

            rigid_group_index += 1

        bone_positions = np.stack(bone_positions, axis=1)                


    if args.show_marker_weights >= 0:
        mw_positions = []
        mw_nodes = []
        mw_meshes = []
        mw_materials = []

        distance_threshold = 0.005
        for i in [args.show_marker_weights]:
            for j in range(markers.get_num_markers()):
                if i == j:
                    continue
                
                mw_positions.append(np.stack((points[:, i], points[:, j]), axis=1))
                distance = np.linalg.norm(points[:, i] - points[:, j], axis=-1)
                std = np.std(distance)

                alpha = np.clip((distance_threshold / std) ** 3, 0, 1)
                mw_color = (np.array([0.5, 0.5, 0.5]) * (1-alpha)) + (np.array([1.0, 0.0, 0.0]) * (alpha))
                mw_materials.append(pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=mw_color,
                    smooth=False,
                ))

                radius = (args.marker_size / 2) * np.clip(alpha, 0.2, 1.0)
                mw_meshes.append(Line3D(radius=radius))
                mw_nodes.append(scene.add(pyrender.Mesh.from_trimesh(mw_meshes[-1].get_mesh())))

        mw_positions = np.stack(mw_positions, axis=1)

    m_nodes = []
    for m_mesh in m_meshes:
        m_nodes.append(scene.add(m_mesh))

    def render_frame(frame):
        # update markers
        marker_frame = frame
        if args.segment_rigid:
            bone_index = 0
            for bone_mesh in bone_meshes:
                bone_nodes[bone_index].mesh = pyrender.Mesh.from_trimesh(
                    bone_mesh.get_mesh(),
                    poses=Line3D.get_matrix(
                        bone_positions[marker_frame, bone_index, 0],
                        bone_positions[marker_frame, bone_index, 1],
                    ),
                    material=bone_materials[bone_index],
                )
                bone_index += 1

        if args.show_marker_weights >= 0:
            mw_index = 0
            for mw_mesh in mw_meshes:
                mw_nodes[mw_index].mesh = pyrender.Mesh.from_trimesh(
                    mw_mesh.get_mesh(),
                    poses=Line3D.get_matrix(
                        mw_positions[marker_frame, mw_index, 0],
                        mw_positions[marker_frame, mw_index, 1],
                    ),
                    material=mw_materials[mw_index],
                )
                mw_index += 1

        m_i = 0
        for m_node in m_nodes:
            if args.segment_rigid:
                color = get_color(rigid_labels[m_i])
            else:
                color = np.array([0, 0, 0])
            
            m_node.mesh.primitives[0].color_0 = color

            m_node.translation = points[marker_frame, m_i]
            m_i += 1

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_frame,
        num_frames=num_frames,
        data_freq=mocap_freq,
        video_fps=args.video_fps,
        video_res=args.video_res,
        video_path=args.video_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_only", action="store_true", help="only use the CPU")
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--description", type=str, help="layout for .c3d file name", default=None)
    parser.add_argument("--id_markers", type=str, help="run identification on markers")
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.03)
    parser.add_argument("--num_drop", type=int, help="number of markers to drop", default=0)
    parser.add_argument("--rand_order", action="store_true", help="randomize parker order (simulates passive markers)")
    parser.add_argument("--segment_rigid", action="store_true", help="segment rigid body")
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--show_marker_weights", type=int, help="show marker weights for vertex", default=-1)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    args = parser.parse_args()

    visualize(args)