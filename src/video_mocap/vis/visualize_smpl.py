import argparse

import numpy as np
import pyrender
from pytorch3d.transforms import axis_angle_to_matrix
from scipy.spatial.transform import Rotation
import torch
import trimesh

from video_mocap.markers.markers import Markers
from video_mocap.utils.smpl import SmplInference
from video_mocap.vis.renderer import VideoMocapRenderer
from video_mocap.vis.scene import VideoMocapScene, SMPL_COLORS


def visualize_smpl(
    filenames,
    video_path,
    marker_size=0.03,
    video_fps=30,
    video_res=(640, 480),
    vis_joints=False,
    hide_smpl=False,
    angle: float=0,
):
    m_meshes = []
    s_meshes = []
    j_meshes = []
    joint_meshes = []
    joint_materials = []
    joints = []

    # separate out the SMPL (mesh) and c3d (marker) files
    smpl_filenames = [x for x in filenames if x.endswith(".npz")]
    c3d_filenames = [x for x in filenames if x.endswith(".c3d")]
    
    scene = VideoMocapScene()

    # SMPL data
    for subject_id in range(len(smpl_filenames)):
        filename = smpl_filenames[subject_id]
        smpl_data = np.load(filename)
        num_frames = smpl_data["trans"].shape[0]

        mocap_freq = 30
        if "mocap_framerate" in smpl_data:
            mocap_freq = smpl_data["mocap_framerate"]
        elif "mocap_frame_rate" in smpl_data:
            mocap_freq = smpl_data["mocap_frame_rate"]

        smpl_inference = SmplInference()
        poses = torch.from_numpy(smpl_data["poses"][:, 3:]).reshape((num_frames, -1, 3))
        betas = torch.repeat_interleave(torch.from_numpy(smpl_data["betas"]).unsqueeze(0), repeats=num_frames, dim=0)
        root_orient = torch.from_numpy(smpl_data["poses"][:, :3]).reshape((num_frames, -1, 3))
        trans = torch.from_numpy(smpl_data["trans"])
        smpl_output = smpl_inference(
            poses=axis_angle_to_matrix(poses).float(),
            betas=betas.float(),
            root_orient=axis_angle_to_matrix(root_orient).float(),
            trans=trans.float(),
        )
        joints.append(smpl_output["joints"])

        # load joint meshes
        if vis_joints:
            num_joints = joints[-1][0].shape[0]
            joint_meshes.append(trimesh.creation.uv_sphere(radius=0.02))
            color = [x / 255.0 for x in SMPL_COLORS[subject_id]]
            joint_materials.append(pyrender.MetallicRoughnessMaterial(baseColorFactor=(color[0], color[1], color[2], 1.0), alphaMode="OPAQUE"))
            j_tfs = np.tile(np.eye(4), (num_joints, 1, 1))
            j_tfs[:, :3, 3] = joints[-1][0]
            j_meshes.append(pyrender.Mesh.from_trimesh(joint_meshes[-1], poses=j_tfs, material=joint_materials[-1]))

        # load SMPL meshes
        s_meshes.append([])
        for s_i in range(num_frames):
            sm = trimesh.Trimesh(
                vertices=smpl_output["vertices"][s_i],
                faces=smpl_inference.smpl.faces,
            )
            color = [x / 255.0 for x in SMPL_COLORS[subject_id]]

            if vis_joints:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(color[0], color[1], color[2], 0.5), alphaMode="BLEND")
            else:
                material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(color[0], color[1], color[2], 1.0), alphaMode="OPAQUE")

            s_meshes[-1].append(pyrender.Mesh.from_trimesh(sm, material=material, wireframe=False))

    # setup lighting
    scene.directional_light.matrix = np.array([
        [0.75, 0.5, 0.433, 0],
        [-0.433, 0.866, -0.2499, 0],
        [-0.5, 0.0, 0.866, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # marker data
    mocap_data = []
    for marker_id in range(len(c3d_filenames)):
        filename = c3d_filenames[marker_id]
        markers = Markers(filename)
        mocap_markers = markers.get_points()

        num_markers = mocap_markers.shape[1]
        mocap_data.append(mocap_markers)

        # load marker meshes
        if num_markers > 0:
            mm = trimesh.creation.uv_sphere(radius=marker_size)
            color = [x / 255.0 for x in SMPL_COLORS[subject_id]]
            marker_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(0.0, 0.0, 0.0, 1.0), alphaMode="OPAQUE")
            tfs = np.tile(np.eye(4), (num_markers, 1, 1))
            tfs[:, :3, 3] = mocap_markers[0]
            m_meshes.append(pyrender.Mesh.from_trimesh(mm, poses=tfs, material=marker_material))
            m_node = scene.add(m_meshes[-1])
    
    if vis_joints:
        j_nodes = []
        for i in range(len(smpl_filenames)):
            j_nodes.append(scene.add(j_meshes[i]))

    if not hide_smpl:
        s_nodes = []
        for i in range(len(smpl_filenames)):
            s_nodes.append(scene.add(s_meshes[i][0]))

    def render_frame(frame):
        # update markers
        marker_frame = frame

        for m_mesh in m_meshes:
            tfs[:, :3, 3] = mocap_markers[marker_frame]
            m_node.mesh = pyrender.Mesh.from_trimesh(mm, poses=tfs, material=marker_material)

        if not hide_smpl:
            for i in range(len(smpl_filenames)):
                s_nodes[i].mesh = s_meshes[i][marker_frame]
                if vis_joints:
                    j_tfs[:, :3, 3] = joints[i][marker_frame]
                    j_nodes[i].mesh = pyrender.Mesh.from_trimesh(joint_meshes[i], poses=j_tfs, material=joint_materials[i])

    quality_mode = "normal"
    if video_path and (video_path.endswith(".png") or video_path.endswith(".mp4")):
        quality_mode = "ultra"

    camera_matrix = [
        [0.750, -0.252, 0.611, 3.4],
        [0.661, 0.291, -0.691, -2.0],
        [-0.004, 0.923, 0.385, 2.3],
        [0, 0, 0, 1],
    ]

    if angle != 0.0:
        rot = Rotation.from_rotvec(np.array([0, 0, -angle]), degrees=True).as_matrix()
        rot_4x4 = np.eye(4)
        rot_4x4[:3, :3] = rot
        camera_matrix = rot_4x4 @ camera_matrix
        light_matrix = scene.directional_light.matrix
        light_matrix = rot_4x4 @ light_matrix
        scene.directional_light.matrix = light_matrix

    renderer = VideoMocapRenderer(
        scene=scene,
        render_frame_fn=render_frame,
        num_frames=num_frames,
        data_freq=mocap_freq,
        video_fps=video_fps,
        video_res=video_res,
        video_path=video_path,
        quality_mode=quality_mode,
        camera_matrix=camera_matrix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", nargs="+", type=str, help="filenames", required=True)
    parser.add_argument("--hide_smpl", action="store_true", help="hide SMPL meshes")
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.01)
    parser.add_argument("--video_fps", type=int, help="video frames per second", default=30)
    parser.add_argument("--video_path", type=str, help="video path", default=None)
    parser.add_argument("--video_res", nargs=2, type=int, help="video resolution", default=(640, 480))
    parser.add_argument("--vis_joints", action="store_true", help="visualize joints")
    args = parser.parse_args()

    visualize_smpl(
        filenames=args.filenames,
        marker_size=args.marker_size,
        video_fps=args.video_fps,
        video_path=args.video_path,
        video_res=args.video_res,
        vis_joints=args.vis_joints,
        hide_smpl=args.hide_smpl,
    )