import numpy as np
import pyrender
import torch
import trimesh

from video_mocap.utils.smpl import SmplInference


class VideoMocapScene(pyrender.Scene):
    def __init__(self):
        """
        Scene template
        """
        super().__init__(ambient_light=[0.2, 0.2, 0.2])

        # create floor mesh
        f_node = self.add(create_floor())

        # setup lighting
        self.directional_light = self.add(
            pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0),
        )


def create_floor():
    vertices = []
    faces = []
    face_colors = []
    tile_size = 1
    for x in range(-20, 20, tile_size):
        for y in range(-20, 20, tile_size):
            offset = len(vertices)
            vertices.append([x, y, 0])
            vertices.append([x, y+tile_size, 0])
            vertices.append([x+tile_size, y, 0])
            vertices.append([x+tile_size, y+tile_size, 0])
            if abs((x//tile_size) + (y//tile_size)) % 2 == 0:
                for _ in range(2):
                    face_colors.append([0.5, 0.5, 0.5])
            else:
                for _ in range(2):
                    face_colors.append([0.2, 0.2, 0.2])

            faces.append([offset + 0, offset + 2, offset + 1])
            faces.append([offset + 1, offset + 2, offset + 3])

    floor_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors,
    )

    floor_mesh = pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)

    return floor_mesh


def create_smpl_meshes(
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    smpl_inference: SmplInference,
):
    device = smpl_inference.device

    output_frame = smpl_inference(
        poses=torch.from_numpy(pose_body).to(device),
        root_orient=torch.from_numpy(root_orient).to(device),
        trans=torch.from_numpy(trans).to(device),
        betas=torch.from_numpy(betas).to(device),
    )
    multimodal_verts = output_frame["vertices"].detach().cpu().numpy()

    meshes = []
    smpl_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.6, 1.0, 1.0], alphaMode="OPAQUE")
    for i in range(trans.shape[0]):
        tm = trimesh.Trimesh(
            vertices=multimodal_verts[i],
            faces=smpl_inference.smpl.faces,
        )
        meshes.append(pyrender.Mesh.from_trimesh(tm, material=smpl_material))

    return meshes


def extract_part_vertices(
    pose_body: np.ndarray,
    root_orient: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray,
    smpl_inference: SmplInference,
    part_joints: np.ndarray=None,
):
    device = smpl_inference.device

    output_frame = smpl_inference(
        poses=torch.from_numpy(pose_body).to(device),
        root_orient=torch.from_numpy(root_orient).to(device),
        trans=torch.from_numpy(trans).to(device),
        betas=torch.from_numpy(betas).to(device),
    )
    multimodal_verts = output_frame["vertices"].detach().cpu().numpy()

    lbs_weights = smpl_inference.get_lbs_weights()
    vertex_ids = torch.argmax(lbs_weights, dim=-1)
    vertex_ids = vertex_ids.detach().cpu().numpy()

    indices = []
    for part_joint in part_joints:
        indices = indices + np.where(vertex_ids==part_joint)[0].tolist()

    points = multimodal_verts[:, indices]

    return points


SMPL_COLORS = [
    [31.0, 119.0, 180.0],
    [255.0, 127.0, 14.0],
    [44.0, 160.0, 44.0],
    [214.0, 39.0, 40.0],
    [148.0, 103.0, 189.0],
    [140.0, 86.0, 75.0],
    [227.0, 119.0, 194.0],
    [127.0, 127.0, 127.0],
    [188.0, 189.0, 34.0],
    [23.0, 190.0, 207.0],
]
