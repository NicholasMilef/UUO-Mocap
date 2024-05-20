import argparse
import os

import numpy as np
import torch
import trimesh

from video_mocap.datasets.dataset_mocap import DatasetMocap
from video_mocap.utils.marker_layout import get_marker_layout, compute_markers_from_layout
from video_mocap.utils.smpl import SmplInferenceGender
from video_mocap.utils.smpl_utils import get_joint_id, get_marker_vertex_id


def export_layout_ply(args):
    device = torch.device("cpu")

    # When using Blender, create a custom material with the Nodes editor and the vertex color
    if args.parts is not None:
        parts = [[get_joint_id(x) for x in args.parts], 1.0]
    else:
        parts = [list(range(0,24)), 1.0]

    dataset = DatasetMocap(
        batch_size=1,
        num_markers=args.num_markers,
        sequence_length=-1,
        parts_set=[parts],
    )
    data = dataset.get_sequence(args.subject, args.sequence)

    pose_body = torch.from_numpy(data["poses"][:, 3:72]).float()
    betas = torch.from_numpy(data["betas"][:10]).float()
    root_orient = torch.from_numpy(data["poses"][:, :3]).float()
    trans = torch.from_numpy(data["trans"]).float()
    gender_one_hot = torch.from_numpy(data["gender_one_hot"]).float()

    smpl = SmplInferenceGender(device)

    output = smpl(
        poses=torch.unsqueeze(pose_body, dim=0),
        betas=torch.unsqueeze(betas, dim=0),
        root_orient=torch.unsqueeze(root_orient, dim=0),
        trans=torch.unsqueeze(trans, dim=0),
        gender_one_hot=torch.unsqueeze(gender_one_hot, dim=0),
        pose2rot=True,
        compute_part_labels=True,
    )

    # compute marker positions
    markers = None
    if args.marker_layout is not None:
        marker_layout = get_marker_layout(args.marker_layout)
        marker_vertex_ids = [get_marker_vertex_id(i) for i in marker_layout]
        markers = compute_markers_from_layout(
            output["vertices"],
            torch.from_numpy(smpl.smpls["male"].faces.astype(np.int32)),
            marker_vertex_ids
        )["marker_pos"][0]
    else:
        if args.num_markers > 0:
            markers = dataset.compute_markers(
                output["vertices"],
                output["vertex_part_labels"],
            )["marker_pos"][0]

    if markers is not None:
        if args.indices is not None:
            markers = markers[args.indices[0]:args.indices[1]:args.indices[2]]
        elif args.frames is not None:
            markers = markers[args.frames]

    # extract vertices
    vertices = output["vertices"][0].detach().cpu().numpy()
    if args.indices is not None:
        vertices = vertices[args.indices[0]:args.indices[1]:args.indices[2]]
    elif args.frames is not None:
        vertices = vertices[args.frames]

    meshes = []
    start_color = np.ones((vertices.shape[1], 4)) * ([0.4, 0.6, 1.0, 1.0])
    end_color = np.ones((vertices.shape[1], 4)) * ([0, 0.15, 1.0, 1.0])

    if args.export_single:
        start_color = end_color

    if args.parts is not None:
        weights = output["vertex_part_labels"][0].detach().cpu().numpy()
        weights_total = np.zeros_like(weights[:, 0])
        start_color = np.ones((vertices.shape[1], 4)) * ([0, 0.15, 1.0, 1.0])
        background_color = np.ones((vertices.shape[1], 4)) * ([0.5, 0.5, 0.5, 0.1])
        for joint in args.parts:
            joint_index = get_joint_id(joint)
            joint_weight = weights[:, joint_index]
            weights_total += joint_weight

        weights_exp = np.expand_dims(weights_total, axis=1)

        start_color = (weights_exp * start_color) + (background_color * (1.0 - weights_exp))
        start_color = np.clip(start_color, 0.0, 1.0)
        end_color = start_color

    # add SMPL meshes
    for i in range(vertices.shape[0]):
        if vertices.shape[0] == 1:
            alpha = 1.0
        else:
            alpha = i / (vertices.shape[0] - 1)
        
        vertex_colors = (alpha * end_color) + ((1 - alpha) * start_color)
        meshes.append(trimesh.Trimesh(
            vertices=vertices[i],
            faces=smpl.smpls["male"].faces,
            vertex_colors=vertex_colors,
            process=False,
        ))

    # add marker meshes
    for i in range(markers.shape[0]):
        spheres = []
        for j in range(markers.shape[1]):
            spheres.append(trimesh.primitives.Sphere(radius=args.marker_size, center=markers[i, j]))
            spheres[-1].visual.vertex_colors[:, 0] = 255
            spheres[-1].visual.vertex_colors[:, 1] = 0
            spheres[-1].visual.vertex_colors[:, 2] = 0

        spheres = trimesh.util.concatenate(spheres)
        meshes.append(spheres)

    if not args.export_single:
        total_meshes = trimesh.util.concatenate(meshes)
        total_meshes.export(args.output)
    else:
        index = 0
        for mesh in meshes:
            temp_filename = args.output
            extension = os.path.splitext(args.output)[-1]
            temp_filename = temp_filename.replace(extension, "_" + str(index) + extension)
            mesh.export(temp_filename)
            index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_single", action="store_true", help="export single meshes")
    parser.add_argument("--frames", type=int, nargs="+", help="indices like in NumPy", default=None)
    parser.add_argument("--indices", type=int, nargs="+", help="indices like in NumPy", default=None)
    parser.add_argument("--marker_layout", type=str, help="use marker layouts", default=None)
    parser.add_argument("--marker_size", type=float, help="marker size", default=0.01)
    parser.add_argument("--num_markers", type=int, help="number of markers")
    parser.add_argument("--output", type=str, help="export mesh name")
    parser.add_argument("--parts", nargs="+", help="names of parts", default=None)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    args = parser.parse_args()

    export_layout_ply(args)