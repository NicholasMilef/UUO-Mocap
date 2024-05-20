import argparse
import os

from pytorch3d.transforms import axis_angle_to_matrix
import torch
import numpy as np
import trimesh

from video_mocap.utils.marker_layout import compute_markers_from_layout
from video_mocap.markers.markers import Markers
from video_mocap.utils.smpl import SmplInference


def export_marker_layout(
    input_dir: str,
    dataset: str,
    subject: str,
    sequence: str,
    frame: int,
    smpl_filename: str,
    part: str,
):
    if part is not None:
        markers = Markers(os.path.join(input_dir, dataset, "mocap_parts___" + part, subject, sequence + ".c3d"))
    else:
        markers = Markers(os.path.join(input_dir, dataset, "mocap", subject, sequence + ".c3d"))
    marker_pos = markers.get_points()
    num_frames = marker_pos.shape[0]

    # source mesh
    smpl_data = np.load(os.path.join(input_dir, dataset, "smpl", subject, sequence + "_stageii.npz"))
    smpl_inference = SmplInference()

    smpl_output = smpl_inference(
        poses=axis_angle_to_matrix(torch.from_numpy(smpl_data["poses"][:, 1:]).float()),
        trans=torch.from_numpy(smpl_data["trans"]).float(),
        betas=torch.from_numpy(smpl_data["betas"])[None].float(),
        root_orient=axis_angle_to_matrix(torch.from_numpy(smpl_data["poses"][:, [0]]).float()),
    )

    vertices_expanded = torch.repeat_interleave(smpl_output["vertices"][:, None], repeats=marker_pos.shape[1], dim=1).float()
    markers_expanded = torch.repeat_interleave(torch.from_numpy(marker_pos[:, :, None]), repeats=6890, dim=2).float()
    distances = torch.norm(vertices_expanded - markers_expanded, dim=-1)
    distances = torch.sum(distances, dim=0)
    marker_vids = torch.argmin(distances, dim=-1).tolist()

    # template mesh
    template_data = np.load(smpl_filename)
    template_smpl_inference = SmplInference()
    template_smpl_output = smpl_inference(
        poses=axis_angle_to_matrix(torch.from_numpy(template_data["poses"][:, 1:]).float()),
        trans=torch.from_numpy(template_data["trans"]).float(),
        betas=torch.from_numpy(template_data["betas"])[None].float(),
        root_orient=axis_angle_to_matrix(torch.from_numpy(template_data["poses"][:, [0]]).float()),
    )

    markers = compute_markers_from_layout(
        vertices=template_smpl_output["vertices"][None],
        triangles=torch.from_numpy(template_smpl_inference.smpl.faces.astype(np.int32)),
        marker_vertex_ids=marker_vids,
        marker_offset=0.0095,
    )["marker_pos"]
    
    # export geometry
    markers = markers[0, frame]
    meshes = []
    for m_i in range(markers.shape[0]):
        mesh = trimesh.creation.icosphere(
            radius=0.02,
            vertex_colors=(0, 0, 0),
        )
        mesh.apply_translation(markers[m_i])
        meshes.append(mesh)

    smpl_mesh = trimesh.Trimesh(
        vertices=template_smpl_output["vertices"][frame],
        faces=template_smpl_inference.smpl.faces,
        vertex_colors=np.array([0.08, 0.5, 1.0]),
        process=False,
    )
    meshes.append(smpl_mesh)

    output_mesh = trimesh.util.concatenate(meshes)

    output_dir = "./paper/teaser/markers"
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, dataset + ".ply")
    if part is not None:
        filename = os.path.join(output_dir, dataset + "___" + part + ".ply")
    else:
        filename = os.path.join(output_dir, dataset + ".ply")

    output_mesh.export(filename)
    print("Saved", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--smpl_filename", type=str, help="SMPL filename", required=True)
    parser.add_argument("--frame", type=int, help="SMPL frame number", required=True)
    args = parser.parse_args()

    export_marker_layout(
        input_dir=args.input_dir,
        dataset="umpm",
        subject="p1_chair_2",
        sequence="p1_chair_2_00000150",
        part="left_arm",
        frame=args.frame,
        smpl_filename=args.smpl_filename,
    )

    export_marker_layout(
        input_dir=args.input_dir,
        dataset="umpm",
        subject="p1_chair_2",
        sequence="p1_chair_2_00000150",
        part=None,
        frame=args.frame,
        smpl_filename=args.smpl_filename,
    )

    export_marker_layout(
        input_dir=args.input_dir,
        dataset="cmu_kitchen_pilot_rb",
        subject="s1",
        sequence="brownies_00000150",
        part=None,
        frame=args.frame,
        smpl_filename=args.smpl_filename,
    )

    export_marker_layout(
        input_dir=args.input_dir,
        dataset="moyo_val",
        subject="20221004_with_com",
        sequence="221004_yogi_nexus_body_hands_03596_Warrior_III_Pose_or_Virabhadrasana_III_-a_00000060",
        part=None,
        frame=args.frame,
        smpl_filename=args.smpl_filename,
    )
