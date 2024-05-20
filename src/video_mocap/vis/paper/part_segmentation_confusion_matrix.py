import argparse
import os

import matplotlib.pyplot as plt
from moshpp.marker_layout.marker_vids import all_marker_vids
import numpy as np
import pandas as pd
import seaborn as sn
from smplx.joint_names import SMPLH_JOINT_NAMES
import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import trimesh

from video_mocap.markers.markers import Markers
from video_mocap.markers.markers_utils import cleanup_markers
from video_mocap.models.marker_segmenter_multimodal import MarkerSegmenterMultimodal
from video_mocap.utils.smpl import SmplInference


def visualize_part_segmentation_confusion_matrix(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.gpu))

    mocap_dir = os.path.join(args.input_dir, args.dataset, "mocap")

    smpl_inference = SmplInference(device)

    num_joints = smpl_inference.smpl.lbs_weights.shape[-1]

    marker_segmenter_model = MarkerSegmenterMultimodal(
        num_parts = num_joints,
        latent_dim = 64,
        sequence_length = 32,
        modalities=["markers"],
    ).to(device)
    marker_segmenter_model.eval()
    marker_segmenter_filename = "./checkpoints/marker_segmenter_multimodal/10_26_s4_40m_rot_fast_l64_hard_mp_full/model110.pth"
    marker_segmenter_model.load_state_dict(
        torch.load(marker_segmenter_filename, map_location=device),
    )

    if args.subjects is None:
        subjects = sorted(os.listdir(mocap_dir))
    else:
        subjects = args.subjects

    # convert SMPL-X marker locations to SMPL
    smplx_mesh = trimesh.load("body_models/model_transfer/meshes/smplx/01.ply", process=False)
    smpl_mesh = trimesh.load("body_models/model_transfer/meshes/smpl/01.ply", process=False)

    smplx_vids = all_marker_vids["smplx"]
    smpl_vids = {}
    for label, vid in smplx_vids.items():
        smplx_pos = smplx_mesh.vertices[vid]

        min_index = None
        min_distance = np.inf
        for i in range(smpl_mesh.vertices.shape[0]):
            smpl_pos = smpl_mesh.vertices[i]
            distance = np.linalg.norm(smplx_pos - smpl_pos)

            if distance < min_distance:
                min_index = i
                min_distance = distance

        smpl_vids[label] = min_index

    vertex_labels = torch.argmax(smpl_inference.smpl.lbs_weights, dim=-1)
    joint_names = SMPLH_JOINT_NAMES[:num_joints]

    # create confusion matrix
    matrix = MulticlassConfusionMatrix(num_classes=num_joints).to(device)
    full_confusion_matrix = np.zeros((num_joints, num_joints))

    with torch.no_grad():
        for subject in subjects:
            sequences = sorted(os.listdir(os.path.join(mocap_dir, subject)))
            sequences = [x for x in sequences if x.endswith(".c3d")]
            for sequence in sequences:
                markers_filename = os.path.join(mocap_dir, subject, sequence)

                mocap_markers = Markers(markers_filename)
                points = mocap_markers.get_points()
                points = np.nan_to_num(points, 0)
                points = cleanup_markers(points)
                mocap_markers.set_points(points)

                num_frames = points.shape[0]

                markers = torch.from_numpy(mocap_markers.get_points()).float().to(device)  # []

                segmented_markers = marker_segmenter_model.forward_sequence(
                    torch.unsqueeze(markers, dim=0),
                    torch.zeros((1, num_frames, 24, 3)).to(device),  # the video modality is unused
                    stride=4 * int(mocap_markers.get_frequency() // 30),
                    center=True,
                )  # [1, M, P]
            
            segmented_markers = torch.argmax(segmented_markers, dim=-1)[0]  # [F, M]

            # get part assignment for each marker
            marker_labels = mocap_markers.get_labels()
            
            missing_indices = []
            marker_index = 0
            for marker_label in marker_labels:
                if marker_label not in smpl_vids.keys():
                    missing_indices.append(marker_index)
                marker_index += 1
            valid_indices = list(range(markers.shape[1]))
            for x in missing_indices:
                if x in valid_indices:
                    valid_indices.remove(x)

            marker_vids = [smpl_vids[x] for x in marker_labels if x in smpl_vids.keys()]
            gt_markers = vertex_labels[marker_vids]  # [M]
            gt_markers_expanded = torch.repeat_interleave(gt_markers[None], dim=0, repeats=num_frames)  # [F, M]

            confusion_matrix = matrix(
                preds = segmented_markers[:, valid_indices],
                target = gt_markers_expanded,
            ).detach().cpu().numpy()

            full_confusion_matrix += confusion_matrix

    normalized_full_confusion_matrix = full_confusion_matrix.astype(np.float32) / full_confusion_matrix.sum(-1)[:, np.newaxis]
    normalized_full_confusion_matrix = np.nan_to_num(normalized_full_confusion_matrix)
    df_cm = pd.DataFrame(normalized_full_confusion_matrix, joint_names, joint_names)
    plt.figure(figsize = (16, 12))
    ax = sn.heatmap(data=df_cm, annot=True, fmt="0.2f")
    ax.axes.set_title("Accuracy for marker segmentation", fontsize=30)
    ax.set_xlabel("Pred Joint Name", fontsize=20)
    ax.set_ylabel("Actual Joint Name", fontsize=20)
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--gpu", type=int, help="GPU ID")
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    args = parser.parse_args()

    visualize_part_segmentation_confusion_matrix(args)
