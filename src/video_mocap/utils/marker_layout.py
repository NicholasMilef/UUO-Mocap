from typing import Dict, List

import numpy as np
import torch
import trimesh


# marker layouts as defined by SOMA
marker_layouts = {
    "cmu_41": [
        "C7",
        "CLAV",
        "LANK",
        "LBHD",
        "LBWT",
        "LELB",
        "LFIN",
        "LFHD",
        "LFRM",
        "LFWT",
        "LHEE",
        "LIWR",
        "LKNE",
        "LMT5",
        "LOWR",
        "LSHN",
        "LSHO",
        "LTHI",
        "LTOE",
        "LUPA",
        "RANK",
        "RBAK",
        "RBHD",
        "RBWT",
        "RELB",
        "RFHD",
        "RFIN",
        "RFWT",
        "RHEE",
        "RIWR",
        "RKNE",
        "RMT5",
        "RSHN",
        "RSHO",
        "RTHI",
        "RTOE",
        "RUPA",
        "STRN",
        "T10",
    ]
}


def compute_markers_from_layout(
    vertices: torch.Tensor,  # [N, F, V, 3]
    triangles: torch.Tensor,  # [Faces, 3]
    marker_vertex_ids: List,  # [N, M]
    marker_offset: float=0.0095,  # consistent with SOMA (9.5mm offset)
) -> Dict:
    batch_size, num_frames, _, _ = vertices.shape

    num_markers = len(marker_vertex_ids)

    marker_pos = np.zeros((batch_size, num_frames, num_markers, 3))    
    marker_normals = np.zeros((batch_size, num_frames, num_markers, 3))
    for batch in range(batch_size): 
        for frame in range(num_frames):
            smoothmesh = trimesh.Trimesh(
                vertices=vertices[batch, frame].detach().cpu().numpy(),
                faces=triangles.detach().cpu().numpy(),
            )            
            marker_pos[batch, frame] = vertices[batch, frame, marker_vertex_ids]
            marker_normals[batch, frame] = smoothmesh.vertex_normals[marker_vertex_ids]

    output = {}

    output["marker_pos"] = marker_pos + (marker_normals * marker_offset)
    output["marker_pos"] = torch.from_numpy(output["marker_pos"])

    return output


def compute_marker_labels_from_layout(
    marker_vertex_ids: List,  # [M]
    lbs_weights: torch.Tensor,  # [V, J]
) -> torch.Tensor:
    marker_labels = torch.argmax(lbs_weights[marker_vertex_ids], dim=-1)
    return marker_labels


def get_marker_layout(marker_layout):
    return marker_layouts[marker_layout]


if __name__ == "__main__":
    print("CMU dataset length", len(marker_layouts["cmu"]))