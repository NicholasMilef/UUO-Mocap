from typing import List

import igl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch3d.structures import Meshes, Pointclouds
import torch

from video_mocap.evaluation.point_mesh_face_distance_single_direction import point_mesh_face_distance_single_directional


def compute_marker_to_surface_distance_old(
    vertices: torch.Tensor,  # [F, M, 3]
    faces: torch.Tensor,  # [F, M, 3]
    markers: torch.Tensor,  # [F, M, 3]G
):
    meshes = Meshes(
        verts=vertices,
        faces=faces,
    )
    points = Pointclouds(markers)
    distance = point_mesh_face_distance_single_directional(meshes, points, min_triangle_area=0)
    return distance


def compute_marker_to_surface_distance(
    vertices: torch.Tensor,  # [F, M, 3]
    faces: torch.Tensor,  # [F, M, 3]
    markers: torch.Tensor,  # [F, M, 3]
):
    num_frames, num_markers, _ = markers.shape
    distances = np.zeros((num_frames, num_markers))
    
    markers_np = markers.detach().cpu().numpy()
    vertices_np = vertices.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    for i in range(num_frames):
        distances[i], _, _ = igl.signed_distance(
            markers_np[i],
            vertices_np[i],
            faces_np[i],
        )
    return torch.tensor(np.mean(np.abs(distances)))


def compute_MPJPE(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
):
    mpjpe = torch.norm(pred_joints - gt_joints, dim=-1)
    return torch.mean(mpjpe)


def compute_MPJPE_joints(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    joints_ids: List[int],
):
    mpjpe = torch.norm(pred_joints[:, joints_ids] - gt_joints[:, joints_ids], dim=-1)
    return torch.mean(mpjpe)


def compute_MPJVE(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    freq: float,
):
    pred_vel = (pred_joints[1:] - pred_joints[:-1]) * freq
    gt_vel = (gt_joints[1:] - gt_joints[:-1]) * freq
    mpjve = torch.norm(pred_vel - gt_vel, dim=-1)
    return torch.mean(mpjve)


def compute_MPJVE_joints(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    freq: float,
    joints_ids: List[int],
):
    pred_vel = (pred_joints[1:] - pred_joints[:-1]) * freq
    gt_vel = (gt_joints[1:] - gt_joints[:-1]) * freq
    mpjve = torch.norm(pred_vel[:, joints_ids] - gt_vel[:, joints_ids], dim=-1)
    return torch.mean(mpjve)


def compute_PA_MPJPE(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
):
    pred_joints_hat = compute_similarity_transform(pred_joints, gt_joints)
    mpjpe = torch.norm(pred_joints_hat - gt_joints, dim=-1)
    return torch.mean(mpjpe)


def compute_PA_MPJPE_joints(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    joints_ids: List[int],
):
    pred_joints_hat = compute_similarity_transform(pred_joints, gt_joints)
    mpjpe = torch.norm(pred_joints_hat[:, joints_ids] - gt_joints[:, joints_ids], dim=-1)
    return torch.mean(mpjpe)


def compute_PA_MPJVE(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    freq: float,
):
    pred_joints_hat = compute_similarity_transform(pred_joints, gt_joints)
    pred_vel = (pred_joints_hat[1:] - pred_joints_hat[:-1]) * freq
    gt_vel = (gt_joints[1:] - gt_joints[:-1]) * freq
    mpjve = torch.norm(pred_vel - gt_vel, dim=-1)
    return torch.mean(mpjve)


def compute_PA_MPJVE_joints(
    pred_joints: torch.Tensor,  # [F, J, 3]
    gt_joints: torch.Tensor,  # [F, J, 3]
    freq: float,
    joints_ids: List[int],
):
    pred_joints_hat = compute_similarity_transform(pred_joints, gt_joints)
    pred_vel = (pred_joints_hat[1:] - pred_joints_hat[:-1]) * freq
    gt_vel = (gt_joints[1:] - gt_joints[:-1]) * freq
    mpjve = torch.norm(pred_vel[:, joints_ids] - gt_vel[:, joints_ids], dim=-1)
    return torch.mean(mpjve)


def compute_V2V(
    pred_vertices: torch.Tensor,  # [F, V, 3]
    gt_vertices: torch.Tensor,  # [F, V, 3]
):
    v2v = torch.norm(pred_vertices - gt_vertices, dim=-1)
    return torch.mean(v2v)


# from https://raw.githubusercontent.com/shubham-goel/4D-Humans/main/hmr2/utils/pose_utils.py
def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)


def plot_error_heatmap(
    filename: str,
    error: np.ndarray,  # [F, J]
):
    ax = sns.heatmap(
        np.transpose(error),
        cmap="viridis",
        vmin=0.0,
        vmax=0.5,
        square=True,
        cbar_kws={"orientation": "horizontal"},
    )
    ax.get_figure().savefig(filename, dpi=300)
    plt.close()
