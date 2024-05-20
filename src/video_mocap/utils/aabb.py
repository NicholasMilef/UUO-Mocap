import torch
import torch


def get_aabb(
    points: torch.Tensor,  # [F, M, 3]
) -> torch.Tensor:  # [F, 3, 2]
    aabb = torch.zeros((points.shape[0], 3, 2), dtype=points.dtype).to(points.device)

    aabb[:, 0, 0] = torch.min(points[..., 0], dim=1)[0]  # x min
    aabb[:, 0, 1] = torch.max(points[..., 0], dim=1)[0]  # x max
    aabb[:, 1, 0] = torch.min(points[..., 1], dim=1)[0]  # y min
    aabb[:, 1, 1] = torch.max(points[..., 1], dim=1)[0]  # y max
    aabb[:, 2, 0] = torch.min(points[..., 2], dim=1)[0]  # z min
    aabb[:, 2, 1] = torch.max(points[..., 2], dim=1)[0]  # z max

    return aabb


def get_aabb_volume(
    aabb: torch.Tensor,  # [F, 3, 2]
):  # [F]
    diff = aabb[:, :, 1] - aabb[:, :, 0]
    volume = diff[:, 0] * diff[:, 1] * diff[:, 2]
    return volume