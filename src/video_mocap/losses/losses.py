import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, input, target):
        return self.kl_div_loss(F.log_softmax(input, dim=-1), target)
    

class LineSegmentLoss(nn.Module):
    """
    Regularization loss for line segment
    Inspired by: https://stackoverflow.com/a/58660434
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        points,  # points on line segment [F, 2, 3]
        markers,  # marker positions [F, M, 3]
    ):
        line = points[:, [0]] - points[:, [1]]
        line_m = markers - points[:, [1]]
        norm_line = torch.norm(line, dim=-1)
        cross = torch.cross(line, line_m, dim=-1)
        norm_cross = torch.norm(cross, dim=-1)
        
        if self.reduction == "mean":
            loss = torch.mean(norm_cross / norm_line)
        elif self.reduction == "sum":
            loss = torch.sum(norm_cross / norm_line)

        return loss


def MarkerLoss(
    markers,  # markers [F, M, 3]
    virtual_markers,  # virtual markers [F, M, 3]
    marker_weights,  # [F, M]
    marker_distance,
):
    marker_loss = (torch.norm(markers - virtual_markers, dim=-1) - marker_distance) ** 2
    marker_loss = marker_loss * marker_weights
    return marker_loss


def ReprojectionLoss():
    return