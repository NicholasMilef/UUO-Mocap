import torch
import torch.nn as nn


class PosDiff(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.linear0 = nn.Linear(3, 128)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 3)

    def forward(self, points):
        x0 = torch.relu(self.linear0(points))
        x1 = torch.relu(self.linear1(x0))
        pos_diff = self.linear2(x1)

        output = {
            "pos_diff": pos_diff,
        }
        return output
