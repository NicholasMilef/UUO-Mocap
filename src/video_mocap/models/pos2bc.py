import torch
import torch.nn as nn


class Pos2BC(nn.Module):
    def __init__(
        self,
        num_vertices=6890,  # number of vertices for SMPL-H mesh
    ):
        super().__init__()
        self.linear0 = nn.Linear(3, 128)
        self.linear1 = nn.Linear(128, 1024)
        self.linear2 = nn.Linear(1024, num_vertices)

    def forward(self, points):
        x0 = torch.relu(self.linear0(points))
        x1 = torch.relu(self.linear1(x0))
        bc_coords = self.linear2(x1)

        output = {
            "barycentric_coords": bc_coords,
        }
        return output
