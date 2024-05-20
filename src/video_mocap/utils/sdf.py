import numpy as np
import torch

from video_mocap.models.pos2bc import Pos2BC
from video_mocap.models.pos_diff import PosDiff
from video_mocap.utils.smpl import SmplInferenceGender


class SDF:
    def __init__(self, device):
        self.data = np.load("./data/smpl_sdf.npz")
        self.points = torch.from_numpy(self.data["points"]).float().to(device)  # [X, Y, Z, 3]
        self.samples = torch.from_numpy(self.data["samples"]).float().to(device)  # [X, Y, Z, 3]

        self.pos2bc = Pos2BC()
        self.pos2bc.load_state_dict(
            torch.load("./checkpoints/barycentric_coords/final_2/pos2bc.pth", map_location=device),
        )
        self.pos2bc.eval()
        self.pos2bc.to(device)

        self.pos_diff = PosDiff()
        self.pos_diff.load_state_dict(
            torch.load("./checkpoints/barycentric_coords/pos_diff3/pos_diff.pth", map_location=device),
        )
        self.pos_diff.eval()
        self.pos_diff.to(device)

        self.smpl_inference = SmplInferenceGender(device=device)
        output = self.smpl_inference(
            poses=torch.zeros((1, 1, 69)).to(device),
            betas=torch.zeros((1, 10)).to(device),
            root_orient=torch.zeros((1, 1, 3)).to(device),
            trans=torch.zeros((1, 1, 3)).to(device),
            gender_one_hot=torch.FloatTensor([[1.0, 0.0]]).to(device),
            pose2rot=True,
        )
        self.vertices = output["vertices"]

        self.min_x = torch.min(self.samples[..., 0]).item()
        self.max_x = torch.max(self.samples[..., 0]).item()
        self.min_y = torch.min(self.samples[..., 1]).item()
        self.max_y = torch.max(self.samples[..., 1]).item()
        self.min_z = torch.min(self.samples[..., 2]).item()
        self.max_z = torch.max(self.samples[..., 2]).item()

    def points_to_barycentric_one_hot(self, points):
        res = self.samples.shape[0:3]

        """
        norm_points = torch.clone(points)
        norm_x = torch.clamp(norm_points[..., [0]], self.min_x, self.max_x)
        norm_y = torch.clamp(norm_points[..., [1]], self.min_y, self.max_y)
        norm_z = torch.clamp(norm_points[..., [2]], self.min_z, self.max_z)

        norm_x = ((norm_x - self.min_x) / (self.max_x - self.min_x)) * 2 - 1
        norm_y = ((norm_y - self.min_y) / (self.max_y - self.min_y)) * 2 - 1
        norm_z = ((norm_z - self.min_z) / (self.max_z - self.min_z)) * 2 - 1
        norm_points = torch.concatenate((norm_x, norm_y, norm_z), dim=-1)  # [M, 3]

        input_norm_points = torch.unsqueeze(torch.permute(self.samples, (3, 0, 1, 2)), dim=0)  # [1, 3, X, Y, Z]
        grid_norm_points = norm_points.reshape((1, -1, 1, 1, 3))  # [1, M, 1, 1, 3]
        surface_points = torch.nn.functional.grid_sample(input=input_norm_points, grid=grid_norm_points, align_corners=True)
        surface_points = torch.permute(surface_points[0, :, :, 0, 0], (1, 0))
        """

        # diff points
        pos_diff = self.pos_diff(points)["pos_diff"]  # [M, P]

        # get barycentric coords
        barycentric_coords_one_hot = torch.softmax(self.pos2bc(pos_diff)["barycentric_coords"], dim=-1)  # [M, V]

        #barycentric_coords_one_hot = torch.zeros((self.batch_size, self.body["v"].shape[0]))
        #barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])
        #barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])
        #barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])

        return barycentric_coords_one_hot

    def barycentric_one_hot_to_points(self, barycentric_one_hot):
        num_markers, num_vertices = barycentric_one_hot.shape
        vertices_expanded = torch.repeat_interleave(torch.unsqueeze(self.vertices[0, 0], dim=0), repeats=num_markers, dim=0)  # [M, V, 3]
        barycentric_one_hot_expanded = torch.repeat_interleave(torch.unsqueeze(barycentric_one_hot, dim=-1), repeats=3, dim=-1)  # [M, V, 3]
        virtual_markers = torch.sum(vertices_expanded * barycentric_one_hot_expanded, dim=1)
        return virtual_markers
