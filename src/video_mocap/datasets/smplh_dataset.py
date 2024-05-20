import numpy as np
import trimesh
import torch
import random
from torch.utils.data import Dataset

from video_mocap.utils.smpl import SmplInferenceGender


class SMPLHDataset(Dataset):
    def __init__(
        self,
        batch_size,
        device,
        parts_set=None,  # list of lists
        shuffle_parts=True,
        seed=None,
    ):
        self.batch_size = batch_size
        self.device = device
        self.shuffle_parts = shuffle_parts
        self.seed = seed

        self.male_bm = SmplInferenceGender(device)
        if parts_set is None:
            self.parts_set = [[list(range(0, self.male_bm.smpls["male"].NUM_JOINTS)), 1.0]]
        else:
            self.parts_set = parts_set

        output = self.male_bm(
            poses=torch.zeros((1, 1, 69)).to(self.device),
            betas=torch.zeros((1, 10)).to(self.device),
            root_orient=torch.zeros((1, 1, 3)).to(self.device),
            trans=torch.zeros((1, 1, 3)).to(self.device),
            gender_one_hot=torch.FloatTensor([[1.0, 0.0]]).to(self.device),
            pose2rot=True,
        )

        # male and female lbs weights are slightly different, so they are averaged
        lbs_weights = (self.male_bm.smpls["male"].lbs_weights + self.male_bm.smpls["female"].lbs_weights) / 2
        triangles = self.male_bm.smpls["male"].faces.astype(np.int32)

        self.face_weights = []
        self.face_weights_p = []
        for parts, weight in self.parts_set:
            face_weights = np.zeros((triangles.shape[0]))
            weight_sums = []
            for part in parts:
                weight_sum = lbs_weights[triangles[:, 0]][:, part] +\
                              lbs_weights[triangles[:, 1]][:, part] +\
                              lbs_weights[triangles[:, 2]][:, part]

                face_weights += (weight_sum.detach().cpu().numpy() / 3)  # / (torch.sum(weight_sum).item() / 3)
                weight_sums.append(weight_sum)

            self.face_weights.append(face_weights)
            self.face_weights_p.append(weight)

        self.body_mesh = trimesh.Trimesh(
            vertices=output["vertices"][0, 0].detach().cpu().numpy(),
            faces=self.male_bm.smpls["male"].faces,
            process=False,
        )

        faces = torch.from_numpy(self.male_bm.smpls["male"].faces.astype(np.int32))
        self.body = {
            "v": output["vertices"][0, 0].detach().cpu(),
            "f": faces.long(),
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.shuffle_parts:
            face_weight = random.choices(self.face_weights, self.face_weights_p)[0]
        else:
            face_weight = self.face_weights[idx % len(self.face_weights)]

        points, face_indices = trimesh.sample.sample_surface(
            self.body_mesh,
            self.batch_size,
            face_weight=face_weight,
            seed=self.seed,
        )

        i0 = self.body["f"][face_indices][:, 0]  # [N]
        i1 = self.body["f"][face_indices][:, 1]  # [N]
        i2 = self.body["f"][face_indices][:, 2]  # [N]

        v0 = self.body["v"][i0]  # [N, 3]
        v1 = self.body["v"][i1]  # [N, 3]
        v2 = self.body["v"][i2]  # [N, 3]

        vertices = torch.stack((v0, v1, v2), axis=1).detach().cpu().numpy()

        barycentric_coords = torch.from_numpy(
            trimesh.triangles.points_to_barycentric(vertices, np.array(points)),
        ).float()  # [N, 3]

        output = {}

        barycentric_coords_one_hot = torch.zeros((self.batch_size, self.body["v"].shape[0]))
        barycentric_coords_one_hot.scatter_(1, i0.unsqueeze(1), barycentric_coords[:, [0]])
        barycentric_coords_one_hot.scatter_(1, i1.unsqueeze(1), barycentric_coords[:, [1]])
        barycentric_coords_one_hot.scatter_(1, i2.unsqueeze(1), barycentric_coords[:, [2]])

        output["barycentric_coords"] = barycentric_coords_one_hot
        output["pos"] = torch.from_numpy(np.array(points)).float()

        triangles = np.zeros((output["pos"].shape[0], 3), dtype=np.int32)
        triangles[:, 0] = i0
        triangles[:, 1] = i1
        triangles[:, 2] = i2
        output["triangles"] = torch.from_numpy(triangles)

        return output
