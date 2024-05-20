import igl
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset

from video_mocap.utils.smpl import SmplInferenceGender


class SMPLHDiffDataset(Dataset):
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

        vertices = output["vertices"].detach().cpu().numpy()
        padding = 0.1  # padding for SDF

        self.min_x = np.min(vertices[0, 0, :, 0]) - padding
        self.max_x = np.max(vertices[0, 0, :, 0]) + padding
        self.min_y = np.min(vertices[0, 0, :, 1]) - padding
        self.max_y = np.max(vertices[0, 0, :, 1]) + padding
        self.min_z = np.min(vertices[0, 0, :, 2]) - padding
        self.max_z = np.max(vertices[0, 0, :, 2]) + padding

        faces = torch.from_numpy(self.male_bm.smpls["male"].faces.astype(np.int32))
        self.body = {
            "v": output["vertices"][0, 0].detach().cpu(),
            "f": faces.long(),
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        samples_x = np.random.uniform(low=self.min_x, high=self.max_x, size=(self.batch_size, 1))  # [N]
        samples_y = np.random.uniform(low=self.min_y, high=self.max_y, size=(self.batch_size, 1))  # [N]
        samples_z = np.random.uniform(low=self.min_z, high=self.max_z, size=(self.batch_size, 1))  # [N]
        samples = np.concatenate((samples_x, samples_y, samples_z), axis=-1)
        distances, face_indices, points = igl.signed_distance(samples, self.body_mesh.vertices, self.body_mesh.faces)

        output = {}
        output["pos"] = torch.from_numpy(np.array(samples)).float()
        output["pos_diff"] = torch.from_numpy(np.array(points)).float()
        return output
