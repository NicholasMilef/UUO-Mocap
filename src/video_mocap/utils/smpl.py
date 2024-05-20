import os
from typing import Dict

import smplx
import torch
import torch.nn as nn


class SmplInference(nn.Module):
    def __init__(
        self,
        device=torch.device("cpu"),
        gender="neutral",
    ):
        super(SmplInference, self).__init__()

        self.body_model_path = "./body_models/"

        self.device = device
        self.gender = gender

        self.smpl = smplx.create(
            self.body_model_path,
            model_type = "smpl",
            gender = gender,
            batch_size = 1,
        ).to(self.device)

    def forward(
        self,
        poses: torch.Tensor,  # [F, 23, 3, 3]
        betas: torch.Tensor,  # [F, 10]
        root_orient: torch.Tensor,  # [F, 1, 3, 3]
        trans: torch.Tensor,  # [F, 3]
    ) -> Dict:
        if betas.shape[1] != 10:
            raise ValueError("Betas array must have 10 beta values")

        output_data = self.smpl(
            body_pose = poses,
            betas = betas,
            global_orient = root_orient,
            transl = trans,
            pose2rot = False,
        )

        output = {}
        output["joints"] = output_data.joints
        output["vertices"] = output_data.vertices
        return output

    def get_lbs_weights(self):
        return self.smpl.lbs_weights


class SmplInferenceGender(nn.Module):
    def __init__(
        self,
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.body_model_path = "./body_models/"

        self.device = device

        self.smpls = {}
        for gender in ["male", "female"]:
            self.smpls[gender] = smplx.create(
                self.body_model_path,
                model_type = "smpl",
                gender = gender,
                batch_size = 1,
                dtype = torch.float32,
            ).to(self.device)

    def forward(
        self,
        poses: torch.Tensor,  # [N, F, 69] or [N, F, 23, 3, 3]
        betas: torch.Tensor,  # [N, 10]
        root_orient: torch.Tensor,  # [N, F, 3] or [N, F, 3, 3]
        trans: torch.Tensor,  # [N, F, 3]
        gender_one_hot: torch.Tensor,  # [N, 2]
        pose2rot: bool=True,
        compute_part_labels: bool=False,
    ) -> Dict:
        if betas.shape[1] != 10:
            raise ValueError("Betas array must have 10 beta values")

        if len(gender_one_hot.shape) != 2:
            raise ValueError("Gender one-hot vector must have 2 dimensions")

        batch_size, num_frames, _ = trans.shape
        vertices = torch.zeros((num_frames)).to(self.device)

        poses_rs = torch.reshape(poses, (-1, poses.shape[-1]))
        root_orient_rs = torch.reshape(root_orient, (-1, root_orient.shape[-1]))
        trans_rs = torch.reshape(trans, (-1, trans.shape[-1]))
        betas_rs = torch.repeat_interleave(torch.unsqueeze(betas, dim=0), dim=0, repeats=num_frames)
        betas_rs = torch.reshape(betas_rs, (-1, betas.shape[-1]))
        gender_one_hot_rs = torch.repeat_interleave(torch.unsqueeze(gender_one_hot, dim=1), dim=1, repeats=num_frames)
        gender_one_hot_rs = torch.reshape(gender_one_hot_rs, (-1, gender_one_hot.shape[-1], 1))

        output_data = {}
        for gender in ["male", "female"]:
            output_data[gender] = self.smpls[gender](
                body_pose = poses_rs,
                betas = betas_rs,
                global_orient = root_orient_rs,
                transl = trans_rs,
                pose2rot = pose2rot,
                dtype = torch.float32,
            )

        joints = output_data["male"]["joints"][:, :24] * gender_one_hot_rs[:, [0], :] +\
                 output_data["female"]["joints"][:, :24] * gender_one_hot_rs[:, [1], :]
        vertices = output_data["male"]["vertices"] * gender_one_hot_rs[:, [0], :] +\
                   output_data["female"]["vertices"] * gender_one_hot_rs[:, [1], :]
        if compute_part_labels:
            vertex_part_labels = self.smpls["male"].lbs_weights * gender_one_hot_rs[[0], [0], :] +\
                                 self.smpls["female"].lbs_weights * gender_one_hot_rs[[0], [1], :]
            vertex_part_labels = torch.repeat_interleave(torch.unsqueeze(vertex_part_labels, 0), repeats=batch_size, dim=0)

        output = {}
        output["joints"] = torch.reshape(joints, (batch_size, num_frames, 24, 3))  # [N, F, J, 3]
        output["vertices"] = torch.reshape(vertices, (batch_size, num_frames, 6890, 3))  # [N, F, V, 3]

        if compute_part_labels:
            output["vertex_part_labels"] = vertex_part_labels  # [N, V, J]

        return output


if __name__ == "__main__":
    poses = torch.zeros((1, 69))
    betas = torch.zeros((1, 10))
    root_orient = torch.zeros((1, 3))
    trans = torch.zeros((1, 3))

    model = SmplInference()
    data = model(
        poses = poses,
        betas = betas,
        root_orient = root_orient,
        trans = trans,
    )
