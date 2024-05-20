import os

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix

from human_body_prior.tools.rotation_tools import aa2matrot


class DatasetSMPLHMotion(Dataset):
    def __init__(
        self,
        split="train",
        sequence_length=32,
        batch_size=16,
        features=[],
        augmentations=[],
        stride=1,
        device=torch.device("cpu"),
        filename=None,
        seed=None,
    ):
        self.split = split
        self.sequence_data = {}
        self.stride = stride

        # create dataset splits
        if self.split == "train":
            self.dataset_names = ["ACCAD", "BioMotionLab_NTroje", "CMU", "EKUT", "Eyes_Japan_Dataset", "KIT", "MPI_Limits"]
        elif self.split == "valid":
            self.dataset_names = ["SFU", "BMLhandball"]

        self.sequence_indices = []
        self.sequence_length = sequence_length
        self.features = features

        # count number of samples
        if filename is None:
            root_dir = "./data/processed/SMPL_H_G/"
            for dataset_name in self.dataset_names:
                subjects = os.listdir(os.path.join(root_dir, dataset_name))
                for subject in subjects:
                    subject_key = dataset_name + "___" + subject
                    self.sequence_data[subject_key] = {}
                    sequences = os.listdir(os.path.join(root_dir, dataset_name, subject))

                    for sequence in sequences:
                        filename = os.path.join(root_dir, dataset_name, subject, sequence)
                        with np.load(filename) as sequence_data:
                            self.sequence_data[subject_key][sequence] = {}
                            for key, value in sequence_data.items():
                                self.sequence_data[subject_key][sequence][key] = value
                            self.sequence_indices.append((subject_key, sequence))
        else:
            subject_key = filename
            self.sequence_data[subject_key] = {}
            with np.load(filename) as sequence_data:
                sequence = filename
                self.sequence_data[subject_key][sequence] = {}
                for key, value in sequence_data.items():
                    self.sequence_data[subject_key][sequence][key] = np.array(value)
                self.sequence_indices.append((subject_key, sequence))


    def __len__(self):
        return len(self.sequence_indices)


    def __getitem__(self, idx):
        subject_key, sequence_key = self.sequence_indices[idx]
        data = self.sequence_data[subject_key][sequence_key]

        output = {}

        output["trans"] = data["trans"]
        output["betas"] = data["betas"][:10]
        output["poses"] = data["poses"]
        if data["gender"] == "male":
            output["gender_one_hot"] = np.array([1.0, 0.0], dtype=np.float32)
        elif data["gender"] == "female":
            output["gender_one_hot"] = np.array([0.0, 1.0], dtype=np.float32)

        num_frames = data["trans"].shape[0]
        if self.sequence_length > 0:
            start_frame = 0
            seq_len = self.sequence_length * self.stride
            if num_frames > seq_len:
                start_frame = np.random.randint(max(num_frames - seq_len, 0))
            end_frame = min(start_frame + seq_len, num_frames)
            indices = np.arange(start_frame, end_frame, self.stride)

            output["trans"] = pad_sequence(output["trans"][indices], self.sequence_length)
            output["poses"] = pad_sequence(output["poses"][indices], self.sequence_length)

        return output


    def get_sequence(self, subject, sequence):
        i = 0
        for (sub, seq) in self.sequence_indices:
            if (sub.split("___")[1] == subject) and (seq == sequence + "_poses.npz"):
                return self.__getitem__(i)
            i += 1
        return None


def pad_sequence(sequence, sequence_length):
    num_frames = sequence_length
    if sequence.shape[0] != sequence_length:
        padding_frames = sequence_length - sequence.shape[0]
        padding = np.repeat(sequence[[-1]], repeats=padding_frames, axis=0)
        sequence = np.concatenate((sequence, padding), axis=0)

    return sequence


def apply_random_rotation_to_pos(
    points: torch.Tensor,  # [N, F, 3]
    angle: torch.Tensor,  # [N]
) -> torch.Tensor:
    batch_size, num_frames, _ = points.shape

    points_homogenous = torch.ones((batch_size, num_frames, 4)).to(points.device)
    points_homogenous[:, :, :3] = points

    up_vector = torch.zeros((batch_size, num_frames, 3)).to(points.device)
    up_vector[:, :, 2] = torch.unsqueeze(-angle, dim=1)  # negative due to PyTorch3d conventions

    random_rotation = torch.zeros((batch_size, num_frames, 4, 4)).to(points.device)
    random_rotation[:, :, :3, :3] = axis_angle_to_matrix(up_vector)
    random_rotation[:, :, 3, 3] = 1.0

    transformed_points = (points_homogenous.unsqueeze(2) @ random_rotation)[:, :, 0, :3]
    return transformed_points


def apply_random_rotation_to_rot(
    rots: torch.Tensor,  # [N, F, 3]
    angle: torch.Tensor,  # [N]
) -> torch.Tensor:
    rots_matrices = axis_angle_to_matrix(rots)

    up_vector = torch.zeros_like(rots)
    up_vector[:, :, 2] = torch.unsqueeze(angle, dim=1)

    random_rotation = axis_angle_to_matrix(up_vector)
    rots_matrices = random_rotation @ rots_matrices
    rots_aa = matrix_to_axis_angle(rots_matrices)
    return rots_aa


def apply_random_translation_to_pos(
    pos: torch.Tensor,  # [N, F, 3]
    std: torch.Tensor,  # [N]
    center: bool=False,
) -> torch.Tensor:
    transformed_points = pos

    if center:
        median = torch.median(pos[:, :, :2], dim=1, keepdim=True).values  # [N, 2]
        transformed_points[:, :, :2] = transformed_points[:, :, :2] - median

    offset = torch.normal(
        torch.zeros_like(pos[:, [0], :2]),
        torch.ones_like(pos[:, [0], :2]) * std,
    )
    transformed_points[:, :, :2] = transformed_points[:, :, :2] + offset

    return transformed_points


def world_to_local_pos(
    pos,  # [N, F, J, 3]
    trans,  # [N, F, 3]
    root_orient,  # [N, F, 3]
):
    local_pos = pos - torch.unsqueeze(trans, dim=2)

    batch_size, num_frames, num_markers, _ = pos.shape
    output = torch.zeros_like(pos)
    for f in range(num_frames):
        root_orient_mat = aa2matrot(root_orient[:, f]).unsqueeze(dim=1)  # [N, 1, 3, 3]
        root_orient_mat = torch.repeat_interleave(
            root_orient_mat, dim=1, repeats=num_markers,
        )  # [N, M, 3, 3]
        inv_root_orient_mat = torch.permute(root_orient_mat, (0, 1, 3, 2))  # [N, M, 3, 3]
        points = (inv_root_orient_mat @ local_pos[:, f].unsqueeze(dim=-1))[..., 0]  # [N, M, 3]
        output[:, f] = points  # [N, M, 3]

    return output
