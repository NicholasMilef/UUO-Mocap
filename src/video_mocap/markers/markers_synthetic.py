import os

import ezc3d
import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix
import torch
from torch.utils.data import DataLoader

from video_mocap.datasets.dataset_mocap import DatasetMocap
from video_mocap.utils.smpl import SmplInference


class MarkersSynthetic:
    """
    Marker class that uses AMASS
    """
    def __init__(self, filename, num_markers=10, shuffle=False, parts=None):
        self.filename = filename

        parts_set = None
        if parts is not None:
            parts_set = [parts]
        
        self.dataset = DatasetMocap(
            batch_size=1,
            num_markers=num_markers,
            sequence_length=-1,
            parts_set=parts_set,
            filename=self.filename,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        self.units = "m"

        smpl = SmplInference(torch.device("cpu"))

        for _, data in enumerate(self.dataloader):
            num_frames = data["poses"].shape[1]

            pose_body = data["poses"][0, :, 3:72].float()
            betas = torch.repeat_interleave(data["betas"], repeats=num_frames, dim=0).float()
            root_orient = data["poses"][0, :, :3].float()
            trans = data["trans"][0, :, :3].float()

            pose_body = axis_angle_to_matrix(pose_body.reshape(num_frames, -1, 3))
            root_orient = axis_angle_to_matrix(root_orient.reshape(num_frames, -1, 3))

            smpl_output = smpl(
                poses=pose_body,
                betas=betas,
                root_orient=root_orient,
                trans=trans,
            )

            self.points = self.dataset.compute_markers(
                vertices=smpl_output["vertices"][None],
                vertex_part_labels=None,
            )["marker_pos"][0].detach().cpu().numpy()  # [F, 10, 3]

        if shuffle:
            self.points_temp = np.zeros_like(self.points)
            for f in range(self.points.shape[0]):
                permutation = np.random.permutation(self.points.shape[1])
                self.points_temp[f] = np.ascontiguousarray(self.points[f, permutation])
            self.points = self.points_temp

        self.freq = int(np.load(filename)["mocap_framerate"])

    def get_points(self):
        return self.points
    
    def set_points(self, points):
        self.points = points

    def get_num_markers(self):
        return self.points.shape[1]

    def __len__(self):
        return self.points.shape[0]

    def get_duration(self):
        return self.freq * self.points.shape[0]

    def get_frequency(self):
        return self.freq
