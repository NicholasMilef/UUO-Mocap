import numpy as np
import torch
from torch.utils.data import DataLoader

from video_mocap.datasets.dataset_mocap import DatasetMocap
from video_mocap.utils.marker_layout import compute_marker_labels_from_layout, compute_markers_from_layout, get_marker_layout
from video_mocap.utils.smpl import SmplInferenceGender
from video_mocap.utils.smpl_utils import get_marker_vertex_id


class MarkersSyntheticStructured:
    """
    Marker class that uses AMASS and structured
    """
    def __init__(self, filename, layout, shuffle=False, parts=None):
        self.filename = filename

        parts_set = None
        if parts is not None:
            parts_set = [parts]
        
        self.dataset = DatasetMocap(
            batch_size=1,
            num_markers=1,  # marker number doesn't really matter
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

        smpl = SmplInferenceGender(torch.device("cpu"))

        for _, data in enumerate(self.dataloader):
            smpl_output = smpl(
                poses=data["poses"][:, :, 3:72].float(),
                betas=data["betas"].float(),
                root_orient=data["poses"][:, :, :3].float(),
                trans=data["trans"][:, :, :3].float(),
                gender_one_hot=data["gender_one_hot"].float(),
                pose2rot=True,
                compute_part_labels=True,
            )

            vertex_ids = [get_marker_vertex_id(x) for x in get_marker_layout(layout)]
            self.points = compute_markers_from_layout(
                vertices=smpl_output["vertices"],
                triangles=torch.from_numpy(smpl.smpls["male"].faces.astype(np.int32)),
                marker_vertex_ids=vertex_ids,
            )["marker_pos"][0].detach().cpu().numpy()

            lbs_weights = 0
            lbs_weights += smpl.smpls["male"].lbs_weights * data["gender_one_hot"][0][0]
            lbs_weights += smpl.smpls["female"].lbs_weights * data["gender_one_hot"][0][1]
            marker_labels = compute_marker_labels_from_layout(
                marker_vertex_ids=vertex_ids,
                lbs_weights=lbs_weights,
            )  # [M]

            points = []
            for part_id in parts_set[0][0]:
                for marker_id in range(marker_labels.shape[0]):
                    if marker_labels[marker_id].item() == part_id:
                        points.append(marker_id)
            self.points = self.points[:, sorted(points)]

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
