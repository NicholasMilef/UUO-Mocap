import ezc3d

import numpy as np


class Markers:
    def __init__(self, filename, shuffle=False):
        self.filename = filename
        self.shuffle=True
        self.c3d = ezc3d.c3d(filename)

        self.units = self.c3d["parameters"]["POINT"]["UNITS"]["value"][0]
        if self.units == "m":
            self.scale_factor = 1
        elif self.units == "cm":
            self.scale_factor = 100
        elif self.units == "mm":
            self.scale_factor = 1000

        self.points = self.c3d["data"]["points"]  # [4, M, F]
        self.points = np.transpose(self.points, (2, 1, 0))[:, :, :3] / self.scale_factor  # [F, M, 3]

        if shuffle:
            self.points_temp = np.zeros_like(self.points)
            for f in range(self.points.shape[0]):
                permutation = np.random.permutation(self.points.shape[1])
                self.points_temp[f] = np.ascontiguousarray(self.points[f, permutation])
            self.points = self.points_temp

        self.freq = int(self.c3d["parameters"]["POINT"]["RATE"]["value"])
        
        if "LABELS" in self.c3d["parameters"]["POINT"]:
            self.labels = self.c3d["parameters"]["POINT"]["LABELS"]["value"]

    def get_points(self):
        return self.points
    
    def set_points(self, points):
        self.points = points

    def get_labels(self):
        return self.labels

    def get_num_markers(self):
        return self.points.shape[1]

    def __len__(self):
        return self.points.shape[0]

    def get_duration(self):
        return self.freq * self.points.shape[0]

    def get_frequency(self):
        return self.freq
