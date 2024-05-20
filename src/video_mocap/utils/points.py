from typing import Dict

import numpy as np


def geometric_median(
    points,  # [N, 3]
):
    num_points = points.shape[0]

    points_exp = np.repeat(np.expand_dims(points, axis=0), repeats=num_points, axis=0)
    points_exp_t = np.repeat(np.expand_dims(points, axis=1), repeats=num_points, axis=1)
    points_diff = np.linalg.norm(points_exp - points_exp_t, axis=-1)
    points_dist = np.sum(points_diff, axis=1)

    return points[np.argmin(points_dist)]


def closest_point(
    query_points,  # [N_0, 3]
    point_cloud,  # [N_1, 3]
) -> Dict:
    n_0 = query_points.shape[0]
    n_1 = point_cloud.shape[0]

    query_points_expanded = np.repeat(
        query_points[None, :, :],
        repeats=n_1,
        axis=0,
    )
    point_cloud_expanded = np.repeat(
        point_cloud[:, None, :],
        repeats=n_0,
        axis=1,
    )
    distances = np.linalg.norm(query_points_expanded - point_cloud_expanded, axis=-1)

    output = {}
    output["vertex_indices"] = np.argmin(distances, axis=0)
    output["distances"] = np.min(distances, axis=0)
    output["points"] = point_cloud[output["vertex_indices"]]
    return output


if __name__ == "__main__":
    points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.3, 0.3, 0.3],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ])

    median = geometric_median(points)
    print(median)
