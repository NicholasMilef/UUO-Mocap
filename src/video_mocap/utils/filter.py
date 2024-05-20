import argparse

import numpy as np
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d, rotation_6d_to_matrix
import torch
from scipy.signal import savgol_filter


def filter_rotations(
    rotations: np.ndarray,  # [F, J, 3, 3]
) -> np.ndarray:
    rotations_shape = rotations.shape

    # filter data
    rotations_flattened = rotations.reshape((num_frames, -1))  # [F, J*3*3]
    rotations_flattened = savgol_filter(rotations_flattened, 11, 3, axis=0, mode="nearest")  # [F, J*3*3]
    output = rotations_flattened.reshape(rotations_shape)

    # correct output
    output_torch = torch.from_numpy(output)
    output_torch = rotation_6d_to_matrix(matrix_to_rotation_6d(output_torch))

    return output_torch.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    data = dict(np.load(args.input))
    num_frames = data["poses"].shape[0]

    poses_mat_torch = torch.from_numpy(data["poses"].reshape((num_frames, -1, 3)))
    poses_mat = axis_angle_to_matrix(poses_mat_torch).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.plot(np.arange(num_frames), poses_mat[:, 0, 0, 0])

    filtered_poses_mat = filter_rotations(poses_mat)

    plt.plot(np.arange(num_frames), filtered_poses_mat[:, 0, 0, 0])
    plt.savefig("test_plot.png")

    filtered_poses_mat_torch = torch.from_numpy(filtered_poses_mat)
    filtered_poses_mat_torch = matrix_to_axis_angle(filtered_poses_mat_torch).reshape((num_frames, -1))
    data["poses"] = filtered_poses_mat_torch.detach().cpu().numpy()

    np.savez(args.output, **data)
