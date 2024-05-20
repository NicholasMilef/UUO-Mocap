import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from pytorch3d.loss import chamfer_distance

from video_mocap.markers.markers import Markers


def plot(name, ax, root, time, plot_type):
    if plot_type == "line":
        ax.plot(
            root[:, 0],
            root[:, 1],
            zs=root[:, 2],
            zdir="z",
        )
    elif plot_type == "points":
        cmap_name = {
            "mocap": plt.cm.Blues,
            "slahmr": plt.cm.Oranges,
        }

        ax.scatter(
            root[:, 0],
            root[:, 1],
            zs=root[:, 2],
            c=time,
            zdir="z",
            vmin=-root.shape[0],
            vmax=root.shape[0],
            cmap=cmap_name[name],
        )        


def viz_root(args):
    # setup plot
    ax = plt.figure().add_subplot(projection="3d")

    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    z_max = -np.inf

    num_iters = 100
    for iteration in range(num_iters):

        if "mocap" in args.methods:
            filename = os.path.join(
                args.input_dir,
                "mocap",
                args.subject,
                args.sequence + ".c3d",
            )
            markers = Markers(filename)
            points = np.nan_to_num(markers.get_points(), 0)  # [F, M, 3]
            root = np.median(points, axis=1)  # [F, 3]

            if args.num_mocap_frames is not None:
                root = root[:args.num_mocap_frames]

            plot("mocap", ax, root, np.arange(root.shape[0]), args.plot_type)

            x_min = np.minimum(x_min, np.min(root[:, 0]))
            x_max = np.maximum(x_max, np.max(root[:, 0]))
            y_min = np.minimum(y_min, np.min(root[:, 1]))
            y_max = np.maximum(y_max, np.max(root[:, 1]))
            z_max = np.maximum(z_max, np.max(root[:, 2]))

        if "slahmr" in args.methods:
            slahmr_dir = os.path.join(
                args.input_dir,
                "comparisons",
                "slahmr",
                args.subject,
                args.sequence,
                "motion_chunks",
            )
            slahmr_files = os.listdir(slahmr_dir)
            slahmr_files = sorted([x for x in slahmr_files if x.endswith("_world_results.npz")])
            slahmr_data = np.load(os.path.join(slahmr_dir, slahmr_files[-1]))

            root = np.zeros_like(slahmr_data["trans"][0])
            time = -np.ones((root.shape[0]), dtype=np.int32)
            for frame in range(slahmr_data["trans"].shape[1]):
                for person in range(slahmr_data["trans"].shape[0]):
                    if slahmr_data["track_mask"][person, frame]:
                        root[frame] = slahmr_data["trans"][person, frame]  # [N_people, F, 3]
                        time[frame] = frame

            # remove 0 columns
            root = root[np.any(root != 0, axis=-1)]
            time = time[time != -1]
            plot("slahmr", ax, root, time, args.plot_type)

            x_min = np.minimum(x_min, np.min(root[:, 0]))
            x_max = np.maximum(x_max, np.max(root[:, 0]))
            y_min = np.minimum(y_min, np.min(root[:, 1]))
            y_max = np.maximum(y_max, np.max(root[:, 1]))

        x_min = np.minimum(x_min, y_min)
        x_max = np.maximum(x_max, y_max)
        y_min = np.minimum(x_min, y_min)
        y_max = np.maximum(x_max, y_max)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0, z_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--methods", type=str, nargs="+", help="methods [mocap, slahmr]", default=["mocap", "slahmr"])
    parser.add_argument("--num_mocap_frames", type=int, default=None)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--plot_type", type=str, help="plot type", default="line")
    args = parser.parse_args()

    viz_root(args)
