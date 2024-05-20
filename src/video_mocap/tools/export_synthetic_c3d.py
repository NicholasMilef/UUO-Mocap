import argparse
import os

import ezc3d
import numpy as np

from video_mocap.markers.markers_synthetic import MarkersSynthetic
from video_mocap.utils.random_utils import set_random_seed


def generate_synthetic_c3d(markers):
    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [markers.get_frequency()]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [markers.get_frequency()]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = tuple(["point" + str(i) for i in range(markers.get_num_markers())])
    c3d["data"]["points"] = np.transpose(markers.get_points(), (2, 1, 0))

    return c3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_markers", nargs="+", type=int, help="number of markers", default=[])
    parser.add_argument("--num_samples", type=int, help="number", default=10)
    parser.add_argument("--parts", nargs="+", type=str, help="parts", default=None)
    parser.add_argument("--subjects", nargs="+", type=str, help="subject names", default=None)
    args = parser.parse_args()

    # generate each .c3d sequence
    input_dir = os.path.join(args.input_dir, args.dataset)
    if args.subjects is None:
        subjects = os.listdir(os.path.join(input_dir, "smpl"))
    else:
        subjects = args.subjects
    sequence_num = 0

    for s_i in range(args.num_samples):
        for subject in subjects:
            sequences = os.listdir(os.path.join(input_dir, "smpl", subject))
            sequences = [x for x in sequences if x.endswith("_stageii.npz")]
            for sequence in sequences:
                input_filename = os.path.join(input_dir, "smpl", subject, sequence)

                set_random_seed(s_i)

                markers = MarkersSynthetic(
                    input_filename,
                    num_markers=max(args.num_markers),
                    parts=args.parts,
                )
                markers_points = markers.get_points()

                for num_markers in args.num_markers:
                    markers.set_points(markers_points[:, :num_markers, :])

                    c3d = generate_synthetic_c3d(markers)

                    output_filename = os.path.join(
                        input_dir,
                        "mocap_synthetic___" + str(s_i) + "_" + str(num_markers),
                        subject,
                        sequence.replace("_stageii.npz", ".c3d"),
                    )

                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    c3d.write(output_filename)

                sequence_num += 1
