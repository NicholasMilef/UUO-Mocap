import argparse
import os

import ezc3d
import numpy as np

from video_mocap.markers.markers_synthetic_structured import MarkersSyntheticStructured
from video_mocap.utils.random_utils import set_random_seed
from video_mocap.utils.smpl_utils import get_joint_id, get_all_joint_ids


def generate_synthetic_structured_c3d(filename, layout, parts=None):
    if parts is not None:
        parts_id = [[get_joint_id(part) for part in parts], 1.0]
    else:
        parts_id = [get_all_joint_ids(), 1.0]

    markers = MarkersSyntheticStructured(
        filename,
        layout,
        parts=parts_id,
    )
    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [markers.get_frequency()]
    c3d["parameters"]["POINT"]["RATE"]["value"] = [markers.get_frequency()]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = tuple(["point" + str(i) for i in range(markers.get_num_markers())])
    c3d["data"]["points"] = np.transpose(markers.get_points(), (2, 1, 0))

    return c3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, help="description")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--layout", type=str, help="layout name", required=True)
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--parts", nargs="+", type=str, help="parts", default=None)
    parser.add_argument("--sequence", type=str, help="sequence", default=None)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    args = parser.parse_args()

    # find all SMPL sequences for subject
    if args.sequence is None:
        sequences = os.listdir(os.path.join(
            args.input_dir,
            args.subject,
        ))
        sequences = [x.split("_poses.npz")[0] for x in sequences if x.endswith("_poses.npz")]
    else:
        sequences = [args.sequence]

    # generate each .c3d sequence
    for sequence in sequences:
        set_random_seed(0)
        input_filename = os.path.join(
            args.input_dir,
            args.subject,
            sequence + "_poses.npz",
        )

        output_filename = os.path.join(
            args.output_dir,
            args.subject,
            sequence + "." + args.description + ".c3d",
        )

        c3d = generate_synthetic_structured_c3d(input_filename, args.layout, args.parts)

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        c3d.write(output_filename)
