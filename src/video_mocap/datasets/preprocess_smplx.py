"""
This script copies subsets of the SMPL-X ground truth from the full sequence versions
"""
import argparse
import os

import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="video_mocap directory", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    freq = 30
    padding = {
        "cmu_kitchen_pilot": [5, 5],
        "cmu_kitchen_pilot_rb": [5, 5],
        "umpm": [5, 5],
        "moyo_train": [2, 2],
        "moyo_val": [2, 2],
    }[args.dataset]
    window = {
        "cmu_kitchen_pilot": 15,
        "cmu_kitchen_pilot_rb": 15,
        "umpm": 15,
        "moyo_train": 3,
        "moyo_val": 3,
    }[args.dataset]

    input_c3d_dir = os.path.join(args.input_dir, args.dataset, "mocap")
    input_smplx_dir = os.path.join(args.input_dir, args.dataset + "_full", "smplx")
    output_smplx_dir = os.path.join(args.input_dir, args.dataset, "smplx")
    os.makedirs(output_smplx_dir, exist_ok=True)

    # split the full sequence SMPL-X
    num_sequences = 0

    offsets = {}
    subjects = os.listdir(input_c3d_dir)
    for subject in sorted(subjects):
        offsets[subject] = {}
        sequences = os.listdir(os.path.join(input_c3d_dir, subject))
        for sequence in sorted(sequences):
            if not sequence.endswith(".c3d"):
                continue

            sequence_name = "_".join(sequence.split(".c3d")[0].split("_")[:-1])
            if sequence_name not in offsets[subject]:
                offsets[subject][sequence_name] = []

            offset = int(sequence.split(".c3d")[0].split("_")[-1])
            offsets[subject][sequence_name].append(offset)
            num_sequences += 1

    # split the full sequence SMPL-X
    subjects = os.listdir(input_smplx_dir)
    progress = tqdm(total=num_sequences)
    for subject in sorted(subjects):
        sequences = os.listdir(os.path.join(input_smplx_dir, subject))
        for sequence in sorted(sequences):
            data = np.load(os.path.join(input_smplx_dir, subject, sequence))
            sequence_name = sequence.split("_00000000_stageii.npz")[0]

            freq = data["mocap_frame_rate"].item()

            if sequence_name not in offsets[subject]:
                print("Skipping", sequence_name, "because cannot find")
                continue

            for offset in offsets[subject][sequence_name]:
                output_name = sequence_name + "_" + str(offset).zfill(8) + "_stageii.npz"

                window_size = int(window * freq)

                output_data = {}
                output_data["betas"] = data["betas"]
                output_data["gender"] = data["gender"]
                output_data["poses"] = data["poses"][offset:offset+window_size]
                output_data["trans"] = data["trans"][offset:offset+window_size]
                output_data["mocap_frame_rate"] = data["mocap_frame_rate"]

                os.makedirs(os.path.join(output_smplx_dir, subject), exist_ok=True)
                np.savez(os.path.join(output_smplx_dir, subject, output_name), **output_data)

                progress.update(1)

    progress.close()
