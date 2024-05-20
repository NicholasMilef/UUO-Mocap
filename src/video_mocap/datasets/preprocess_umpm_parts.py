"""
This script is for processing the UMPM dataset
The labels for the markers is not consistent with MoSh, so we change them here.
Expected input file structure:

--base_dir
----p1_chair_2d
------Groundtruth
--------p1_chair_2.c3d
------Video
--------p1_chair_2_r.avi
----p1_grab_2d
"""
import argparse
import os
import json
import warnings

import ezc3d
import numpy as np
from tqdm import tqdm
from video_mocap.datasets.preprocess_utils import get_c3d_duration, get_c3d_freq, get_downsampled_indices, get_video_duration, preprocess_videos, shuffle_c3d
from video_mocap.utils.random_utils import set_random_seed


body_parts = {
    "left_arm": ["UMPM_LWREXT", "UMPM_LWRTOP", "UMPM_LWRLOW", "UMPM_LELBTOP", "UMPM_LELBEXT", "UMPM_LELBLOW", "UMPM_LSHLD"],
    "right_arm": ["UMPM_RWREXT", "UMPM_RWRTOP", "UMPM_RWRLOW", "UMPM_RELBTOP", "UMPM_RELBEXT", "UMPM_RELBLOW", "UMPM_RSHLD"],
    "left_leg": ["UMPM_LTOPLEG", "UMPM_LKNEEFR", "UMPM_LKNEEBK", "UMPM_LKNEEIS", "UMPM_LANKFR", "UMPM_LANKBK", "UMPM_LANKIS"],
    "right_leg": ["UMPM_RTOPLEG", "UMPM_RKNEEFR", "UMPM_RKNEEBK", "UMPM_RKNEEIS", "UMPM_RANKFR", "UMPM_RANKBK", "UMPM_RANKIS"],
    "left_shoulder": ["UMPM_LSHLD", "UMPM_BNECK", "UMPM_FRNECK", "UMPM_LELBTOP", "UMPM_LELBEXT", "UMPM_LELBLOW"],
    "right_shoulder": ["UMPM_RSHLD", "UMPM_BNECK", "UMPM_FRNECK", "UMPM_RELBTOP", "UMPM_RELBEXT", "UMPM_RELBLOW"],
    "left_forearm": ["UMPM_LWREXT", "UMPM_LWRTOP", "UMPM_LWRLOW", "UMPM_LELBTOP", "UMPM_LELBEXT", "UMPM_LELBLOW"],
    "right_forearm": ["UMPM_RWREXT", "UMPM_RWRTOP", "UMPM_RWRLOW", "UMPM_RELBTOP", "UMPM_RELBEXT", "UMPM_RELBLOW"],
    "left_lower_leg": ["UMPM_LKNEEFR", "UMPM_LKNEEBK", "UMPM_LKNEEIS", "UMPM_LANKFR", "UMPM_LANKBK", "UMPM_LANKIS"],
    "right_lower_leg": ["UMPM_RKNEEFR", "UMPM_RKNEEBK", "UMPM_RKNEEIS", "UMPM_RANKFR", "UMPM_RANKBK", "UMPM_RANKIS"],
    "left_ankle": ["UMPM_LANKFR", "UMPM_LANKBK", "UMPM_LANKIS"],
    "right_ankle": [ "UMPM_RANKFR", "UMPM_RANKBK", "UMPM_RANKIS"],
    "head": ["UMPM_FHEAD", "UMPM_RHEAD", "UMPM_LHEAD"],
}


def fix_label(label):
    label = label.upper()
    if label == "LKNSSBK":
        label = "LKNEEBK"
    return "UMPM_" + label


def preprocess_c3d_data(
    input_filename,
    output_dir,
    subject,
    sequence,
    dataset_name,
    window,
    duration,
    padding,
    freq,
):
    data = ezc3d.c3d(input_filename)
    o_labels = data["parameters"]["POINT"]["LABELS"]["value"]

    mocap_freq = data["parameters"]["POINT"]["RATE"]["value"][0]

    for parts_name in body_parts.keys():
        index = 0
        indices = {}
        labels = {}

        markers = data["data"]["points"]  # [4, M, F]
        _, num_markers, num_frames = markers.shape

        markers = markers[:, :, get_downsampled_indices(mocap_freq, freq, num_frames)]
        num_frames = markers.shape[-1]

        parts_list = body_parts[parts_name]
        for label in data["parameters"]["POINT"]["LABELS"]["value"]:
            if label.startswith("*"):
                index += 1
                continue
        
            if ":" in label:
                subject_initial = label.split(":")[0]
                if subject_initial not in labels:
                    labels[subject_initial] = []
                    indices[subject_initial] = []

                fixed_label = fix_label(label.split(":")[1])
                if parts_list is not None and fixed_label not in parts_list:
                    index += 1
                    continue

                labels[subject_initial].append(fixed_label)
                indices[subject_initial].append(index)
            else:
                if "" not in labels:
                    labels[""] = []
                    indices[""] = []

                fixed_label = fix_label(label)
                if parts_list is not None and fixed_label not in parts_list:
                    index += 1
                    continue

                labels[""].append(fixed_label)
                indices[""].append(index)
            index += 1

        if window is None:
            seq_len = int(duration * freq)
        else:
            seq_len = int(window * freq)

        total_length = int(duration * freq)
        if duration == -1:
            seq_len = num_frames
            total_length = num_frames
            padding = [0, 0]

        for subject_initial in labels.keys():
            for frame in range(int(padding[0] * freq), total_length - seq_len + 1, seq_len):
                c3d = ezc3d.c3d()
                c3d["parameters"]["POINT"]["UNITS"]["value"] = data["parameters"]["POINT"]["UNITS"]["value"]
                c3d["parameters"]["POINT"]["RATE"]["value"] = np.array([float(freq)])

                seq_markers = markers[:, indices[subject_initial], frame:frame+seq_len]

                if subject_initial != "":
                    subject_name = subject + "_" + subject_initial
                else:
                    subject_name = subject

                c3d["data"]["points"] = seq_markers
                c3d["parameters"]["POINT"]["LABELS"]["value"] = labels[subject_initial]

                if parts_list:
                    output_filename = os.path.join(output_dir, dataset_name, "mocap_parts___" + parts_name, subject_name, sequence)
                else:
                    output_filename = os.path.join(output_dir, dataset_name, "mocap", subject_name, sequence)

                output_filename_seq = output_filename.replace(".c3d", "_" + str(frame).zfill(8) + ".c3d")

                invalid_c3d = False
                for subframe in range(c3d["data"]["points"].shape[-1]):
                    count = np.count_nonzero(c3d["data"]["points"][:3, :, subframe]==0)
                    if count == c3d["data"]["points"][:3, :, subframe].size:
                        warnings.warn("WARNING: skipping sequence" + str(output_filename_seq) + " because of invalid marker positions")
                        invalid_c3d = True
                        break

                if not invalid_c3d:
                    os.makedirs(os.path.dirname(output_filename_seq), exist_ok=True)
                    c3d.write(output_filename_seq)

            with open(os.path.join(os.path.dirname(output_filename), "settings.json"), "w", encoding="utf-8") as f:
                json.dump({"gender": "neutral"}, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, help="target frequency", default=30)
    parser.add_argument("--input_dir", type=str, help="input directory of UMPM dataset", required=True)
    parser.add_argument("--output_dir", type=str, help="base directory of video_mocap datasets", required=True)
    parser.add_argument("--padding", nargs=2, type=int, help="padding for the animation", default=[5, 5])
    parser.add_argument("--window", type=int, help="duration for each subsequence in seconds", default=None)
    args = parser.parse_args()

    set_random_seed(0)

    dataset_name = "umpm"

    num_videos = 0
    num_mocap = 0

    padding = args.padding
    subjects = os.listdir(os.path.join(args.input_dir))

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}
    for subject in sorted(subjects):
        min_times[subject] = {}
        mocap_freq[subject] = {}
        parts_map[subject] = {}
        if not os.path.exists(os.path.join(args.input_dir, subject, "Groundtruth")):
            continue

        sequences = os.listdir(os.path.join(args.input_dir, subject, "Groundtruth"))
        sequences = [x for x in sequences if x.endswith(".c3d")]
        sequences = [x for x in sequences if not x.endswith("_ik.c3d")]
        sequences = [x for x in sequences if not x.endswith("_vm.c3d")]

        for sequence in sorted(sequences):
            sequence_name = sequence.split(".")[0]

            # get c3d time
            min_times[subject][sequence_name] = get_c3d_duration(os.path.join(args.input_dir, subject, "Groundtruth", sequence))
            mocap_freq[subject][sequence_name] = get_c3d_freq(os.path.join(args.input_dir, subject, "Groundtruth", sequence))
            parts_map[subject][sequence_name] = list(body_parts.keys())[num_mocap % len(body_parts.keys())]

            videos = os.listdir(os.path.join(args.input_dir, subject, "Video"))
            videos = [x for x in videos if x.startswith(sequence_name)]
            for video in videos:
                min_times[subject][sequence_name] = min(
                    min_times[subject][sequence_name],
                    get_video_duration(os.path.join(args.input_dir, subject, "Video", video)),
                )
                num_videos += 1

            min_time = min_times[subject][sequence_name] - padding[0] - padding[1]

            num_subsequences = int(min_time / args.window)
            min_times[subject][sequence_name] = args.window * num_subsequences + padding[0] + padding[1]
            num_mocap += 1

    # process mocap
    print("Processing mocap...")
    progress = tqdm(total=num_mocap)
    for subject in subjects:
        if not os.path.exists(os.path.join(args.input_dir, subject, "Groundtruth")):
            continue

        sequences = os.listdir(os.path.join(args.input_dir, subject, "Groundtruth"))
        sequences = [x for x in sequences if x.endswith(".c3d")]
        sequences = [x for x in sequences if not x.endswith("_ik.c3d")]
        sequences = [x for x in sequences if not x.endswith("_vm.c3d")]

        for sequence in sequences:
            input_filename = os.path.join(args.input_dir, subject, "Groundtruth", sequence)

            # sequences p2-p4 contain two subjects each
            preprocess_c3d_data(
                input_filename=input_filename,
                output_dir=args.output_dir,
                subject=subject,
                sequence=sequence,
                dataset_name=dataset_name,
                window=args.window,
                duration=min_times[subject][sequence.split(".")[0]],
                padding=args.padding,
                freq=args.freq,
            )
            progress.update(1)
    progress.close()
