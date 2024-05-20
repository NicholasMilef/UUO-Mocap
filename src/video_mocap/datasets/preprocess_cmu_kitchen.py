"""
This script is for processing the CMU kitchen mocap data
The labels for the markers is not consistent with MoSh, so we change them here.
Expected input file structure:

--base_dir
----s1
------salad.c3d
----s2
"""
import argparse
import json
import os
import warnings

import numpy as np
import ezc3d
from tqdm import tqdm
from video_mocap.datasets.preprocess_utils import get_c3d_freq, get_video_duration, preprocess_videos, shuffle_c3d
from video_mocap.utils.random_utils import set_random_seed


body_parts = {
    #"left_arm": ["LWRA", "LWRB", "LFIN", "LTHMB", "LELB", "LFRM", "NEWLSHO", "LUPA"],
    "right_arm": ["RWRA", "RWRB", "RFIN", "RTHMB", "RELB", "RFRM", "NEWRSHO", "RUPA"],
    "left_leg": ["LFWT", "LTHI", "LKNE", "LSHN", "LANK", "LHEE", "LTOE", "LMT5", "LMT1", "LRSTBEEF"],
    #"right_leg": ["RFWT", "RTHI", "RKNE", "RSHN", "RANK", "RHEE", "RTOE", "RMT5", "RMT1", "RRSTBEEF"],
    "left_shoulder": ["LELB", "LFRM", "NEWLSHO", "LUPA", "LSHO"],
}
parts_index = 0

def cleanup_markers(markers):
    # markers: [4, M, F]
    for frame in range(markers.shape[2]-1, -1, -1):
        count = np.count_nonzero(markers[:3, :, frame]==0)
        if count != markers[:3, :, frame].size:
            break
    markers = markers[:, :, :frame+1]
    return markers


def get_c3d_duration(filename, freq):
    data = ezc3d.c3d(filename)
    markers = data["data"]["points"]  # [4, M, F]
    mocap_freq = data["parameters"]["POINT"]["RATE"]["value"][0]
    stride = int(mocap_freq / freq)

    markers = cleanup_markers(markers)
    markers = markers[:, :, ::stride]
    total_time = markers.shape[2] / freq
    return total_time


def preprocess_c3d_data(
    input_filename,
    output_filename,
    window,
    duration,
    remove_backpack,
    padding,
    freq,
    shuffle,
    parts,
):
    global parts_index

    data = ezc3d.c3d(input_filename)
    labels = data["parameters"]["POINT"]["LABELS"]["value"]

    o_labels = data["parameters"]["POINT"]["LABELS"]["value"]

    markers = data["data"]["points"]  # [4, M, F]
    markers = cleanup_markers(markers)

    mocap_freq = data["parameters"]["POINT"]["RATE"]["value"][0]
    stride = int(mocap_freq / freq)
    markers = markers[:, :, ::stride]

    _, num_markers, num_frames = markers.shape

    backpack_labels = [
        "LBWT",
        "NEWLBAC",
        "NEWRBAC",
        "RBAC",
        "RBWT",
        "T10",
        "T8",
    ]

    if window is None:
        seq_len = int(duration * freq)
    else:
        seq_len = int(window * freq)

    total_length = int(duration * freq)
    if duration == -1:
        seq_len = num_frames
        total_length = num_frames
        padding = [0, 0]

    for frame in range(int(padding[0] * freq), total_length - seq_len + 1, seq_len):
        parts_list = None
        if parts:
            parts_name = list(body_parts.keys())[parts_index % len(body_parts)]
            parts_list = body_parts[parts_name]

        c3d = ezc3d.c3d()
        c3d["parameters"]["POINT"]["UNITS"]["value"] = data["parameters"]["POINT"]["UNITS"]["value"]
        c3d["parameters"]["POINT"]["RATE"]["value"] = np.array([float(freq)])

        indices = []
        labels = []
        for i in range(num_markers):
            if "cook" in o_labels[i].split(":")[0]:
                if remove_backpack and o_labels[i].split(":")[-1] in backpack_labels:
                    continue

                if parts_list is not None and o_labels[i].split(":")[-1] not in parts_list:
                    continue

                indices.append(i)
                labels.append(o_labels[i].split(":")[-1])

        seq_markers = markers[:, indices, frame:frame+seq_len]

        c3d["data"]["points"] = seq_markers
        c3d["parameters"]["POINT"]["LABELS"]["value"] = labels

        if shuffle:
            c3d = shuffle_c3d(c3d)

        output_filename_seq = output_filename.replace(".c3d", "_" + str(frame).zfill(8) + ".c3d")

        invalid_c3d = False
        for subframe in range(c3d["data"]["points"].shape[-1]):
            count = np.count_nonzero(c3d["data"]["points"][:3, :, subframe]==0)
            if count == c3d["data"]["points"][:3, :, subframe].size:
                warnings.warn("skipping sequence" + str(output_filename_seq) + " because of invalid marker positions")
                invalid_c3d = True
                break

        if not invalid_c3d:
            output_filename_seq_window = output_filename_seq
            if parts:
                output_filename_seq_window = output_filename_seq_window.replace("mocap_parts___", "mocap_parts___" + parts_name)
            os.makedirs(os.path.dirname(output_filename_seq_window), exist_ok=True)
            c3d.write(output_filename_seq_window)
            parts_index += 1

            with open(os.path.join(os.path.dirname(output_filename_seq_window), "settings.json"), "w", encoding="utf-8") as f:
                json.dump({"gender": "neutral"}, f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", type=int, help="target frequency", default=30)
    parser.add_argument("--input_dir", type=str, help="input directory of CMU kitchen dataset", required=True)
    parser.add_argument("--output_dir", type=str, help="base directory of video_mocap datasets", required=True)
    parser.add_argument("--padding", nargs=2, type=int, help="padding for the animation", default=[5, 5])
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--remove_backpack", action="store_true", help="removes markers on backpack")
    parser.add_argument("--skip_video", action="store_true", help="skips video processing")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--window", type=int, help="duration for each subsequence in seconds", default=None)
    args = parser.parse_args()

    set_random_seed(0)

    if args.remove_backpack:
        dataset_name = "cmu_kitchen_pilot_rb"
    else:
        dataset_name = "cmu_kitchen_pilot"

    if args.use_full:
        dataset_name = dataset_name + "_full"
    if args.shuffle:
        dataset_name = dataset_name + "_shuffle"

    num_videos = 0
    num_mocap = 0

    padding = args.padding
    if args.use_full:
        padding = [0, 0]

    subjects = os.listdir(os.path.join(args.input_dir, "mocap"))

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}
    for subject in sorted(subjects):
        min_times[subject] = {}
        mocap_freq[subject] = {}
        parts_map[subject] = {}

        sequences = os.listdir(os.path.join(args.input_dir, "mocap", subject))
        sequences = [x for x in sequences if x.endswith(".c3d")]
        for sequence in sorted(sequences):
            sequence_name = sequence.split(".")[0]

            # get c3d time
            min_times[subject][sequence_name] = get_c3d_duration(os.path.join(args.input_dir, "mocap", subject, sequence), args.freq)
            mocap_freq[subject][sequence_name] = get_c3d_freq(os.path.join(args.input_dir, "mocap", subject, sequence))

            videos = os.listdir(os.path.join(args.input_dir, "videos", subject))
            videos = [x for x in videos if x.startswith(sequence_name)]
            for video in videos:
                min_times[subject][sequence_name] = min(
                    min_times[subject][sequence_name],
                    get_video_duration(os.path.join(args.input_dir, "videos", subject, video)),
                )
                num_videos += 1
            min_time = min_times[subject][sequence_name] - padding[0] - padding[1]

            if not args.use_full:
                num_subsequences = int(min_time / args.window)
                min_times[subject][sequence_name] = args.window * num_subsequences + padding[0] + padding[1]
            elif args.use_full:
                min_times[subject][sequence_name] = -1
            num_mocap += 1

    # process mocap
    print("Processing mocap...")
    progress = tqdm(total=num_mocap)
    for subject in subjects:
        sequences = os.listdir(os.path.join(args.input_dir, "mocap", subject))
        sequences = [x for x in sequences if x.endswith(".c3d")]
        for sequence in sequences:
            input_filename = os.path.join(args.input_dir, "mocap", subject, sequence)

            output_filename = os.path.join(args.output_dir, dataset_name, "mocap", subject, sequence)
            if args.parts:
                for part_name in body_parts.keys():
                    output_filename = os.path.join(args.output_dir, dataset_name, "mocap_parts___", subject, sequence + "")

            preprocess_c3d_data(
                input_filename=input_filename,
                output_filename=output_filename,
                window=args.window,
                duration=min_times[subject][sequence.split(".")[0]],
                remove_backpack=args.remove_backpack,
                padding=padding,
                freq=args.freq,
                shuffle=args.shuffle,
                parts=args.parts,
            )
            progress.update(1)
    progress.close()

    # process video
    if not args.skip_video:
        print("Processing video...")
        progress = tqdm(total=num_videos)
        subjects = os.listdir(os.path.join(args.input_dir, "videos"))
        for subject in subjects:
            sequences = os.listdir(os.path.join(args.input_dir, "videos", subject))
            sequences = [x for x in sequences if x.endswith(".avi")]
            for sequence in sequences:
                input_filename = os.path.join(args.input_dir, "videos", subject, sequence)
                output_filename = os.path.join(args.output_dir, dataset_name, "videos", subject, sequence)
                preprocess_videos(
                    input_filename=input_filename,
                    output_filename=output_filename,
                    window=args.window,
                    duration=min_times[subject][sequence.split(".")[0]],
                    padding=padding,
                    mocap_freq=args.freq,
                )
                progress.update(1)
        progress.close()
