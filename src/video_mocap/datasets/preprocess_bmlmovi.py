"""
This script is for processing the MOYO dataset.
The entire dataset (including images) should be downloaded.
The dataset should be uncompressed (except images).

Expected input file structure:

--bmlmovi_train_full
----src
------mocap
--------F_v3d_Subject_1.mat
--------F_v3d_Subject_2.mat
--------F_v3d_Subject_3.mat
------videos
--------F_PG1_Subject_1_L.avi
--------F_PG1_Subject_2_L.avi
--------F_PG1_Subject_3_L.avi
"""

import argparse
import json
import os
import pickle
import warnings
import zipfile
from zipfile import ZipFile

import cv2
import ezc3d
import numpy as np
import scipy.io
from tqdm import tqdm

from video_mocap.datasets.preprocess_cmu_kitchen import cleanup_markers
from video_mocap.datasets.preprocess_utils import get_video_frequency, preprocess_videos, shuffle_c3d
from video_mocap.utils.random_utils import set_random_seed


body_parts = {
    "left_arm": ["LUPA", "LELB", "LIEL", "LFRM", "LIWR", "LOWR", "LOHAND", "LIHAND"],
    "right_arm": ["RUPA", "RELB", "RIEL", "RFRM", "RIWR", "ROWR", "ROHAND", "RIHAND"],
    "left_leg": ["LTOE", "LMT5", "LMT1", "LHEL", "LANK", "LSHN", "LKNI", "LKNE", "LTHI"],
    "right_leg": ["RTOE", "RMT5", "RMT1", "RHEL", "RANK", "RSHN", "RKNI", "RKNE", "RTHI"],
    "left_shoulder": ["LFSH", "LBSH", "LUPA", "LELB", "LIEL"],
    "right_shoulder": ["RFSH", "RBSH", "RUPA", "RELB", "RIEL"],
}
parts_index = 0


def preprocess_mat_data(
    input_filename,
    output_dir,
    subject,
    sequence,
    dataset_name,
    window,
    duration,
    padding,
    shuffle,
    freq,
    parts,
    mocap_data,
    mocap_units,
    mocap_freq,
    mocap_labels,
):
    global parts_index

    o_labels = mocap_labels

    data = np.transpose(mocap_data, (2, 1, 0))
    data = np.concatenate((data, np.zeros((1, data.shape[1], data.shape[2]))), axis=0)
    if mocap_units == "mm":
        data = data / 1000.0

    markers = data # [4, M, F]
    markers = cleanup_markers(markers)

    stride = int(mocap_freq / freq)

    markers = markers[:, :, ::stride]
    _, _, num_frames = markers.shape

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
        c3d["parameters"]["POINT"]["UNITS"]["value"] = mocap_units
        c3d["parameters"]["POINT"]["RATE"]["value"] = np.array([float(freq)])

        indices = []
        labels = []

        index = 0
        for label in mocap_labels:
            if label.startswith("*"):
                index += 1
                continue
        
            if ":" in label:
                fixed_label = label.split(":")[1]
            else:
                fixed_label = label

            if parts_list is not None and o_labels[index].split(":")[-1] not in parts_list:
                index += 1
                continue

            labels.append(fixed_label)
            indices.append(index)
            index += 1

        seq_markers = markers[:, indices, frame:frame+seq_len]

        c3d["data"]["points"] = seq_markers
        c3d["parameters"]["POINT"]["LABELS"]["value"] = labels

        if shuffle:
            # not implemented
            c3d = shuffle_c3d(c3d)

        if parts_list:
            output_filename = os.path.join(output_dir, dataset_name, "mocap_parts___" + parts_name, subject, sequence)
        else:
            output_filename = os.path.join(output_dir, dataset_name, "mocap", subject, sequence)
        
        output_filename_seq = output_filename + "_" + str(frame).zfill(8) + ".c3d"

        invalid_c3d = False
        for subframe in range(c3d["data"]["points"].shape[-1]):
            count = np.count_nonzero(c3d["data"]["points"][:3, :, subframe]==0)
            if count == c3d["data"]["points"][:3, :, subframe].size:
                warnings.warn("WARNING: skipping sequence" + str(output_filename_seq) + " because of invalid marker positions")
                invalid_c3d = True
                break

        if not invalid_c3d:
            output_filename_seq_window = output_filename_seq
            os.makedirs(os.path.dirname(output_filename_seq_window), exist_ok=True)
            c3d.write(output_filename_seq_window)
            parts_index += 1

            with open(os.path.join(os.path.dirname(output_filename_seq_window), "settings.json"), "w", encoding="utf-8") as f:
                json.dump({"gender": "neutral"}, f, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_video", action="store_true", help="skips video creation")
    parser.add_argument("--convert_smplx", action="store_true", help="convert SMPL-X")
    parser.add_argument("--freq", type=int, help="target frequency", default=30)
    parser.add_argument("--input_dir", type=str, help="input directory BMLmovi repository directory", required=True)
    parser.add_argument("--padding", nargs=2, type=int, help="padding for the animation", default=[0, 0])
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--skip_video", action="store_true", help="skips video processing")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--split", type=str, help="dataset split [train, val]", required=True)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--window", type=int, help="duration for each subsequence in seconds", default=3)
    args = parser.parse_args()

    set_random_seed(0)

    dataset_name = "bmlmovi_" + args.split
    if args.use_full:
        dataset_name = dataset_name + "_full"

    num_videos = 0
    num_mocap = 0

    padding = args.padding
    if args.use_full:
        padding = [0, 0]

    data_dir = os.path.join(args.input_dir, dataset_name + "_full", "src")
    input_mocap_dir = os.path.join(args.input_dir, dataset_name + "_full", "src", "mocap")
    input_video_dir = os.path.join(args.input_dir, dataset_name + "_full", "src", "videos")
    temp_video_dir = os.path.join(args.input_dir, dataset_name + "_full", "videos")

    output_mocap_dir = os.path.join(args.input_dir)
    output_video_dir = os.path.join(args.input_dir, dataset_name, "videos")

    camera_name = "CP1"

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}

    subjects = []
    subjects_mocap = [x.split(".mat")[0].split("F_v3d_")[-1] for x in os.listdir(input_mocap_dir)]
    for subject in subjects_mocap:
        if os.path.exists(os.path.join(input_video_dir, "F_PG1_" + subject + "_L.avi")):
            subjects.append(subject)

    mocap_data = {}
    for subject in subjects:
        mocap_data[subject] = {}
        data_mat = scipy.io.loadmat(os.path.join(input_mocap_dir, "F_v3d_" + subject + ".mat"))
        for key in data_mat.keys():
            if key.startswith("Subject"):
                d_types = data_mat[key][0, 0][2][0, 0].dtype.names

                data = data_mat[key][0, 0][2][0, 0][0, 0]
                d_type_index = 0
                for d_type in d_types:
                    mocap_data[subject][d_type] = np.array(data[d_type_index])
                    d_type_index += 1

        motions = []
        for motion_index in range(mocap_data[subject]["motions_list"].shape[0]):
            motion = mocap_data[subject]["motions_list"][motion_index][0].item()
            motions.append(motion.replace("/", "_"))
        mocap_data[subject]["motions_list"] = motions

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}
    for subject in subjects:
        min_times[subject] = {}
        mocap_freq[subject] = {}
        parts_map[subject] = {}
        
        sequences = [x for x in mocap_data[subject]["motions_list"]]

        fps_video = get_video_frequency(os.path.join(input_video_dir, "F_PG1_" + subject + "_L.avi"))
        if fps_video == 0.0:
            import pdb; pdb.set_trace()

        sequence_index = 0
        for sequence in sequences:
            # get c3d time
            mocap_freq[subject][sequence] = 120
            parts_map[subject][sequence] = list(body_parts.keys())[num_mocap % len(body_parts.keys())]

            min_time_mocap = mocap_data[subject]["flags120"][sequence_index]
            min_time_mocap = (min_time_mocap[1] - min_time_mocap[0]) / 120.0
            min_time_video = mocap_data[subject]["flags30"][sequence_index]
            min_time_video = (min_time_video[1] - min_time_video[0]) / fps_video
            min_times[subject][sequence] = min(min_time_mocap, min_time_video)

            min_time = min_times[subject][sequence] - padding[0] - padding[1]

            if not args.use_full:
                num_subsequences = int(min_time / args.window)
                min_times[subject][sequence] = args.window * num_subsequences + padding[0] + padding[1]
            elif args.use_full:
                min_times[subject][sequence] = -1

            num_mocap += 1

            sequence_index += 1

    # split videos into sequences
    if args.create_video:
        progress = tqdm(total=len(subjects))
        for subject in subjects:
            sequences = [x for x in mocap_data[subject]["motions_list"]]

            input_filename = os.path.join(input_video_dir, "F_PG1_" + subject + "_L.avi")
            input_video = cv2.VideoCapture(input_filename)
            res = (
                int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            freq = input_video.get(cv2.CAP_PROP_FPS)
            num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

            images = []
            while True:
                ret, image = input_video.read()

                if ret:
                    images.append(image)
                else:
                    break
            input_video.release()

            sequence_index = 0
            for sequence in sequences:
                start, end = mocap_data[subject]["flags30"][sequence_index]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                output_filename = os.path.join(temp_video_dir, subject, sequence + "_L.avi")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)

                output_video = cv2.VideoWriter(output_filename, fourcc, freq, res)
                for frame in range(start, end):
                    output_video.write(images[frame])
                output_video.release()

                sequence_index += 1
            progress.update(1)
        progress.close()

    # process SMPL-X
    if args.convert_smplx:
        print("Processing SMPL-X...")
        for subject in subjects:
            smplx_dir_subject = os.path.join(args.input_dir, "data", "MOYO", subject, "mosh", args.split)
            sequences = os.listdir(smplx_dir_subject)
            for sequence in sequences:
                try:
                    smplx_data = pickle.load(open(os.path.join(smplx_dir_subject, sequence), "rb"))
                except:
                    print("Cannot read sequence", sequence)

                o_frame_rate = 60
                stride = int(o_frame_rate / args.freq)

                output = {}
                output["poses"] = smplx_data["fullpose"][::stride]
                output["betas"] = smplx_data["betas"]
                output["mocap_frame_rate"] = 30
                output["trans"] = smplx_data["trans"][::stride]
                output["gender"] = np.array("female")

                output_filename = os.path.join(smplx_dir, subject, sequence.replace("_stageii.pkl", "_00000000_stageii.npz"))
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                np.savez(output_filename, **output)


    # process mocap
    print("Processing mocap...")
    progress = tqdm(total=num_mocap)
    for subject in subjects:
        sequences = [x for x in mocap_data[subject]["motions_list"]]
        sequence_index = 0
        for sequence in sequences:
            input_filename = os.path.join(input_mocap_dir, subject, sequence)

            start, end = mocap_data[subject]["flags120"][sequence_index]

            mocap_labels = []
            for mocap_label in mocap_data[subject]["markerName"][0]:
                if mocap_label[0].dtype == np.float64:
                    mocap_labels.append("")
                else:
                    mocap_labels.append(mocap_label.item())

            preprocess_mat_data(
                input_filename=input_filename,
                output_dir=output_mocap_dir,
                subject=subject,
                sequence=sequence,
                dataset_name=dataset_name,
                window=args.window,
                duration=min_times[subject][sequence] + (1.0/args.freq),
                padding=args.padding,
                freq=args.freq,
                shuffle=args.shuffle,
                parts=args.parts,
                mocap_data=mocap_data[subject]["markerLocation"][start:end],
                mocap_units="mm",
                mocap_freq=120,
                mocap_labels=mocap_labels,
            )

            progress.update(1)
            sequence_index += 1
    progress.close()

    # process video
    if not args.skip_video and not args.use_full:
        print("Processing video...")
        progress = tqdm(total=num_mocap)
        for subject in subjects:
            sequences = [x for x in mocap_data[subject]["motions_list"]]
            sequence_index = 0

            for sequence in sequences:
                input_filename = os.path.join(temp_video_dir, subject, sequence + "_L.avi")
                output_filename = os.path.join(output_video_dir, subject, sequence + ".avi")

                # skip videos without corresponding mocap
                if sequence.split(".")[0] not in min_times[subject]:
                    print("Skipping", sequence, "because couldn't find corresponding mocap")
                    continue

                preprocess_videos(
                    input_filename,
                    output_filename,
                    window=args.window,
                    duration=min_times[subject][sequence] + (1.0/args.freq),  # the for loop expects padding
                    padding=padding,
                    mocap_freq=args.freq,
                )

                progress.update(1)
            sequence_index += 1
        progress.close()

