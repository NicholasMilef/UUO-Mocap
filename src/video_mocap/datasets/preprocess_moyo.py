"""
This script is for processing the MOYO dataset.
The entire dataset (including images) should be downloaded.
The dataset should be uncompressed (except images).

Expected input file structure:

--moyo_toolkit
----data
------MOYO
------20220923_20220926_with_hands
--------cameras
--------images
--------mosh
--------mosh_smpl
--------pressure
--------vicon
------20221004_with_com
--------cameras
--------com
--------images
--------mosh
--------mosh_smpl
--------pressure
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
from tqdm import tqdm

from video_mocap.datasets.preprocess_cmu_kitchen import cleanup_markers
from video_mocap.datasets.preprocess_utils import get_c3d_freq, get_video_duration, preprocess_videos, shuffle_c3d
from video_mocap.utils.random_utils import set_random_seed

VALID_MARKERS = {
    "20220923_20220926_with_hands": ["ARIEL", "C7", "CLAV", "LANK", "LBHD", "LBSH", "LBWT", "LELB", "LFHD", "LFRM", "LFSH", "LFWT", "LHEL", "LIDX3", "LIDX6", "LIEL", "LIHAND", "LIWR", "LKNE", "LKNI", "LMID0", "LMID6", "LMT1", "LMT5", "LOHAND", "LOWR", "LPNK3", "LPNK6", "LRNG3", "LRNG6", "LSHN", "LTHI", "LTHM3", "LTHM6", "LTOE", "LUPA", "MBWT", "MFWT", "RANK", "RBHD", "RBSH", "RBWT", "RELB", "RFHD", "RFRM", "RFSH", "RFWT", "RHEL", "RIDX3", "RIDX6", "RIEL", "RIHAND", "RIWR", "RKNE", "RKNI", "RMID0", "RMID6", "RMT1", "RMT5", "ROHAND", "ROWR", "RPNK3", "RPNK6", "RRNG3", "RRNG6", "RSHN", "RTHI", "RTHM3", "RTHM6", "RTOE", "RUPA", "STRN", "T10"],
    "20221004_with_com": ["C7", "CLAV", "LANK", "LASI", "LBHD", "LELB", "LFHD", "LFIN", "LFRM", "LHEE", "LKNE", "LPSI", "LSHO", "LTHI", "LTIB", "LTOE", "LUPA", "LWRA", "LWRB", "RANK", "RASI", "RBAK", "RBHD", "RELB", "RFHD", "RFIN", "RFRM", "RHEE", "RKNE", "RPSI", "RSHO", "RTHI", "RTIB", "RTOE", "RUPA", "RWRA", "RWRB", "STRN", "T10"],
}


body_parts = {
    "left_arm": ["LUPA", "LELB", "LIEL", "LFRM", "LIWR", "LOWR", "LOHAND", "LIHAND"],
    "right_arm": ["RUPA", "RELB", "RIEL", "RFRM", "RIWR", "ROWR", "ROHAND", "RIHAND"],
    "left_leg": ["LTOE", "LMT5", "LMT1", "LHEL", "LANK", "LSHN", "LKNI", "LKNE", "LTHI"],
    "right_leg": ["RTOE", "RMT5", "RMT1", "RHEL", "RANK", "RSHN", "RKNI", "RKNE", "RTHI"],
    "left_shoulder": ["LFSH", "LBSH", "LUPA", "LELB", "LIEL"],
    "right_shoulder": ["RFSH", "RBSH", "RUPA", "RELB", "RIEL"],
}
parts_index = 0

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
):
    global parts_index
    data = ezc3d.c3d(input_filename)

    o_labels = data["parameters"]["POINT"]["LABELS"]["value"]

    markers = data["data"]["points"]  # [4, M, F]
    markers = cleanup_markers(markers)

    mocap_freq = data["parameters"]["POINT"]["RATE"]["value"][0]
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
        c3d["parameters"]["POINT"]["UNITS"]["value"] = data["parameters"]["POINT"]["UNITS"]["value"]
        c3d["parameters"]["POINT"]["RATE"]["value"] = np.array([float(freq)])

        indices = []
        labels = []

        index = 0
        for label in data["parameters"]["POINT"]["LABELS"]["value"]:
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

            if fixed_label in VALID_MARKERS[subject]:
                labels.append(fixed_label)
                indices.append(index)
            index += 1

        seq_markers = markers[:, indices, frame:frame+seq_len]

        c3d["data"]["points"] = seq_markers
        c3d["parameters"]["POINT"]["LABELS"]["value"] = labels

        if shuffle:
            c3d = shuffle_c3d(c3d)

        if parts_list:
            output_filename = os.path.join(output_dir, dataset_name, "mocap_parts___" + parts_name, subject, sequence)
        else:
            output_filename = os.path.join(output_dir, dataset_name, "mocap", subject, sequence)
        
        output_filename_seq = output_filename.replace(".c3d", "_" + str(frame).zfill(8) + ".c3d")

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
    parser.add_argument("--input_dir", type=str, help="MOYO repository directory", required=True)
    parser.add_argument("--output_dir", type=str, help="base directory of video_mocap datasets", required=True)
    parser.add_argument("--padding", nargs=2, type=int, help="padding for the animation", default=[2, 2])
    parser.add_argument("--parts", action="store_true")
    parser.add_argument("--skip_video", action="store_true", help="skips video processing")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--split", type=str, help="dataset split [train, val]", required=True)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--window", type=int, help="duration for each subsequence in seconds", default=None)
    args = parser.parse_args()

    set_random_seed(0)

    dataset_name = "moyo_" + args.split
    if args.use_full:
        dataset_name = dataset_name + "_full"
    if args.shuffle:
        dataset_name = dataset_name + "_shuffle"

    num_videos = 0
    num_mocap = 0

    padding = args.padding
    if args.use_full:
        padding = [0, 0]

    data_dir = os.path.join(args.input_dir, "data", "MOYO")
    video_dir = os.path.join(args.output_dir, "moyo_" + args.split, "videos")
    video_dir_full = os.path.join(args.output_dir, "moyo_" + args.split + "_full", "videos")
    smplx_dir = os.path.join(args.output_dir, dataset_name, "smplx")

    camera_name = "YOGI_Cam_06"

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}

    subjects = [
        "20220923_20220926_with_hands",
        "20221004_with_com",
    ]    

    # create videos
    if args.create_video:
        for subject in subjects:
            video_dir_subject = os.path.join(video_dir_full, subject)
            os.makedirs(os.path.join(video_dir_subject), exist_ok=True)

            sequences = os.listdir(os.path.join(data_dir, subject, "images", args.split))
            for sequence in sequences:
                zip_path = os.path.join(data_dir, subject, "images", args.split, sequence)        
                output_video_filename = os.path.join(video_dir_subject, sequence.replace(".zip", ".avi"))
                if os.path.exists(output_video_filename):
                    print("Skipped", output_video_filename)
                    continue
                
                if zipfile.is_zipfile(zip_path):
                    zip_file = ZipFile(zip_path)
                else:
                    raise ValueError("zip file cannot be opened (it may be corrupted)")

                zip_file_names = sorted(zip_file.namelist())

                images = []
                for name in zip_file_names:
                    if camera_name in name and name.endswith(".jpg"):
                        image_data = np.frombuffer(zip_file.read(name), np.uint8)
                        image = cv2.imdecode(image_data, cv2.IMREAD_UNCHANGED)
                        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                        images.append(image)
                zip_file.close()

                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                output_video = cv2.VideoWriter(
                    output_video_filename,
                    fourcc,
                    30,
                    (images[0].shape[1], images[0].shape[0]),
                )
                for i in range(len(images)):
                    output_video.write(images[i])
                output_video.release()
                print("Saved", output_video_filename)

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

    # get min time
    min_times = {}
    mocap_freq = {}
    parts_map = {}
    for subject in sorted(subjects):            
        min_times[subject] = {}
        mocap_freq[subject] = {}
        parts_map[subject] = {}    

        mocap_dir_subject = os.path.join(data_dir, subject, "pressure", args.split, "pressure_mat_c3d")
        video_dir_subject = os.path.join(video_dir_full, subject)

        sequences = os.listdir(os.path.join(mocap_dir_subject))
        sequences = [x for x in sequences if x.endswith(".c3d")]

        for sequence in sorted(sequences):
            sequence_name = sequence.split(".")[0]

            # get c3d time
            min_times[subject][sequence_name] = get_c3d_duration(os.path.join(mocap_dir_subject, sequence), args.freq)
            mocap_freq[subject][sequence_name] = get_c3d_freq(os.path.join(mocap_dir_subject, sequence))
            parts_map[subject][sequence_name] = list(body_parts.keys())[num_mocap % len(body_parts.keys())]

            videos = os.listdir(os.path.join(video_dir_subject))
            videos = [x for x in videos if x.startswith(sequence_name)]
            for video in videos:
                min_times[subject][sequence_name] = min(
                    min_times[subject][sequence_name],
                    get_video_duration(os.path.join(video_dir_full, subject, video)),
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
        mocap_dir_subject = os.path.join(data_dir, subject, "pressure", args.split, "pressure_mat_c3d")
        video_dir_subject = os.path.join(video_dir_full, subject)

        sequences = os.listdir(mocap_dir_subject)
        sequences = [x for x in sequences if x.endswith(".c3d")]
        for sequence in sequences:
            input_filename = os.path.join(mocap_dir_subject, sequence)

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
                shuffle=args.shuffle,
                parts=args.parts,
            )
            progress.update(1)
    progress.close()

    # process video
    if not args.skip_video and not args.use_full:
        print("Processing video...")
        progress = tqdm(total=num_videos)
        subjects = os.listdir(video_dir_full)
        for subject in subjects:
            sequences = os.listdir(os.path.join(video_dir_full, subject))
            for sequence in sequences:
                input_filename = os.path.join(video_dir_full, subject, sequence)
                output_filename = os.path.join(video_dir, subject, sequence)

                # skip videos without corresponding mocap
                if sequence.split(".")[0] not in min_times[subject]:
                    print("Skipping", sequence, "because couldn't find corresponding mocap")
                    continue

                preprocess_videos(
                    input_filename,
                    output_filename,
                    window=args.window,
                    duration=min_times[subject][sequence.split(".")[0]],
                    padding=padding,
                    mocap_freq=args.freq,
                )
                progress.update(1)
        progress.close()

