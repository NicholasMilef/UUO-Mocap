import os

import cv2
import ezc3d
import numpy as np


def get_downsampled_indices(source_freq, target_freq, num_frames):
    original_index = 0
    downsampled_indices = []
    while original_index < num_frames:
        if round(original_index) < num_frames:
            downsampled_indices.append(round(original_index))
        else:
            break
        original_index += source_freq / target_freq

    return np.array(downsampled_indices, dtype=np.int32)


def shuffle_c3d(c3d):
    markers = np.array(c3d["data"]["points"])  # [4, M, F]
    _, num_markers, num_frames = markers.shape
    c3d["parameters"]["POINT"]["LABELS"]["value"] = ["marker_" + str(x) for x in range(num_markers)]
    for frame in range(num_frames):
        indices = np.random.permutation(num_markers)
        c3d["data"]["points"][:, :, frame] = np.array(markers[:, indices, frame])

    return c3d


def get_c3d_duration(filename):
    data = ezc3d.c3d(filename)
    markers = data["data"]["points"]  # [4, M, F]
    freq = data["parameters"]["POINT"]["RATE"]["value"][0]
    total_time = markers.shape[2] / freq
    return total_time


def get_c3d_freq(filename):
    data = ezc3d.c3d(filename)
    freq = data["parameters"]["POINT"]["RATE"]["value"][0]
    return freq


def get_video_duration(filename):
    video = cv2.VideoCapture(filename)
    freq = video.get(cv2.CAP_PROP_FPS)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return num_frames / freq


def get_video_frequency(filename):
    video = cv2.VideoCapture(filename)
    freq = video.get(cv2.CAP_PROP_FPS)
    return freq


def preprocess_videos(
    input_filename,
    output_filename,
    window,
    duration,
    padding,
    mocap_freq,
    video_freq=30,
):
    input_video = cv2.VideoCapture(input_filename)
    res = (
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    freq = input_video.get(cv2.CAP_PROP_FPS)
    num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    if freq == video_freq:
        frame_indices = np.arange(num_frames)
    else:
        frame_indices = get_downsampled_indices(freq, video_freq, num_frames)
        freq = video_freq

    if window is None:
        seq_len = int(duration * freq)
    else:
        seq_len = int(window * freq)

    total_length = int(duration * freq)
    if duration == -1:
        seq_len = num_frames
        total_length = num_frames

    images = []
    image_index = 0
    while True:
        ret, image = input_video.read()

        if ret:
            if image_index in frame_indices:
                images.append(image)
        else:
            break

        image_index += 1

    input_video.release()

    for frame in range(int(padding[0] * freq), total_length - seq_len, seq_len):
        mocap_frame = int(frame * (mocap_freq / freq))

        dirname = os.path.dirname(output_filename)
        filename = os.path.basename(output_filename)
        output_filename_seq = filename.split(".")[0] + "_" + str(mocap_frame).zfill(8) + "." + ".".join(filename.split(".")[1:])
        output_filename_seq = os.path.join(dirname, output_filename_seq)
        os.makedirs(os.path.dirname(output_filename_seq), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter(output_filename_seq, fourcc, freq, res)
        for i in range(frame, frame + seq_len):
            output_video.write(images[i])
        output_video.release()
