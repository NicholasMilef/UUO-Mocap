import argparse
import os
import joblib

import cv2
import numpy as np

from video_mocap.utils.img_smpl_utils import joints_2d_labels, get_foot_contacts


def visualize_2d_joints(args):

    file_extensions = [".avi", ".mp4", ".mpg"]
    for file_extension in file_extensions:
        smpl_video_filename = os.path.join(
            args.input_dir,
            "videos",
            args.subject,
            args.sequence + file_extension,
        )
        if os.path.exists(smpl_video_filename):
            break

    smpl_img_filename = os.path.join(
        args.input_dir,
        "comparisons",
        "4d_humans",
        args.subject,
        args.sequence,
        "results",
        "demo_" + args.sequence + ".pkl",
    )

    # get source video data
    input_video = cv2.VideoCapture(smpl_video_filename)
    res = (
        int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    freq = input_video.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_video = cv2.VideoWriter(args.video_path, fourcc, freq, res)

    # SMPL data from monocular
    smpl_img_data = joblib.load(smpl_img_filename)
    joints_2d = np.zeros((len(smpl_img_data), 45, 2))
    frame = 0
    for key in sorted(list(smpl_img_data.keys())):
        for i in range(45):
            try:
                joints_2d[frame, i] = smpl_img_data[key]["2d_joints"][0][i*2:i*2+2]
            except:
                pass
        frame += 1

    foot_contacts = get_foot_contacts(joints_2d, freq)

    index = 0
    while True:
        ret, frame = input_video.read()

        if ret:
            key = sorted(list(smpl_img_data.keys()))[index]
            try:
                joints = smpl_img_data[key]["2d_joints"][0]
                center = smpl_img_data[key]["center"][0]
                for i in range(45):
                    diff = (res[0] - res[1]) / 2
                    j_x = joints[i*2] * res[0]
                    j_y = (joints[i*2+1] * res[0]) - diff

                    if i in [joints_2d_labels["l_toe_in"], joints_2d_labels["l_toe_out"]]:
                        if foot_contacts[index, 0] == 1:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                    elif i in [joints_2d_labels["r_toe_in"], joints_2d_labels["r_toe_out"]]:
                        if foot_contacts[index, 1] == 1:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)
                    else:
                        color = (0, 255, 0)

                    frame = cv2.circle(
                        img=frame,
                        center=(int(j_x), int(j_y)),
                        radius=1,
                        color=color,
                        thickness=-10,
                    )

                bbox_pt1 = smpl_img_data[key]["tracked_bbox"][0][0:2]
                bbox_pt2 = bbox_pt1 + smpl_img_data[key]["tracked_bbox"][0][2:4]
                frame = cv2.rectangle(
                    img=frame,
                    pt1=bbox_pt1.astype(np.int32).tolist(),
                    pt2=bbox_pt2.astype(np.int32).tolist(),
                    color=[0, 0, 255],
                )
            except:
                pass
            output_video.write(frame)
        else:
            break

        index += 1

    input_video.release()
    output_video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input directory", required=True)
    parser.add_argument("--sequence", type=str, help="sequence", required=True)
    parser.add_argument("--subject", type=str, help="subject", required=True)
    parser.add_argument("--video_path", type=str, help="video path", required=True)
    args = parser.parse_args()

    markers = visualize_2d_joints(args)
