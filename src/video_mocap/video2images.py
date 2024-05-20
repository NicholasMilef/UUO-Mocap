import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def video2images(filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(filename)

    frame_num = 0
    progress = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        filename_wo_ext = Path(filename).stem
        output_filename = os.path.join(output_dir, filename_wo_ext + "_f" + str(frame_num).zfill(8) + ".jpg")
        cv2.imwrite(output_filename, frame)
        frame_num += 1
        progress.update(1)

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="input directory", required=True)
    args = parser.parse_args()

    images_dir = os.path.join(args.input_dir, "images")
    videos_dir = os.path.join(args.input_dir, "videos")

    subjects = os.listdir(videos_dir)
    for subject in subjects:
        sequences = os.listdir(os.path.join(videos_dir, subject))
        for sequence in sequences:
            try:
                video2images(
                    os.path.join(videos_dir, subject, sequence),
                    os.path.join(images_dir, subject, Path(sequence).stem),
                )
            except:
                pass
