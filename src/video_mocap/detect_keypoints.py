import argparse
from typing import List

import cv2
import numpy as np 


# constants
CIRCLE_WIDTH = 10

def detect_keypoints(
    img: np.ndarray,
) -> np.ndarray:
    """
    Args:
        img: [H, W, 3]

    Returns:
        np.ndarray: [N, 3]
    """
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grayscale = cv2.blur(grayscale, (3, 3))
    #grayscale = cv2.bilateralFilter(grayscale, 3, 75, 75)

    #cv2.imshow("test", grayscale)
    #cv2.waitKey(0)

    markers_2d = cv2.HoughCircles(
        grayscale,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=4,
        param1=30,
        param2=5,
        minRadius=2,
        maxRadius=4,
    )
    
    grayscale = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if markers_2d is None:
        return None

    return np.uint16(np.around(markers_2d[0]))


def overlay_circles(
    img: np.ndarray,
    markers_2d: np.ndarray,
):
    """
    Args:
        img: [H, W, 3]
        np.ndarray: [N, 3]
        
    Returns:
        np.ndarray: [H, W, 3]
    """
    overlay_img = np.array(img)

    if markers_2d is None:
        return overlay_img

    for i in range(markers_2d.shape[0]):
        cv2.circle(
            overlay_img,
            (markers_2d[i, 0], markers_2d[i, 1]),
            markers_2d[i, 2],
            (0, 0, 255),
            1,
        )

    return overlay_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="input video filename", required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.filename)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if frame is None:
            break

        height, width = frame.shape[0:2]
        r_width = 640
        r_height = int((r_width / width) * height)

        frame = cv2.resize(frame, dsize=(r_width, r_height), interpolation=cv2.INTER_CUBIC)

        markers_2d = detect_keypoints(frame)
        overlay_frame = overlay_circles(frame, markers_2d)

        cv2.imshow("window-name", overlay_frame)
        frame_num += 1
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
