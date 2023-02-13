import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

from ap_loader import load_alphapose_json
import viz3d

BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
GRAY_DARK = (32, 32, 32)
WHITE = (255, 255, 255)
CYAN = (255, 128, 128)


def rg(ratio):
    if ratio > 0.9: 
        return GREEN
    elif ratio > 0.5:
        return YELLOW
    else:
        return RED
    #(255*(1-ratio), 255*ratio, 0)


def render_skeleton(image, pose):
    for segment in viz3d.skeleton:
        i0, i1 = [viz3d.KEYPOINTS[kp] for kp in segment]
        x0, y0, score0 = pose["keypoints"][3*i0:3*i0+3]
        x1, y1, score1 = pose["keypoints"][3*i1:3*i1+3]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(image, [x0, y0], [x1, y1], rg(
            min(score0, score1)), 2, cv2.LINE_AA)
        cv2.circle(image, [x0, y0], 4, rg(score0), -1)
        cv2.circle(image, [x1, y1], 4, rg(score1), -1)

    cx, cy, w, h = pose["box"]
    box = np.array([
        [
            [cx, cy],
            [cx+w, cy],
            [cx+w, cy+h],
            [cx, cy+h],
        ]
    ], dtype=np.int32)
    cv2.polylines(image, box, True, WHITE, 1)
    cv2.putText(image,
                str(pose["idx"]),
                np.array([cx, cy - 2], dtype=np.int32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25, WHITE, 2)


def render_output(input_video, input_json, outfile):
    # read alphapose json
    print("Reading", input_json)
    posedata = load_alphapose_json(input_json)

    # read input
    input_stream = cv2.VideoCapture(str(input_video))

    # get input video properties
    width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_stream.get(cv2.CAP_PROP_FPS))
    # frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # open output video file
    print("Writing to:", outfile)
    output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (width, height))

    num_frame = 0
    while input_stream.isOpened():
        ret, image = input_stream.read()

        if not ret:
            break

        # read image
#        image = cv2.cvtColor(frame, cv2.COLOR_RGB2RGB)

        if num_frame < len(posedata):
            for pose in posedata[num_frame]["objs"]:
                render_skeleton(image, pose)

        # write output
        output.write(image)
        if num_frame % 24 == 0:
            print(".", end="", flush=True)
        num_frame += 1

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analyze-com")
    parser.add_argument("input_video",
                        help="Input video file, eg. S01_01_oe.mp4")
    parser.add_argument("input_json",
                        help="AlphaPose result file, eg. alphapose-results.json")
    args = parser.parse_args()

    video_path = Path(args.input_video)
    jsonpath = Path(args.input_json)

    if not video_path.is_file():
        print("Not a file", str(video_path))
        sys.exit(1)

    if not jsonpath.is_file():
        print("Not a file", str(jsonpath))
        sys.exit(1)

    outfile_name = f"{video_path.stem}-alphapose{video_path.suffix}"
    render_output(video_path, jsonpath, outfile_name)
