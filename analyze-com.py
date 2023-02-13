import argparse
import sys
from pathlib import Path
import re

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.interpolate

from ffmpeg_sync.index_xlsx import read_index_xlsx
import indexfiles

BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
GRAY_DARK = (32, 32, 32)


def com_velocity(c, fps=240):
    # x-y plane
    return np.sqrt(np.diff(c[:, 0])**2+np.diff(c[:, 1])**2) * fps

    # x-y-z plane
    #return np.sqrt(np.diff(c[:, 0])**2+np.diff(c[:, 1])**2+np.diff(c[:, 2])**2) * fps

    # x-plane
    #return np.diff(c[:, 0]) * fps

def nan_partition(x):
    ns = np.isnan(x)
    d = np.diff(ns)


def plot_v(v, fps=240, title="CoM"):
    # discrete diff starts at 1 frame
    t = (1 + np.arange(v.shape[0])) / fps
    plt.plot(t, v)
    plt.title(title)
    plt.xlabel("sec")
    plt.ylabel("m/s")
    plt.grid()
    plt.show()


def compute_velocity(input_com_npy, fps=240):
    com = np.load(input_com_npy)
    return com_velocity(com, fps)


def overlay_transparent(bg_img, img_to_overlay_t):
    """See: https://stackoverflow.com/questions/56356857/how-to-correctly-overlay-two-images-with-opencv
    """
    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    mask = cv2.medianBlur(a, 5)

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(
        bg_img.copy(), bg_img.copy(), mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img = cv2.add(img1_bg, img2_fg)

    return bg_img


def render_logo(image):
    a = cv2.imread("a.jpeg")


def render_output(input_video, outfile, v_arr):
    # read input
    input_stream = cv2.VideoCapture(input_video)

    # get input video properties
    width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_stream.get(cv2.CAP_PROP_FPS))
    # frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # build velocity curve points
    Nv = v_arr.shape[0]
    interp_v = scipy.interpolate.interp1d(np.arange(Nv), v_arr)
    curve_x = np.arange(width - 1) / (width - 1) * (Nv - 1)
    curve_y = interp_v(curve_x)
    CURVE_BOTTOM = 340
    curve_y /= np.max(curve_y)
    curve_y *= CURVE_BOTTOM - 10
    curve_points = np.array([(x + 1, int(CURVE_BOTTOM - y))
                             for (x, y) in enumerate(curve_y) if np.isfinite(y)])

    # open output video file
    output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (width, height))

    print("Processing %s" % input_video)
    num_frame = 0
    while input_stream.isOpened():
        ret, frame = input_stream.read()

        if not ret:
            break

        # read image
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # render visualization
        # image
    #    cv2.circle(image, [20, 20], 10, YELLOW, 4)
        if num_frame > 1:
            frame_v = v_arr[num_frame - 1]
            # velocity as text
            cv2.putText(image, "%.2f m/s" % frame_v, (10, CURVE_BOTTOM + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, YELLOW, 2, cv2.LINE_AA)

            # velocity curve
            cv2.polylines(image, [curve_points], False, YELLOW, 2, cv2.LINE_AA)

            # vertical marker
            hx = 1 + (width - 1) * (1 + num_frame) // Nv
            cv2.line(image, [hx, 10], [hx, CURVE_BOTTOM], GREEN, 2)

        # write output
        output.write(image)
        if num_frame % 24 == 0:
            print(".", end="", flush=True)
        num_frame += 1

    print("\nDone.")


def get_input_files(index_dir_path, throw_id):
    output_dir = index_dir_path.joinpath("Output")
    print("*** get input files")
    coms = list((f"*_{throw_id}-com.npy"))
    if len(coms) == 0:
        raise Exception("No CoM trajectory file found in: %s" %
                        str(output_dir))
    if len(coms) > 1:
        raise Exception(
            "CoM trajectories for multiple subjects found in: %s" % str(output_dir))
    return coms


def process_files(input_dir, cam_id="oe"):
    index_file_paths = indexfiles.glob_index_files(input_dir)
    xlsx_cols = [
        "Throw",
        "Camera",
        # "XOTOFrame",
        # "RLTDFrame",
        # "BLTDFrame",
        # "ReleaseFrame"
    ]

    for indexfile_path in index_file_paths:
        try:
            rows, headers = read_index_xlsx(indexfile_path, xlsx_cols)
        except Exception as e:
            print(e)
            sys.exit(1)

        if len(rows) == 0:
            print(" * No event timings for (all columns are required): %s" %
                  indexfile_path)
            continue

        p = Path(indexfile_path)
        subject_id = p.parent.name
        output_dir = p.parent.joinpath("Output")

        for row in rows:
            throw_id, camera_id = row

            if camera_id != cam_id:
                continue

            com_fname = f"{subject_id}_{throw_id}-com.npy"
            input_com = output_dir.joinpath(com_fname)
            if not input_com.is_file():
                print(
                    f"Skipping... No CoM file found for: {subject_id}_{throw_id}")
                continue

#            input_video, input_com = get_input_files(p.parent, row[0])
            video_fname = f"{subject_id}_{throw_id}_{cam_id}-sync.mp4"
            input_video = p.parent.joinpath("Sync", video_fname)

            if not input_com.is_file():
                print(
                    f"Skipping... No video file (camera={cam_id}) found for: {subject_id}_{throw_id}")
                continue

            # compute velocity (3D)
            v_arr = compute_velocity(input_com)
            print(v_arr)

            # render on video
            outfile = output_dir.joinpath("{subject_id}_{throw_id}-CoM-velocity.mp4")
            print("Processing {input_video} & {input_com} --> {outfile}")
#            render_output(str(input_video), str(outfile), v_arr)
            plot_v(v_arr, fps=240, title=f"CoM: {subject_id} - throw: {throw_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analyze-com")
    parser.add_argument("input_dir",
                        help="Input directory file, eg. 2023-02-16/")
#    parser.add_argument("input_com",
#                        help="Input directory file, eg. 2023-02-16/")
    args = parser.parse_args()

    process_files(args.input_dir)
    sys.exit(0)

    outfile = "analysis-test.mp4"
    input_com = args.input_com
    v_arr = compute_velocity(input_com)
    render_output(args.input_video, outfile, v_arr)
