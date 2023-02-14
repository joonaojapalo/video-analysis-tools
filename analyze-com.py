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
GRAY_DARK_ALPHA = (32, 32, 32, 166)
WHITE = (255, 255, 255)
WHITE_ALPHA = (255, 255, 255, 128)


def com_velocity(c, fps, method="window", window=0.05):
    # create output array
    output = np.zeros(c.shape[0])
    if method == "diffxy":
        # x-y plane
        output[1:] = np.sqrt(np.diff(c[:, 0])**2+np.diff(c[:, 1])**2) * fps
        output[0] = output[1]
    elif method == "diffxyz":
        # x-y-z plane
        output[1:] = np.sqrt(np.diff(c[:, 0])**2 +
                             np.diff(c[:, 1])**2+np.diff(c[:, 2])**2) * fps
        output[0] = output[1]
    elif method == "diffx":
        # x-plane
        output[1:] = np.diff(c[:, 0]) * fps
        output[0] = output[1]
    elif method == "window":
        n = int(fps * window)
        xpp = c[n:, 0]
        xpn = c[:-n, 0]
        offset_0 = n // 2
        offset_1 = n - offset_0
        output[offset_0:-offset_1] = (xpp - xpn) * fps / n
        output[:offset_0] = output[n]
        output[-offset_1:] = output[-1]
    else:
        raise ValueError("Invalid method: %s" % (method))
    return output


def compute_velocity(input_com_npy, fps, method="window", window=0.05):
    com = np.load(input_com_npy)
    return com_velocity(com, fps, method, window)


def nan_partition(x):
    """
    Parameters:
    x (np.array)

    Returns:
    (list of arrays, list of arrays)
    """
    ns = np.isnan(x)
    d = np.zeros(ns.shape)
    d[1:] = np.cumsum(np.diff(ns))
    i = int(ns[0])
    partitions = []
    indices = []
    for k in range(int(d.max())+1):
        # k is a partition?
        if (k + i) % 2 == 0:
            partitions.append(x[d == k])
            indices.append(np.where(d == k)[0])
    return partitions, indices


def plot_v(v, fps=240, title="CoM"):
    # discrete diff starts at 1 frame
    t = (1 + np.arange(v.shape[0])) / fps
    #plt.plot(t, v)
    plt.plot(v)
    plt.title(title)
    plt.xlabel("sec")
    plt.ylabel("m/s")
    plt.grid()
    plt.show()


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


def get_scale(v_arr):
    # high & low not-nan value boundaries
    ma = np.max(v_arr[np.isfinite(v_arr)])
    mi = np.min(v_arr[np.isfinite(v_arr)])
    hh = int(np.ceil(ma))
    ll = int(np.floor(mi))
    return ma, mi, hh, ll


def compute_grid_lines(v_arr, CURVE_BOTTOM, CURVE_HEIGHT, width):
    h_grid_lines = []

    ma, mi, hh, ll = get_scale(v_arr)

    d = ma - mi
    n_grid_lines = hh-ll+1
    scale = np.linspace(ll, hh, n_grid_lines)
    scale_labels = []

    for k in range(n_grid_lines):
        gy = int(CURVE_BOTTOM - k * CURVE_HEIGHT / (n_grid_lines - 1))
        scale_labels.append(gy)
        h_grid_lines.append(np.array([[30, gy], [width-1, gy]]))

    return h_grid_lines, scale, scale_labels


def render_output(input_video, outfile, v_arr):
    # read input
    input_stream = cv2.VideoCapture(input_video)

    # get input video properties
    width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_stream.get(cv2.CAP_PROP_FPS))
    # frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # build list of non-nan curve parts
    ys, xs = nan_partition(v_arr)

    CURVE_BOTTOM = 400
    CURVE_TOP = 50
    CURVE_HEIGHT = CURVE_BOTTOM - CURVE_TOP

    # get scale grid lines
    h_grid_lines, scale, scale_labels = compute_grid_lines(
        v_arr, CURVE_BOTTOM, CURVE_HEIGHT, width)

    # get curve value bounds
    ma, mi, hh, ll = get_scale(v_arr)

    curves = []
    for xa, ya in zip(xs, ys):
        # build velocity curve points
        Nv = ya.shape[0]
        curve_segment_width = int(width * xa.shape[0] / v_arr.shape[0])
        interp_v = scipy.interpolate.interp1d(np.arange(Nv), ya)
        curve_x = np.arange(curve_segment_width) / \
            (curve_segment_width) * (Nv - 1)
        curve_y = interp_v(curve_x)

        # scale curve y
        curve_y = (curve_y - mi) / (hh - ll) + (mi - ll) / (hh - ll)
        curve_y *= CURVE_HEIGHT

        orig_x = int(xa[0] / v_arr.shape[0] * width)
        curve_points = np.array([(x + 1 + orig_x, int(CURVE_BOTTOM - y))
                                 for (x, y) in enumerate(curve_y)], dtype=int)
        curves.append(curve_points)

    # open output video file
    output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(
        'm', 'p', '4', 'v'), fps, (width, height))

    num_frame = 0

    font = cv2.FONT_HERSHEY_DUPLEX

    while input_stream.isOpened():
        ret, image = input_stream.read()

        if not ret:
            break

        if num_frame > 1:
            frame_v = v_arr[num_frame - 1]

            # gridlines
            cv2.polylines(image, h_grid_lines, False,
                          GRAY_DARK_ALPHA, 2, cv2.LINE_AA)

            # velocity curve
            cv2.polylines(image, curves, False, YELLOW, 2, cv2.LINE_AA)

            # vertical marker
            hx = 1 + (width - 1) * (1 + num_frame) // Nv
            cv2.line(image, [hx, 10], [hx, CURVE_BOTTOM], GRAY_DARK_ALPHA, 5)
            cv2.line(image, [hx, 10], [hx, CURVE_BOTTOM], WHITE, 2)

            # gridline labels
            for scale_y, label in zip(scale_labels, scale):
                cv2.putText(image, str(label),
                            (5, scale_y + 5),
                            font,
                            0.6, GRAY_DARK_ALPHA, 5, cv2.LINE_AA)
                cv2.putText(image, str(label),
                            (5, scale_y + 5),
                            font,
                            0.6, WHITE, 1, cv2.LINE_AA)

            # velocity
            text = "%.2f m/s" % frame_v
            text_size = cv2.getTextSize(text, font, 1.2, 1)[0]
            cv2.putText(image, text,
                        (width - text_size[0] - 10,  text_size[1] + 10),
                        font, 1.2, GRAY_DARK_ALPHA, 5, cv2.LINE_AA)
            cv2.putText(image, text,
                        (width - text_size[0] - 10,  text_size[1] + 10),
                        font, 1.2, WHITE, 1, cv2.LINE_AA)

            # title
            text = "Velocity of athlete's center-of-mass:"
            cv2.putText(image, text, (10, 25), font, 1.0,
                        GRAY_DARK_ALPHA, 5, cv2.LINE_AA)
            cv2.putText(image, text, (10, 25), font,
                        1.0, WHITE, 1, cv2.LINE_AA)

        # write output
        output.write(image)
        if num_frame % 24 == 0:
            print(".", end="", flush=True)
        num_frame += 1

    print()


def get_input_files(index_dir_path, throw_id):
    output_dir = index_dir_path.joinpath("Output")
    coms = list((f"*_{throw_id}-com.npy"))
    if len(coms) == 0:
        raise Exception("No CoM trajectory file found in: %s" %
                        str(output_dir))
    if len(coms) > 1:
        raise Exception(
            "CoM trajectories for multiple subjects found in: %s" % str(output_dir))
    return coms


def process_files(input_dir, use_cam_id="oe", use_subject_id=None, use_throw_id=None, capture_fps=240, plot_only=False):
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

            if camera_id != use_cam_id:
                continue

            if use_subject_id is not None and use_subject_id != subject_id:
                continue

            if use_throw_id is not None and use_throw_id != throw_id:
                continue

            com_fname = f"{subject_id}_{throw_id}-com.npy"
            input_com = output_dir.joinpath(com_fname)
            if not input_com.is_file():
                print(
                    f"Skipping... No CoM file found for: {subject_id}_{throw_id}")
                continue

#            input_video, input_com = get_input_files(p.parent, row[0])
            video_fname = f"{subject_id}_{throw_id}_{use_cam_id}-sync.mp4"
            input_video = p.parent.joinpath("Sync", video_fname)

            if not input_com.is_file():
                print(
                    f"Skipping... No video file (camera={use_cam_id}) found for: {subject_id}_{throw_id}")
                continue

            # compute velocity (3D)
            v_arr = compute_velocity(input_com, capture_fps)

            if plot_only:
                plot_v(v_arr, fps=240,
                       title=f"CoM: {subject_id} - throw: {throw_id}")
            else:
                # render on video
                outfile_name = f"{subject_id}_{throw_id}-CoM-velocity.mp4"
                outfile = output_dir.joinpath(outfile_name)
                print(f"Creating video:")
                print(f"  - input (video): {input_video}")
                print(f"  - input (CoM): {input_com}")
                print(f"  - output: {outfile}")
                render_output(str(input_video), str(outfile), v_arr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analyze-com")
    parser.add_argument("input_dir",
                        help="Input directory file, eg. 2023-02-16/")
    parser.add_argument("--subject", "-S",
                        default=None,
                        help="Process only subject with id, eg. KE101")
    parser.add_argument("--trial", "-T",
                        default=None,
                        help="Process only trial with id, eg. 01")
    parser.add_argument("--cam", "-C",
                        default=None,
                        help="Use video from camera, eg. 'oe'")
    parser.add_argument("--capture-fps",
                        dest="capture_fps",
                        default=240,
                        help="Original video capture FPS. Default: 240")
    parser.add_argument("--plot-only",
                        dest="plot_only",
                        default=False,
                        action="store_true",
                        help="Show curve plot only.")
    args = parser.parse_args()

    process_files(args.input_dir, use_subject_id=args.subject,
                  use_throw_id=args.trial, capture_fps=args.capture_fps)
