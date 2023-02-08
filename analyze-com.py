import argparse
import sys

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
    # np.sqrt(np.diff(c[:, 0])**2+np.diff(c[:, 1])**2+np.diff(c[:, 2])**2) * fps
    return np.diff(c[:, 0]) * fps


def plot_v(v, fps=240):
    # discrete diff starts at 1 frame
    t = (1 + np.arange(v.shape[0])) / fps
    plt.plot(t, v)
    plt.title(input_com)
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


def render_output(input_video, outfile, v):
    # read input
    input_stream = cv2.VideoCapture(input_video)

    # get input video properties
    width = int(input_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_stream.get(cv2.CAP_PROP_FPS))
    # frame_count = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    # build velocity curve points
    Nv = v.shape[0]
    interp_v = scipy.interpolate.interp1d(np.arange(Nv), v)
    curve_x = np.arange(width - 1) / (width - 1) * (Nv - 1)
    curve_y = interp_v(curve_x)
    CURVE_BOTTOM = 340
    curve_y /= np.max(curve_y)
    curve_y *= CURVE_BOTTOM - 10
    curve_points = np.array([(x + 1, int(CURVE_BOTTOM - y))
                             for (x, y) in enumerate(curve_y)])

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
            frame_v = v[num_frame - 1]
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


def process_files(input_dir):
    index_file_paths = indexfiles.glob_index_files(input_dir)
    xlsx_cols = [
        "XOTOFrame",
        "RLTDFrame",
        "BLTDFrame",
        "ReleaseFrame"
    ]

    print(index_file_paths)

    for indexfile_path in index_file_paths:
        try:
            data, headers = read_index_xlsx(indexfile_path, xlsx_cols)
        except Exception as e:
            print(e)
            sys.exit(1)

        print(data)
        # compute velocity (3D)
        input_com = "S1_02-com.npy"
        v = compute_velocity(input_com)

        # render on video
        input_video = "2023-01-18\\Subjects\\S1\\Sync\\S1_03_oe-sync.mp4"
        outfile = "analysis-test.mp4"
        render_output(input_video, outfile, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("analyze-com")
    parser.add_argument("input_dir",
                        help="Input directory file, eg. 2023-02-16/")
    args = parser.parse_args()

    process_files(args.input_dir)
