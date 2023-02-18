import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from datasource import DataSource
from keypoints import KEYPOINTS

skeleton = [
    ("Neck", "RShoulder"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("RShoulder", "RElbow"),
    ("LElbow", "LWrist"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LHip"),
    ("RShoulder", "RHip"),
    ("LHip", "LKnee"),
    ("RHip", "RKnee"),
    ("RHip", "LHip"),
    ("LKnee", "LAnkle"),
    ("RKnee", "RAnkle"),
    #("LAnkle", "LHeel"),
    #("RAnkle", "RHeel"),
    #("LHeel", "LBigToe"),
    #("RHeel", "RBigToe"),
    #("LAnkle", "LBigToe"),
    #("RAnkle", "RBigToe"),
]


def getcoords(arr, kp0, kp1=None):
    i0 = KEYPOINTS[kp0]
    if kp1:
        i1 = KEYPOINTS[kp1]
        return [np.take(arr, (3*i0 + axis, 3*i1 + axis)) for axis in range(3)]
    else:
        return [arr[3*i0 + axis] for axis in range(3)]


def plot_skeleton(arr):
    for (i0, i1) in skeleton:
        k0 = KEYPOINTS[i0]
        k1 = KEYPOINTS[i1]
        coords = [np.take(arr, (3*k0 + axis, 3*k1 + axis))
                  for axis in range(3)]
        ax.plot3D(coords[0], coords[1], coords[2], '-', linewidth=4.0)
    ax.plot3D(arr[0::3], arr[1::3], arr[2::3], 'o', markersize=1.5)


def compute_head(arr):
    # (np.array(getcoords(arr, "LEar")) + np.array(getcoords(arr, "REar"))) / 2
    return np.array(getcoords(arr, "Head"))


def plot_head(arr):
    head = compute_head(arr)
    ax.plot3D([head[0], head[0]], [head[1], head[1]],
              [head[2], head[2]], 'o', markersize=20)


def plot_com(arr):
    ax.plot3D([arr[0], arr[0]], [arr[1], arr[1]],
              [arr[2], arr[2]], 'ko', markersize=10, alpha=0.5)
    ax.plot3D([arr[0], arr[0]], [arr[1], arr[1]],
              [arr[2], arr[2]], 'wo', markersize=5, alpha=0.5)

def progress(n, tot):
    percent_ready = (100 * (n + 1)) // tot
    print("\b\b\b\b\b%3d %%" % percent_ready, end="", flush=True)

def animate(frame_counter, n_frames, frame_start, fps, posearr, comarr):
    frame = frame_start + frame_counter * 240 // fps
    t = frame_counter/n_frames
    azim = -5 - 80*t
    elev = 45 - 35*t
    ax.clear()
    ax.view_init(azim=azim, elev=elev)
    plot_skeleton(posearr[frame])
    plot_head(posearr[frame])

    if comarr is not None:
        plot_com(comarr[frame])

    ax.set_aspect('equal')


def process_dir(input_dir, subject, trial, output_fps=120):
    datasource = DataSource(input_dir, subject, trial)

    if len(datasource.outputs) == 0:
        print("No files to process.")
        return

    print("Files to process:")
    for output in datasource.outputs:
        summary_com = "yes" if output.com_path else "no"
        print(f"  {output.path} (CoM: {summary_com})")
    print()

    for output in datasource.outputs:
        print(f"Creating 3d visualization for: {output.path.name}")
        output_video_name = f"{output.path.stem}-3d-stickfigure.mp4"
        output_video = output.path.parent.joinpath(output_video_name)
        process_file(output.path,
                     output_video,
                     output.com_path,
                     output_fps=output_fps)


def process_file(input, output, input_com=None, frame=None, output_fps=120):
    # load pose data
    posearr = np.load(input)

    # load CoM data (if exists)
    if input_com:
        comarr = np.load(input_com)
    else:
        comarr = None

    frame_start = 0

    if frame:
        frame = int(frame)
        plot_skeleton(posearr[frame])
        plot_head(posearr[frame])

        if comarr is not None:
            plot_com(comarr[frame])

        ax.set_aspect('equal')
        plt.show()
    else:
        # animate full video
        output_fps = 120
        n_video_frames = (len(posearr) - frame_start) * output_fps // 240
        anim_args = (n_video_frames, frame_start, output_fps, posearr, comarr)
        ani = FuncAnimation(fig, animate,
                            frames=n_video_frames,
                            fargs=anim_args,
                            interval=2*1000/output_fps,
                            repeat=True)

        if output:
            print("  - processing video:      ", end="")
            ani.save(output, progress_callback=progress)
            print()
            print(f"  - wrote {n_video_frames} frames to file: {output}")
            print()
        else:
            plt.show()


usage = """
Visualize single input file and save output video:

  viz3d .\S1_01-pos.npy -o S1_01-skeleton.mp4
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("viz3d.py", usage=usage)
    parser.add_argument("input",
                        help="Input file, eg. worldpos.npy")
    parser.add_argument("--com",
                        help="Input CoM file, eg. worldpos-com.npy")
    parser.add_argument("-o", "--output",
                        default=None,
                        help="File name for output video.")
    parser.add_argument("-f", "--frame",
                        default=None,
                        help="Frame to inspect. Only for single input file.")
    parser.add_argument("-S", "--subject",
                        default=None,
                        help="Subject identifier. Eg. 'S1' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-T", "--trial",
                        default=None,
                        help="Trial identifier. Eg. '01' as in 'S1_01_oe/alphapose-results.json'")
    args = parser.parse_args()

    input = Path(args.input)

    # setup 3d plot
    fig = plt.figure(figsize=(11.1, 8.3), dpi=72)
    plt.tight_layout(pad=1)
    ax = plt.axes(projection='3d')

    if input.is_file():
        print(f"Creating 3d visualization for: {args.input}")
        process_file(args.input, args.output, args.com, frame=args.frame)
    elif input.is_dir():
        process_dir(args.input, subject=args.subject, trial=args.trial)
