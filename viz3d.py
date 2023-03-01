import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from datasource import DataSource
from keypoints import KEYPOINTS
import com

skeleton = [
    ("LShoulder", "RShoulder"),
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


def plot_skeleton(ax, arr):
    for (i0, i1) in skeleton:
        k0 = KEYPOINTS[i0]
        k1 = KEYPOINTS[i1]
        coords = [np.take(arr, (3*k0 + axis, 3*k1 + axis))
                  for axis in range(3)]
        ax.plot3D(coords[0], coords[1], coords[2], '-', linewidth=4.0)
#    ax.plot3D(arr[0::3], arr[1::3], arr[2::3], 'o', markersize=1.5)


def compute_head(arr):
    return (np.array(getcoords(arr, "Head")) + np.array(getcoords(arr, "Neck"))) / 2.0


def plot_head(ax, arr, radius=0.08, resolution=8):
    head = compute_head(arr)
    [cx, cy, cz] = head[0:3]
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = cx + radius * np.outer(np.cos(u), np.sin(v))
    y = cy + radius * np.outer(np.sin(u), np.sin(v))
    z = cz + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    # plot the ball surface
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.50)


def plot_point(ax, point, **kwargs):
    ax.plot3D([point[0], point[0]],
              [point[1], point[1]],
              [point[2], point[2]],
              **kwargs)


def plot_com(ax, arr):
    plot_point(ax, arr, marker='o', color="black", markersize=10, alpha=0.5)
    plot_point(ax, arr, marker='o', color="white", markersize=5, alpha=0.5)


def progress(n, tot):
    percent_ready = (100 * (n + 1)) // tot
    print("\b\b\b\b\b%3d %%" % percent_ready, end="", flush=True)


def animate(frame_counter, n_frames, frame_start, fps, posearr, comarr, segment_comarr, per_anim_frame, axs):
    frame = frame_start + frame_counter * 240 // fps
    #t = frame_counter/n_frames
    # azim = 0 #- 80*t
    # elev = 0 #- 35*t
    fig.suptitle("Frame: %i" % (frame_counter * per_anim_frame))
    for ax, view, title in axs:
        azim, elev = view
        ax.clear()
        ax.view_init(azim=azim, elev=elev)
        plot_skeleton(ax, posearr[frame])
        plot_head(ax, posearr[frame])

        if comarr is not None:
            s0, s1 = max(0, frame-30), min(frame+30, len(comarr))
            ax.plot3D(comarr[s0:s1, 0],
                      comarr[s0:s1, 1],
                      comarr[s0:s1, 2],
                      color="red", linewidth=2.0, alpha=0.8)
            plot_com(ax, comarr[frame])

        for k, scom in enumerate(segment_comarr):
            plot_point(ax, scom[frame],
                       markersize=7,
                       color="white",
                       marker=".")
            plot_point(ax, scom[frame], markersize=4, color="red", marker=".")

        ax.set_title(title)
        ax.set_aspect('equal')

    # set padding
    plt.subplots_adjust(left=0, bottom=0, right=1.0,
                        top=0.9, wspace=0, hspace=0)


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


def process_file(input, output, input_com=None, frame=None, output_fps=60):
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
        ax = plt.axes(projection='3d')
        plot_skeleton(posearr[frame])
        plot_head(posearr[frame])

        if comarr is not None:
            plot_com(comarr[frame])

        ax.set_aspect('equal')
        plt.show()
    else:
        # define views
        FRONT = (0, 0)
        SIDE = (-90, 0)
        TOP = (0, 90)

        axs = [
            (fig.add_subplot(131, projection='3d'), FRONT, "Front"),
            (fig.add_subplot(132, projection='3d'), SIDE, "Side"),
            (fig.add_subplot(133, projection='3d'), TOP, "Top")
        ]

        # animate full video
        n_video_frames = (len(posearr) - frame_start) * output_fps // 240
        frames_per_animframe = 240 / output_fps

        # compute segment coms
        segment_coms = com.compute_segment_com(posearr)

        anim_args = (
            n_video_frames,
            frame_start,
            output_fps,
            posearr,
            comarr,
            segment_coms,
            frames_per_animframe,
            axs
        )

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
    parser.add_argument("--output-fps",
                        type=int,
                        default=60,
                        help="Output video fps. Default: 60")
    args = parser.parse_args()

    input = Path(args.input)

    # setup 3d plot
    fig = plt.figure(figsize=(20, 8), dpi=72)
    plt.tight_layout(pad=1)

    if input.is_file():
        print(f"Creating 3d visualization for: {args.input}")
        process_file(args.input, args.output, args.com, frame=args.frame)
    elif input.is_dir():
        process_dir(args.input,
                    subject=args.subject,
                    trial=args.trial,
                    output_fps=args.output_fps)
