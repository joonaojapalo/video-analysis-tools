import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    ("RHip", "Rknee"),
    ("RHip", "LHip"),
    ("LKnee", "LAnkle"),
    ("Rknee", "RAnkle"),
    ("LAnkle", "LHeel"),
    ("RAnkle", "RHeel"),
    ("LHeel", "LBigToe"),
    ("RHeel", "RBigToe"),
    ("LAnkle", "LBigToe"),
    ("RAnkle", "RBigToe"),
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
        ax.plot3D(coords[0], coords[1], coords[2], '-', linewidth=2.0)
    ax.plot3D(arr[0::3], arr[1::3], arr[2::3], 'o', markersize=0.5)


def compute_head(arr):
    return (np.array(getcoords(arr, "LEar")) + np.array(getcoords(arr, "REar"))) / 2


def plot_head(arr):
    head = compute_head(arr)
    ax.plot3D([head[0], head[0]], [head[1], head[1]],
              [head[2], head[2]], 'o', markersize=20)


def animate(frame_counter, n_frames, frame_start, fps):
    frame = frame_start + frame_counter * 240 // fps
    t = frame_counter/n_frames
    azim = -5 - 80*t
    elev = 45 - 35*t
    ax.clear()
    ax.view_init(azim=azim, elev=elev)
    plot_skeleton(posearr[frame])
    plot_head(posearr[frame])
    ax.set_aspect('equal')


usage = """
  python viz3d.py .\S1_01-pos.npy --save
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("viz3d.py", usage=usage)
    parser.add_argument("input",
                        help="Input file, eg. worldpos.npy")
    parser.add_argument("-o", "--output",
                        default=None,
                        help="File name for output video.")
    parser.add_argument("-f", "--frame",
                        default=None,
                        help="Frame to inspect.")
    args = parser.parse_args()

    if args.output:
        print("WARNING: Output file already exist:", args.output)

    # load
    posearr = np.load(args.input)
    frame_start = 0

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if args.frame:
        frame = int(args.frame)
        plot_skeleton(posearr[frame])
        plot_head(posearr[frame])
        ax.set_aspect('equal')
        plt.show()
    else:
        # animate full video
        fps = 120
        n_video_frames = (len(posearr) - frame_start) * fps // 240
        ani = FuncAnimation(fig, animate,
                            frames=n_video_frames,
                            fargs=(n_video_frames, frame_start, fps),
                            interval=4*1000/fps,
                            repeat=True)

        if args.output:
            print("Processing video...")
            ani.save(args.output)
            print()
            print(f"Wrote {n_video_frames} frames to file: {args.output}")
            print()
        else:
            plt.show()
