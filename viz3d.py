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
]


def getcoords(arr, kp0, kp1=None):
    i0 = KEYPOINTS[kp0]
    if kp1:
        i1 = KEYPOINTS[kp1]
        return [np.take(arr, (3*i0 + axis, 3*i1 + axis)) for axis in range(3)]
    else:
        return [arr[3*i0 + axis] for axis in range(3)]


def compute_head(arr):
    return (np.array(getcoords(arr, "Head")) + np.array(getcoords(arr, "Neck"))) / 2.0


class SkeletonView:
    def __init__(self, axes) -> None:
        self.artists = self.clear()
        self.axes = axes
        self.head_mesh = self._precompute_head_mesh()

    def clear(self):
        self.artists = {
            "skeleton": [],
            "head": []
        }

    def _precompute_head_mesh(self, radius=0.08, resolution=5):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    def _get_head_data(self, arr):
        head = compute_head(arr)
        [cx, cy, cz] = head[0:3]
        x = cx + self.head_mesh[0]
        y = cy + self.head_mesh[1]
        z = cz + self.head_mesh[2]
        return x, y, z

    def _get_skeleton_data(self, arr):
        xyz = []
        for (i0, i1) in skeleton:
            k0 = KEYPOINTS[i0]
            k1 = KEYPOINTS[i1]
            coords = [np.take(arr, (3*k0 + axis, 3*k1 + axis))
                      for axis in range(3)]
            xyz.append(coords)
        return xyz

    def _render_skeleton(self, arr):
        xyz = self._get_skeleton_data(arr)
        for x, y, z in xyz:
            lines, = self.axes.plot3D(x, y, z,
                                      '-',
                                      linewidth=4.0)
            self.artists["skeleton"].append(lines)

    def _set_data_skeleton(self, arr):
        for artist, (i0, i1) in zip(self.artists["skeleton"], skeleton):
            k0 = KEYPOINTS[i0]
            k1 = KEYPOINTS[i1]
            coords = [np.take(arr, (3*k0 + axis, 3*k1 + axis))
                      for axis in range(3)]
            artist.set_data_3d(coords[0], coords[1], coords[2])

    def _render_head(self, arr):
        x, y, z = self._get_head_data(arr)
        # plot the ball surface
        surf = self.axes.plot_surface(x, y, z,
                                      color='lightgray',
                                      shade=True,
                                      alpha=0.50)
        self.artists["head"] = surf

    def _set_data_head(self, arr):
        self.artists["head"].remove()
        self._render_head(arr)

    def render(self, viewdata, frame: int) -> None:
        self.clear()
        self._render_skeleton(viewdata.pose[frame])
        self._render_head(viewdata.pose[frame])

    def set_data(self, viewdata, frame):
        if not self.artists["skeleton"]:
            return

        arr = viewdata.pose[frame]
        self._set_data_skeleton(arr)
        self._set_data_head(arr)


class ComView:
    def __init__(self, axes) -> None:
        self.axes = axes
        self.clear()

    def clear(self):
        self.artists = {
            "point": None,
            "trace": None
        }

    def _get_com_data(self, comarr, frame):
        s0, s1 = max(0, frame-30), min(frame+30, len(comarr))
        trace_x = comarr[s0:s1, 0]
        trace_y = comarr[s0:s1, 1]
        trace_z = comarr[s0:s1, 2]
        point = comarr[frame]
        return point, trace_x, trace_y, trace_z

    def render(self, viewdata, frame: int) -> None:
        self.clear()

        if viewdata.com is None:
            return

        point, trace_x, trace_y, trace_z = self._get_com_data(
            viewdata.com,
            frame
        )

        lines, = self.axes.plot3D(trace_x, trace_y, trace_z,
                                  color="red",
                                  linewidth=2.0,
                                  alpha=0.8)
        self.artists["trace"] = lines
        self.artists["point"] = [
            plot_point(self.axes, point, marker='o', color="black",
                       markersize=10, alpha=0.5),
            plot_point(self.axes, point, marker='o', color="white",
                       markersize=5, alpha=0.5)
        ]

    def set_data(self, viewdata, frame):
        if "trace" not in self.artists:
            return

        if "point" not in self.artists:
            return

        if viewdata.com is None:
            return

        point, trace_x, trace_y, trace_z = self._get_com_data(
            viewdata.com,
            frame
        )

        self.artists["trace"].set_data_3d(trace_x, trace_y, trace_z)
        self.artists["point"][0].set_data_3d(point)
        self.artists["point"][1].set_data_3d(point)


class SegmentComView:
    def __init__(self, axes) -> None:
        self.artists = self.clear()
        self.axes = axes

    def clear(self):
        self.artists = {
            "outer": [],
            "inner": []
        }

    def render(self, viewdata, frame: int) -> None:
        self.clear()
        for scom in viewdata.segments:
            a = plot_point(self.axes,
                           scom[frame],
                           markersize=7,
                           color="white", marker=".")
            b = plot_point(self.axes,
                           scom[frame],
                           markersize=4,
                           color="red", marker=".")
            self.artists["outer"].append(a)
            self.artists["inner"].append(b)

    def set_data(self, viewdata, frame: int) -> None:
        for k, scom in enumerate(viewdata.segments):
            self.artists["outer"][k].set_data_3d(scom[frame])
            self.artists["inner"][k].set_data_3d(scom[frame])


class ViewData:
    def __init__(self, posearr, comarr, segment_arr) -> None:
        self.pose = posearr
        self.com = comarr
        self.segments = segment_arr
        self._bounds = None

    def get_bounds(self, frame=None):
        if frame:
            xs = self.pose[frame][0::3]
            ys = self.pose[frame][1::3]
            zs = self.pose[frame][2::3]
            xs = xs[np.isfinite(xs)]
            ys = ys[np.isfinite(ys)]
            zs = zs[np.isfinite(zs)]
            if len(xs) == 0:
                xs = np.zeros(1)
            if len(ys) == 0:
                ys = np.zeros(1)
            if len(zs) == 0:
                zs = np.zeros(1)
            return (xs.min(0), xs.max(0)), (ys.min(0), ys.max(0)), (zs.min(0), zs.max(0))
        else:
            if not self._bounds:
                finite = np.isfinite(self.pose).all(1)
                mins = self.pose[finite].min(0)
                maxs = self.pose[finite].max(0)
                self._bounds = [(mins[dim::3].min(), maxs[dim::3].max())
                                for dim in range(3)]
            return self._bounds


class View:
    FRONT = (0, 0)
    SIDE = (-90, 0)
    TOP = (0, 90)

    def __init__(self, posearr: np.ndarray, comarr: np.ndarray, segment_comarr: np.ndarray):
        self.fig = plt.figure(figsize=(20, 8), dpi=72)
        plt.tight_layout(pad=1)
        self.views = self.create_views(self.fig)

        self.data = ViewData(posearr, comarr, segment_comarr)
        self.skeletons = []
        self.coms = []
        self.segment_coms = []

        for axes, (azim, elev), title in self.views:
            axes.set_title(title)
            self.skeletons.append(SkeletonView(axes))
            self.coms.append(ComView(axes))
            self.segment_coms.append(SegmentComView(axes))

    def create_views(self, fig):
        return [
            (fig.add_subplot(131, projection='3d'), self.FRONT, "Front"),
            (fig.add_subplot(132, projection='3d'), self.SIDE, "Side"),
            (fig.add_subplot(133, projection='3d'), self.TOP, "Top")
        ]

    def get_artists(self) -> list:
        # return artists in canonical order
        artists = []

        for i in range(len(self.views)):
            artists.extend(self.skeletons[i].artists)
            artists.extend(self.coms.artists)
            artists.extend(self.segment_coms.artists)

        return artists

    def _set_limits(self, axes, frame=None):
        if not frame:
            bbox_frame = self.data.get_bounds(frame)
            bbox_full = self.data.get_bounds()
            cx = sum(bbox_frame[0]) / 2
            min_y, max_y = bbox_full[1]
            min_z, max_z = bbox_full[2]
            axes.set_xlim([cx - 1, cx + 1])
            axes.set_ylim([min_y, max_y])
            axes.set_zlim([min_z, max_z])

    def render(self, frame=0) -> None:
        for i, (axes, (azim, elev), title) in enumerate(self.views):
            axes.view_init(azim=azim, elev=elev)
            self.skeletons[i].render(self.data, frame)
            self.coms[i].render(self.data, frame)
            self.segment_coms[i].render(self.data, frame)
            self._set_limits(axes)
            axes.set_aspect('equal')

        self.fig.suptitle("Frame: %i" % (frame,))
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1.0,
                            top=0.9,
                            wspace=0, hspace=0)

    def set_data(self, frame: int) -> None:
        self.fig.suptitle("Frame: %i" % (frame,))
        bbox_frame = self.data.get_bounds(frame)
        cx = sum(bbox_frame[0]) / 2

        for i, (axes, (azim, elev), title) in enumerate(self.views):
            self.skeletons[i].set_data(self.data, frame)
            self.coms[i].set_data(self.data, frame)
            self.segment_coms[i].set_data(self.data, frame)
            axes.set_xlim([cx - 1, cx + 1])


def plot_point(ax, point, **kwargs):
    lines, = ax.plot3D([point[0], point[0]],
                       [point[1], point[1]],
                       [point[2], point[2]],
                       **kwargs)
    return lines


def progress(n, tot):
    percent_ready = (100 * (n + 1)) // tot
    print("\b\b\b\b\b%3d %%" % percent_ready, end="", flush=True)


def animate(frame_counter, n_frames, frame_start, fps, frames_per_animframe, view):
    frame = frame_start + frame_counter * 240 // fps
    view.set_data(frame)
    progress(frame_counter, n_frames)


def process_dir(input_dir, subject, trial, output_fps=120, trim_start=0):
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
                     output_fps=output_fps,
                     frame_start=trim_start)


def process_file(input, output, input_com=None, frame=None,
                 output_fps=60,
                 frame_start=0):
    # load pose data
    posearr = np.load(input)

    # load CoM data (if exists)
    if input_com:
        comarr = np.load(input_com)
    else:
        comarr = None

    if frame:
        frame = int(frame)
        view = View(posearr, comarr, segment_coms)
        view.render(frame)
        plt.show()
    else:
        # animate full video
        n_video_frames = (len(posearr) - frame_start) * output_fps // 240
        frames_per_animframe = 240 / output_fps

        # compute segment coms
        segment_coms = com.compute_segment_com(posearr)

        view = View(posearr, comarr, segment_coms)
        view.render(0)

        anim_args = (
            n_video_frames,
            frame_start,
            output_fps,
            frames_per_animframe,
            view
        )

        ani = FuncAnimation(view.fig, animate,
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

    OR

  viz3d 2023-02-15 -S S1 -T 01
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
    parser.add_argument("--trim-start",
                        type=int,
                        dest="trim_start",
                        default=0,
                        help="How many frames to trim from start. Default: 0")
    parser.add_argument("--trim-end",
                        type=int,
                        dest="trim_end",
                        default=None,
                        help="How many frames to trim from end. Default: 0")
    args = parser.parse_args()

    input = Path(args.input)

    # setup 3d plot
    #fig = plt.figure(figsize=(20, 8), dpi=72)
    # plt.tight_layout(pad=1)

    if input.is_file():
        print(f"Creating 3d visualization for: {args.input}")
        process_file(args.input, args.output, args.com,
                     frame=args.frame,
                     start_frame=args.trim_start)
    elif input.is_dir():
        process_dir(args.input,
                    subject=args.subject,
                    trial=args.trial,
                    output_fps=args.output_fps,
                    trim_start=args.trim_start)
