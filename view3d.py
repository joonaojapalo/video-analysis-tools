from typing import List
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from com import SegmentCenterOfMass
from keypoints import KEYPOINTS

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

required = [
    "LAnkle",
    "LElbow",
    "LHip",
    "LKnee",
    "LShoulder",
    "LWrist",
    "RAnkle",
    "RElbow",
    "RHip",
    "RKnee",
    "RShoulder",
    "RWrist"
]


def getcoords(arr, kp0, kp1=None):
    i0 = KEYPOINTS[kp0]
    if kp1:
        i1 = KEYPOINTS[kp1]
        return [np.take(arr, (3*i0 + axis, 3*i1 + axis)) for axis in range(3)]
    else:
        return [arr[3*i0 + axis] for axis in range(3)]


# def compute_head(arr):
#    return (np.array(getcoords(arr, "Head")) + np.array(getcoords(arr, "Neck"))) / 2.0

def get_skeleton_data(arr):
    xyz = []
    for (i0, i1) in skeleton:
        k0 = KEYPOINTS[i0]
        k1 = KEYPOINTS[i1]
        coords = [np.take(arr, (3*k0 + axis, 3*k1 + axis))
                  for axis in range(3)]
        xyz.append(coords)
    return xyz


def get_head_pos(segments: List[SegmentCenterOfMass]):
    segment = next(s for s in segments if s.name == "head")
    return segment.pos


def not_any_nan(arr):
    return not np.isnan(np.array(arr).ravel()).any()


def has_skeleton_data(viewdata, frame: int) -> bool:
    pos_skeleton = viewdata.pose[frame]
    pos_head = get_head_pos(viewdata.segments)[frame]

    has_skeleton = not_any_nan(get_skeleton_data(pos_skeleton))
    has_head = not_any_nan(pos_head)

    return has_skeleton and has_head


def format_point_3d(point):
    return (
        [point[0], point[0]],
        [point[1], point[1]],
        [point[2], point[2]]
    )


def plot_point(ax, point, **kwargs):
    lines, = ax.plot3D(*format_point_3d(point), **kwargs)
    return lines


class ViewData:
    def __init__(self, posearr, comarr, segments: List[SegmentCenterOfMass]) -> None:
        self.pose = posearr
        self.com = comarr
        self.segments = segments
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
                self._bounds = []

                for dim in range(3):
                    x = self.pose[:, dim::3]
                    y = x[np.isfinite(x)]
                    self._bounds.append((y.min(), y.max()))

            return self._bounds

    def get_frame_count(self):
        return len(self.pose)


class SkeletonView:
    def __init__(self, axes) -> None:
        self.clear()
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

    def _get_head_data(self, pos_head):
        [cx, cy, cz] = pos_head[0:3]
        x = cx + self.head_mesh[0]
        y = cy + self.head_mesh[1]
        z = cz + self.head_mesh[2]
        return x, y, z

    def _render_skeleton(self, arr):
        xyz = get_skeleton_data(arr)
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

    def _render_head(self, pos_head):
        x, y, z = self._get_head_data(pos_head)
        # plot the ball surface
        surf = self.axes.plot_surface(x, y, z,
                                      color='lightgray',
                                      shade=True,
                                      alpha=0.50)
        self.artists["head"] = surf

    def _set_data_head(self, pos_head):
        self.artists["head"].remove()
        self._render_head(pos_head)

    def _should_render(self):
        return len(self.artists["skeleton"]) == 0 or self.artists["head"] is None

    def render(self, viewdata, frame: int) -> None:
        if not has_skeleton_data(viewdata, frame):
            return

        pos_skeleton = viewdata.pose[frame]
        pos_head = get_head_pos(viewdata.segments)[frame]

        self.clear()
        self._render_skeleton(pos_skeleton)
        self._render_head(pos_head)

    def set_data(self, viewdata, frame):
        if not has_skeleton_data(viewdata, frame):
            return

        if self._should_render():
            self.render(viewdata, frame)
            return

        pos_skeleton = viewdata.pose[frame]
        pos_head = get_head_pos(viewdata.segments)[frame]

        self._set_data_skeleton(pos_skeleton)
        self._set_data_head(pos_head)


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

    def _should_render(self):
        return any([v is None for v in self.artists.values()])

    def render(self, viewdata, frame: int) -> None:
        self.clear()

        if viewdata.com is None:
            return

        point, trace_x, trace_y, trace_z = self._get_com_data(
            viewdata.com,
            frame
        )

        # has data?
        if not not_any_nan(point):
            return

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
        if viewdata.com is None:
            return

        if self._should_render():
            self.render(viewdata, frame)
            return

        point, trace_x, trace_y, trace_z = self._get_com_data(
            viewdata.com,
            frame
        )

        self.artists["trace"].set_data_3d(trace_x, trace_y, trace_z)
        self.artists["point"][0].set_data_3d(format_point_3d(point))
        self.artists["point"][1].set_data_3d(format_point_3d(point))


class SegmentComView:
    def __init__(self, axes) -> None:
        self.clear()
        self.axes = axes

    def clear(self):
        self.artists = {
            "outer": [],
            "inner": []
        }

    def _should_render(self):
        return any([len(v) == 0 for v in self.artists.values()])

    def render(self, viewdata: ViewData, frame: int) -> None:
        self.clear()

#        if not not_any_nan([segment_com.pos[frame] for segment_com in viewdata.segments]):
#            return

        for segment_com in viewdata.segments:
            a = plot_point(self.axes,
                           segment_com.pos[frame],
                           markersize=7,
                           color="white", marker=".")
            b = plot_point(self.axes,
                           segment_com.pos[frame],
                           markersize=4,
                           color="red", marker=".")
            self.artists["outer"].append(a)
            self.artists["inner"].append(b)

    def set_data(self, viewdata, frame: int) -> None:
        if self._should_render():
            self.render(viewdata, frame)
            return

        for k, segment_com in enumerate(viewdata.segments):
            p = segment_com.pos[frame]
            self.artists["outer"][k].set_data_3d([p[0],p[0]],[p[1],p[1]],[p[2],p[2]])
            self.artists["inner"][k].set_data_3d([p[0],p[0]],[p[1],p[1]],[p[2],p[2]])


class View:
    FRONT = (0, 0)
    SIDE = (-90, 0)
    TOP = (0, 90)

    def __init__(self, posearr: np.ndarray, comarr: np.ndarray, segments: List[SegmentCenterOfMass]):
        self.fig = plt.figure(figsize=(20, 8), dpi=72)
        plt.tight_layout(pad=1)
        self.views = self.create_views(self.fig)

        self.data = ViewData(posearr, comarr, segments)
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
        bbox_full = self.data.get_bounds()
        if frame:
            bbox_frame = self.data.get_bounds(frame)
        else:
            bbox_frame = bbox_full
        cx = sum(bbox_frame[0]) / 2
        min_y, max_y = bbox_full[1]
        min_z, max_z = bbox_full[2]
        axes.set_xlim([cx - 1, cx + 1])
        axes.set_ylim([min_y, max_y])
        axes.set_zlim([min_z, max_z])

    def find_first_frame(self):
        for frame in range(self.data.get_frame_count()):
            if has_skeleton_data(self.data, frame):
                return frame

    def render(self, frame=0) -> None:
        for i, (axes, (azim, elev), title) in enumerate(self.views):
            axes.view_init(azim=azim, elev=elev)
            axes.set_aspect('equal')
            self._set_limits(axes)

        self.fig.suptitle("Frame: %i" % (frame,))
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1.0,
                            top=0.9,
                            wspace=0, hspace=0)

        if not has_skeleton_data(self.data, frame):
            return

        for i, (axes, (azim, elev), title) in enumerate(self.views):
            self.skeletons[i].render(self.data, frame)
            self.coms[i].render(self.data, frame)
            self.segment_coms[i].render(self.data, frame)

    def set_data(self, frame: int) -> None:
        self.fig.suptitle("Frame: %i" % (frame,))

        if not has_skeleton_data(self.data, frame):
            return

        bbox_frame = self.data.get_bounds(frame)
        cx = sum(bbox_frame[0]) / 2

        for i, (axes, (azim, elev), title) in enumerate(self.views):
            self.skeletons[i].set_data(self.data, frame)
            self.coms[i].set_data(self.data, frame)
            self.segment_coms[i].set_data(self.data, frame)
            axes.set_xlim([cx - 1, cx + 1])
            axes.set_aspect('equal')
