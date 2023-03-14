import sys
import os
import copy

from pylab import plot, show, text, xlim, ylim, title, subplot

from alphapose_json import load_alphapose_json
from pose_tracker import *
from keypoint_tools import box_from_keypoints
from pose_tracker import harmonize_indices

fn = sys.argv[1] if len(
    sys.argv) > 1 else "..\\javelin-data\\2023-01-18\Subjects\S1\Pose\S1_08_ot-sync\\alphapose-results.json"
s = load_alphapose_json(fn)
h = copy.deepcopy(s)
h, pose_idx_events = harmonize_indices(h)
print("Loaded", fn)


def xy(box):
    x, y, ww, hh = box
    w = ww//2
    h = hh//2
    return [x-w, x+w, x+w, x-w, x-w], [y-h, y-h, y+h, y+h, y-h]


colors = ["red", "blue", "green", "orange", "cyan", "pink"]


def plot_frame(sequence, f0, f1=None):
    xlim([0, 1920])
    ylim([1080, 0])
    print("Plotting frames",f0,f1)
    for frame in range(f0, f1 if f1 else f0+1):
        if len(sequence) < frame:
            continue
        for o in sequence[frame]["objs"]:
            idx = int(o["idx"])
            color = colors[idx % len(colors)]
            b = box_from_keypoints(o["keypoints"])
            xs, ys = xy(b)
            plot(xs, ys, '-', color=color, linewidth=0.75, alpha=0.5)
            text(b[0], b[1], o["idx"], ha="center", va="bottom",
                 fontweight="semibold", color=color)
            text(b[0], b[1], frame, ha="center",
                 va="top", color=color, fontsize=6)


f0 = int(sys.argv[2]) if len(sys.argv) > 2 else 140
f1 = int(sys.argv[3]) if len(sys.argv) > 3 else None
#f1 = min(f1, len(sequence)) if f1 else f0+1
subplot(2, 1, 1)
title(os.path.realpath(fn).split(os.path.sep)[-2])
plot_frame(s, f0, f1)
subplot(2, 1, 2)
plot_frame(h, f0, f1)
show()
