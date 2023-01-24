import os
from collections import defaultdict
import pprint

import numpy as np

from keypoint_tools import keypoint_diff
from sequence_tools import find_by_idx
from datatools import quickload


def inspect(sequence, pose_idx):
    stat = defaultdict(float)
    prev = None
    ds = []

    for frame in sequence:
        obj = find_by_idx(frame, pose_idx)
        if not obj:
            continue

        if prev:
            d = keypoint_diff(obj["keypoints"], prev["keypoints"])
            print("Frame %s: keypoint diff=%.3f" % (frame["image_id"], d))
            if d < 5:
                stat["diff_lt_5"] += 1
            if d > 100:
                stat["diff_exceed_100"] += 1
            if d > 200:
                stat["diff_exceed_200"] += 1
            if not np.isnan(d):
                ds.append(d)
            else:
                stat["nan"] += 1

        prev = obj

    return stat, np.array(ds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("poi_detector.py")
    parser.add_argument("id", type=str)
    parser.add_argument("pose_idx", type=int)
    args = parser.parse_args()

    # load data
    sequence = quickload(args.id)
    stat, ds  = inspect(sequence, args.pose_idx)
    pprint.pprint(stat)

    print("mean diff: %.2f +- %.2f" % (ds.mean(), ds.std()))


