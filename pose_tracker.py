from collections import defaultdict
import pprint

import numpy as np

from nearestbox import intersect
from keypoint_tools import box_from_keypoints

# bbox overlap for pose matching
BBOX_OVERLAP_THRESHOLD = 0.80

__all__ = ["harmonize_indices"]


def get_pose_idx_events(sequence, return_stat=False):
    """Match pose index 
    """
    stat = defaultdict(int)

    # index matching
    last_frame_idxs = None
    last_frame_objs = []

    index_matches = []

    for frame in sequence:
        frame_idxs = set(pose["idx"] for pose in frame["objs"])
        frame_objs = dict((pose["idx"], pose) for pose in frame["objs"])
        image_id = frame["image_id"]

        if last_frame_idxs:
            appear_idx = frame_idxs.difference(last_frame_idxs)
            dropped_idx = last_frame_idxs.difference(frame_idxs)

            if dropped_idx:
                # case 1: idx changed (drop)
                stat["drop"] += len(dropped_idx)

            if appear_idx:
                # case 1: idx changed (appear)
                stat["appear"] += len(appear_idx)

            if dropped_idx and appear_idx:
                # case 2: idx--pose switch
                stat["match_candidate"] += len(appear_idx)

                # build last frame obj bbox list
                dropped_boxes = []
                drop_ilookup = {}
                for i, d_idx in enumerate(dropped_idx):
                    pose = last_frame_objs[d_idx]
                    box = box_from_keypoints(pose["keypoints"])
                    dropped_boxes.append(box)
                    drop_ilookup[i] = d_idx

                # compute bbox overlaps between dropped and appeared poses
                appear_boxes = []
                appear_ilookup = {}
                for i, a_idx in enumerate(appear_idx):
                    pose = frame_objs[a_idx]
                    box = box_from_keypoints(pose["keypoints"])
                    appear_boxes.append(box)
                    appear_ilookup[i] = a_idx

                # overlaps[point/4][box]
                overlaps = intersect(appear_boxes, dropped_boxes)
                if not np.any(overlaps >= BBOX_OVERLAP_THRESHOLD):
                    continue

#                print("Frame %s -- match candidates: %s --> %s" %
#                        (image_id, dropped_idx, appear_idx))

                for aix, dd in enumerate(overlaps):
                    for dix, ratio in enumerate(dd):
                        i0 = drop_ilookup[dix]
                        i1 = appear_ilookup[aix]
                        if ratio > BBOX_OVERLAP_THRESHOLD:
                            index_matches.append([image_id, i1, i0])
                            stat["index_change"] += 1
#                            print("  overlap: %i -> %i : %.2f" %
#                                    (i0, i1, ratio))

        # update indices
        last_frame_idxs = frame_idxs
        last_frame_objs = frame_objs

    if return_stat:
        return index_matches, stat

    return index_matches

def parse_frame_num(image_id):
    return int(image_id.split(".")[0])

def within(x0, x1, x):
    return x0 <= x and x < x1

def remap_idx_inplace(sequence, pose_idx_changes):
    open_intervals = {}             # orig_idx -> [start_frame, vidx]
    intervals = defaultdict(list)   # orig_idx -> [[f0, f1, orig_id, vidx]]
    vidx_seq = 10000
    # iterate over index change events
    for frame, i1, i0 in pose_idx_changes:
        frame_num = parse_frame_num(frame)
#        print("Frame %i, update %i -> %i" % (frame_num, i0, i1))

        if i0 in open_intervals:
            # open interval closes
            start_frame, int_vidx = open_intervals[i0]
            interval = [start_frame, frame_num, i0, int_vidx]
            intervals[i0].append(interval)
            del open_intervals[i0]
#            print("    interval", interval)
            # ...new interval opens
            open_intervals[i1] = [frame_num, int_vidx]

        if i0 not in intervals:
            # new vid
            vidx_seq += 1
            # first interval closes
            interval = [0, frame_num - 1, i0, vidx_seq]
            intervals[i0].append(interval)
#            print("    first interval", [0, frame_num, i0, vidx_seq])
            # ...new interval opens
            open_intervals[i1] = [frame_num, vidx_seq]
#            print("    new virtual_idx (%i) for idx: %i" % (vidx_seq, i0))
#        print("    open intervals", open_intervals)

    # close open remaining intervals
    last_frame = len(sequence)
    for orig_id, open_interval in open_intervals.items():
        start_frame, int_vidx = open_interval
        interval = [start_frame, last_frame, orig_id, int_vidx]
        intervals[orig_id].append(interval)
#        print("Close remaining open int.", interval)
#    pprint.pprint(intervals)

    # remap sequence
    for frame in sequence:
        frame_num = parse_frame_num(frame["image_id"])
        for obj in frame["objs"]:
            idx = obj["idx"]

            if idx not in intervals:
                continue

            for f0, f1, orig_idx, vidx in intervals[idx]:
                # for each idx: belongs to interval  { orig_idx: [frame0, frame1, vidx] }
                if within(f0, f1, frame_num):
#                    print("frame %i : remap idx %i -> %i" % (frame_num, idx, vidx))
                    # remap idx
                    obj["idx"] = vidx

def harmonize_indices(sequence):
    pose_idx_events = get_pose_idx_events(sequence)
    remap_idx_inplace(sequence, pose_idx_events)
    return sequence, pose_idx_events


if __name__ == "__main__":
    from ap_loader import load_alphapose_json
    import argparse

    parser = argparse.ArgumentParser("poi_detector.py")
    parser.add_argument("input_json")
    args = parser.parse_args()

    # load data
    print("Opening %s" % args.input_json)

    sequence = load_alphapose_json(args.input_json)

    pose_idx_events, stat = get_pose_idx_events(sequence, return_stat=True)
    pprint.pprint(stat)
    pprint.pprint(pose_idx_events)
    remap_idx_inplace(sequence, pose_idx_events)
