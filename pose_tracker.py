from collections import defaultdict
import pprint

import numpy as np

from nearestbox import intersect
from keypoint_tools import box_from_keypoints

# bbox overlap for pose matching
BBOX_OVERLAP_THRESHOLD = 0.75

__all__ = ["harmonize_indices"]


def build_boxes(idxs, obj_lookup):
    """Build boxes from by obj_lookup with respective list inverse lookup"""
    boxes = []
    ilookup = {}
    for i, idx in enumerate(idxs):
        pose = obj_lookup[idx]
        box = box_from_keypoints(pose["keypoints"])
        boxes.append(box)
        ilookup[i] = idx
    return boxes, ilookup


def parse_frame_number(image_id):
    return int(image_id.replace(".jpg", ""))


def get_pose_idx_events(sequence, return_stat=False, verbose=False, max_gap=2):
    """Get index change events.

    Parameters:
    sequence (list)     : Alphapose sequence.
    return_stat (bool)  : Should return statistics as additional value
    verbose (book)      : Verbose logging. Default: False.
    max_gap (int)       : Detect missing poses at most that max_gap frames back.

    Returns:
    (index_matches, index_duplicates, [stats])
    """
    stat = defaultdict(int)

    # index matching
    last_frame_idxs = set()
    last_frame_objs = {}

    # gap tracking via "wanted list"
    wanted_idx = {}  # { frame: [ idxs ]}

    index_changes = []
    index_duplicates = []

    for frame in sequence:
        # find pose indices for objects in frame
        frame_idxs = set(pose["idx"] for pose in frame["objs"])
        frame_objs = dict((pose["idx"], pose) for pose in frame["objs"])
        image_id = frame["image_id"]
        frame_num = parse_frame_num(image_id)

        # find new and dropped pose indices
        appear_idx = frame_idxs.difference(last_frame_idxs)
        dropped_idx = last_frame_idxs.difference(frame_idxs)

        if dropped_idx:
            stat["drop"] += len(dropped_idx)

        if appear_idx:
            stat["appear"] += len(appear_idx)

        if appear_idx:
            # compute bbox overlaps between dropped and appeared poses
            appear_boxes, appear_ilookup = build_boxes(
                appear_idx, frame_objs)

        if dropped_idx:
            # build last frame obj bbox list
            dropped_boxes, drop_ilookup = build_boxes(
                dropped_idx, last_frame_objs)

            # add to wanted list
            wanted_idx[frame_num] = (
                dropped_idx, dropped_boxes, drop_ilookup)

        if appear_idx:
            # case 1: duplicate
            # build boxes for existing objects
            current_idxs = frame_idxs.difference(appear_idx)
            current_boxes, current_ilookup = build_boxes(
                current_idxs, frame_objs)
            overlaps = intersect(current_boxes, appear_boxes)

            # find duplicate
            if np.any(overlaps >= BBOX_OVERLAP_THRESHOLD):
                for cix, dd in enumerate(overlaps):
                    for aix, ratio in enumerate(dd):
                        idx_appear = appear_ilookup[aix]
                        idx_current = current_ilookup[cix]
                        if ratio > BBOX_OVERLAP_THRESHOLD:
                            if verbose:
                                print(" *** Duplicate %s, %i with %i (overlap: %.3f)" %
                                        (image_id, idx_appear, idx_current, ratio))

                            if frame_objs.get(idx_appear) is None:
                                # already deleted, duplicate with multiple other poses
                                continue

                            # clear duplicate
                            del frame_objs[idx_appear]
                            frame_idxs.remove(idx_appear)
                            appear_idx.remove(idx_appear)
                            stat["duplicate"] += 1
                            index_duplicates.append([image_id, idx_appear])

        if verbose and wanted_idx:
            print("Frame %s -- wanted list: %s" % (image_id, wanted_idx))

        if wanted_idx and appear_idx:
            for gap in range(max_gap):
                wanted_frame = frame_num - gap

                if wanted_frame not in wanted_idx:
                    # no dropped boxes in frame
                    continue

                # case 2: pose index switch from A to B
                _dropped_idx, _dropped_boxes, _drop_ilookup = wanted_idx[wanted_frame]

                stat["match_candidate"] += len(appear_idx)

                # overlaps[point/4][box]
                overlaps = intersect(appear_boxes, _dropped_boxes)
                if not np.any(overlaps >= BBOX_OVERLAP_THRESHOLD):
                    continue

                if verbose:
                    print("Frame %s -- match candidates: %s --> %s" %
                        (image_id, _dropped_idx, appear_idx))

                has_index_changes = False
                for aix, dd in enumerate(overlaps):
                    for dix, ratio in enumerate(dd):
                        i0 = _drop_ilookup[dix]
                        i1 = appear_ilookup[aix]
                        if ratio > BBOX_OVERLAP_THRESHOLD:
                            index_changes.append([image_id, i1, i0])
                            has_index_changes = True
                            stat["index_change"] += 1
                            if verbose:
                                print("  (%s) overlap: %i -> %i : %.2f" %
                                    (image_id, i0, i1, ratio))
                if has_index_changes:
                    break

        # clean "wanted" list by max_gap
        stale_frames = [
            f for f in wanted_idx.keys() if frame_num - f > max_gap
        ]
        for stale_frame in stale_frames:
            del wanted_idx[stale_frame]


        # update indices
        last_frame_idxs = frame_idxs
        last_frame_objs = frame_objs

    if return_stat:
        return index_changes, index_duplicates, stat

    return index_changes, index_duplicates


def parse_frame_num(image_id):
    return int(image_id.split(".")[0])


def within(x0, x1, x):
    return x0 <= x and x < x1


def remap_idx_inplace(sequence, pose_idx_changes, duplicates, virtual_idx_init=10000):
    open_intervals = {}             # orig_idx -> [start_frame, vidx]
    intervals = defaultdict(list)   # orig_idx -> [[f0, f1, orig_id, vidx]]

    # vidx = virtual index (artificial and not used by alphapose)
    virtual_idx_seq = virtual_idx_init

    # iterate over index change events
    for frame, i1, i0 in pose_idx_changes:
        frame_num = parse_frame_num(frame)
        #print("Frame %i, update %i -> %i" % (frame_num, i0, i1))

        if i0 in open_intervals:
            # open interval closes
            start_frame, int_vidx = open_intervals[i0]
            interval = [start_frame, frame_num, i0, int_vidx]
            intervals[i0].append(interval)
            del open_intervals[i0]
            #print("    interval", interval)
            # ...new interval opens
            open_intervals[i1] = [frame_num, int_vidx]

        if i0 not in intervals:
            # first interval closes
            # new vid
            virtual_idx_seq += 1
            interval = [0, frame_num, i0, virtual_idx_seq]
            intervals[i0].append(interval)
            #print("    first interval", interval)
            # ...new interval opens
            open_intervals[i1] = [frame_num, virtual_idx_seq]
            #print("    new virtual_idx (%i) for idx: %i (frame %i)" % (virtual_idx_seq, i0, frame_num))
        #print("    open intervals", open_intervals)

    # close open remaining intervals
    last_frame = len(sequence)
    for orig_id, open_interval in open_intervals.items():
        start_frame, int_vidx = open_interval
        interval = [start_frame, last_frame + 1, orig_id, int_vidx]
        intervals[orig_id].append(interval)
        #print("Close remaining open int.", interval, "last frame:", last_frame)
    # print("Intervals:")
    # pprint.pprint(intervals)

    dupes_by_frame = defaultdict(list)
    for image_id, idx in duplicates:
        dupes_by_frame[image_id].append(idx)

    # remap sequence
    for frame in sequence:
        frame_num = parse_frame_num(frame["image_id"])

        # remove duplicates
        frame_dupe_indices = dupes_by_frame.get(frame["image_id"])
        if frame_dupe_indices:
            frame["objs"] = [obj for obj in frame["objs"]
                             if obj["idx"] not in frame_dupe_indices]

        for obj in frame["objs"]:
            idx = obj["idx"]

            if idx not in intervals:
                continue

            for f0, f1, orig_idx, vidx in intervals[idx]:
                #                print("apply interval (%i) %i -> %i, [%i..%i]" % (frame_num, vidx, orig_idx, f0, f1,))
                # for each idx: belongs to interval  { orig_idx: [frame0, frame1, vidx] }
                if within(f0, f1, frame_num):
                    #                    print("  ... done")
                    #                    print("frame %i : remap idx %i -> %i" % (frame_num, idx, vidx))
                    # remap idx
                    obj["idx"] = vidx


def harmonize_indices(sequence):
    pose_idx_events, duplicates = get_pose_idx_events(sequence)
    remap_idx_inplace(sequence, pose_idx_events, duplicates)
    return sequence, pose_idx_events


if __name__ == "__main__":
    from alphapose_json import load_alphapose_json
    import argparse

    parser = argparse.ArgumentParser("pose_tracker.py")
    parser.add_argument("input_json")
    args = parser.parse_args()

    # load data
    print("Opening %s" % args.input_json)

    sequence = load_alphapose_json(args.input_json)

    changes, duplicates, stat = get_pose_idx_events(sequence, return_stat=True,
                                                    verbose=True)
    pprint.pprint(stat)
    print("Index changes (img, to, from):")
    pprint.pprint(changes)
    print("Duplicates changes (img, idx):")
    pprint.pprint(duplicates)
    remap_idx_inplace(sequence, changes, duplicates)
