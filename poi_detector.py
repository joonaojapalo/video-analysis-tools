"""
Person of interest (POI) detector.

POI = the one moving most in horizontal direction
"""

from collections import defaultdict
import pprint

import numpy as np

from ap_loader import load_alphapose_json
from pose_tracker import harmonize_indices
from keypoint_tools import box_from_keypoints
import shellcolors as sc

__all__ = ["detect_poi"]

POI_MOVEMENT_THRESHOLD = 0.6

def parse_frame_num(frame):
    return int(frame["image_id"].split(".")[0])

def detect_poi(sequence, policy="horizontal"):
    # {idx -> { frames_in_motion, tot_x, tot_y, v_x?}}
    idx_movement = defaultdict(lambda: defaultdict(float))
    prev_centroids = {}

    for i, frame in enumerate(sequence):
        centroids = {}
        for obj in frame["objs"]:
            idx = obj["idx"]
            centroid = box_from_keypoints(obj["keypoints"])[:2]
            if idx in prev_centroids:
                dx = centroid[0] - prev_centroids[idx][0]
                dy = centroid[1] - prev_centroids[idx][1]
                if abs(dx) > 50:
                    frame_num = parse_frame_num(frame)
                    sc.print_warn("Too high dx (%.2f): idx=%i (frame %i)" % (dx, idx, frame_num))
                if abs(dy) > 50:
                    frame_num = parse_frame_num(frame)
                    sc.print_warn("Too high dy (%.2f): idx=%i (frame %i)" % (dy, idx, frame_num))
#               print("Frame %i -- move %i: %.2f, %.2f" % (i, idx, dx, dy))
                idx_movement[idx]["frames_in_motion"] += 1
                idx_movement[idx]["tot_x"] += dx
                idx_movement[idx]["tot_y"] += dy
            centroids[idx] = centroid
        prev_centroids = centroids

#    pprint.pprint(idx_movement)

    def tot_x_key(e):
        return -e[1]["tot_x"]

    idx_lookup = np.array(list(idx_movement.keys()), dtype=int)

    # find most horizontal motion (tot_x)
    if policy == "horizontal":
        poi_metrics = np.array([x["tot_x"] for x in idx_movement.values()])
    else:
        raise NotImplementedError(f"policy={policy}")

    mx = np.mean(np.abs(poi_metrics))
    stdx = np.std(np.abs(poi_metrics))

    if 1.5 * stdx < abs(mx):
        msg = ", ".join(f"{x} ({idx})" for x,idx in zip(poi_metrics, idx_lookup))
        sc.print_warn(f"Ambiguous person of interest detection: {msg}")
        print("  poi_metrics: mean=%.2f (std. %.2f)" % (mx, stdx))

    # get idxs with most movement
    pois = np.abs(poi_metrics) / np.abs(poi_metrics).sum() > POI_MOVEMENT_THRESHOLD
    return idx_lookup[pois]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("poi_detector.py")
    parser.add_argument("input_json")
    args = parser.parse_args()

    # load data
    print("Opening %s" % args.input_json)
    sequence = load_alphapose_json(args.input_json)
    # "pose-estimation/S1_01_oe/alphapose-results.json"

    # harmonize pose idx
    harmonize_indices(sequence)
    pois = detect_poi(sequence)

    if len(pois) > 1:
        print("WARNING: multiple POI candidates found", pois)

    print(pois)
