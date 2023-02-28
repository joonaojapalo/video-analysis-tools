"""
Person of interest (POI) detector.

POI = the one moving most in horizontal direction
"""

from collections import defaultdict

import numpy as np

from ap_loader import load_alphapose_json
from pose_tracker import harmonize_indices
from keypoint_tools import box_from_keypoints

__all__ = ["detect_poi"]

POI_MOVEMENT_THRESHOLD = 0.6

def parse_frame_num(frame):
    return int(frame["image_id"].split(".")[0])

def detect_poi(sequence, policy="horizontal", return_warnings=False):
    # { idx: { frames_in_motion, tot_x, tot_y } }
    idx_movement = defaultdict(lambda: defaultdict(float))
    prev_centroids = {}
    warnings = []

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
                    warn_msg = "Frame %i -- Too high horizontal bbox movement (%.2f px) of pose_idx: %i" % (frame_num, dx, idx)
                    warnings.append(warn_msg)
                if abs(dy) > 50:
                    frame_num = parse_frame_num(frame)
                    warn_msg = "Frame %i -- Too high vertical bbox movement (%.2f px) of pose_idx: %i" % (frame_num, dx, idx)
                    warnings.append(warn_msg)
                idx_movement[idx]["frames_in_motion"] += 1
                idx_movement[idx]["tot_x"] += dx
                idx_movement[idx]["tot_y"] += dy
            centroids[idx] = centroid
        prev_centroids = centroids

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

    if len(poi_metrics) > 1 and 1.5 * stdx < abs(mx):
        msg = ", ".join(f"{x} ({idx})" for x,idx in zip(poi_metrics, idx_lookup))
        warnings.append(f"Ambiguous person of interest detection: {msg}")
        print("  poi_metrics: mean=%.2f (std. %.2f)" % (mx, stdx))

    # get idxs with most movement
    pois = np.abs(poi_metrics) / np.abs(poi_metrics).sum() > POI_MOVEMENT_THRESHOLD
    if return_warnings:
        return idx_lookup[pois], warnings
    else:
        return idx_lookup[pois]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("poi_detector.py")
    parser.add_argument("input_json")
    args = parser.parse_args()

    # load data
    print("Opening %s" % args.input_json)
    sequence = load_alphapose_json(args.input_json)

    # harmonize pose idx
    harmonize_indices(sequence)
    pois, warnings = detect_poi(sequence, return_warnings=True)

    if len(pois) > 1:
        print("  - WARNING: multiple POI candidates found", pois)
    
    for w in warnings:
        print("  - WARNING: %s" % w)

    print(pois)
