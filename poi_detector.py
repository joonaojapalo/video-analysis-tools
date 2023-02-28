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

POI_TOTAL_MOVEMENT_THRESHOLD = 0.6


def parse_frame_num(frame):
    return int(frame["image_id"].split(".")[0])


def detect_poi(sequence, policy="horizontal",
               return_warnings=False,
               relative_speed_threshold=0.1):
    # { idx: { frames_in_motion, tot_x, tot_y } }
    idx_movement = defaultdict(lambda: defaultdict(float))
    prev_bboxes = {}
    warnings = []

    for i, frame in enumerate(sequence):
        bboxes = {}
        for obj in frame["objs"]:
            idx = obj["idx"]
            bbox = box_from_keypoints(obj["keypoints"])
            if idx in prev_bboxes:
                # bbox movement
                delta = bbox - prev_bboxes[idx]
                dx, dy, dw, dh = delta
                bbox_width_change = abs(dw / prev_bboxes[idx][2])
                bbox_height_change = abs(dh / prev_bboxes[idx][2])

                # scale centroid movement to height (assume vertical axis
                # parallel to gravity). hence, for 180cm person, 10 %
                # change on 240 fps equals to speed 43 m/s (155 km/h)
                # which can be safely assumed to be caused by detection 
                # issue
                ref_height = prev_bboxes[idx][3]
                relative_horizontal_change = abs(dx) / ref_height
                relative_vertical_change = abs(dy) / ref_height

                if bbox_width_change < 0.2 and relative_horizontal_change > relative_speed_threshold:
                    frame_num = parse_frame_num(frame)
                    warn_msg = "Frame %i -- Too quick horizontal bbox movement (%d%%, %dpx) of pose_idx: %i" % (
                        frame_num,
                        100 * relative_horizontal_change,
                        dx,
                        idx
                    )
                    warnings.append(warn_msg)

                if  bbox_height_change < 0.2 and relative_vertical_change > relative_speed_threshold:
                    frame_num = parse_frame_num(frame)
                    warn_msg = "Frame %i -- Too quick vertical bbox movement (%d%%, %.2fpx) of pose_idx: %i" % (
                        frame_num,
                        100 * relative_vertical_change,
                        dy,
                        idx
                    )
                    warnings.append(warn_msg)
                idx_movement[idx]["frames_in_motion"] += 1
                idx_movement[idx]["tot_x"] += dx
                idx_movement[idx]["tot_y"] += dy
            bboxes[idx] = bbox
        prev_bboxes = bboxes

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
        msg = ", ".join(f"{x} ({idx})" for x,
                        idx in zip(poi_metrics, idx_lookup))
        warnings.append(f"Ambiguous person of interest detection: {msg}")
        print("  poi_metrics: mean=%.2f (std. %.2f)" % (mx, stdx))

    # get idxs with most movement
    pois = np.abs(poi_metrics) / \
        np.abs(poi_metrics).sum() > POI_TOTAL_MOVEMENT_THRESHOLD
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
