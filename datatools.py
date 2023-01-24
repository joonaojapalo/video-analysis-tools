import os

import numpy as np

from ap_loader import load_alphapose_json
from pose_tracker import harmonize_indices
from sequence_tools import select_sequence_idx

__all__ = ["load", "quickload"]

def load(path):
    print("Opening %s" % path)
    sequence = load_alphapose_json(path)
    harmonize_indices(sequence)
    return sequence


def quickload(video_id, idx=None, numpy=False):
    path = os.path.join("pose-estimation", video_id, "alphapose-results.json")
    sequence = load(path)

    if idx:
        # pick single person by idx
        poi_sequence = select_sequence_idx(sequence, idx)
        if numpy:
            return np.array([ f["keypoints"] if f else [np.NaN] * 78 for f in poi_sequence])
        else:
            return poi_sequence
    else:
        return sequence
