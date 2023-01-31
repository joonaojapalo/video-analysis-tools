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


def quickload(subject_id, video_id, idx=None, numpy=False):
    path = os.path.join("2023-01-18", "Subjects", subject_id, "Pose", video_id, "alphapose-results.json")
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
