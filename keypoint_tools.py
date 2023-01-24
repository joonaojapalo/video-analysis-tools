import numpy as np
from nearestbox import p2box

__all__ = ["box_from_keypoints", "keypoint_diff"]

#def yolobbox2bbox(x, y, w, h):
#    x1, y1 = x-w/2, y-h/2
#    x2, y2 = x+w/2, y+h/2
#    return [[x1, y1], [x2, y2]]
#
#
#def yolobbox2points(x, y, w, h):
#    x1, y1 = x-w/2, y-h/2
#    x2, y2 = x+w/2, y+h/2
#    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
#

def keypoints2arr(keypoints):
    a = np.asarray(keypoints)
    return np.concatenate(([a[0::3]],[a[1::3]]), axis=0).T


def box_from_keypoints(keypoints):
    arr = keypoints2arr(keypoints)
    return p2box(arr)


def keypoint_diff(keypoints1, keypoints2):
    """compute RMSE difference between 2 poses
    """
    return np.sqrt(np.sum((np.asarray(keypoints1) - np.asarray(keypoints2))**2))
