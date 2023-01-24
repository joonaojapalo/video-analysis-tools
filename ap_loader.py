import json
import os
from collections import defaultdict
from pylab import plot, show, array, imshow, imread, figure, gca
from matplotlib.animation import FuncAnimation

# alphapose-result.json objects
#  - 'image_id',
#  - 'category_id',
#  - 'keypoints', (N=3x136/frame)
#  - 'score',
#  - 'box', (x, y, w, h)
#  - 'idx'

__all__ = ["load_alphapose_json"]


def load_alphapose_json(fname):
    """
    """
    objs = json.load(open(fname))
    obj_idx = defaultdict(list)
    img_idx = defaultdict(list)
    for i, obj in enumerate(objs):
        iid = obj["image_id"]
        obj_idx[iid].append(obj)
        img_idx[iid].append(i)
    img_ids = sorted(img_idx.keys(), key=lambda x: int(x.split(".")[0]))
    return [{"image_id": iid, "objs": obj_idx[iid]} for iid in img_ids]
