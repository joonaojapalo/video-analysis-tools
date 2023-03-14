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


def image_id(image_num):
    return f"{image_num}.jpg"


def empty_frame(image_num):
    return [{"image_id": image_id(image_num), "objs": []}]


def parse_image_num(image_id):
    return int(image_id.split(".")[0])


def pad_start(iids):
    image_lookup = dict((parse_image_num(iid), iid) for iid in iids)
    image_num_max = max(image_lookup.keys())
    pad = []
    for n in range(0, image_num_max + 1):
        if n not in image_lookup:
            pad.append(image_id(n))
        else:
            pad.append(image_lookup[n])
    return pad


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
    img_ids = sorted(img_idx.keys(), key=parse_image_num)
    img_ids = pad_start(img_ids)

    return [{"image_id": iid, "objs": obj_idx.get(iid, [])} for iid in img_ids]
