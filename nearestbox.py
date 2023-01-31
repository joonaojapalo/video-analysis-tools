import numpy as np

__all__ = ["nbb", "intersect", "p2box"]

def relu(arr):
    return arr * (arr > 0)

def nbb(points, boxes):
    """Compute distance of points to boxes.

        Returns:
            [[float]]   - distances from points (rows) to boxes (columns)

        Example:

        boxes = [[[10,10], [20,20]], [[50,50], [60,60]]]
        nbb([[15,15], [57,52], [20, 70]], boxes)
    """
    mins = np.array([ [min(b[0][0], b[1][0]), min(b[0][1], b[1][1])] for b in boxes])
    maxs = np.array([ [max(b[0][0], b[1][0]), max(b[0][1], b[1][1])] for b in boxes])
    ds = np.zeros([len(points), len(boxes)])
    for i, (x, y) in enumerate(points):
        min_ds = np.sqrt(relu(mins[:,0] - x) ** 2 + relu(mins[:,1] - y) ** 2)
        max_ds = np.sqrt(relu(x - maxs[:,0]) ** 2 + relu(y - maxs[:,1]) ** 2)
        ds[i,:] = min_ds + max_ds
    return ds

def intersect(boxes1, boxes2):
    """Compute distance of points to boxes.

        Parameters;
        boxes1  : Array of yolo boxes [[x,y,h,w], ...]
        boxes2  : Array of yolo boxes

        Returns:
        (np.array(float))   : intersection rations [0..1] from boxes1 (rows) to
                              boxes2 (columns)

        Example:

        boxes1 = [[15,15,10,10], [55,55,10,10]]
        boxes2 = [[20,15,10,5], [20,70,10,10], [16,16,10,10]]
        interect(boxes1, boxes2)
    """
    xywh1 = np.asarray(boxes1)
    interection_ratios = np.zeros([len(boxes1), len(boxes2)])

    if interection_ratios.size == 0:
        return interection_ratios

    for i,box2 in enumerate(boxes2):
        x2,y2,w2,h2 = box2
        # intersection box widths & heights
        wp = relu(0.5*(xywh1[:,2] + w2) - np.abs(x2 - xywh1[:,0]))
        hp = relu(0.5*(xywh1[:,3] + h2) - np.abs(y2 - xywh1[:,1]))
        # boxes1 area
        A = xywh1[:,2]*xywh1[:,3]
        interection_ratios[:,i] = wp * hp / A
    return interection_ratios


def p2box(points):
    """create a box from array of points
    """
    points = np.asarray(points)
    mns = points.min(0)
    mxs = points.max(0) 
    d = mxs - mns
    c = mns + d/2
    return np.concatenate((c, d))
