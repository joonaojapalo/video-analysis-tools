import os
import glob
import re
import itertools
from numpy import sqrt

from dltx import dlt_calibrate, dlt_reconstruct

# regular expressions
cam_file_re = re.compile("calibration_camera-([A-Za-z]+).txt")

# threshold (in meters) to detect camera pairs producing unstable reconstructions
# compared to reconstruction from all available cameras
THREHOLD_UNSTABILITY = 0.01

def read_calibration_ssv(fd, marker_column="Marker", columns=["X", "Y"]):
    """ssv = space-separated-values
    """
    header = [h.lower() for h in next(fd).split()]
    marker_idx = header.index(marker_column.lower())
    col_idxs = [header.index(c.lower()) for c in columns]

    output = {}
    for line in fd:
        cols = line.split()

        if len(cols) == 0:
            continue

        if len(cols) < max(col_idxs):
            raise ValueError("Corrupted calibration file")

        marker = cols[marker_idx]
        output[marker] = [float(cols[i].replace(",", ".")) for i in col_idxs]
    return output


def read_calibration_files(path):
    points_fns = glob.glob(os.path.join(path, "calibration_camera-*.txt"))
    world_fn = os.path.join(path, "calibration_world.txt")

    if len(points_fns) == 0:
        raise Exception("No camera calibration files found")

    with open(world_fn) as fd:
        world = read_calibration_ssv(fd, "Point", ["X", "Y", "Z"])

    cams = {}
    for fn in points_fns:
        basename = os.path.basename(fn)
        m = cam_file_re.match(basename)

        if m is None:
            raise Exception(
                "Invalid camera calibraton file name: %s" % basename)

        cam_id = m.group(1)
        with open(fn) as fd:
            cams[cam_id] = read_calibration_ssv(fd, "Marker")

    return world, cams


def _detect_unstable_cam_pairs(n_dims, cam_ids, markers, camera_calibration, cam_xy_by_id):
    # list of unstable camera pairs. this data can be used to assess quality of
    # 2-camera reconstructions
    print("[calibration] detecting unstable camera pairs...")

    # marker ground-truth world positions
    pos_gt = {}

    for marker_idx in markers:
        Ls = [camera_calibration[cam_id] for cam_id in cam_ids]
        point = [cam_xy_by_id[cam_id][marker_idx] for cam_id in cam_ids]
        pos = dlt_reconstruct(n_dims, len(Ls), Ls, point)
        pos_gt[marker_idx] = pos

    unstable_cam_pairs = []
    all_cam_pairs = itertools.combinations(cam_ids, 2)

    for cam_id0, cam_id1 in all_cam_pairs:
        Ls = [
            camera_calibration[cam_id0],
            camera_calibration[cam_id1]
        ]

        for marker_idx in markers:
            point = [
                cam_xy_by_id[cam_id0][marker_idx],
                cam_xy_by_id[cam_id1][marker_idx]
            ]

            # reconstruct point
            pos = dlt_reconstruct(n_dims, 2, Ls, point)

            # compute distance to ground-truth position
            v_delta = pos_gt[marker_idx] - pos
            delta = sqrt(v_delta.dot(v_delta))

            if delta > THREHOLD_UNSTABILITY:
                print("[calibration] unstable cam pair: %s..%s (distance from calibration marker ground truth: %.1fmm)" % (cam_id0, cam_id1, delta * 1000))
                unstable_cam_pairs.append(set([cam_id0, cam_id1]))
                break

    return unstable_cam_pairs


def load_calibration(path, return_unstable_cams=False):
    calib_world, calib_cams = read_calibration_files(path)

    if len(calib_cams) == 0:
        raise Exception("No camera calibration files read: %s" % path)

    markers = calib_world.keys()

    cam_ids = []
    cam_xy_by_id = {}
    camera_calibration = {}

    print("Camera calibration error values:")
    for cam_id, cam_xy in calib_cams.items():
        use_markers = [x for x in markers if x in cam_xy]
        cam_arr = [cam_xy[x] for x in use_markers]
        cam_ids.append(cam_id)
        cam_xy_by_id[cam_id] = cam_xy

        world_arr = [calib_world[x] for x in use_markers]
        n_dims = len(world_arr[0])
        assert n_dims == 3

        # dlt calibration
        L, err = dlt_calibrate(n_dims, world_arr, cam_arr)
        print("  Camera '%s': %.3f" % (cam_id, err))
        camera_calibration[cam_id] = L

    unstable_cam_pairs = _detect_unstable_cam_pairs(n_dims, cam_ids, markers, camera_calibration, cam_xy_by_id)
    print()

    if return_unstable_cams:
        return camera_calibration, cam_ids, unstable_cam_pairs
    else:
        return camera_calibration, cam_ids
