import os
import glob
import re

from dltx import dlt_calibrate

# regular expressions
cam_file_re = re.compile("calibration_camera-([A-Za-z]+).txt")


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


def load_calibration(path):
    calib_world, calib_cams = read_calibration_files(path)

    if len(calib_cams) == 0:
        raise Exception("No camera calibration files read: %s" % path)

    markers = calib_world.keys()
#    pprint.pprint(world_arr)

    cam_ids = []
    camera_calibration = {}

    print("Camera calibration error values:")
    for cam_id, cam_values in calib_cams.items():
        use_markers = [x for x in markers if x in cam_values]
        cam_arr = [cam_values[x] for x in use_markers]
        cam_ids.append(cam_id)

        world_arr = [calib_world[x] for x in use_markers]
        n_dims = len(world_arr[0])
        assert n_dims == 3

        # dlt calibration
        L, err = dlt_calibrate(n_dims, world_arr, cam_arr)
        print("  Camera '%s': %.3f" % (cam_id, err))
        camera_calibration[cam_id] = L
    print()

    return camera_calibration, cam_ids

def is_opposite(cam):
    """
    Arguments:

    cam         : camera calibration matrix (3x4)
    """
    pass
