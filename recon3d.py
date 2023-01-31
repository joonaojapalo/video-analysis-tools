import os
import sys
import re
import glob
import pprint
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.signal
from dltx import dlt_reconstruct, dlt_calibrate

import cmdline
import shellcolors as sc
from ap_loader import load_alphapose_json
from pose_tracker import harmonize_indices
from poi_detector import detect_poi
from sequence_tools import select_sequence_idx
import fps_interpolate
from keypoints import KEYPOINTS
import kalmanfilt as kf
from nanmedianfilt import nanmedianfilt

DEFAULT_FPS = 50

# regular expressions
cam_file_re = re.compile("calibration_camera-([a-z]+).txt")
dir_re = re.compile("([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)")

# execution statistics
stats = defaultdict(int)


def read_calibration_ssv(fd, marker_column="Marker", columns=["X", "Y"]):
    header = [h.lower() for h in next(fd).split()]
    marker_idx = header.index(marker_column.lower())
    col_idxs = [header.index(c.lower()) for c in columns]

    output = {}
    for line in fd:
        cols = line.split()

        if len(cols) < max(col_idxs):
            raise ValueError("Corrupted calibration file")

        marker = cols[marker_idx]
        output[marker] = [float(cols[i]) for i in col_idxs]
    return output


class AlphaposeCameraSet:
    def __init__(self, subject_id, trial_id, paths_by_cam, subject_dir):
        self.subject_dir = subject_dir
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.paths_by_cam = paths_by_cam

    def get_cam_ids(self):
        return list(self.paths_by_cam.keys())

    def __repr__(self) -> str:
        return "AlphaposeTrial('%s', '%s', subject_dis='%s', cams=%s)" % (self.subject_id, self.trial_id, self.subject_dir, self.paths_by_cam)


class DataSource:
    def __init__(self, root_path):
        self.root_path = root_path
        self.alphapose_camera_sets = []
        self.parse_root_path(root_path)

    def parse_root_path(self, root_path):
        subjects = os.path.join(root_path, "Subjects", "*")

        initdata = {}

        for subject_dir in glob.glob(subjects):
            subject_id = subject_dir.split(os.path.sep)[-1]
            initdata[subject_id] = defaultdict(
                lambda: {"subject_dir": None, "cams": {}})

            # sync_glob = os.path.join("Sync",
            #                        "%s_%s_*.mp4")
            pose_glob = os.path.join(subject_dir,
                                     "Pose",
                                     "%s_*_*" % (subject_id),
                                     "alphapose-results.json")
            for pose_fn in glob.glob(pose_glob):
                posedir = pose_fn.split(os.path.sep)[-2]
                m = re.match(dir_re, posedir)
                if not m:
                    sc.print_warn(f"Invalid pose directory name: {posedir}")
                    continue
                parts = m.groups()
                assert len(parts) == 3
                dir_subject_id, trial_id, cam_id = parts
                assert dir_subject_id == subject_id
                initdata[subject_id][trial_id]["cams"][cam_id] = pose_fn
                initdata[subject_id][trial_id]["subject_dir"] = subject_dir

        # build AlphaposeTrials
        for subject_id, trials in initdata.items():
            for trial_id, data in trials.items():
                ap = AlphaposeCameraSet(
                    subject_id, trial_id, data["cams"], subject_dir=data["subject_dir"])
                self.alphapose_camera_sets.append(ap)

    def get_calibration_files(self, dir="Calibration"):
        cal_dir = os.path.join(self.root_path, dir)
        points_fns = glob.glob(os.path.join(
            cal_dir, "calibration_camera-*.txt"))
        world_fn = os.path.join(cal_dir, "calibration_world.txt")

        if len(points_fns) == 0:
            raise Exception("No camera calibration files found")

        return points_fns, world_fn


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


def seq_as_array(poi_sequence):
    N = 78
    return np.array([f["keypoints"] if f else [np.NaN] * N for f in poi_sequence])


def impute_filter():
    pass


def drop_low_scores(keypoint_arr, visibility_threshold):
    # keypoint score filter
    for i in range(2, 78, 3):
        idx_arr = keypoint_arr[:, i] < visibility_threshold
#        print(f"  {KEYPOINTS_INV[(i-2)/3]} ({i}) dropped by low score: {np.sum(idx_arr)}")
        keypoint_arr[idx_arr, i-2:i] = np.NaN
    return keypoint_arr


def filter_data(poi_sequence, fps, visibility_threshold=0.5):
    # build keypoint array
    keypoint_arr = seq_as_array(poi_sequence)
    keypoint_arr = drop_low_scores(keypoint_arr, visibility_threshold)

    # TODO: kNN imputer (sklearn)
    # TODO: regression imputation... (GPR?)
    # median filter (each x,y column)

    # init butterworth
    b, a = scipy.signal.butter(4, 16, 'low', fs=fps)
    median_filter_window = 15
    for k in range(2):
        for i in range(k, 78, 3):
            mf = nanmedianfilt(keypoint_arr[:, i], median_filter_window)
#            mf[np.isnan(mf)] = 0
            keypoint_arr[:, i] = mf#scipy.signal.filtfilt(b, a, mf)
            # kf.kalmanfilt1d(keypoint_arr[:, i],
            #                time_step=0.2,
            #                mesurement_noise=2.0)
            #scipy.signal.medfilt(keypoint_arr[:, i], [median_filter_window])

    return keypoint_arr


def calib_by_cam_ids(camera_calibration, cam_ids):
    return [camera_calibration[cam_id] for cam_id in cam_ids]


def is_valid(point):
    return not np.isnan(point).any()


def reconstruct_3d(posedata, cam_ids, camera_calibration, n_cams_min=2):
    """Map set of 2D posedata coordinates to 3D world coordinates.

    Parameters:
    posedata (dict)
    cam_ids (list)
    camera_calibration (dict)
    n_cams_min (int): Minimum number of cameras that must see point
    """
    assert n_cams_min > 1

    if (len(cam_ids) == 0):
        raise ValueError("no cam_ids")

    # check between camera frame numbers
    n_frames_ref = len(posedata[cam_ids[0]])

    world_pos = np.empty([n_frames_ref, 3*26])
    world_pos[:] = np.NaN

    for frame in range(n_frames_ref):
        # reconstruct person
        print(".", end="")
        for kp in range(26):
            # select only cameras that have observation
            point = []
            point_cam_ids = []
            for cam_id in cam_ids:
                campoint = posedata[cam_id][frame, kp*3:kp*3+2]

                if np.isnan(campoint).any():
                    continue

                point.append(campoint)
                point_cam_ids.append(cam_id)

            n_cams = len(point_cam_ids)
            stats[f"keypoint_{n_cams}_cams"] += 1

            if n_cams < n_cams_min:
                continue

            if is_valid(point):
                # compose calibration matrix array
                Ls = calib_by_cam_ids(camera_calibration, point_cam_ids)

                # reconstruct point
                pos = dlt_reconstruct(3, len(Ls), Ls, point)
                world_pos[frame, kp*3:kp*3+3] = pos
    print()
    return world_pos


def check_frame_count(posedata, cam_ids):
    if not cam_ids:
        sc.print_warn("No cam_ids.")
        return False

    n_frames_ref = len(posedata[cam_ids[0]])
    has_mismatch = any(
        [len(posedata[cam_id]) != n_frames_ref for cam_id in cam_ids])

    if not has_mismatch:
        return True

    sc.print_warn(
        "Inequal number of frames in pose data from cameras. Could this be due to fps mismatch?")
    print("Perhaps you would like to use ffmpeg-sync.yml by '--sync' flag.")
    should_continue = cmdline.input_boolean_prompt(
        "Continue anyway? (y/N)")
    return should_continue


def read_ffmpeg_fps_config(sync_file_dir, cam_ids):
    """Parse ffmpeg-sync.yml config file.

    Returns:
    dict {"cam_id": fps, ...}
    """
    import ffmpeg_sync_config

    if not sync_file_dir:
        raise Exception("Sync file directory path missing.")

    conf = ffmpeg_sync_config.read_conf(sync_file_dir)
    if not conf:
        raise Exception("Sync file not found from '%s'" % sync_file_dir)

    cameras = conf.get("cameras", {})
    cam_fps = {}
    for cam_id in cam_ids:
        camconf = cameras.get(cam_id)
        if not camconf:
            raise Exception(
                f"ffmpeg-sync.yml: configuration missing for camera: {cam_id}")
        fps = camconf.get("fps")
        if not camconf:
            raise Exception(
                f"ffmpeg-sync.yml: fps configuration missing for camera: {cam_id}")
        cam_fps[cam_id] = fps
    return cam_fps


def get_target_fps(cam_ids, sync_file_dir):
    if not sync_file_dir:
        return DEFAULT_FPS

    # read ffmpeg config from file
    cam_fps = read_ffmpeg_fps_config(sync_file_dir, cam_ids)
    if cam_fps:
        return max(cam_fps.values())
    else:
        print("No camera FPS configuration. Using default: %i..." %
              DEFAULT_FPS)
        return DEFAULT_FPS


def interpolate_cams(posedata, cam_ids, sync_file_dir=None, verbose=True):
    target_fps = DEFAULT_FPS
    if sync_file_dir:
        # read ffmpeg config from file
        cam_fps = read_ffmpeg_fps_config(sync_file_dir, cam_ids)
        if cam_fps:
            target_fps = max(cam_fps.values())
        else:
            print("No camera FPS configuration. Using default: %i..." %
                  DEFAULT_FPS)
    else:
        cam_fps = {}

    if verbose:
        print(f"Interpolating to target fps: {target_fps}:")

    for cam_id in cam_ids:
        input_fps = cam_fps.get(cam_id, DEFAULT_FPS)

        if input_fps == target_fps:
            continue

        interp = fps_interpolate.interpolate(
            posedata[cam_id], input_fps, target_fps)
        posedata[cam_id] = interp
        if verbose:
            print(f"  Interpolated '{cam_id}' {input_fps} -> {target_fps}")

    if verbose:
        print()

    return posedata


def compute_com(world_pos):
    """Compute CoM from 3D position coordinates using model by Dempster"""
    trunk_weight = 0.678
    lower_extr_weight = 0.161
    rprox_trunk = 0.626
    rprox_lower_extr = 0.447

    # TODO: port to Halpe keypoint
    KEYPOINTS[""]

    # calculate segment CoMs
    #   = head, arms & trunk
    #   = greater trochanter (proximal) to glenohumeral joint (distal)
    # TODO: com_trunk_cam1 = segmentEndsCam1(:,1:2) + rprox_trunk * (segmentEndsCam1(:,3:4) - segmentEndsCam1(:,1:2));

    # LowerExtremity = greater trochanger (proximal) to medial malleolus (distal)
    # TODO: com_lower_extr_cam1 = segmentEndsCam1(:,1:2) + rprox_lower_extr * (segmentEndsCam1(:,5:6) - segmentEndsCam1(:,1:2));

    # Whole body
    ### TODO: com = trunk_weight * com_trunk_cam1 + 2 * (lower_extr_weight * com_lower_extr_cam1);
    return com


usage = """
  python recon3d.py ./2013-01-13

  Define calibration directory:
  python recon3d.py -c ./2013-01-13/Calibration ./2013-01-13
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("poi_detector.py", usage=usage)
    parser.add_argument("-o", "--output", default="worldpos")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-Y", "--sync", dest="sync_file_dir",
                        help="Path to directory containing 'ffmpeg-sync.yml' (only for fps interpolation).")
    parser.add_argument("--com", action="store_true")
    parser.add_argument("-S", "--subject",
                        default="*",
                        help="Subject identifier. Eg. 'S1' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-T", "--trial",
                        default="*",
                        help="Trial identifier. Eg. '01' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-c", "--calibration",
                        dest="calib_path",
                        help="Path to directory containing calibration files (calibration_*.txt)")
    parser.add_argument("input_directory",
                        metavar="INPUT",
                        help="Input directory for input data files.")
    args = parser.parse_args()

    # read directory structure
    datasource = DataSource(args.input_directory)

    # build camera calibration path
    default_clib_path = os.path.join(args.input_directory, "Calibration")
    calib_path = args.calib_path if args.calib_path else default_clib_path
    cam_path = os.path.realpath(calib_path)

    print(f"Loading calibration from: {cam_path}")
    camera_calibration, cam_ids = load_calibration(cam_path)

    if args.verbose:
        print("camera_calibration", camera_calibration)

    posedata = {}

    if not datasource.alphapose_camera_sets:
        sc.print_fail("No alphapose-results.json files found")
        sys.exit(1)

    sync_file_dir = args.sync_file_dir if args.sync_file_dir else args.input_directory

    # load data
    print("Opening pose data files for subjects.")

    for camset in datasource.alphapose_camera_sets:
        cam_ids = camset.get_cam_ids()
        fps = get_target_fps(cam_ids, sync_file_dir)

        # load pose data
        print("Opening pose data files:")
        for cam_id, pose_path in camset.paths_by_cam.items():
            print(f"  File '{pose_path}'")
            print(f"    - camera: {cam_id}")
            sequence = load_alphapose_json(pose_path)

            # pose tracking
            harmonize_indices(sequence)

            # find sequence for person-of-interest
            pois = detect_poi(sequence)

            if len(pois) > 1:
                sc.print_warn("Multiple POI candidates found", pois)
            else:
                poi = pois[0]

            # select only data for person of interest
            poi_sequence = select_sequence_idx(sequence, poi)
            keypoint_arr = filter_data(poi_sequence, fps)
            posedata[cam_id] = keypoint_arr
            n_frames, n_cols = keypoint_arr.shape
            n_keypoints = n_cols // 3
            print(
                f"    - detected POI={poi}, frames={n_frames}, keypoints={n_keypoints}")
        print()

        if not cam_ids:
            sc.print_fail("Inconsistent input data: no camera ids found.")
            sys.exit(1)

        # do fps interpolation
        posedata = interpolate_cams(posedata, cam_ids, sync_file_dir)

        if not check_frame_count(posedata, cam_ids):
            sc.print_fail("Frame count check fail.")
            sys.exit(0)

        # make 3D recostructions
        world_pos = reconstruct_3d(posedata, cam_ids, camera_calibration, fps)

        # write to disk
        output_dir = os.path.join(camset.subject_dir, "Output")
        output_filename = f"{camset.subject_id}_{camset.trial_id}"
        output_path = os.path.join(output_dir, output_filename)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if os.path.isfile(output_path):
            print(f"File exists, overwriting: {output_path}...")
        np.save(output_path, world_pos)

        if args.verbose:
            print()
            print("Statistics:")
            pprint.pprint(stats)
            print()

        sc.print_ok(
            f"World position coordinates written to: {output_path}.npy ({world_pos.shape[0]})")

        if args.com:
            com = compute_com(world_pos)
            fncom = f"{args.output}-com"
            # np.save(fncom)
            #sc.print_ok(f"CoM coordinates written to: {fncom}.npy ({com.shape[0]})")

    print()
