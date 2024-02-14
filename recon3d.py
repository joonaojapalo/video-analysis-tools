import os
import sys
import pprint
from collections import defaultdict
from pathlib import Path
import itertools

import numpy as np
import scipy.signal
from dltx import dlt_reconstruct
from calibration import load_calibration

import cmdline
import shellcolors as sc
import fps_interpolate
from nanmedianfilt import nanmedianfilt
import progress
import com
from datasource import AlphaposeDataSource, SIMIDataSource, DataSourceException
from outputwriter import CSVOuputWriter, NumpyOutputWriter
from offset_table import apply_offset_table, read_offset_table

DEFAULT_FPS = 50

# execution statistics
stats = defaultdict(int)


def impute_filter():
    pass


def drop_low_scores(keypoint_arr, visibility_threshold):
    # keypoint score filter
    for i in range(2, 78, 3):
        idx_arr = keypoint_arr[:, i] < visibility_threshold
#        print(f"  {KEYPOINTS_INV[(i-2)/3]} ({i}) dropped by low score: {np.sum(idx_arr)}")
        keypoint_arr[idx_arr, i-2:i] = np.NaN
    return keypoint_arr


def filter_data_median(arr, median_filter_window=9, dimensions=2):
    """Filter (x,y) dimensions in array dimension-wise.
    """
    # TODO: kNN imputer (sklearn)
    # TODO: regression imputation... (GPR?)

    for k in range(dimensions):
        for i in range(k, 78, 3):
            mf = nanmedianfilt(arr[:, i], median_filter_window)
            arr[:, i] = mf
            # kf.kalmanfilt1d(keypoint_arr[:, i],
            #                time_step=0.2,
            #                mesurement_noise=2.0)
    return arr


def filter_data_butterworth4(arr, freq, fps, dimensions=3):
    """Butterworth filter array dimension-wise.
    """
    # init butterworth
    b, a = scipy.signal.butter(4, freq, 'lowpass', fs=fps)
    for k in range(dimensions):
        for i in range(k, 78, 3):
            # pick only non-nan values to filter
            finite_idx = np.isfinite(arr[:, i])
            arr[finite_idx, i] = scipy.signal.filtfilt(
                b, a, arr[finite_idx, i])
    return arr


def calib_by_cam_ids(camera_calibration, cam_ids):
    return [camera_calibration[cam_id] for cam_id in cam_ids]


def is_valid(point):
    return not np.isnan(point).any()


def reconstruct_3d(posedata, cam_ids, camera_calibration, n_cams_min=2,
                   use_combinations=True,
                   unstable_cam_pairs=None):
    """Map set of 2D posedata coordinates to 3D world coordinates.

    Arguments:

        posedata (dict)
        cam_ids (list)
        camera_calibration (dict)
        n_cams_min (int)            : Minimum number of cameras that must see point
        unstable_cam_pairs (list)   : List of sets of cam_ids which produce unstable reconstructions
    """
    assert n_cams_min > 1

    if (len(cam_ids) == 0):
        raise ValueError("no cam_ids")

    # check between camera frame numbers
    n_frames_ref = len(posedata[cam_ids[0]])

    world_pos = np.empty([n_frames_ref, 3*26])
    world_pos[:] = np.NaN

    print("Reconstructing:", flush=True)

    for frame in range(n_frames_ref):
        if frame % (n_frames_ref // 50) == 0:
            # print progress indicator
            progress.progress(frame, n_frames_ref)

        # reconstruct keypoints
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

            if n_cams == 2 and set(point_cam_ids) in unstable_cam_pairs:
                stats[f"num_frames_droppoed_as_unstable"] += 1
                continue

            stats[f"enough_cam_frames"] += 1

            # reconstruct all camera pair combinations
            if use_combinations:
                pair_pos = []
                for pair in itertools.combinations(range(n_cams), n_cams_min):
                    Ls = calib_by_cam_ids(camera_calibration,
                                          np.take(point_cam_ids, pair))
                    pair_point = [point[p] for p in pair]

                    # reconstruct point
                    pos = dlt_reconstruct(3, len(Ls), Ls, pair_point)
                    pair_pos.append(pos)

                # median of cam pair reconstructions
                world_pos[frame, kp*3:kp*3+3] = np.median(pair_pos, 0)
            else:
                # compose calibration matrix array
                Ls = calib_by_cam_ids(camera_calibration, point_cam_ids)

                # reconstruct point
                pos = dlt_reconstruct(3, len(Ls), Ls, point)

                # median of cam pair reconstructions
                world_pos[frame, kp*3:kp*3+3] = pos
    progress.complete()
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


def pad_posedata(posedata, sync_file_dir):
    cam_ids = posedata.keys()
    if sync_file_dir:
        # read ffmpeg config from file
        cam_fps = read_ffmpeg_fps_config(sync_file_dir, cam_ids)
    else:
        cam_fps = {}

    durations = dict((cam_id, len(posearr) / cam_fps[cam_id])
                     for cam_id, posearr in posedata.items()
                     )
    target_duration = max(durations.values())
    for cam_id in posedata.keys():
        if durations[cam_id] != target_duration:
            duration = durations[cam_id]
            fps = cam_fps[cam_id]
            pad_frames = round((target_duration - duration) * fps)
            print("Padding camera '%s' duration: %.2f sec to %.2f (%i frames)" % (cam_id,
                                                                                  duration,
                                                                                  target_duration,
                                                                                  pad_frames))
            # create bigger array
            cols = posedata[cam_id].shape[1]
            padded = np.empty((pad_frames, cols))
            padded[:] = np.NaN
            posedata[cam_id] = np.concatenate((posedata[cam_id], padded))


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
        print(f"Interpolating to target fps: {target_fps}:", flush=True)

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


def parse_com_exclude_segments(exclude):
    if not exclude:
        return []

    NAMES = ["hands", "legs", "forearms", "shanks"]
    ex = exclude.split(",")

    for e in ex:
        if e not in NAMES:
            raise ValueError("Invalid CoM exclude segment: %s" % e)
    return ex


usage = """
  python recon3d.py ./2013-01-13

  Define calibration directory:
  python recon3d.py -c ./2013-01-13/Calibration ./2013-01-13
"""


def build_parser():
    import argparse

    parser = argparse.ArgumentParser("reco3d.py", usage=usage)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-Y", "--sync", dest="sync_file_dir",
                        help="Path to directory containing 'ffmpeg-sync.yml' (only for fps interpolation).")
    parser.add_argument("--com", action="store_true")
    parser.add_argument("-S", "--subject",
                        default=None,
                        help="Subject identifier. Eg. 'S1' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-T", "--trial",
                        default=None,
                        help="Trial identifier. Eg. '01' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-c", "--calibration",
                        dest="calib_path",
                        help="Path to directory containing calibration files (calibration_*.txt)")
    parser.add_argument("input_directory",
                        metavar="INPUT",
                        help="Input directory for input data files.")
    parser.add_argument("-D", "--datasource",
                        default="alphapose",
                        help="Datasource: alphapose | simi")
    parser.add_argument("--visibility",
                        type=float,
                        default=0.5,
                        help="Pose detection score visibility threshold 0.0-1.0 (higher is strictier). Default: 0.5")
    parser.add_argument("--median",
                        type=int,
                        default=9,
                        help="Median filter (pre reconstruction) window length. Default: 9")
    parser.add_argument("--median-post",
                        dest="median_post",
                        type=int,
                        default=3,
                        help="Median filter (post reconstruction) window length. Default: 3")
    parser.add_argument("--freq",
                        type=float,
                        default=16,
                        help="Butterworth (4th order) filter frequency (post reconstruction). Default: 16.0")
    parser.add_argument("--min-cams",
                        dest="n_cams_min",
                        type=int,
                        default=2,
                        help="Minimum number of cameras to use for reconstructing each joint position. Default: 2")
    parser.add_argument("--cam-pair",
                        dest="cam_combinations",
                        action="store_true",
                        default=False,
                        help="In reconstruction, use median point indivudual reconstruction from camera pairs that the point is visible.")
    parser.add_argument("--com-exclude",
                        dest="com_exclude",
                        default="",
                        help="Comma-separate list of segments to exclude from CoM computation. Available values: hands, legs, forearms, shanks.")
    parser.add_argument("--output-format",
                        dest="output_format",
                        default="npy",
                        help="Output file format: npy (default) or csv.")
    parser.add_argument("--offset-table",
                        dest="offset_table_path",
                        help="Path to landmark frame offset table CSV file.")
    return parser


def build_datasource(datasource_type, input_directory, subject, trial, args={}):
    if datasource_type == 'alphapose':
        return AlphaposeDataSource(input_directory, subject, trial)
    elif datasource_type == 'simi':
        return SIMIDataSource(input_directory, subject, trial, **args)
    else:
        raise Exception("Invalid datasource: " + datasource_type)


def build_output_writer(writer_type, output_path):
    if writer_type == 'npy':
        return NumpyOutputWriter(output_path)
    elif writer_type == 'csv':
        return CSVOuputWriter(output_path)
    else:
        raise Exception("Invalid output writer: " + writer_type)


def build_com_model_name(datasource):
    if datasource == 'simi':
        return 'dempster-kihu'
    elif datasource == 'alphapose':
        return'dempster-alphapose'
    else:
        raise Exception("Invalid center-of-mass model: " + datasource)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # parse & validate com_exclude
    com_exclude = parse_com_exclude_segments(args.com_exclude)

    # read directory structure
    print("Trial", args.trial)
    datasource = build_datasource(args.datasource,
                                  args.input_directory,
                                  args.subject,
                                  args.trial)

    # build camera calibration path
    default_clib_path = os.path.join(args.input_directory, "Calibration")
    calib_path = args.calib_path if args.calib_path else default_clib_path
    cam_path = os.path.realpath(calib_path)

    print(f"Loading calibration from: {cam_path}")
    camera_calibration, cam_ids, unstable_cam_pairs = load_calibration(cam_path,
                                                                       return_unstable_cams=True)

    if args.verbose:
        print("camera_calibration", camera_calibration)

    posedata = {}

    if not datasource.camera_sets:
        sc.print_fail("No input files found")
        sys.exit(1)

    sync_file_dir = args.sync_file_dir if args.sync_file_dir else args.input_directory

    # load data
    print("Opening pose data files for subjects.")

    outputfiles = []

    for camset in datasource.camera_sets:
        cam_ids = camset.get_cam_ids()

        fps = get_target_fps(cam_ids, sync_file_dir)

        # load pose data
        print("Opening pose data files:")
        invalid_cams = []
        for cam_id in cam_ids:
            try:
                # read keypoint array
                keypoint_arr = camset.get_keypoint_array(cam_id)
            except DataSourceException as e:
                print("WARN: DataSourceException -- ", e)
                invalid_cams.append(cam_id)
                continue
            
            if args.offset_table_path:
                print("[frame-offset]: applying table %s" % (args.offset_table_path,))
                keypoint_arr = apply_offset_table(keypoint_arr, args.offset_table_path)

            keypoint_arr = drop_low_scores(keypoint_arr, args.visibility)

            if args.median > 0:
                print("[pre-median] Filtering...", flush=True)
                keypoint_arr = filter_data_median(keypoint_arr, args.median)

            # add camera posedata
            posedata[cam_id] = keypoint_arr

        # remove invalid camera views
        for invalid_cam_id in invalid_cams:
            print("Dropping cam_id: %s" % (invalid_cam_id,))
            del cam_ids[cam_ids.index(invalid_cam_id)]

        print()

        if not cam_ids:
            sc.print_fail("Inconsistent input data: no camera ids found.")
            sys.exit(1)

        # pad posedata tails to equal length
        pad_posedata(posedata, sync_file_dir)

        # do fps interpolation
        print("posedata keys", posedata.keys())
        print("camids",cam_ids)
        posedata = interpolate_cams(posedata, cam_ids, sync_file_dir)

        if not check_frame_count(posedata, cam_ids):
            sc.print_fail("Frame count check fail.")
            continue

        # make 3D recostructions
        world_pos = reconstruct_3d(posedata,
                                   cam_ids,
                                   camera_calibration,
                                   n_cams_min=args.n_cams_min,
                                   use_combinations=args.cam_combinations,
                                   unstable_cam_pairs=unstable_cam_pairs)

        frames_with_data = stats["enough_cam_frames"]
        if frames_with_data == 0:
            sc.print_fail(f"ERROR: No reconstructed points.")
            continue

        # post median filter
        if args.median_post > 0:
            print("[post-median] Filtering...", flush=True)
            filter_data_median(world_pos, args.median_post, dimensions=3)

        # post Butterworth
        if args.freq > 0:
            try:
                print("[butterworth-lowpass] applying filter (cutoff: %.1f Hz)..." %
                      args.freq,
                      flush=True
                      )
                world_pos = filter_data_butterworth4(
                    world_pos,
                    args.freq,
                    fps,
                    dimensions=3
                )
            except ValueError as err:
                print("[butterworth-lowpass] ERROR:", err)
                print("[butterworth-lowpass] skipping to next camera set...")
                continue
        else:
            print("[butterworth-lowpass] skipped.")

        if args.verbose:
            print()
            print("Statistics:")
            pprint.pprint(stats)
            print()

        # write to disk
        output_dir = datasource.get_output_dir(camset.subject_dir)
        output_filename = datasource.get_output_basename(camset.subject_id,
                                                         camset.trial_id)
        output_path = os.path.join(output_dir, output_filename)
        output_writer = build_output_writer(args.output_format, output_path)
        output_writer.write(world_pos)
        outputfiles.append(output_writer.get_path())

        if args.com:
            print("Computing CoM...", flush=True)
            com_model = build_com_model_name(args.datasource)
            com_trajectory = com.compute(world_pos, com_exclude, com_model)
            output_writer.write_com(com_trajectory)
            outputfiles.append(output_writer.get_com_path())

    for output_path in outputfiles:
        sc.print_ok(f"Coordinates written to: {output_path}")

    print()
