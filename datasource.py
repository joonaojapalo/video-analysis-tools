import os
import re
import glob
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np

import shellcolors as sc
from alphapose_json import load_alphapose_json
from pose_tracker import harmonize_indices
from poi_detector import detect_poi
from sequence_tools import select_sequence_idx

from simi.SimiDataFile import SimiDataFile
from keypoints import KEYPOINTS

# regular expressions
ap_dir_re = re.compile("([A-Za-z0-9v]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)")
simi_file_re = re.compile("([A-Za-z0-9v]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)(-[a-zV-]*)?.p")
cam_file_re = re.compile("calibration_camera-([a-z]+).txt")
output_file_re = re.compile("([A-Za-z0-9v]+)_([A-Za-z0-9]+)-(ap|simi).npy")

class DataSourceException (Exception):
    pass


def seq_as_array(poi_sequence):
    N = 78
    return np.array([f["keypoints"] if f else [np.NaN] * N for f in poi_sequence])


def parse_output_paths(subject_dirs, trial, method="ap"):
    outputs = []

    for subject_id, subject_dir in subject_dirs.items():
        output_glob = os.path.join(subject_dir,
                                    "Output",
                                    "%s_*" % (subject_id)
                                    )
        for output_path in glob.glob(output_glob):
            p = Path(output_path)
            match = output_file_re.match(p.name)

            if not match:
                continue

            trial_id = match.groups()[1]
            method_name = match.groups()[2]

            if trial and trial_id != trial:
                continue

            if method and method_name != method:
                continue

            com_path = p.parent.joinpath(f"{p.stem}-com.npy")
            com = None

            if com_path.is_file():
                com = com_path

                outputs.append(Output(subject_id,
                                      trial_id,
                                      output_path,
                                      com))
    return outputs


class CameraSet:
    def __init__(self, subject_id, trial_id, paths_by_cam, subject_dir):
        self.subject_dir = subject_dir
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.paths_by_cam = paths_by_cam


def as_float(simi_float_arr):
    return np.array([v.as_float() for v in simi_float_arr])


class SIMICameraSet (CameraSet):
    # map from Halpe landmark name to SIMI/KIHU file format landmark names
    # @ prefix denotes virtual landmark which is computed based on
    # other landmarks
    HALPE_SIMI_MAP = {
        "Head": "head",
        "LAnkle": "left ankle-bone",
        "LElbow": "left elbow",
        "LHip": "left hip",
        "LKnee": "left knee",
        "LShoulder": "left shoulder",
        "LWrist": "left hand",
#        "Neck": "@neck",
        "RAnkle": "right ankle-bone",
        "RElbow": "right elbow",
        "RHip": "right hip",
        "RKnee": "right knee",
        "RShoulder": "right shoulder",
        "RWrist": "right hand"
    }

    def __init__(self, frame_width, frame_height, subject_id, trial_id, paths_by_cam, subject_dir):
        super().__init__(subject_id, trial_id, paths_by_cam, subject_dir)

        self.virtual_landmarks = {
            "@head": self._processor_head,
            "@neck": self._processor_neck
        }
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def _is_none(self, keypoint_name):
        return any(k is None for k in keypoint_name)
    
    def _are_all_none(self, data, keypoint_names):
        kp_data = map(lambda kp:data[kp], keypoint_names)

        if any(map(lambda kp:self._is_none(kp), kp_data)):
            return True

        return False

    def _processor_neck(self, data, ctx={}):
        if self._are_all_none(data, ("left shoulder", "right shoulder")):
            return None

        v_shoulder_l = as_float(data["left shoulder"])
        v_shoulder_r = as_float(data["right shoulder"])

        return (v_shoulder_l + v_shoulder_r) / 2

    def _processor_head(self, data, ctx={}):
        v_neck = self._processor_neck(data, ctx)

        if v_neck is None:
            return None
        
        if self._is_none(data["head"]):
            print("WARN: head missing. SIMI file row: " + str(ctx.get("rownum")))
            return None

        v_head = np.array(as_float(data["head"]))

        # inverse Dempster
        return v_neck + (v_head - v_neck) / 0.483

    def get_subject_id(self):
        return self.subject_id

    def get_cam_ids(self):
        return list(self.paths_by_cam.keys())

    def _process(self, simi_name, row_data, ctx={}):
        if simi_name is None:
            return None

        if simi_name.startswith("@"):
            # compute virtual landmarks denoted by @ prefix
            return self.virtual_landmarks[simi_name](row_data, ctx)

        if self._is_none(row_data[simi_name]):
            return None

        return as_float(row_data[simi_name])

    def _simi_to_frame_coord(self, vec):
        return [
            vec[0] * self.frame_width,
            vec[1] * self.frame_height,
        ]

    def get_keypoint_array(self, cam_id):
        assert cam_id in self.paths_by_cam
        data = SimiDataFile.from_file(self.paths_by_cam[cam_id])

        arr = []
        print("samples", data.get_samples(), "cam_id", cam_id)
        for num_row in range(data.get_samples()):
            row = []
            ctx = {
                "rownum": num_row + 11
            }

            for halpe_name in KEYPOINTS.keys():
                simi_name = self.HALPE_SIMI_MAP.get(halpe_name)

                vec = self._process(simi_name, data[num_row], ctx)

                if vec is None:
                    row.extend([
                        np.NaN,
                        np.NaN,
                        np.NaN
                    ])
                else:
                    frame_vec = self._simi_to_frame_coord(vec)
                    row.extend([
                        frame_vec[0], # x
                        frame_vec[1], # y
                        1.0,    # visibility
                    ])

            arr.append(row)

        # [[kp0_x, kp0_y, kp0_vis, ....]]
        return np.array(arr)

    def __repr__(self) -> str:
        return "SIMICameraSet('%s', '%s', subject_dis='%s', cams=%s)" % (self.subject_id, self.trial_id, self.subject_dir, self.paths_by_cam)


class AlphaposeCameraSet (CameraSet):
    def __init__(self, subject_id, trial_id, paths_by_cam, subject_dir):
        super().__init__(subject_id, trial_id, paths_by_cam, subject_dir)

    def get_subject_id(self):
        return self.subject_id
    
    def get_cam_ids(self):
        return list(self.paths_by_cam.keys())
    
    def get_keypoint_array(self, cam_id):
        assert cam_id in self.paths_by_cam

        pose_path = self.paths_by_cam[cam_id]
        print(f"  Pose data: {Path(pose_path).parent.name}")
        print(f"    - camera: {cam_id}")
        print(f"    - path: '{pose_path}'")
        sequence = load_alphapose_json(pose_path)

        # pose tracking
        harmonize_indices(sequence)

        # find sequence for person-of-interest
        pois, poi_warnings = detect_poi(sequence, return_warnings=True)

        for warn_msg in poi_warnings:
            print(f"    - WARN: {warn_msg}")

        if len(pois) == 0:
            print("    - FAILED detection: No POI found")
            # remove camera from 3d reconstruction set
            raise DataSourceException("No person-of-interest (POI) found")
        elif len(pois) > 1:
            print("    - FAILED detection: Multiple POI candidates found", pois)
            # remove camera from 3d reconstruction set
            raise DataSourceException("Multiple person-of-interest (POI) candidates found")
        else:
            poi = pois[0]

        # select only data for person of interest
        poi_sequence = select_sequence_idx(sequence, poi)

        # build keypoint array
        keypoint_arr = seq_as_array(poi_sequence)        

        # log data statistics
        n_frames, n_cols = keypoint_arr.shape
        n_keypoints = n_cols // 3
        print(f"    - detected POI={poi}, frames={n_frames}, keypoints={n_keypoints}")

        return keypoint_arr

    def __repr__(self) -> str:
        return "AlphaposeCameraSet('%s', '%s', subject_dis='%s', cams=%s)" % (self.subject_id, self.trial_id, self.subject_dir, self.paths_by_cam)


class Output:
    def __init__(self, subject_id, trial_id, path, com_path=None):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.path = Path(path)
        self.com_path = None
        if com_path:
            self.com_path = Path(com_path)

    def __repr__(self) -> str:
        return "Output('%s', '%s', '%s', com_path=%s)" % (self.subject_id, self.trial_id, self.path, self.com_path)


class DataSource:
    def __init__(self, root_path, subject=None, trial=None):
        self.root_path = root_path
        self.subject = subject
        self.trial = trial
        self.camera_sets = []
        self.outputs = []
        self.parse_root_path(root_path, subject=subject, trial=trial)

    def parse_root_path(self, root_path, subject, trial):
        subjects = os.path.join(root_path, "Subjects", "*")

        subject_dirs = {}

        for subject_dir in glob.glob(subjects):
            subject_id = subject_dir.split(os.path.sep)[-1]
            if subject and subject_id != subject:
                continue
            subject_dirs[subject_id] = subject_dir

        self._parse_pose_paths(subject_dirs, trial)
        self._parse_output_paths(subject_dirs, trial)

    @staticmethod
    def get_output_basename(subject_id, trial_id):
        return f"{subject_id}_{trial_id}"

    @staticmethod
    def get_output_dir(subject_dir):
        return os.path.join(subject_dir, "Output")

    def get_calibration_files(self, dir="Calibration"):
        cal_dir = os.path.join(self.root_path, dir)
        points_fns = glob.glob(os.path.join(
            cal_dir, "calibration_camera-*.txt"))
        world_fn = os.path.join(cal_dir, "calibration_world.txt")

        if len(points_fns) == 0:
            raise Exception("No camera calibration files found")

        pointfile_paths = {}
        for fn in points_fns:
            basename = os.path.basename(fn)
            match = cam_file_re.match(basename)

            if match is None:
                raise Exception(
                    "Invalid camera calibraton file name: %s" % basename)

            cam_id = match.group(1)
            pointfile_paths[cam_id] = fn

        return pointfile_paths, world_fn

    def _parse_output_paths(self, subject_dirs, trial):
        raise NotImplementedError

    def _parse_pose_paths(self, subject_dirs, trial):
        raise NotImplementedError


class SIMIDataSource (DataSource):
    def __init__(self, root_path, subject=None, trial=None, params={"frame_x": 1920, "frame_y": 1080}):
        self.frame_width = params["frame_x"]
        self.frame_height = params["frame_y"]
        super().__init__(root_path, subject, trial)

    def _parse_output_paths(self, subject_dirs, trial):
        outputs = parse_output_paths(subject_dirs, trial, method="simi")
        self.outputs.extend(outputs)

    @staticmethod
    def get_output_basename(subject_id, trial_id):
        return f"{subject_id}_{trial_id}-simi"

    def _parse_pose_paths(self, subject_dirs, trial):
        initdata_pose = {}
        for subject_id, subject_dir in subject_dirs.items():
            initdata_pose[subject_id] = defaultdict(
                lambda: {
                    "subject_dir": None,
                    "cams": {}
                })

            pose_glob = os.path.join(subject_dir,
                                     "SIMI",
                                     "%s_*_*.p" % (subject_id))
            print("[datasource-simi] Searching input files: %s" % (pose_glob,)) # TODO: fix me!
            for pose_fn in glob.glob(pose_glob):
                simifn = pose_fn.split(os.path.sep)[-1]
                m = re.match(simi_file_re, simifn)
                if not m:
                    sc.print_warn(f"[datasource-simi] Invalid SIMI file name: {simifn}")
                    continue
                parts = m.groups()
                file_subject_id, trial_id, cam_id, _ = parts
                assert file_subject_id == subject_id
                print(" SIMI", file_subject_id, trial_id, cam_id)
                if trial and trial_id != trial:
                    continue
                initdata_pose[subject_id][trial_id]["cams"][cam_id] = pose_fn

        # build
        for subject_id, trials in initdata_pose.items():
            for trial_id, data in trials.items():
                camset = SIMICameraSet(
                    self.frame_width,
                    self.frame_height,
                    subject_id,
                    trial_id,
                    data["cams"],
                    subject_dir=subject_dirs[subject_id],
                )
                # populate!
                self.camera_sets.append(camset)


class AlphaposeDataSource (DataSource):
    def __init__(self, root_path, subject=None, trial=None):
        super().__init__(root_path, subject, trial)

    @staticmethod
    def get_output_basename(subject_id, trial_id):
        return f"{subject_id}_{trial_id}-ap"

    def _parse_output_paths(self, subject_dirs, trial):
        outputs = parse_output_paths(subject_dirs, trial, method="ap")
        self.outputs.extend(outputs)

    def _parse_pose_paths(self, subject_dirs, trial):
        initdata_pose = {}
        for subject_id, subject_dir in subject_dirs.items():
            initdata_pose[subject_id] = defaultdict(
                lambda: {
                    "subject_dir": None,
                    "cams": {}
                })

            pose_glob = os.path.join(subject_dir,
                                     "Pose",
                                     "%s_*_*" % (subject_id),
                                     "alphapose-results.json")
            for pose_fn in glob.glob(pose_glob):
                posedir = pose_fn.split(os.path.sep)[-2]
                m = re.match(ap_dir_re, posedir)
                if not m:
                    sc.print_warn(f"Invalid pose directory name: {posedir}")
                    continue
                parts = m.groups()
                dir_subject_id, trial_id, cam_id = parts
                assert dir_subject_id == subject_id
                if trial and trial_id != trial:
                    continue
                initdata_pose[subject_id][trial_id]["cams"][cam_id] = pose_fn

        # build AlphaposeTrials
        for subject_id, trials in initdata_pose.items():
            for trial_id, data in trials.items():
                ap = AlphaposeCameraSet(
                    subject_id,
                    trial_id,
                    data["cams"],
                    subject_dir=subject_dirs[subject_id]
                )
                self.camera_sets.append(ap)
