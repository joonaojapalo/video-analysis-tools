import os
import re
import glob
from collections import defaultdict
from pathlib import Path

import shellcolors as sc

# regular expressions
dir_re = re.compile("([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)")
cam_file_re = re.compile("calibration_camera-([a-z]+).txt")
output_file_re = re.compile("([A-Za-z0-9]+)_([A-Za-z0-9]+).npy")


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
        self.alphapose_camera_sets = []
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

    def _parse_output_paths(self, subject_dirs, trial):
        outputs = defaultdict(dict)  # {subject_id: {trial_id: path}}
        outputs_com = defaultdict(dict)

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

                if trial and trial_id != trial:
                    continue

                com_path = p.parent.joinpath(f"{p.stem}-com.npy")
                com = None
                if com_path.is_file():
                    com = com_path
                self.outputs.append(Output(subject_id,
                                           trial_id,
                                           output_path,
                                           com))

    @staticmethod
    def get_output_basename(subject_id, trial_id):
        return f"{subject_id}_{trial_id}"

    @staticmethod
    def get_output_dir(subject_dir):
        return os.path.join(subject_dir, "Output")

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
                m = re.match(dir_re, posedir)
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
                self.alphapose_camera_sets.append(ap)

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
