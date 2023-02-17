import os
import re
import glob
from collections import defaultdict

import shellcolors as sc

# regular expressions
dir_re = re.compile("([A-Za-z0-9]+)_([A-Za-z0-9]+)_([A-Za-z0-9]+)")


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
    def __init__(self, root_path, subject=None, trial=None):
        self.root_path = root_path
        self.subject = subject
        self.trial = trial
        self.alphapose_camera_sets = []
        self.parse_root_path(root_path, subject=subject, trial=trial)

    def parse_root_path(self, root_path, subject, trial):
        subjects = os.path.join(root_path, "Subjects", "*")

        initdata = {}

        for subject_dir in glob.glob(subjects):
            subject_id = subject_dir.split(os.path.sep)[-1]

            if subject and subject_id != subject:
                continue

            initdata[subject_id] = defaultdict(
                lambda: {"subject_dir": None, "cams": {}})

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
                if trial and trial_id != trial:
                    continue
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
