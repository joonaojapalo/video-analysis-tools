import subprocess
import os
import glob
import sys
import argparse
import time
from pathlib import Path

import shellcolors as sc
import csc_alphapose.ssh as ssh

ALPHAPOSE_PATH = os.environ.get("ALPHAPOSE_PATH")

perf_times = []

parser = argparse.ArgumentParser(
    prog='File feeder to AlphaPose',
    description='Inputs multiple files to AlphaPose')
parser.add_argument('input')
parser.add_argument("--local", action="store_true", default=False,
                    help="Use local installation")
parser.add_argument('-r', '--reldir',
                    default=os.path.join("..", "Pose"),
                    help="Output directory relative to input file. (local usage only)")
parser.add_argument('-d', '--dryrun',
                    action="store_true",
                    help="Dry run. No actual data transfers.")
parser.add_argument('--cancel', type=int,
                    help="Cancel sbatch job. eg. --cancel=112233")


class RemoteMapping:
    def __init__(self, connection, jobid=None):
        self.connection = connection
        if jobid:
            self.jobid = jobid
        else:
            self.jobid = self._generate_local_job_id(
                self.connection.local_basepath)

    def _generate_local_job_id(self, local_basepath):
        return f"{local_basepath.name}_02"

    def get_jobdir(self):
        return self.connection.get_job_dir(self.jobid)

    def get_jobdir_input(self):
        return self.connection.get_jobdir(self.jobid, "input")

    def get_jobdir_output(self):
        return self.connection.get_jobdir(self.jobid, "output")


def transfer_input_to_csc(connection, remote, dry_run=False):
    jobdir_input = remote.get_jobdir_input()

    # create directory
    ssh.run_command(connection, f"mkdir -p {jobdir_input}")
    sc.print_ok(f"Created remote job input directory: {jobdir_input}")

    # transfer command
    rsync_cmd = [
        "rsync",
        "-av", "--progress",
        "--include='/Subjects'",
        "--include='Sync/*.mp4'",
        "--include='*/'",
        "--exclude='*'",
        "--prune-empty-dirs",
        ".",
        f"{connection.user}@{connection.host}:{jobdir_input}"
    ]

    if dry_run:
        rsync_cmd.append("--list-only")

    cwd = str(connection.local_basepath)
    subprocess.run(rsync_cmd, cwd=cwd, check=True)


def run_alphapose_local(input_video, outdir):
    if not ALPHAPOSE_PATH:
        sc.print_fail(
            " ** ALPHAPOSE_PATH env is unset (set to AlphaPose installation base directory")
        sys.exit(1)

    print("ALPAPOSE_PATH path: %s" % ALPHAPOSE_PATH)
    ap_config_path = os.path.join("configs", "halpe_26",
                                  "resnet", "256x192_res50_lr1e-3_1x.yaml")
    ap_model_path = os.path.join("pretrained_models",
                                 "halpe26_fast_res50_256x192.pth")
    cmd = [
        sys.executable,
        os.path.join("scripts", "demo_inference.py"),
        "--cfg", ap_config_path,
        "--checkpoint", ap_model_path,
        "--video", os.path.realpath(input_video),
        "--outdir", os.path.realpath(outdir),
        "--detector", "yolo",
        "--save_video",
        "--pose_track"
    ]

    print(" ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, cwd=ALPHAPOSE_PATH)
    perf_times.append(time.time() - t0)


def split_filename(fn):
    parts = os.path.basename(fn).split(os.path.extsep)
    return os.path.extsep.join(parts[0:-1]), parts[-1] if len(parts) > 1 else None


def get_basedir(fn):
    return os.path.sep.join(fn.split(os.path.sep)[:-1])


def run_local(args):
    sc.print_bold("Pose estimation:")
    for input_file in glob.glob(args.input):
        input_basename, ext = split_filename(input_file)
        input_basedir = get_basedir(input_file)

        if ext != 'mp4':
            sc.print_warn("WARNING: Not a mp4 file: %s" % input_file)

        # output to sibling directory "Pose"
        sibling_dir = os.path.join(input_basedir, args.reldir, input_basename)
        outdir = os.path.realpath(sibling_dir)

        Path(outdir).mkdir(parents=True, exist_ok=True)
        print(f"  {input_file} --> {outdir}")

        # run alphapose
        run_alphapose_local(input_file, outdir)


def run_csc(args):
    remote_basedir = "/scratch/project_2006605/alphapose-jobs/"

    conn = ssh.Connection("ojapjoil", "mahti.csc.fi",
                          args.input, remote_basedir)
    remote = RemoteMapping(conn, "2023-01-30_01")  # TODO: read jobid

    print("Local JOBID:", remote.jobid)

    try:
        sc.print_bold("Transfering input videos.")
        transfer_input_to_csc(conn, remote, dry_run=args.dryrun)
        sc.print_ok("Succesfully transferred input files.")

    except subprocess.CalledProcessError as e:
        sc.print_fail("File trasnsfer failure: %s" % e)
        sys.exit(1)

    # run remote job prepare script
    ssh.prepare_alphapose_jobs(conn, remote.jobid)
    sc.print_ok("Succesfully prepared sbatch job.")

    # start sbatch
    sjobid = ssh.sbatch(conn, remote.jobid)
    sc.print_ok("Succesfully queued sbatch job: %s" % sjobid)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.local:
        run_local(args)
        print()
        print("Processing times", perf_times)
    else:
        run_csc(args)
