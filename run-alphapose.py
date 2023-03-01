import subprocess
import os
import glob
import sys
import argparse
import time
from pathlib import Path
import json

import shellcolors as sc
import csc.connection
import csc.ssh
import csc.rsync

ALPHAPOSE_PATH = os.environ.get("ALPHAPOSE_PATH")

perf_times = []


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


def write_job_status(inputdir, local_jobid, sbatch_jobid):
    # write sbatch jobid & local jobid to
    data = {
        "local_jobid": local_jobid,
        "sbatch_jobid": sbatch_jobid
    }

    status_fn = f"job-{local_jobid}.json"
    status_path = os.path.join(inputdir, status_fn)
    with open(status_path, "w") as fd:
        json.dump(data, fd, indent=2)
    return status_path


def run_csc(args):
    remote = csc.connection.get_remote_mapping(args.input, None, args.seq)

    print("Local JOBID:", remote.jobid)

    try:
        sc.print_bold("Transfering input videos.")
        csc.rsync.upload(remote.connection,
                         remote,
                         subject=args.subject,
                         trial=args.trial,
                         dry_run=args.dryrun
                         )
        sc.print_ok("Succesfully transferred input files.")

    except subprocess.CalledProcessError as e:
        sc.print_fail("File transfer failure: %s" % e)
        sys.exit(1)

    # run remote job prepare script
    csc.ssh.prepare_alphapose_jobs(remote.connection, remote.jobid)
    sc.print_ok("Succesfully prepared sbatch job.")

    # start sbatch
    sbatch_jobid = csc.ssh.sbatch(remote.connection, remote.jobid)
    sc.print_ok("Succesfully queued sbatch job: %s" % sbatch_jobid)

    # write jobid to file
    status_path = write_job_status(args.input, remote.jobid, sbatch_jobid)
    sc.print_ok("Status written to file: %s" % status_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='File feeder to AlphaPose',
        description='Inputs multiple files to AlphaPose')
    parser.add_argument('input')
    parser.add_argument("--local", action="store_true", default=False,
                        help="Use local installation")
    parser.add_argument('-r', '--reldir',
                        default=os.path.join("..", "Pose"),
                        help="Output directory relative to input file. (local usage only)")
    parser.add_argument('--seq',
                        default=1,
                        type=int,
                        help="Batch sequence number. Default: 1.")
    parser.add_argument('-d', '--dryrun',
                        action="store_true",
                        help="Dry run. No actual data transfers.")
    parser.add_argument("--subject", "-S",
                        default="*",
                        help="Process only subject with id, eg. KE101.")
    parser.add_argument("--trial", "-T",
                        default="*",
                        help="Process only trial with id, eg. 01")

    args = parser.parse_args()

    if args.local:
        run_local(args)
        print()
        print("Processing times", perf_times)
    else:
        run_csc(args)
