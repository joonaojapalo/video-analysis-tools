import subprocess
import os
import glob
import sys
import argparse
import time
from pathlib import Path

ALPHAPOSE_PATH = os.environ.get("ALPHAPOSE_PATH")

perf_times = []

parser = argparse.ArgumentParser(
    prog='File feeder to AlphaPose',
    description='Inputs multiple files to AlphaPose')
parser.add_argument('input_files')
parser.add_argument('-r', '--reldir',
                    default=os.path.join("..", "Pose"),
                    help="Output directory relative to input file.")


def run_alphapose(input_video, outdir):
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
    parts = os.path.basename(input_file).split(os.path.extsep)
    return os.path.extsep.join(parts[0:-1]), parts[-1] if len(parts) > 1 else None


def get_basedir(fn):
    return os.path.sep.join(fn.split(os.path.sep)[:-1])


if __name__ == "__main__":
    args = parser.parse_args()

    if not ALPHAPOSE_PATH:
        print(
            " ** ALPHAPOSE_PATH env is unset (set to AlphaPose installation base directory")
        sys.exit(1)

    print("Pose estimation:")
    for input_file in glob.glob(args.input_files):
        input_basename, ext = split_filename(input_file)
        input_basedir = get_basedir(input_file)

        if ext != 'mp4':
            print("WARNING: Not a mp4 file: %s" % input_file)

        # output to sibling directory "Pose"
        sibling_dir = os.path.join(input_basedir, args.reldir, input_basename)
        outdir = os.path.realpath(sibling_dir)
        print(outdir)
        sys.exit(0)

        Path(outdir).mkdir(parents=True, exist_ok=True)
        print(f"  {input_file} --> {outdir}")

        # run alphapose
        run_alphapose(input_file, outdir)

    print()
    print("Processing times", perf_times)
