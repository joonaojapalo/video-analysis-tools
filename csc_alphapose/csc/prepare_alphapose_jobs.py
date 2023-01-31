#/bin/env python3
"""
Script to run mp4:s through Yolo7 StrogSORT pipeline
Written by Timo Rantalainen 2022 tjrantal at gmail.command
GPL 3.0 or newer licence applies
"""

import os  # File listing
from pathlib import Path
import argparse


def transform_sync_to_pose(input_path):
    """
    Arguments:
    input_path (pathlib.Path):  path to transform

    Returns:
    pathlib.Path
    """
    replace = {
        "Sync": "Pose"
    }
    replaced_parts = (replace.get(part, part) for part in input_path.parts)
    return Path(*replaced_parts)


def get_alphapose_commands(input_basedir, output_basedir, alphapose_dir, path_transformers=[]):
    patterns = [
        "*.mp4",
        "*.avi",
    ]

    input_basepath = Path(input_basedir)
    output_basepath = Path(output_basedir)
    alphapose_path = Path(alphapose_dir).absolute()

    commands = []
    for pattern in patterns:
        for path in input_basepath.rglob(pattern):
            cmd = build_alphapose_command(
                Path(path),
                output_basepath,
                input_basepath,
                alphapose_path,
                path_transformers=path_transformers
            )
            commands.append(cmd)

    return commands


def build_alphapose_command(path, output_basepath, input_basepath, alphapose_path, path_transformers=[]):
    # Create execution command
    target_path = path.parent.relative_to(input_basepath)

    # run path transformers
    for transform in path_transformers:
        target_path = transform(target_path)

    name_out = path.name[:-len(path.suffix)]
    out_path = output_basepath.joinpath(target_path, name_out)
    target_file = out_path.joinpath("alphapose-results.json")

    # alphapose
    alphapose_prog = alphapose_path.joinpath("scripts", "demo_inference.py")
    alphapose_config = alphapose_path.joinpath("configs", "halpe_26", "resnet",
                                                "256x192_res50_lr1e-3_1x.yaml")
    alphapose_model = alphapose_path.joinpath("pretrained_models",
                                               "halpe26_fast_res50_256x192.pth")

    if not os.path.isfile(target_file):
        cmd = [
            "python3", str(alphapose_prog),
            "--cfg", str(alphapose_config),
            "--checkpoint", str(alphapose_model),
            "--gpus", "2",
            "--pose_track",
            "--video", str(path.absolute()),
            "--outdir", str(out_path.absolute())
        ]
        # Create the output folder structure beforehand here
        os.makedirs(out_path, exist_ok=True)
        return " ".join(cmd) + '\n'


usage = """
  python3 S01_alphapose.py /scratch/project_2001930/PRE_GoPro_data/ -o /scratch/project_2006605/b2r_pilot/alphapose/hienoo/
"""

parser = argparse.ArgumentParser(description="Job file generator.")
parser.add_argument("input_dir")
parser.add_argument("-o", "--outdir",
                    required=True,
                    help="Output root directory. eg. '/scratch/project_2006605/b2r_pilot/alphapose/hienoo/'")
parser.add_argument("-f", "--outputfile",
                    default="alphapose-jobs.txt",
                    help="Output file name.")
parser.add_argument("-J", "--jobid",
                    default="default",
                    help="Job local id (eg. 2023-02-16_001).")

if __name__ == "__main__":
    # parse command line args
    args = parser.parse_args()

    # Process input directory tree
    alphapose_dir = "/projappl/project_2006605/AlphaPose/"
    indir = args.input_dir
    outdir = args.outdir
    commands = get_alphapose_commands(indir, outdir, alphapose_dir,
                                      path_transformers=[transform_sync_to_pose])

    # This is where the commands get saved to
    with open(args.outputfile, "w") as commandFile:
        commandFile.writelines(commands)

    with open("alphapose-job.sh.template") as template_fd:
        with open(os.path.join(args.jobid, "alphapose-job.sh"), "w") as output_fd:
            for line in template_fd:
                output_fd.write(line.replace("{{JYU_JOBFILE}}", args.outfile))

    num_lines = sum(1 for line in open(args.outputfile))
    print(f"Number of command written: {num_lines}\n")
    print("ALPHAPOSE JOB FILE CREATION: OK")
