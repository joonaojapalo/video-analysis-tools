# /bin/env python3
"""
Script to run mp4:s through Yolo7 StrogSORT pipeline
Written by Timo Rantalainen 2022 tjrantal at gmail.command
GPL 3.0 or newer licence applies
"""

import os  # File listing
import sys
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


def get_alphapose_commands(input_basepath, output_basepath, alphapose_path, path_transformers=[]):
    patterns = [
        "*.mp4",
        "*.avi",
    ]

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
            if cmd:
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
    alphapose_prog = os.path.join("scripts", "demo_inference.py")
    alphapose_config = alphapose_path.absolute().joinpath("configs", "halpe_26", "resnet",
                                                          "256x192_res50_lr1e-3_1x.yaml")
    alphapose_model = alphapose_path.absolute().joinpath("pretrained_models",
                                                         "halpe26_fast_res50_256x192.pth")

    if not os.path.isfile(target_file):
        cmd = [
            "(",
            f"cd {alphapose_path.absolute()};",
            "pwd;"
            "python3", str(alphapose_prog),
            "--cfg", str(alphapose_config),
            "--checkpoint", str(alphapose_model),
            "--gpus", "0",
            "--pose_track",
            "--video", str(path.absolute()),
            "--outdir", str(out_path.absolute()),
            ")"
        ]
        # Create the output folder structure beforehand here
        os.makedirs(out_path, exist_ok=True)
        return " ".join(cmd) + '\n'
    else:
        print("Target file alerady exists", target_file)


usage = """
  python3 S01_alphapose.py /scratch/project_2001930/PRE_GoPro_data/ -o /scratch/project_2006605/b2r_pilot/alphapose/hienoo/
"""

parser = argparse.ArgumentParser(description="Job file generator.")
parser.add_argument("input_dir")
parser.add_argument("-o", "--outdir",
                    default="output",
                    help="Output root directory. eg. '/scratch/project_2006605/b2r_pilot/alphapose/hienoo/'")
parser.add_argument("-f", "--outputfile",
                    default="alphapose-jobs.txt",
                    help="Output file name.")
parser.add_argument("-J", "--jobid",
                    required=True,
                    help="Job local id (eg. 2023-02-16_001).")
parser.add_argument("-D", "--jobdir",
                    default=".",
                    help="Job sandbox directory")


def get_alphapose_path():
    DEFAULT = "/projappl/project_2006605/AlphaPose/"
    apdir = os.environ.get("ALPHAPOSE_PATH", DEFAULT)
    return Path(apdir)
  

if __name__ == "__main__":
    # parse command line args
    args = parser.parse_args()

    sandbox = Path(args.jobdir, args.jobid)
    input_path = Path(args.input_dir)

    if not sandbox.is_dir():
        raise Exception("Invalid job directory: %s" % str(sandbox))

    if not input_path.is_dir():
        raise Exception("Invalid video input directory: %s" % str(input_path))

    print("Reading input from", input_path)
    alphapose_path = get_alphapose_path()
    output_path = sandbox.joinpath(args.outdir)

    # Process input directory tree
    commands = get_alphapose_commands(input_path, output_path, alphapose_path,
                                      path_transformers=[transform_sync_to_pose])

    # This is where the commands get saved to
    outputfile = sandbox.joinpath(args.outputfile)
    with open(outputfile, "w") as commandFile:
        commandFile.writelines(commands)

    # create batch file from template
    NUM_ARRAY_ITEMS = 32
    n_commands = len(commands)

    n_arr_items = min(n_commands, NUM_ARRAY_ITEMS)

    if n_arr_items == 0:
        print("No commands to run.")
        print("ALPHAPOSE JOB FILE CREATION: NO COMMANDS")
        sys.exit(0)
    else:
        with open("alphapose-job.sh.template") as template_fd:
            with open(sandbox.joinpath("alphapose-job.sh"), "w") as output_fd:

                jobs_per_item = n_commands // n_arr_items
                for line in template_fd:
                    line= line.replace("{{JOBFILE}}", args.outputfile)
                    line = line.replace("{{ARRAY_ITEMS}}", str(n_arr_items))
                    line = line.replace("{{JOBS_PER_ITEM}}", str(jobs_per_item))
                    output_fd.write(line)

    num_lines = sum(1 for line in open(outputfile))
    print(f"{num_lines} commands written to {outputfile}")
    print("ALPHAPOSE JOB FILE CREATION: OK")
