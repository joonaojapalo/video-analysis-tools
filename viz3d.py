import argparse
from pathlib import Path

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

from datasource import AlphaposeDataSource, SIMIDataSource
import progress
import com
from view3d import View


def animate(frame_counter, n_frames, frame_start, fps, frames_per_animframe, view):
    frame = frame_start + frame_counter * 240 // fps
    view.set_data(frame)

def build_data_source(type):
    if (type == "alphapose"):
        return AlphaposeDataSource
    elif type == "simi":
        return SIMIDataSource


def process_dir(input_dir, subject, trial, com_model, output_fps=120, trim_start=0, datasource_name="alphapose"):
    DataSource = build_data_source(datasource_name)
    datasource = DataSource(input_dir, subject, trial)

    if len(datasource.outputs) == 0:
        print("No files to process.")
        return

    print("Files to process:")
    for output in datasource.outputs:
        summary_com = "yes" if output.com_path else "no"
        print(f"  {output.path} (CoM: {summary_com})")
    print()

    for output in datasource.outputs:
        print(f"Creating 3d visualization for: {output.path.name}")
        output_video_name = f"{output.path.stem}-3d-stickfigure.mp4"
        output_video = output.path.parent.joinpath(output_video_name)
        process_npy_file(output.path,
                         output_video,
                         com_model,
                         output.com_path,
                         output_fps=output_fps,
                         frame_start=trim_start)


def process_npy_file(input, output, com_model, input_com=None, frame=None,
                     output_fps=60,
                     frame_start=0):
    # load pose data
    posearr = np.load(input)

    # load CoM data (if exists)
    if input_com:
        comarr = np.load(input_com)
    else:
        comarr = None

    # animate
    process_array(posearr, comarr, output, com_model, frame, output_fps, frame_start)


query_keypoints = """
SELECT *
FROM
    markers
WHERE
    subject_id=? AND trial_id=?
ORDER BY
    relative_frame;
"""


def process_sqlite(input, output, subject, trial, com_model,
                   frame=None,
                   output_fps=60,
                   frame_start=0):
    import sqlite3

    # load pose data from sqlite
    # TODO: ...
    conn = sqlite3.connect(str(input))
    with conn:
        conn.execute(query_keypoints, (subject, trial))
        posearr = np.load(input)

    # CoM data not supported
    comarr = None

    # animate
    process_array(posearr, comarr, output, com_model, frame, output_fps, frame_start)


def process_array(posearr, comarr, output, com_model,
                  frame=None,
                  output_fps=60,
                  frame_start=0):
    if frame:
        frame = int(frame)
        segment_coms = com_model.compute_segment_com(posearr)
        view = View(posearr, comarr, segment_coms)
        view.render(frame)
        plt.show()
    else:
        # animate full video
        n_video_frames = (len(posearr) - frame_start) * output_fps // 240
        frames_per_animframe = 240 / output_fps

        # compute segment coms
        segment_coms = com_model.compute_segment_com(posearr)

        view = View(posearr, comarr, segment_coms)
        view.render(0)

        anim_args = (
            n_video_frames,
            frame_start,
            output_fps,
            frames_per_animframe,
            view
        )

        ani = FuncAnimation(view.fig, animate,
                            frames=n_video_frames,
                            fargs=anim_args,
                            interval=2*1000/output_fps,
                            repeat=True)

        if output:
            print("  - processing video:      ", end="")
            ani.save(output, progress_callback=progress.progress)
            progress.complete()
            print(f"  - wrote {n_video_frames} frames to file: {output}")
            print()
        else:
            plt.show()


def detect_input_type(input_path):
    if input.is_file():
        # sqlite
        import sqlite3
        try:
            conn = sqlite3.connect(str(input_path))
            return "sqlite"
        except:
            pass

        # .npy
        return "numpy"
    elif input.is_dir():
        return "directory"


def build_com_model(com_model_name):
    if com_model_name == 'dempster-alphapose':
        return com.DempsterAlphapose()
    elif com_model_name == 'dempster-kihu':
        return com.DempsterKIHU()
    else:
        raise Exception("Invalid center-of-mass model: " + com_model_name)



usage = """
Visualize single input file and save output video:

  viz3d .\\S1_01-pos.npy -o S1_01-skeleton.mp4

    OR

  viz3d 2023-02-15 -S S1 -T 01
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser("viz3d.py", usage=usage)
    parser.add_argument("input",
                        help="Input file, eg. worldpos.npy")
    parser.add_argument("--com",
                        help="Input CoM file, eg. worldpos-com.npy")
    parser.add_argument("-M", "--com-model",
                        dest="com_model",
                        default="dempster-alphapose",
                        help="CoM model name: dempster-kihu | dempster-alphapose. Default: dempster-alphapose")
    parser.add_argument("-o", "--output",
                        default=None,
                        help="File name for output video.")
    parser.add_argument("-f", "--frame",
                        default=None,
                        help="Frame to inspect. Only for single input file.")
    parser.add_argument("-S", "--subject",
                        default=None,
                        help="Subject identifier. Eg. 'S1' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("-T", "--trial",
                        default=None,
                        help="Trial identifier. Eg. '01' as in 'S1_01_oe/alphapose-results.json'")
    parser.add_argument("--output-fps",
                        type=int,
                        default=60,
                        help="Output video fps. Default: 60")
    parser.add_argument("--trim-start",
                        type=int,
                        dest="trim_start",
                        default=0,
                        help="How many frames to trim from start. Default: 0")
    parser.add_argument("--trim-end",
                        type=int,
                        dest="trim_end",
                        default=None,
                        help="How many frames to trim from end. Default: 0")
    parser.add_argument("--datasource", "-D",
                        default="alphapose",
                        help="Input data source name: alphapose, simi. Default: alphapose")
    args = parser.parse_args()

    input = Path(args.input)

    # input
    input_type = detect_input_type(input)

    # com model
    com_model = build_com_model(args.com_model)

    if input_type == "numpy":
        print(f"Creating 3d visualization for: {args.input}")
        process_npy_file(args.input, args.output,
                         com_model, args.com,
                         frame=args.frame,
                         start_frame=args.trim_start)
    elif input_type == "directory":
        process_dir(args.input,
                    subject=args.subject,
                    trial=args.trial,
                    com_model=com_model,
                    output_fps=args.output_fps,
                    trim_start=args.trim_start,
                    datasource_name=args.datasource)
    elif input_type == "sqlite":
        raise NotImplemented("SQLite")
