import re
import subprocess

__all__ = ["parse_fps"]

fps_re = re.compile(b"(\\d+)/(\\d+)")

def parse_fps(video_path):
    """Parse video stream FPS from {video_path}."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        "-show_entries",
        "stream=r_frame_rate",
        video_path
    ]

    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as e:
        return None

    if len(output) > 20:
        raise Exception("Too long output from ffmpeg probe query: %s" % " ".join(cmd))

    # strip eol
    output_str = output.strip()
    match = fps_re.match(output_str)
    if not match:
        raise Exception("Invalid ffmpeg fps probe output format: %s" % output_str)

    # evaluate exact fps (format: 50/1)
    [nom, denom] = match.groups()
    fps = int(nom) / int(denom)
    return fps
