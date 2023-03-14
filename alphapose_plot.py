import os
import sys
import glob

from pylab import plot, show, array, imshow, imread, figure, gca
from matplotlib.animation import FuncAnimation
import imageio

from keypoint_tools import box_from_keypoints

from alphapose_json import load_alphapose_json

colors = [
    "red",
    "blue",
    "yellow",
    "green",
    "olive",
    "purple",
    "white",
]


def plot_keypoints(obj, color=None):
    xy = array([obj["keypoints"][k:k+2]
                for k in range(0, len(obj["keypoints"]), 3)])
    plot(xy[:, 0], xy[:, 1], 'o', color=color)


def plot_frame(image, frame, box_idx=None):
    ax = gca()
    ax.clear()
    iid = frame["image_id"]

    imshow(image)
    for obj in frame["objs"]:
        idx = obj["idx"]
        color = colors[idx % len(colors)]
        print("image: %s, idx: %i (%s)" % (iid, idx, color))
        plot_keypoints(obj, color=color)

        # plot bbox
        x, y, w, h = box_from_keypoints(obj["keypoints"])
        print("    x,y = (%i, %i)" % (x,y))
        plot([x-w/2, x+w/2, x+w/2, x-w/2, x-w/2],
                [y-h/2, y-h/2, y+h/2, y+h/2, y-h/2],
                '-',
                color=color, linewidth=3.0 if idx == box_idx else 1)


VIDEO_FRAME_SKIP = 5


def animate(fid):
    print("-" * 24)
    plot_frame(frames[fid * VIDEO_FRAME_SKIP])


def read_ap_dir(path):
    jsonfile = os.path.join(path, "alphapose-results.json")
    videofile = glob.glob(os.path.join(path, "*.mp4"))[0]
    return jsonfile, videofile


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print()
        print("Usage: alphapose_plot PATH_TO_POSE_DIR FRAME_NUMBER [POSE_INDEX]")
        print()
        sys.exit(1)

    path = sys.argv[1]
    frame_idx = int(sys.argv[2])
    box_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # load alphapose data & video
    jsonfile, videofile = read_ap_dir(path)
    frames = load_alphapose_json(jsonfile)
    print("Total frames: %i" % len(frames))

    fig = figure(figsize=(11.1, 8.3), dpi=72)
    print("plot frame: %i" % frame_idx)

    #    im = imread(os.path.join('hippos/alphapose/h2ve/vis/', iid))
    # extract video frame image
    vid = imageio.get_reader(videofile,  'ffmpeg')
    im = vid.get_data(frame_idx)
    plot_frame(im, frames[frame_idx], box_idx)
    show()
    fig.savefig("apload-%s.png" % frame_idx)

#    n_video_frames = divmod(len(frames), VIDEO_FRAME_SKIP)[0]
#    ani = FuncAnimation(fig, animate, frames=n_video_frames,
#                        interval=50, repeat=False)
#    ani.save('javelin-he2ve.mp4')
