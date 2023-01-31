# Video 3D reconstruction tools

## Setup

1. Clone this repository.
2. Add `.\bin\` into PATH environment variable.


### Usage

```sh
# sync and trim videos
ffmpeg-sync Subjects\

# run pose estimation
run-alphapose Subjects\**\Sync\*.mp4

# reconstruct all subjects & throws, also compute CoM
recon3d --com .\2023-01-13\

# visualize recostruction into video file (--save)
viz3d --save .\S1_01-pos.npy.\2023-01-13\Subjects\S1\Output\S1_01.npy
```
