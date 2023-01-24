# Processing toolchain

## Overview

- ffmpeg-sync
- run-alphapose
- recond3d

### Usage

```sh
# sync and trim videos
ffmpeg-sync Subjects\

# run pose estimation
run-alphapose Subjects\**\Sync\*.mp4

python .\recon3d.py .\pipeline\data\hippos-01\ -o S1_01-pos

python .\viz3d.py .\S1_01-pos.npy --save
```
