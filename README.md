# Video 3D reconstruction tools

## Setup

1. Clone this repository.
2. Add `.\bin\` into `PATH` environment variable.
3. Add repository directory into `JAVELIN_TOOLKIT_PATH` environment variable.

### Usage

```sh
# sync and trim videos
ffmpeg-sync 2023-02-15\

# run pose estimation
run-alphapose 2023-02-15\
# (write down Local JOBID, format 2023-02-15_01, 1st output line)

# start remote job
ssh user@host.com
./start_alphapose_batch.sh <Local JOBID> # replace <Local JOBID> from the previous phase
# (write down sbatch job id)
exit

# wait for and download results
csc-jobs download 2023-02-15\job-2023-02-15_01.json

# reconstruct all subjects & throws, also compute CoM
recon3d --com 2023-01-15

# write analysis to video
analyze-com 
```

## Inspection

Check Alphapose job status:
```sh
csc-jobs status path\to\job-<LocalJOBID>.json # replace <LocalJOBID>
```

Visualize 3d reconstructions:
```sh
# visualize recostruction into video file (--save)
viz3d --save .\S1_01-pos.npy.\2023-01-13\Subjects\S1\Output\S1_01.npy
```

## Configuration

See docs for ffmpeg-sync.

`connection.yml`: create from template file `connection.yml.dist` and setup with your  server credentials.
