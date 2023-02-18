# Video 3D reconstruction tools

## Setup

1. Clone this repository.
2. Install Python libraries denoted by `requirements.txt` into environment of choise.
3. Add `.\bin\` into `PATH` environment variable.
4. Add repository directory into `JAVELIN_TOOLKIT_PATH` environment variable.

## Server setup

Pose estimation is to be done on computation-focused server infrastructure.

1. Install [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/INSTALL.md).
2. Set `ALPHAPOSE_PATH` environment variable to point to AlphaPose installation.
3. Create a *sandbox directory* for alphapose jobs with proper permissions.
4. Copy files (scp) from `csc/remote_tools/` into the created *sandbox directory*.

### Usage

```sh
# sync and trim videos
ffmpeg-sync 2023-02-15\

# run pose estimation
run-alphapose 2023-02-15\
# (write down Local JOBID, format 2023-02-15_01, 1st output line)

# start remote job
ssh user@host.com
./start_alphapose_batch.sh 2023-02-15_01 # replace "2023-02-15_01" with the Local JOBID output by run-alphapose
# (write down sbatch job id)
exit

# write sbatch job id from remote shell output into the local
# job file eg. 2023-02-15\job-2023-02-15_01.json which was created by
# run-alphapose tool

# wait for and download results
csc-jobs download 2023-02-15\job-2023-02-15_01.json

# reconstruct all subjects & throws, also compute CoM
recon3d --com 2023-01-15

# write analysis to video
analyze-com 2023-01-15
```

## Inspection

Check Alphapose job status:
```sh
csc-jobs status path\to\job-<LocalJOBID>.json # replace <LocalJOBID>
```

Visualize AlphaPose results:
```sh
viz_alphapose .\Sync\S1_08_ve-sync.mp4 .\Pose\S1_08_ve-sync\alphapose-results.json
```

Visualize 3d reconstructions:
```sh
# visualize skeleton and CoM
viz3d .\2023-01-13\Subjects\S1\Output\S1_01.npy --com .\2023-01-13\Subjects\S1\Output\S1_01-com.npy

# save all reconstructions in directory tree to video files
viz3d .\2023-01-13\

# save single subject reconstructions in directory tree to video files
viz3d .\2023-01-13\ -S S01
```

## Reconstruction parameters

```sh
# median filter window length (for pose data)
recon3d --com 2023-01-15 --median 16

# LP filter cut-off frequency
recon3d --com 2023-01-15 --freq 50

# minimum number of required cameras for 3D reconstruction
recon3d --com 2023-01-15 --min-cams 3

# exclude body segments from CoM computation
recon3d --com 2023-01-15 --com-exclude hands
recon3d --com 2023-01-15 --com-exclude legs
recon3d --com 2023-01-15 --com-exclude forearms
recon3d --com 2023-01-15 --com-exclude forearms
```


## Configuration

See docs for ffmpeg-sync.

`connection.yml`: create from template file `connection.yml.dist` and setup with your  server credentials.
