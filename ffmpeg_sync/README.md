# `ffmpeg-sync` tool

Synchronize and trim bunch of consistently named videos.

## Usage

Simple run:
```sh
python .\ffmpeg-sync.py .\athletes
```

Define configuration to use:
```sh
python .\ffmpeg-sync.py -c ..\my-conf.yml .\athletes
```

Define output directory (default: output):
```sh
python .\ffmpeg-sync.py -o .\processed .\athletes
```

Example directory contents of `./athletes/athletel/:

```
athlete1_indices.xlsx
athlete1_trial1_cam1.mp4
athlete1_trial1_cam2.mp4
athlete1_trial2_cam1.mp4
athlete1_trial2_cam2.mp4
```

## Configuration

A [YAML](https://yaml.org/) file `ffmpeg-sync.yml` defines configuration.
Configuration includes camera ids and columns for Excel definition files.
Please see example configuration file [ffmpeg-sync.yml](./examples/ffmpeg-sync.yml).

## *_indices.xlsx

Trim and sync definitions are defined in an Excel file (*_indices.xlsx). It must be located in each video-directory beside video files. Headers that are defined in `ffmpeg-sync.yml` configuration file named must exists in `_indices.xlsx` file. Default header column names
are: "Throw", "Camera" and "Frame".

Input video names is constructed as `{subject}_{column1}_{column2}.mp4`.
