import os
import shellcolors

import yaml

FILENAME = "ffmpeg-sync.yml"

def get_config_path(path):
    if os.path.isdir(path):
        return os.path.join(path, FILENAME)
    else:
        return path


def read_conf(path="."):
    fn = get_config_path(path)
    if os.path.isfile(fn):
        with open(fn) as fd:
            return yaml.load(fd, Loader=yaml.Loader)
    else:
        shellcolors.print_warn("Config file doesn't exist: %s" % fn)


def read_xlsx_cols(conf, default_cols=["Throw", "Camera", "Frame"]):
    xlsx_cols = default_cols

    # default confs
    if not conf:
        return xlsx_cols

    # excel columns from config
    excel_conf = conf.get("excel", {})
    for i, xlsx_col in enumerate(["column1", "column2", "farme_column"]):
        col_name = excel_conf.get(xlsx_col)
        if col_name:
            xlsx_cols[i] = col_name

    return xlsx_cols