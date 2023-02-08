import os

import yaml

import shellcolors
from . import ssh

__all__ = ["read_conf"]


def get_toolkit_path():
    # connection conf path
    path = os.environ.get("JAVELIN_TOOLKIT_PATH")

    if not path:
        path = os.path.abspath(".")

    return os.path.join(path, "connection.yml")


def read_conf():
    fn = get_toolkit_path()

    if os.path.isfile(fn):
        with open(fn) as fd:
            conf = yaml.load(fd, Loader=yaml.Loader)
            if not conf["connection"]:
                raise Exception("field 'connection' missing: %s" % fn)
            return conf
    else:
        raise Exception("Config file doesn't exist: %s" % fn)


def get_connection(local_basedir):
    conf = read_conf()["connection"]
    conn = ssh.Connection(conf["username"], conf["host"],
                          local_basedir, conf["remote_dir"])
    return conn
