import os
from pathlib import Path
import json

import yaml

from . import ssh
from . import connection

__all__ = ["read_conf", "Connection"]


class Connection:
    def __init__(self, user, host, local_basedir, remote_basedir):
        self.user = user
        self.host = host
        self.local_basepath = Path(local_basedir)
        self.remote_basedir = remote_basedir

        if not self.local_basepath.is_dir():
            raise ValueError("Local base dir not exist: %s" % local_basedir)

    def get_jobdir(self, local_jobid, subdir=""):
        return os.path.join(self.remote_basedir,
                            local_jobid, subdir).replace(os.path.sep, "/")


class RemoteMapping:
    def __init__(self, connection, local_jobid=None, sbatch_jobid=None):
        self.connection = connection
        self.sbatch_jobid = sbatch_jobid
        if local_jobid:
            self.jobid = local_jobid
        else:
            self.jobid = self._generate_local_job_id(
                self.connection.local_basepath)

    def _generate_local_job_id(self, local_basepath):
        return f"{local_basepath.name}_01"

    def get_jobdir(self):
        return self.connection.get_job_dir(self.jobid)

    def get_jobdir_input(self):
        return self.connection.get_jobdir(self.jobid, "input")

    def get_jobdir_output(self):
        return self.connection.get_jobdir(self.jobid, "output")


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
    return connection.Connection(conf["username"],
                                 conf["host"],
                                 local_basedir,
                                 conf["remote_dir"])


def get_remote_mapping(local_basedir, jobfile=None):
    conn = get_connection(local_basedir)

    if jobfile:
        if not os.path.isfile(jobfile):
            raise ValueError("File expected: %s" % jobfile)
        with open(jobfile) as fd:
            job_handle = json.load(fd)
            local_jobid = job_handle["local_jobid"]
            sbatch_jobid = job_handle["sbatch_jobid"]
        return RemoteMapping(conn, local_jobid, sbatch_jobid)
    else:
        return RemoteMapping(conn)
