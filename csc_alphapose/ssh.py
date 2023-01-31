import subprocess
import os
from pathlib import Path


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


def ssh_command(connection, command):
    cmd = [
        "ssh",
        f"{connection.user}@{connection.host}",
        command
    ]

    return subprocess.check_output(cmd)


SUCCESS_MESSAGE_JOBFILE = b"ALPHAPOSE JOB FILE CREATION: OK"


def ssh_prepare_alpohapose_jobs(connection, jobid):
    prepare_cmd = [
        f"cd {connection.remote_basedir};"
        f"python3 prepare_alphapose_jobs.py -J {jobid} -o {jobid}/output -f {jobid}/alphapose-jobs.txt {jobid}/input/"
    ]

    cmd = " ".join(prepare_cmd)
    print("Running remote command:", cmd)
    output = ssh_command(connection, cmd)

    if output.find(SUCCESS_MESSAGE_JOBFILE) < 0:
        raise Exception("Alphapose job file prepare failed.")


def ssh_sbatch(connection, jobid):
    prepare_cmd = [
        f"cd {connection.remote_basedir};"
        f"sbatch {jobid}/alphapose-job.sh"
    ]

    cmd = " ".join(prepare_cmd)
    print("Running remote command:", cmd)
    output = ssh_command(connection, cmd)

    if output.find(SUCCESS_MESSAGE_JOBFILE) < 0:
        raise Exception("Alphapose job file prepare failed.")
