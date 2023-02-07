import subprocess

import shellcolors as sc
from . import ssh


class RemoteMapping:
    def __init__(self, connection, jobid=None):
        self.connection = connection
        if jobid:
            self.jobid = jobid
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


def upload(connection, remote, dry_run=False):
    """Sync local data to remote directory.
    """
    jobdir_input = remote.get_jobdir_input()

    # create directory
    ssh.run_command(connection, f"mkdir -p {jobdir_input}")
    sc.print_ok(f"Created remote job input directory: {jobdir_input}")

    # transfer command
    rsync_cmd = [
        "rsync",
        "-av", "--progress",
        "--include='/Subjects'",
        "--include='Sync/*.mp4'",
        "--include='*/'",
        "--exclude='*'",
        "--prune-empty-dirs",
        ".",
        f"{connection.user}@{connection.host}:{jobdir_input}"
    ]

    if dry_run:
        rsync_cmd.append("--list-only")

    cwd = str(connection.local_basepath)
    subprocess.run(rsync_cmd, cwd=cwd, check=True)
    return jobdir_input


def download(connectio, remote, dry_run=False):
    pass
