import subprocess

import shellcolors as sc
from . import ssh


def upload(connection, remote, subject="*", trial="*", dry_run=False):
    """Sync local data to remote directory.
    """
    jobdir_input = remote.get_jobdir_input()

    # create directory
    ssh.run_command(connection, f"mkdir -p {jobdir_input}")
    sc.print_ok(f"Created remote job input directory: {jobdir_input}")

    rule = f"Sync/{subject}_{trial}_*.mp4"

    # transfer command
    rsync_cmd = [
        "rsync",
        "-av", "--progress",
        "--include='*/'",
        f"--include='{rule}'",
        "--exclude='*'",
        "--prune-empty-dirs",
        ".",
        f"{connection.user}@{connection.host}:{jobdir_input}"
    ]

    if dry_run:
        rsync_cmd.append("--list-only")

    print(" ".join(rsync_cmd))
    cwd = str(connection.local_basepath)
    subprocess.run(rsync_cmd, cwd=cwd, check=True)
    return jobdir_input


def download(remote, dry_run=False):
    # rsync -av --progress ojapjoil@mahti.csc.fi:/scratch/project_2006605/alphapose-jobs/18.1.2023_01/output/ .
    jobdir_output = remote.get_jobdir_output()

    # transfer command
    rsync_cmd = [
        "rsync",
        "-av", "--progress",
        "--prune-empty-dirs",
        f"{remote.connection.user}@{remote.connection.host}:{jobdir_output}/",
        ".",
    ]

    if dry_run:
        rsync_cmd.append("--list-only")

    cwd = str(remote.connection.local_basepath)
    subprocess.run(rsync_cmd, cwd=cwd, check=True)
    return jobdir_output
