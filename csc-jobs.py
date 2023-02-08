import time
import argparse
from pathlib import Path

import shellcolors as sc
import csc.ssh
import csc.rsync
import csc.connection


def csc_job_status(conn, jobid):
    stata = csc.ssh.sacct(conn, jobid)
    print()
    sc.print_bold("Array job status:")
    for arraid, status in stata:
        if status == "FAILED":
            print(
                f"  {arraid} -> {status}")
        elif status == "COMPLETED":
            print(
                f"  {arraid} -> {status}")
        elif status == "PENDING":
            print(
                f"  {arraid} -> {status}")
        else:
            print(f"  {arraid} -> {status}")
    print()


SBATCH_COMPLETED = "COMPLETED"
SBATCH_FAILED = "FAILED"
SBATCH_RUNNING = "FAILED"
SBATCH_PENDING = "PENDING"


def get_job_status(remote):
    stata = csc.ssh.sacct(remote.connection, remote.sbatch_jobid)

    if all(status == SBATCH_COMPLETED for (arrid, status) in stata):
        return SBATCH_COMPLETED
    elif any(status == SBATCH_FAILED for (arrid, status) in stata):
        return SBATCH_FAILED
    elif all(status == SBATCH_PENDING for (arrid, status) in stata):
        return SBATCH_PENDING
    elif any(status in (SBATCH_RUNNING, SBATCH_PENDING) for (arrid, status) in stata):
        return SBATCH_RUNNING
    else:
        return "UNKNOWN"


class JobException (Exception):
    def __init__(self, message, jobid) -> None:
        super().__init__(message)
        self.jobid = jobid


def watch(remote, wait_for_status=SBATCH_COMPLETED):
    while True:
        status = get_job_status(remote)
        if status == wait_for_status:
            print()
            break
        elif status == SBATCH_FAILED:
            print()
            raise JobException("Job failed", remote.sbatch_jobid)
        print(".", end="")
        time.sleep(5)


def csc_download(remote):
    try:
        print(f"Waiting for job {remote.sbatch_jobid} to complete.")
        watch(remote)
    except JobException as e:
        sc.print_fail(e)
        return

    csc.rsync.download(remote)
    print("Download succesfully completed")


def csc_watch(remote):
    try:
        print(f"Waiting for job {remote.sbatch_jobid} to complete.")
        watch(remote)
        print("Completed.")
    except JobException as e:
        print(f"Job failed")
        return


def process_command(remote, command):
    command = command.lower()
    if command == "status":
        csc_job_status(remote)
    elif command == "watch":
        csc_watch(remote)
    elif command == "download":
        csc_download(remote)
    else:
        sc.print_fail(f"Invalid command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CSC/AlphaPose job manager',
        description='Remote manager for AlphaPose jobs.')
    parser.add_argument('command', help="Commands: status, watch, download")
    parser.add_argument('jobfile')

    args = parser.parse_args()

    jobfile_path = Path(args.jobfile)

    # connect
    remote = csc.connection.get_remote_mapping(
        jobfile_path.parent, jobfile_path)

    # process
    process_command(remote, args.command)
