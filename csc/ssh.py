import subprocess
import re

# remote shell output pattern
SUCCESS_MESSAGE_JOBFILE = b"ALPHAPOSE JOB FILE CREATION: OK"


def run_command(connection, command):
    cmd = [
        "ssh",
        f"{connection.user}@{connection.host}",
        command
    ]

    return subprocess.check_output(cmd)


def prepare_alphapose_jobs(connection, jobid):
    prepare_cmd = [
        f"cd {connection.remote_basedir};",
        f"mkdir -p {jobid}/log_job;",
        f"mkdir -p {jobid}/log_error;",
        f"python3 prepare_alphapose_jobs.py -J {jobid} {jobid}/input/"
    ]

    cmd = " ".join(prepare_cmd)
    print("Running remote command:", cmd)
    output = run_command(connection, cmd)

    if output.find(SUCCESS_MESSAGE_JOBFILE) < 0:
        raise Exception("Alphapose job file prepare failed.")


def sbatch(connection, jobid, verbose=True):
    """Submit Slurm job.

    Returns:
    int: remote job id
    """
    prepare_cmd = [
        f"cd {connection.remote_basedir}/{jobid};",
        f"sbatch alphapose-job.sh"
    ]

    cmd = " ".join(prepare_cmd)

    if verbose:
        print("Running remote command:", cmd)

    output = run_command(connection, cmd)
    match = re.search(b"Submitted batch job (\d+)", output)

    if not match:
        raise Exception("Failed to submit a job.")

    return int(match.groups()[0])


def sacct(connection, jobid, verbose=False):
    prepare_cmd = [
        f"sacct -j {jobid}"
    ]

    cmd = " ".join(prepare_cmd)
    if verbose:
        print("Running remote command:", cmd)
    output = run_command(connection, cmd)

    if output.find(b"JobID") < 0 or output.find(b"State") < 0:
        raise Exception("Invalid output format.")

    lines = [line.split() for line in output.split(b"\n")]

    # parse header
    idx_jobid = lines[0].index(b"JobID")
    idx_status = lines[0].index(b"State")
    stata = []
    for i, ln in enumerate(lines[2:]):
        if len(ln) != 7:
            continue
        stata.append((ln[idx_jobid].decode(), ln[idx_status].decode()))

    return stata


def scancel(connection):
    raise NotImplementedError("scancel")
