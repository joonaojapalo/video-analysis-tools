import subprocess
import os
import glob
import sys
import argparse
from pathlib import Path
import json

import shellcolors as sc
import csc.ssh
import csc.connection


def csc_job_status(conn, jobid):
    stata = csc.ssh.sacct(conn, jobid)
    print()
    sc.print_bold("Array job status:")
    for arraid, status in stata:
        if status == "FAILED":
            print(
                f"  {arraid} -> {sc.ShellColors.FAIL}{status}{sc.ShellColors.ENDC}")
        elif status == "COMPLETED":
            print(
                f"  {arraid} -> {sc.ShellColors.OKGREEN}{status}{sc.ShellColors.ENDC}")
        elif status == "PENDING":
            print(
                f"  {arraid} -> {sc.ShellColors.OKCYAN}{status}{sc.ShellColors.ENDC}")
        else:
            print(f"  {arraid} -> {status}")
    print()



def process_command(conn, command):
    if command == "status":
        csc_job_status(conn, sbatch_jobid)
    else:
        sc.print_fail(f"Invalid command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CSC/AlphaPose job manager',
        description='Remote manager for AlphaPose jobs.')
    parser.add_argument('command')
    parser.add_argument('jobfile')

    args = parser.parse_args()

    localpath = Path(args.jobfile)

    with open(localpath) as jobfile:
        job_handle = json.load(jobfile)
        local_jobid = job_handle["local_jobid"]
        sbatch_jobid = job_handle["sbatch_jobid"]
        print(f"Job: {local_jobid}/{sbatch_jobid}")

    # connect
    conn = csc.connection.get_connection(localpath.parent)

    # process
    process_command(conn, args.command)
