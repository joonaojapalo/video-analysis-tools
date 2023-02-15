#!/bin/env bash

if [ -z "$1" ]; then
	echo "  ** Job name missing"
	exit 1
fi

LOCAL_JOBID=$1
echo "Starting $LOCAL_JOBID"
(cd /scratch/project_2006605/alphapose-jobs/$LOCAL_JOBID && sbatch alphapose-job.sh)

