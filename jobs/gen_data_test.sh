#!/bin/bash

# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

#SBATCH --job-name=IRIS_gen
#SBATCH --output=gen_data_test.out
#SBATCH --partition=general-gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100"

LOCAL_DATA_DIR="/dev/shm/job_data_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DATA_DIR"
PID=0
function winddown
{
    echo "Winding down SLURM job..."
    if [ $PID -ne 0 ]; then
        echo "Killing MPI process $PID..."
        kill -TERM "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
    fi

    echo "Removing local data at $LOCAL_DATA_DIR..."
    rm -rf "$LOCAL_DATA_DIR"
    echo "Wind-down complete."
}
trap winddown EXIT SIGTERM SIGINT

module purge
module load openmpi/5.0.5-noucx
source ~/IRIS/iris_venv/bin/activate
mpirun python gen_data_test.py "$LOCAL_DATA_DIR" &

PID=$!
wait $PID