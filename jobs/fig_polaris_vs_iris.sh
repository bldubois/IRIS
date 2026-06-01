#!/bin/bash

# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

#SBATCH --job-name=IRIS_fig
#SBATCH --output=fig_polaris_vs_iris.out
#SBATCH --partition=general-gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

module purge
module load gcc
module load openmpi/5.0.5-noucx

export IRIS_MPI_RANKS=3
export POLARIS_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export POLARIS_PATH="$HOME/POLARIS"
export PATH="$POLARIS_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$POLARIS_PATH/lib/lib:$POLARIS_PATH/lib/lib64:${LD_LIBRARY_PATH:-}"
source ~/IRIS/iris_venv/bin/activate
mpirun -np "$IRIS_MPI_RANKS" --oversubscribe --bind-to none python fig_polaris_vs_iris.py
