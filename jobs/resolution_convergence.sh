#!/bin/bash

# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

#SBATCH --job-name=IRIS_test
#SBATCH --output=resolution_convergence.out
#SBATCH --partition=general-gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

module purge
module load gcc
module load openmpi/5.0.5-noucx

source ~/IRIS/iris_venv/bin/activate
mpirun python resolution_convergence.py
