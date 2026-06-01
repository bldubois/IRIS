#!/bin/bash

# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

#SBATCH --job-name=IRIS_fig
#SBATCH --output=fig_sims_overview.out
#SBATCH --partition=general-gpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

module purge
source ~/IRIS/iris_venv/bin/activate
python fig_sims_overview.py