# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Loss Trajectory figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import visualization as vz


checkpoint_directories = [f'~/IRIS/models/reverter_{n + 1}' for n in range(6)]
vz.loss_trajectory(checkpoint_directories=checkpoint_directories,
                   path='~/IRIS/output/loss_trajectory.png')