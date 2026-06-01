# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Failure Modes figure from the IRIS paper (DuBois et al., 2026).
"""

import os
import torch

from iris import arepo_processing as ap
from iris import reversion as rv
from iris import observation as ob
from iris import visualization as vz


reader = ap.Reader(path='/path/to/wrong_physics',
                   dataset_type=ap.StandardDataset)
wrong_dataset = reader.dataset
observer = ob.IteratedSyntheticObserver(hyper=wrong_dataset.hyper)
reverter = rv.Reverter(hyper=wrong_dataset.hyper)
reverter.load_state_dict(
    torch.load(os.path.expanduser('~/IRIS/models/reverter_1/chp_32.pt'), weights_only=True, map_location='cpu'))

for i in range(4):
    vz.failure_modes(reverter=reverter,
                     wrong_dataset=wrong_dataset,
                     observer=observer,
                     hyper=wrong_dataset.hyper,
                     path=f'~/IRIS/output/failure_modes_{i}.png')