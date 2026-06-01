# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the SEDIGISM Reversions figure from the IRIS paper (DuBois et al., 2026).
"""

import os
import torch

from iris import hyper as hp
from iris import arepo_processing as ap
from iris import reversion as rv
from iris import visualization as vz

hyper = hp.SEDIGISM_13C16O()
hyper.validate()
reader = ap.Reader(path='/path/to/training_data_1',
                   dataset_type=ap.SyntheticallyObservedDataset,
                   hyper=hyper)

reverters = []
for i in range(4):
    reverter = rv.Reverter(hyper=hyper)
    reverter.load_state_dict(
        torch.load(os.path.expanduser(f'~/IRIS/models/reverter_{i + 1}/chp_32.pt'), weights_only=True, map_location='cpu'))
    reverters.append(reverter)

vz.true_reversions(reverters=reverters,
                   hyper=hyper,
                   path='~/IRIS/output/true_reversions.png')
