# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Train the IRIS `Reverter`.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

import os

from iris import hyper as hp
from iris import reversion
from iris import observation
from iris import training
from iris import arepo_processing as ap


n = os.getenv('SLURM_ARRAY_TASK_ID')
hyper = hp.SEDIGISM_13C16O()
hyper.validate()
paths = ['/path/to/training_data_1', '/path/to/training_data_2']
litter_path = ['/path/to/litter_data_1', '/path/to/litter_data_2']

reader = ap.Reader(path=paths,
                   hyper=hyper,
                   dataset_type=ap.SyntheticallyObservedDataset,
                   litter_path=litter_path,
                   litter_type=ap.CPUBatchObservedDataset)
dataset = reader.dataset
litter = reader.litter
noise = observation.Noise(hyper=hyper, fade=True)

reverter = reversion.Reverter(hyper=hyper)
# reverter.load_state_dict(torch.load('path/to/checkpoint', weights_only=True, map_location='cpu'))
reverter, rank = training.train_reverter(reverter=reverter,
                                         dataset=dataset,
                                         noise=noise,
                                         litter=litter,
                                         hyper=hyper,
                                         checkpoint_directory='~/IRIS/models',
                                         checkpoint_name=f'reverter_{n}')
