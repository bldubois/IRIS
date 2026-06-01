# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Generate the IRIS training dataset.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

import sys
import os

from iris import hyper as hp
from iris import arepo_processing_write as ap

local_cache = sys.argv[1]
n = os.getenv('SLURM_ARRAY_TASK_ID')
hyper = hp.SEDIGISM_13C16O()
hyper.validate()

writer = ap.Writer(path=f'/path/to/training_data_{n}',
                   snapshot_directory='/path/to/snapshot/directory',
                   ssh_key_path='/path/to/ssh/key',
                   remote_address='user@remote.server.edu',
                   local_cache=local_cache,
                   hyper=hyper,
                   dataset_type=ap.SyntheticallyObservedDataset,
                   gpu_interpolate=True,
                   gpu_normalize=False,
                   verbose=True,
                   transfer='optically thick')

# writer = ap.Writer(path=f'/path/to/training_data_{n}',
#                    snapshot_paths=['/path/to/snapshot_1', '/path/to/snapshot_2'],
#                    hyper=hyper,
#                    dataset_type=ap.SyntheticallyObservedDataset,
#                    gpu_interpolate=True,
#                    gpu_normalize=False,
#                    verbose=True)
