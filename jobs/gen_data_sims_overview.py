# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Generate the standard dataset for the Simulation Overview figure (and other figures)
from the IRIS paper (DuBois et al., 2026).

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

import sys

from iris import hyper as hp
from iris import arepo_processing_write as ap


local_cache = sys.argv[1]
hyper = hp.SEDIGISM_13C16O(r_steps=2048)
hyper.writer_hyper.points_per_snapshot = 1
hyper.dataset_hyper.CMZ_scale_factor = None
hyper.dataset_hyper.CMZ_scale_range = None
hyper.coordinate_hyper.theta_zero = 270.
hyper.validate()
reader = ap.Reader(path='/path/to/training_data_1',
                   dataset_type=ap.SyntheticallyObservedDataset)

writer = ap.Writer(path='/path/to/sims_overview_data',
                   snapshot_paths=['/path/to/snapshot.hdf5'],
                   hyper=hyper,
                   dataset_type=ap.StandardDataset,
                   units_from=reader.dataset,
                   gpu_interpolate=True,
                   gpu_normalize=False,
                   verbose=True)

# writer = ap.Writer(path='/path/to/sims_overview_data',
#                    snapshot_directory='/path/to/snapshot/directory',
#                    ssh_key_path='/path/to/ssh/key',
#                    remote_address='user@remote.server.edu',
#                    local_cache=local_cache,
#                    hyper=hyper,
#                    dataset_type=ap.StandardDataset,
#                    units_from=reader.dataset,
#                    gpu_interpolate=True,
#                    gpu_normalize=False,
#                    verbose=True)
