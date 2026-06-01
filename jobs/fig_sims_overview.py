# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Simulation Overview figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import arepo_processing as ap
from iris import visualization as vz


reader = ap.Reader(path='/path/to/sims_overview_data',
                   dataset_type=ap.StandardDataset)
dataset = reader.dataset
snapshot_path = '/path/to/snapshot.hdf5'
vz.sims_overview(snapshot_path=snapshot_path,
                 dataset=dataset,
                 hyper=None,
                 path='~/IRIS/output/sims_overview.png')