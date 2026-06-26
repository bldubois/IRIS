# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Formal vs Smooth Integration figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import hyper as hp
from iris import arepo_processing as ap
from iris import visualization as vz


hyper = hp.SEDIGISM_13C16O(r_steps=2048)
hyper.observer_hyper.out_blur_fwhm = None
hyper.validate()
reader = ap.Reader(path='/path/to/sims_overview_data',
                   dataset_type=ap.StandardDataset,
                   hyper=hyper)
dataset = reader.dataset

vz.formal_vs_smooth(dataset=dataset,
                    hyper=hyper,
                    plus_dust=True,
                    path='~/IRIS/output/formal_vs_smooth.png')
