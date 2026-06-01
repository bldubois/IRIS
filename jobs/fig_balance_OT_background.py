# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the OT Balance Background Temperature Comparison figures from the IRIS paper (DuBois et al., 2026).
"""

from iris import hyper as hp
from iris import arepo_processing as ap
from iris import visualization as vz


hyper = hp.SEDIGISM_13C16O()
hyper.observer_hyper.out_blur_fwhm = None
hyper.validate()
reader = ap.Reader(path='/path/to/sims_overview_data',
                   dataset_type=ap.StandardDataset,
                   hyper=hyper)
dataset = reader.dataset

left_background = 0
right_background = 2.73
title = 'balance_OT_0K_vs_CMB'
# left_background = 2.73
# right_background = 5.0
# title = 'balance_OT_CMB_vs_5K'
vz.balance_OT_background(dataset=dataset,
                         hyper=hyper,
                         left_background=left_background,
                         right_background=right_background,
                         plus_dust=True,
                         path=f'~/IRIS/{title}.png')
