# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Dust vs. No Dust figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import hyper as hp
from iris import arepo_processing as ap
from iris import visualization as vz


hyper = hp.SEDIGISM_13C16O()
title = 'no_dust_vs_dust'
dust_name = 'Standard Dust Opacity'
color_scale_center = 0
color_norm_min = 0
color_norm_max = 5.0
cbar_ticks = (-0.1, 0.0, 0.1, 1.0, 2.5, 5.0)

# title = 'no_dust_vs_dust_x100'
# dust_name = r'Dust Opacity $\times$ 100'
# kappa_dust = hyper.observer_hyper.kappa_dust[0]
# kappa_dust *= 100
# hyper.observer_hyper.kappa_dust = [kappa_dust]
# color_scale_center = 2e-2
# color_norm_min = -0.1

hyper.observer_hyper.out_blur_fwhm = None
hyper.validate()
reader = ap.Reader(path='/path/to/sims_overview_data',
                   dataset_type=ap.StandardDataset,
                   hyper=hyper)
dataset = reader.dataset

vz.no_dust_vs_dust(dataset=dataset,
                   hyper=hyper,
                   dust_name=dust_name,
                   color_scale_center=color_scale_center,
                   color_norm_min=color_norm_min,
                   color_norm_max=color_norm_max,
                   cbar_ticks=cbar_ticks,
                   path=f'~/IRIS/output/{title}.png')
