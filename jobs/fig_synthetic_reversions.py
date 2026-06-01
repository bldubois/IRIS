# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the Synthetic Reversions figure from the IRIS paper (DuBois et al., 2026).
"""

import os
import torch

from iris import arepo_processing as ap
from iris import reversion as rv
from iris import visualization as vz

reader = ap.Reader(path='/path/to/test_data',
                   dataset_type=ap.SyntheticallyObservedDataset)
dataset = reader.dataset
reader = ap.Reader(path='/path/to/full_cone',
                   dataset_type=ap.CPUBatchObservedDataset)
full_cone_dataset = reader.dataset

reverter = rv.Reverter(hyper=dataset.hyper)
reverter.load_state_dict(
    torch.load(os.path.expanduser('~/IRIS/models/reverter_1/chp_32.pt'), weights_only=True, map_location='cpu'))

vz.compute_synthetic_reversions(reverter=reverter,
                                dataset=dataset,
                                full_cone_dataset=full_cone_dataset)
vz.synthetic_reversions(reverter=reverter,
                        pure_top_down_paths=['pure_top_down1.np', 'pure_top_down2.np'],
                        pure_lv_paths=['pure_lv1.np', 'pure_lv2.np'],
                        full_cone_top_down_paths=['fc_top_down1.np', 'fc_top_down2.np'],
                        full_cone_lv_paths=['fc_lv1.np', 'fc_lv2.np'],
                        noise_top_down_paths=['noise_top_down1.np', 'noise_top_down2.np'],
                        noise_lv_paths=['noise_lv1.np', 'noise_lv2.np'],
                        hyper=dataset.hyper,
                        full_cone_hyper=full_cone_dataset.hyper,
                        noise_hyper=full_cone_dataset.hyper,
                        path='~/IRIS/output/synthetic_reversions.png')