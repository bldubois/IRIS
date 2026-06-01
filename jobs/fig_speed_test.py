# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the IRIS-SO Speed Test figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import visualization as vz


vz.speed_test(speed_data_paths=['~/IRIS/data/speed_test_A.json',
                                '~/IRIS/data/speed_test_B.json'],
              path='~/IRIS/output/speed_test.png')
