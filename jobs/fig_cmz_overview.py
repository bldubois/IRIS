# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the CMZ Overview figure from the IRIS paper (DuBois et al., 2026).
"""

from iris import hyper as hp
from iris import visualization as vz


hyper = hp.SEDIGISM_13C16O()
hyper.validate()
vz.cmz_overview(hyper, path='~/IRIS/output/cmz_overview.png')