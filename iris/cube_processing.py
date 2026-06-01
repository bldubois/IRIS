# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Utilities for loading and processing spectral cubes.

Some utilities in this module use the `spectral_cube` and `reproject` packages.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import typing

import numpy as np
import torch
from astropy import units
from astropy.wcs import WCS
from spectral_cube import SpectralCube
from reproject import reproject_interp

if typing.TYPE_CHECKING:
    from . import hyper as hp


def load_fits_with_bounds(path: str,
                          lon_steps: int,
                          lon_min: float,
                          lon_max: float,
                          lat_steps: int,
                          lat_min: float,
                          lat_max: float,
                          v_min: float,
                          v_max: float) -> np.ndarray:
    """
    Reads a spectral cube saved as a FITS file and converts it to a NumPy array.

    Opens the FITS file, then reprojects and reinterpolates the spectral cube within a
    new set of specified bounds using the `reproject` and `spectral_cube` libraries.
    Ensures that the cube is expressed within Cartesian galactic longitude/latitude
    coordinates consistent with the IRIS spherical coordinate system. Automatically
    corrects, for instance, spectral cubes expressed in terms of a sine projection.
    Does not reinterpolate the velocity dimension of the cube.

    Args:
        path: Path to the FITS file on the disk.
        lon_steps: Number of galactic longitude steps. Longitude values are linearly spaced.
        lon_min: Minimal longitude bound.
        lon_max: Maximal longitude bound.
        lat_steps: Number of galactic latitude steps. Latitude values are linearly spaced.
        lat_min: Minimal latitude bound.
        lat_max: Maximal latitude bound.
        v_min: Minimal velocity bound.
        v_max: Maximal velocity bound.

    Returns:
        The spectral cube as a NumPy array.
    """
    cube = SpectralCube.read(path, format='fits')
    cube = cube.with_spectral_unit(units.km / units.s).spectral_slab(
        v_min * units.km / units.s, v_max * units.km / units.s)

    v_steps = cube.shape[0]
    new_wcs = WCS(naxis=2)
    new_wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    new_wcs.wcs.cunit = ['deg', 'deg']
    new_wcs.wcs.crpix = [(lon_steps + 1) / 2,
                         (lat_steps + 1) / 2]
    new_wcs.wcs.crval = [(lon_min + lon_max) / 2,
                         (lat_min + lat_max) / 2]
    new_wcs.wcs.cdelt = [(lon_max - lon_min) / (lon_steps - 1),
                         (lat_max - lat_min) / (lat_steps - 1)]

    raw = np.empty((v_steps, lat_steps, lon_steps), dtype=np.float32)
    for i in range(v_steps):
        slice, footprint = reproject_interp((cube.filled_data[i, :, :].value, cube[i].wcs.celestial),
                                             new_wcs,
                                             shape_out=(lat_steps, lon_steps))
        raw[i, :, :] = slice.astype(np.float32)
    return raw.transpose((2, 1, 0))

def make_cube(lon_steps: int,
              lat_steps: int,
              v_steps: int,
              hyper: hp.Hyper,
              units: str = 'iris',
              verbose: bool = False) -> torch.Tensor:
    """
    Makes a spectral cube of specified shape from the cube path or paths and
    configuration parameters specified in a [hyperparameters object][iris.hyper.Hyper].

    Sources a spectral cube or spectral cubes from either a list
    `hyper.cube_hyper.data_path` of NumPy .np files or a
    user-specified mapping function `hyper.cube_hyper.fits_map`
    that returns a list of FITS file objects. Designed to process a list of
    cubes in case IRIS is configured to observe and revert multiple spectral lines
    in combination. Interpolates cubes onto the shared grid, applies any denoising
    steps, specified as a list `hyper.cube_hyper.clean_noise` of
    user-defined functions with input and output of a spectral cube, and converts
    into the desired units.

    Args:
        lon_steps: Number of galactic longitude steps. Longitude values are linearly spaced.
        lat_steps: Number of galactic latitude steps. Latitude values are linearly spaced.
        v_steps: Number of velocity steps. Velocity values are linearly spaced.
        hyper: A hyperparameters object.
        units: One of `'iris', 'T K'`.
        verbose: If `True`, prints cube statistics.

    Returns:
        The cube or stacked cubes, with dimensions
        `1` (batch)`, n_lines` (channel)`, lon_steps` (longitude)`,
        lat_steps `(latitude)`, v_steps `(velocity).
        If there are multiple cubes, they are stacked along the channel dimension, for
        ease of use with observation and reversion.

    Raises:
        ValueError: If `units` is not one of `'iris', 'T K'`.
    """
    if hyper.cube_hyper.fits_map is not None:
        raw = hyper.cube_hyper.fits_map()
    else:
        raw = [np.load(os.path.expanduser(path)) for path in hyper.cube_hyper.data_path]

    raw = [torch.tensor(cube, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0) for cube in raw]

    if hyper.cube_hyper.conversion_raw_to_T_K is not None:
        for i in range(hyper.observer_hyper.n_lines):
            conversion = hyper.cube_hyper.conversion_raw_to_T_K[i]
            if conversion is not None:
                raw[i] = conversion(raw[i], hyper)

    if verbose:
        for i in range(len(raw)):
            print(f'Cube {i + 1} Max:\t{raw[i].max():.4f}', flush=True)

    sized = [torch.nn.functional.interpolate(cube, size=(lon_steps, lat_steps, v_steps)) for cube in raw]

    if hyper.cube_hyper.clean_noise is not None:
        for i in range(hyper.observer_hyper.n_lines):
            clean_noise = hyper.cube_hyper.clean_noise[i]
            if clean_noise is not None:
                sized[i] = clean_noise(sized[i], hyper)

    stacked = torch.cat(sized, dim=1)

    if units == 'iris':
        temperature = hyper.dataset_hyper._temperature_iris_per_SI
        units_corrected = stacked * temperature
    elif units == 'T K':
        units_corrected = stacked
    else:
        raise ValueError("Invalid units provided to make_cube: must be 'iris' or 'T K'.")

    return units_corrected

def make_default_cube(hyper: hp.Hyper,
                      units: str = 'iris',
                      numpy: bool = False,
                      verbose: bool = False) -> torch.Tensor | np.ndarray:
    """
    Makes a spectral cube of default shape from the cube path or paths and
    configuration parameters specified in a [hyperparameters object][iris.hyper.Hyper].

    A wrapper for [`make_cube`][iris.cube_processing.make_cube].

    Args:
        hyper: A hyperparameters object.
        units: One of `'iris', 'T K'`.
        numpy: If `True`, converts to a NumPy output for convenience.
        verbose: If `True`, prints cube statistics.

    Returns:
        The cube or stacked cubes, with dimensions
        `1` (batch)`, n_lines` (channel)`, lon_steps` (longitude)`,
        lat_steps `(latitude)`, v_steps `(velocity).
        If there are multiple cubes, they are stacked along the channel dimension, for
        ease of use with observation and reversion.
    """
    lon_steps = hyper.coordinate_hyper.lon_steps
    lat_steps = hyper.coordinate_hyper.lat_steps
    v_steps = hyper.cube_hyper.v_steps
    cube = make_cube(lon_steps=lon_steps,
                     lat_steps=lat_steps,
                     v_steps=v_steps,
                     hyper=hyper,
                     units=units,
                     verbose=verbose)
    if numpy:
        cube = cube.detach().numpy()
    return cube

def corrected_antenna_temperature_to_raleigh_jeans_temperature(cube: torch.Tensor,
                                                               hyper: hp.Hyper) -> torch.Tensor:
    r"""
    A convenience function for converting corrected antenna temperature to Raleigh-Jeans temperature.

    Designed to be used as an entry in `hyper.cube_hyper.conversion_raw_to_T_K`.
    Not used for IRIS SEDIGISM, as the SEDIGISM $^{13}\text{CO}$ cubes are published in units of
    Raleigh-Jeans temperature.

    Args:
        cube: A cube to convert.
        hyper: A hyperparameters object.

    Returns:
        The corrected cube.
    """
    return cube / hyper.cube_hyper.beam_efficiency

def intensity_to_raleigh_jeans_temperature(I: torch.Tensor,
                                           hyper: hp.Hyper,
                                           nu_ul: float) -> torch.Tensor:
    r"""
    A convenience function for converting intensity in Jy/sr to Raleigh-Jeans temperature in K.

    Designed to be used as an entry in `hyper.cube_hyper.conversion_raw_to_T_K`,
    but must in this case be used via a lambda pattern to specify `nu_ul`.

    Args:
        I: The input cube of intensity in Jy/sr.
        hyper: A hyperparameters object.
        nu_ul: The transition frequency in Hz.

    Returns:
        The output cube of Raleigh-Jeans brightness temperature in K.
    """
    c = hyper.observer_hyper.c
    k = hyper.observer_hyper.k
    v_min = hyper.cube_hyper.v_min * 1000
    v_max = hyper.cube_hyper.v_max * 1000
    batch, channel, lon_steps, lat_steps, v_steps = I.shape
    v = torch.linspace(v_min, v_max, v_steps)
    nu = (1 + v / c) * nu_ul
    Trj = c * c * I * 1e-26 / (2 * k * nu * nu)
    return Trj
