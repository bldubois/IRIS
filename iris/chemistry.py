# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
r"""
A module for computing chemical and spectral data of tracer molecules.

Parses files in the .dat file format specified by the Leiden Atomic and Molecular Database (LAMDA).
Determines transition frequencies, Einstein coefficients, collision rates,
and other key chemical information for each observed tracer. Computes the level
balance systems for each tracer over a grid of $\text{H}_2$ abundances and temperatures
using a non-LTE, optically thin assumption. Then computes emission and absorption
coefficients for each tracer over a grid of gas density, $\text{H}_2$ abundance, and temperature.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import gc
import typing

import numpy as np
import scipy
import torch

if typing.TYPE_CHECKING:
    import mpi4py
    from . import hyper as hp


def _parse_lamda(path: str) -> TracerData:
    """
    Retrieves chemical data for a single tracer molecule by parsing a file
    in the Leiden Atomic and Molecular Database (LAMDA) .dat format.

    Args:
        path: The path on disk to a .dat file.

    Returns:
        A [`TracerData`][iris.chemistry.TracerData] dictionary containing:

            * `molecular_weight`
            * `num_levels`
            * `energies`
            * `weights`
            * `A_ij`
            * `nu`
            * `K_ij_H`
            * `T_H`
            * `K_ij_H2`
            * `T_H2`
            * `K_ij_para_H2`
            * `T_para_H2`
            * `K_ij_ortho_H2`
            * `T_ortho_H2`
            * `K_ij_He`
            * `T_He`

    Raises:
        RuntimeError: If there is an error parsing the .dat file.
    """
    with open(os.path.expanduser(path), 'r') as file:
        lines = file.readlines()

    n = 0
    found = False
    for line in lines:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Molecular ID not found.')

    found = False
    n += 1
    for line in lines:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Molecular weight not found.')
    molecular_weight = float(lines[n].strip())

    found = False
    n += 1
    for line in lines[n:]:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Number of energy levels not found.')
    num_levels = int(lines[n].strip())

    found = False
    n += 1
    for line in lines[n:]:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Energy-weight data not found.')

    energies = np.zeros((num_levels,), dtype=np.float32)
    weights = np.zeros((num_levels,), dtype=np.float32)
    found = False
    for line in lines[n:]:
        n += 1
        if line.strip() == '!NUMBER OF RADIATIVE TRANSITIONS':
            found = True
            break
        data = [float(x) for x in line.split()]
        i = int(data[0] - 1)
        energies[i] = data[1]
        weights[i] = data[2]
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Energy-weight data incomplete.')

    found = False
    for line in lines[n:]:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Transition data not found.')

    A_ij = np.zeros((num_levels, num_levels), dtype=np.float32)
    nu = np.zeros((num_levels, num_levels), dtype=np.float32)
    found = False
    for line in lines[n:]:
        n += 1
        if line.strip().startswith('!'):
            found = True
            break
        data = [float(x) for x in line.split()]
        i = int(data[1] - 1)
        j = int(data[2] - 1)
        A_ij[i, j] = data[3]
        nu[i, j] = data[4]
    if not found:
        raise RuntimeError('Error reading LAMDA .dat file. Transition data incomplete.')

    collisions = []
    partners = []
    temperatures = []
    while True:
        found = False
        for line in lines[n:]:
            n += 1
            if line.strip().startswith('!'):
                found = True
                break
        if not found:
            raise RuntimeError('Error reading LAMDA .dat file. Collisions data not found.')
        partners.append(int(lines[n].strip()[0]))

        found = False
        for line in lines[n:]:
            n += 1
            if line.strip().startswith('!'):
                found = True
                break
        if not found:
            raise RuntimeError('Error reading LAMDA .dat file. Collisions data not found.')

        found = False
        for line in lines[n:]:
            n += 1
            if line.strip().startswith('!'):
                found = True
                break
        if not found:
            raise RuntimeError('Error reading LAMDA .dat file. Collision data not found.')

        found = False
        for line in lines[n:]:
            n += 1
            if line.strip().startswith('!'):
                found = True
                break
        if not found:
            raise RuntimeError('Error reading LAMDA .dat file. Collision data not found.')
        T = np.array([float(x) for x in lines[n].split()], np.float32)
        temperatures.append(T)

        found = False
        for line in lines[n:]:
            n += 1
            if line.strip().startswith('!'):
                found = True
                break
        if not found:
            raise RuntimeError('Error reading LAMDA .dat file. Collision data not found.')

        K_ij = np.zeros((num_levels, num_levels, len(T)), dtype=np.float32)
        done = True
        for line in lines[n:]:
            if line.strip().startswith('!'):
                if not lines[n + 1].strip().startswith('!'):
                    done = False
                break
            data = np.array([float(x) for x in line.split()], dtype=np.float32)
            i = int(data[1] - 1)
            j = int(data[2] - 1)
            K_ij[i, j, :] = data[3:]
            n += 1
        collisions.append(K_ij)

        if done:
            break

    K_ij_H = None
    T_H = None
    K_ij_H2 = None
    T_H2 = None
    K_ij_para_H2 = None
    T_para_H2 = None
    K_ij_ortho_H2 = None
    T_ortho_H2 = None
    K_ij_He = None
    T_He = None
    for partner, K_ij, T in zip(partners, collisions, temperatures):
        if partner == 1:
            K_ij_H2 = K_ij
            T_H2 = T
        elif partner == 2:
            K_ij_para_H2 = K_ij
            T_para_H2 = T
        elif partner == 3:
            K_ij_ortho_H2 = K_ij
            T_ortho_H2 = T
        elif partner == 5:
            K_ij_H = K_ij
            T_H = T
        elif partner == 6:
            K_ij_He = K_ij
            T_He = T

    return {'molecular_weight': molecular_weight,
            'num_levels': num_levels,
            'energies': energies,
            'weights': weights,
            'A_ij': A_ij,
            'nu': nu,
            'K_ij_H': K_ij_H,
            'T_H': T_H,
            'K_ij_H2': K_ij_H2,
            'T_H2': T_H2,
            'K_ij_para_H2': K_ij_para_H2,
            'T_para_H2': T_para_H2,
            'K_ij_ortho_H2': K_ij_ortho_H2,
            'T_ortho_H2': T_ortho_H2,
            'K_ij_He': K_ij_He,
            'T_He': T_He}

def _compute_single_molecule(data: TracerData, hyper: hp.Hyper) -> TracerData:
    r"""
    Computes spectral data for a single tracer molecule not included in its LAMDA file.

    Computes the Einstein B coefficients. Reinterpolates LAMDA collision rates over a shared
    temperature grid. Generates an all-partner collision tensor over dimensions of
    $\text{H}_2$ abundance and temperature. Computes excitational collision rates
    from the de-excitational rates using the detailed balance. Computes a tensor of
    level balance matrices over $\text{H}_2$ abundance and temperature dimensions.

    For the $\text{H}_2$ abundance dimension, all H atoms are assumed to exist in the
    binary of either H or $\text{H}_2$. In contrast, IRIS is designed to operate with
    AREPO simulations that model a three-species H chemistry of H, $\text{H}_2$,
    and $\text{H}^+$. Effectively, the IRIS approximation treats $\text{H}^+$ collisions as
    H collisions, which is not strictly accurate, since H and $\text{H}^+$ collision rates
    may differ drastically. This simplification is not particularly problematic, however,
    since $\text{H}^+$ is very low-abundance in regions where CO or other complex tracers
    are high-abundance. Ignoring $\text{H}^+$ prevents having to model another dimension
    in the [emission and absorption grids][iris.chemistry.make_observability_grids],
    which would add orders of magnitude of complexity.

    This function then sets up the level balance system using a non-LTE, optically thin assumption.
    A true optically thin assumption supposes that the medium absorbs no radiative energy,
    and thus is balanced only by collisions and spontaneous emission, which eliminates the
    need for a costly intensity dimension in the grid. This is a weaker assumption
    than the assumption that the radiative transfer of the spectral line is optically thin.
    Depending upon the transfer mode enabled
    (see [`TransferProcessor`][iris.observation.TransferProcessor]), IRIS can compute both
    stimulated emission and absorption of the spectral line, and does not treat the
    line radiative transfer itself as optically thin.

    In computing line-of-sight transfer, we may expect that the spectral line
    may become optically thick along a minority of lines within the plane of the galactic disk.
    Of relevance to the level balance at any point along this line of sight, however, is whether
    the average behavior over all solid angle is locally optically thick. In many feasible cases,
    an optically thin assumption may still provide a good approximation to these level balances
    even along optically thick lines of sight.

    A variant of the optically thin level balance is the full-escape-probability assumption with
    background intensity. This balance assumes that local reabsorption of the spectral line itself
    is negligible, but absorption of (or stimulated emission by) some external background radiation
    is not. IRIS is intended to be used with the true optically thin assumption with no background.
    IRIS does have the capability, however, of computing the balance with the addition of a fixed,
    blackbody background by specifying `hyper.observer_hyper.T_continuum`.

    It is recommended `T_continuum` be set to `None` (i.e. disregarded) as opposed to the temperature
    of the CMB. The CMB is still involved separately in computation of the line radiative transfer
    and continuum subtraction. In many cases, however, the intensity of the CMB may only be a
    minority contribution to the background intensity at any point at which the level balance
    is computed. Therefore, setting `T_continuum = T_cmb` only provides an appearance of greater
    accuracy as opposed to a true enhancement.

    The `T_continuum` option is left as enabled, however, for use in gauging the efficacy of the
    true optically thin level balance assumption. A side-by-side of the same synthetic observation
    computed with `T_continuum = None` versus with a range of expected values shows negligible
    difference, indicating that the optically thin assumption is a sufficient approximation
    for the IRIS use-case. See the IRIS paper for further discussion.

    Args:
        data: A parsed LAMDA file.
        hyper: A hyperparameters object.

    Returns:
        A [`TracerData`][iris.chemistry.TracerData] dictionary containing:

            * `K`
            * `A`
            * `abundance_H2`
            * `d_ab`
            * `T`
            * `dT`
            * `num_levels`
            * `nu_ul`
            * `A_ul`
            * `B_ul`
            * `B_lu`
            * `weight_upper`
            * `weight_lower`
            * `molecular_weight`

    Raises:
        RuntimeError: If no $\text{H}_2$ collision rates found in the LAMDA file.
    """
    # Get data.
    k_b = hyper.observer_hyper.k
    h = hyper.observer_hyper.h
    c = hyper.observer_hyper.c

    ortho_to_para_H2_ratio = hyper.observer_hyper.ortho_to_para_H2_ratio
    abundance_He = hyper.observer_hyper.abundance_He
    abundance_H2_steps = data['abundance_H2_steps']
    # Max H2 abundance per total H number density is 0.5.
    abundance_H2 = np.linspace(0, .5, abundance_H2_steps, dtype=np.float32)
    d_ab = .5 / (abundance_H2_steps - 1)
    abundance_H2 = np.expand_dims(abundance_H2, axis=(0, 1, 3))

    interpolation_max_T = data['interpolation_max_T']
    T_steps = data['T_steps']
    T = np.linspace(0, interpolation_max_T, T_steps, dtype=np.float32)
    dT = interpolation_max_T / (T_steps - 1)

    num_levels = data['num_levels']
    energies = data['energies'] * 100 * h * c         # from cm^-1 to Joules
    weights = data['weights']
    weight_upper = weights[data['transition'][0]]
    weight_lower = weights[data['transition'][1]]
    A_ij = data['A_ij']
    nu = data['nu'] * 1e9      # in Hz
    A_ul = A_ij[data['transition'][0], data['transition'][1]]
    nu_ul = nu[data['transition'][0], data['transition'][1]]

    # Compute Einstein B coefficients.
    B_ij = np.zeros((num_levels, num_levels), dtype=np.float32)
    epsilon = 1e-15
    for i in range(num_levels):
        for j in range(i):
            if A_ij[i, j] > epsilon:
                B_ij[i, j] = c * c / 2 / (h * nu[i, j] * nu[i, j] * nu[i, j]) * A_ij[i, j]
    for i in range(num_levels):
        for j in range(i + 1, num_levels):
            if A_ij[j, i] > epsilon:
                B_ij[i, j] = weights[j] / weights[i] * B_ij[j, i]
    B_ul = B_ij[data['transition'][0], data['transition'][1]]
    B_lu = B_ij[data['transition'][1], data['transition'][0]]

    # For diagnostics only:
    # Compute intensities from the continuum.
    # Use to gauge stability of an optically thin assumption on the level balances.
    J_ij = np.zeros((num_levels, num_levels), dtype=np.float32)
    T_continuum = hyper.observer_hyper.T_continuum
    if T_continuum is not None:
        for i in range(1, num_levels):
            for j in range(i):
                if A_ij[i, j] > epsilon:
                    J_ij[i, j] = 2 * h * nu[i, j] * nu[i, j] * nu[i, j] / c / c / (
                            np.exp(h * nu[i, j] / k_b / T_continuum) - 1)

    # Get collision rates.
    K_ij_H = None
    T_H = data['T_H']
    if data['K_ij_H'] is not None:
        K_ij_H = data['K_ij_H'] / 1e6                 # in m^3/s

    T_para_H2 = data['T_para_H2']
    if data['K_ij_para_H2'] is not None:
        K_ij_para_H2 = data['K_ij_para_H2'] / 1e6     # in m^3/s
    elif data['K_ij_H2'] is not None:
        K_ij_para_H2 = data['K_ij_H2'] / 1e6          # in m^3/s
    else:
        raise RuntimeError('No H2 collision rates provided in .dat file.')

    T_ortho_H2 = data['T_ortho_H2']
    if data['K_ij_ortho_H2'] is not None:
        K_ij_ortho_H2 = data['K_ij_ortho_H2'] / 1e6   # in m^3/s
    elif data['K_ij_H2'] is not None:
        K_ij_ortho_H2 = data['K_ij_H2'] / 1e6         # in m^3/s
    else:
        raise RuntimeError('No H2 collision rates provided in .dat file.')

    K_ij_He = None
    T_He = data['T_He']
    if data['K_ij_He'] is not None:
        K_ij_He = data['K_ij_He'] / 1e6               # in m^3/s

    # Reinterpolate collision rates over a shared temperature grid.
    fine_temperature_grid = np.meshgrid(np.arange(num_levels),
                                        np.arange(num_levels),
                                        T,
                                        indexing='ij')
    if K_ij_H is not None:
        coarse_temperature_grid = (np.arange(num_levels), np.arange(num_levels), T_H)
        K_ij_H_fine = scipy.interpolate.interpn(coarse_temperature_grid,
                                                K_ij_H,
                                                fine_temperature_grid,
                                                method='linear',
                                                bounds_error=False,
                                                fill_value=None)
        K_ij_H_fine = np.expand_dims(K_ij_H_fine, axis=2)
    coarse_temperature_grid = (np.arange(num_levels), np.arange(num_levels), T_para_H2)
    K_ij_para_H2_fine = scipy.interpolate.interpn(coarse_temperature_grid,
                                                  K_ij_para_H2,
                                                  fine_temperature_grid,
                                                  method='linear',
                                                  bounds_error=False,
                                                  fill_value=None)
    K_ij_para_H2_fine = np.expand_dims(K_ij_para_H2_fine, axis=2)
    coarse_temperature_grid = (np.arange(num_levels), np.arange(num_levels), T_ortho_H2)
    K_ij_ortho_H2_fine = scipy.interpolate.interpn(coarse_temperature_grid,
                                                   K_ij_ortho_H2,
                                                   fine_temperature_grid,
                                                   method='linear',
                                                   bounds_error=False,
                                                   fill_value=None)
    K_ij_ortho_H2_fine = np.expand_dims(K_ij_ortho_H2_fine, axis=2)
    if K_ij_He is not None:
        coarse_temperature_grid = (np.arange(num_levels), np.arange(num_levels), T_He)
        K_ij_He_fine = scipy.interpolate.interpn(coarse_temperature_grid,
                                                 K_ij_He,
                                                 fine_temperature_grid,
                                                 method='linear',
                                                 bounds_error=False,
                                                 fill_value=None)
        K_ij_He_fine = np.expand_dims(K_ij_He_fine, axis=2)

    # Compute the all-partners collision tensor over dimensions of i, j, H2 abundance, temperature.
    # All H atoms not bound as H2 are assumed to be H. H+ is ignored.
    # This is a tradeoff. H and H+ collision rates are very different, so accuracy is reduced.
    # But eliminating a dimension of H+ abundance makes the grids reasonably sized.
    # This is generally okay because H+ and complex tracers are not mutually abundant.
    abundance_para_H2 = 1 / (ortho_to_para_H2_ratio + 1)
    abundance_ortho_H2 = 1 - abundance_para_H2
    K_ij = abundance_H2 * (abundance_para_H2 * K_ij_para_H2_fine + abundance_ortho_H2 * K_ij_ortho_H2_fine)
    if K_ij_H is not None:
        abundance_H = 1 - 2 * abundance_H2
        K_ij += abundance_H * K_ij_H_fine
    if K_ij_He is not None:
        K_ij += abundance_He * K_ij_He_fine

    # Compute excitational collision rates from de-excitational
    # collision rates using the detailed balance.
    # Note that the detailed balance, while derived from LTE,
    # is a microscopic property of molecular geometry, and so holds out of LTE also.
    for i in range(1, num_levels):
        for j in range(0, i):
            coeff = weights[i] / weights[j] * np.exp((energies[j] - energies[i]) / k_b / T[1:])
            K_ij[j, i, :, 1:] = K_ij[i, j, :, 1:] * coeff

    # Compute the level balance matrices.
    K = np.zeros(K_ij.shape, dtype=np.float32)
    for i in range(1, num_levels):
        for j in range(num_levels):
            if i != j:
                K[i, j] = K_ij[j, i]
                K[i, i] -= K_ij[i, j]
    A = np.zeros((num_levels, num_levels), dtype=np.float32)
    for i in range(1, num_levels):
        for j in range(0, i):
            A[i, i] -= A_ij[i, j] + J_ij[i, j] * B_ij[i, j]
            A[i, j] = J_ij[i, j] * B_ij[j, i]
        for k in range(i + 1, num_levels):
            A[i, i] -= J_ij[k, i] * B_ij[i, k]
            A[i, k] = A_ij[k, i] + J_ij[k, i] * B_ij[k, i]

    # Convert to torch tensors.
    K = torch.tensor(K, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    abundance_H2 = torch.tensor(abundance_H2.squeeze(axis=(0, 1, 3)), dtype=torch.float32)
    d_ab = torch.tensor(d_ab, dtype=torch.float32)
    T = torch.tensor(T, dtype=torch.float32)
    dT = torch.tensor(dT, dtype=torch.float32)
    nu_ul = torch.tensor(nu_ul, dtype=torch.float32)
    A_ul = torch.tensor(A_ul, dtype=torch.float32)
    molecular_weight = torch.tensor(data['molecular_weight'], dtype=torch.float32)
    return {'K': K,
            'A': A,
            'abundance_H2': abundance_H2,
            'd_ab': d_ab,
            'T': T,
            'dT': dT,
            'num_levels': num_levels,
            'nu_ul': nu_ul,
            'A_ul': A_ul,
            'B_ul': B_ul,
            'B_lu': B_lu,
            'weight_upper': weight_upper,
            'weight_lower': weight_lower,
            'molecular_weight': molecular_weight}

def _compute_molecular_data(hyper: hp.Hyper) -> list[TracerData]:
    """
    Computes spectral data for each tracer molecule specified in the
    [hyperparameters object][iris.hyper.Hyper].

    Args:
        hyper: A hyperparameters object.

    Returns:
        A list of [`TracerData`][iris.chemistry.TracerData] dictionaries containing:

            * `K`
            * `A`
            * `abundance_H2`
            * `d_ab`
            * `T`
            * `dT`
            * `num_levels`
            * `nu_ul`
            * `A_ul`
            * `B_ul`
            * `B_lu`
            * `weight_upper`
            * `weight_lower`
            * `molecular_weight`
            * `transition`
            * `abundance_H2_steps`
            * `interpolation_max_T`
            * `T_steps`
    """
    chem_data = [_parse_lamda(path) for path in hyper.observer_hyper.chem_path]
    for i in range(hyper.observer_hyper.n_lines):
        chem_data[i]['transition'] = hyper.observer_hyper.transition[i]
        chem_data[i]['abundance_H2_steps'] = hyper.observer_hyper.abundance_H2_steps[i]
        chem_data[i]['interpolation_max_T'] = hyper.observer_hyper.interpolation_max_T[i]
        chem_data[i]['T_steps'] = hyper.observer_hyper.T_steps[i]
    molecular_data = [_compute_single_molecule(data, hyper) for data in chem_data]
    return molecular_data

def _make_population_grids(hyper: hp.Hyper,
                           units: str,
                           node_comm: mpi4py.MPI.Intracomm | None = None) -> tuple[list[TracerData], int | None]:
    r"""
    Solves the level populations of each tracer over a grid of
    total H number density, $\text{H}_2$ abundance, and temperature.

    Solves for the populations of all tracer levels over a grid of
    total H atom number density, $\text{H}_2$ abundance, and temperature,
    expressed per tracer abundance, where tracer abundance is expressed as a fraction of
    total H atom number density, as the system is linear with respect to this value.
    Then isolates the populations of the upper and lower energy levels of the line transition.
    Solving per abundance allows the abundance factor to be applied as the final step of
    [observability determination][iris.observation.ObservabilityProcessor.forward],
    which reduces the backpropagation overhead of
    [`SyntheticObserver`][iris.observation.SyntheticObserver] in
    [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
    Grid dimensions are configured as follows:

    * Gas mass density is constantly proportional to total number density of H atoms, which
    is bounded between `0` and `hyper.observer_hyper.interpolation_max_N_H_TOT`,
    with `hyper.observer_hyper.N_H_TOT_steps` spaced linearly in `arcsinh(N_H_TOT)` space.
    * $\text{H}_2$ abundance, expressed as a fraction of total H atom number density, is bounded
    between the absolute extrema `0, 0.5`, with `hyper.observer_hyper.abundance_H2_steps`
    linearly spaced steps.
    * Temperature is bounded between `0` and `hyper.observer_hyper.interpolation_max_T`
    with `hyper.observer_hyper.T_steps` linearly spaced steps.

    Note that all line emission and absorption will be linear (not constant) outside of these bounds.

    Args:
        hyper: A hyperparameters object.
        units: The units in which to compute population grids. One of `'iris', 'processing'`.
        node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support,
            if used during [dataset writing][iris.arepo_processing_write.Writer].

    Returns:
        A tuple `population_grids, gpu` containing a list of
            [`TracerData`][iris.chemistry.TracerData] dictionaries and a GPU access key, if available.
            Each `TracerData` dictionary contains:

            * `upper_population_per_abundance`
            * `lower_population_per_abundance`
            * `rho`
            * `dN_bolic`
            * `abundance_H2`
            * `d_ab`
            * `T`
            * `dT`
            * `nu_ul`
            * `A_ul`
            * `B_ul`
            * `B_lu`
            * `weight_upper`
            * `weight_lower`
            * `molecular_weight`

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """
    if units == 'iris':
        mass = hyper.dataset_hyper._mass_iris_per_SI
        time = hyper.dataset_hyper._time_iris_per_SI
        frequency = 1 / time
        length = hyper.dataset_hyper._length_iris_per_SI
        volume = length * length * length
        iris_number_unit = hyper.dataset_hyper.iris_number_unit
        number_density = 1 / (iris_number_unit * volume)
        temperature = hyper.dataset_hyper._temperature_iris_per_SI
    elif units == 'processing':
        length = 100 / hyper.writer_hyper.length_cm_per_processing
        volume = length * length * length
        iris_number_unit = hyper.dataset_hyper.iris_number_unit
        number_density = 1 / (iris_number_unit * volume)
        velocity = 100 / hyper.writer_hyper.velocity_cm_per_s_per_processing
        time = length / velocity
        frequency = 1 / time
        mass = 1000 / hyper.writer_hyper.mass_g_per_processing
        temperature = 1 / hyper.writer_hyper.temperature_K_per_processing
    else:
        raise ValueError("Invalid units provided to _make_population_grids. Must be 'iris' or 'processing'.")
    
    m_He = hyper.observer_hyper.m_He
    abundance_He = hyper.observer_hyper.abundance_He
    m_H = hyper.observer_hyper.m_H
    number_ism_molecular_mass = (m_H + abundance_He * m_He) * (mass * iris_number_unit)

    molecular_data = _compute_molecular_data(hyper)
    population_grids = []

    # If GPU support is available and node_comm is not None,
    # attempt to query the GPU manager and request the access key.
    gpu = None
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        if node_comm is not None:
            node_size = node_comm.Get_size()
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)
                device = f'cuda:{gpu}'

    # Compute population grids for each spectral line.
    for i in range(hyper.observer_hyper.n_lines):
        # Get tracer data.
        data = molecular_data[i]

        K = data['K'] / number_density / time
        A = data['A'] / time
        abundance_H2 = data['abundance_H2']
        d_ab = data['d_ab']
        T = data['T'] * temperature
        dT = data['dT'] * temperature
        nu_ul = data['nu_ul'] * frequency
        A_ul = data['A_ul'] / time
        B_ul = data['B_ul'] * time / mass
        B_lu = data['B_lu'] * time / mass
        weight_upper = data['weight_upper']
        weight_lower = data['weight_lower']
        num_levels = data['num_levels']
        upper_level = hyper.observer_hyper.transition[i][0]
        lower_level = hyper.observer_hyper.transition[i][1]

        # Make the N_H_TOT grid in arcsinh space.
        N_H_TOT_steps = hyper.observer_hyper.N_H_TOT_steps[i]
        interpolation_max_N_H_TOT = hyper.observer_hyper.interpolation_max_N_H_TOT[i]
        bolic_normalization = hyper.observer_hyper.bolic_normalization[i]
        max_bolic = torch.asinh(torch.tensor(interpolation_max_N_H_TOT / bolic_normalization))
        N_H_TOT_bolic = torch.linspace(0, max_bolic, N_H_TOT_steps, dtype=torch.float32)
        dN_bolic = max_bolic / (N_H_TOT_steps - 1)
        N_H_TOT = torch.sinh(N_H_TOT_bolic) * bolic_normalization * number_density
        rho = N_H_TOT * number_ism_molecular_mass

        # Chunk the level systems over N_H_TOT for manageable throughput.
        A[0, :] = torch.ones(A.shape[1], dtype=torch.float32)
        K = torch.permute(K, (2, 3, 0, 1)).contiguous()
        max_level_throughput_resolution = hyper.observer_hyper.max_level_throughput_resolution

        systems_per_density = abundance_H2.shape[0] * T.shape[0]
        total_systems = N_H_TOT_steps * systems_per_density
        max_systems_per_chunk = max(1, max_level_throughput_resolution // (num_levels * num_levels))
        chunk_systems = min(total_systems, max_systems_per_chunk)
        upper_population_per_abundance = torch.empty(total_systems, dtype=torch.float32, device=device)
        lower_population_per_abundance = torch.empty(total_systems, dtype=torch.float32, device=device)

        with torch.no_grad():
            for start in range(0, total_systems, chunk_systems):
                end = min(start + chunk_systems, total_systems)
                flat_indices = torch.arange(start, end, dtype=torch.int64)
                density_indices = torch.div(flat_indices, systems_per_density, rounding_mode='floor')
                rem = flat_indices.remainder(systems_per_density)
                abundance_indices = torch.div(rem, T.shape[0], rounding_mode='floor')
                temperature_indices = rem.remainder(T.shape[0])

                # Define the level systems.
                # (N_H_TOT * K + [1, A]) @ level_populations_per_abundance = [N_H_TOT, 0, 0, ...]
                # where tracer abundance is expressed per N_H_TOT.
                # See chemistry._compute_single_molecule for the definitions of K, A.
                # They are not the same as K_ij, A_ij.
                chunk_N_H_TOT = N_H_TOT[density_indices]
                equilibrium_matrices = (
                    chunk_N_H_TOT.unsqueeze(dim=-1).unsqueeze(dim=-1)
                    * K[abundance_indices, temperature_indices]
                    + A.unsqueeze(dim=0)).detach()
                constants = torch.zeros((end - start, num_levels), dtype=torch.float32)
                constants[:, 0] = chunk_N_H_TOT

                # Solve the level systems and isolate the upper and lower energy levels.
                # level_populations_per_abundance has dimensions
                # N_H_TOT_chunk * abundance_H2 * T, levels.
                equilibrium_matrices = equilibrium_matrices.to(device=device)
                constants = constants.to(device=device)

                level_populations_per_abundance = torch.linalg.solve(equilibrium_matrices, constants)
                upper_population_per_abundance[start:end] = level_populations_per_abundance[:, upper_level]
                lower_population_per_abundance[start:end] = level_populations_per_abundance[:, lower_level]

                del equilibrium_matrices, constants, level_populations_per_abundance

        # Reshape populations to N_H_TOT, abundance_H2, T.
        upper_population_per_abundance = upper_population_per_abundance.view(
            N_H_TOT_steps, abundance_H2.shape[0], T.shape[0])
        lower_population_per_abundance = lower_population_per_abundance.view(
            N_H_TOT_steps, abundance_H2.shape[0], T.shape[0])
        population_data = {'upper_population_per_abundance': upper_population_per_abundance,
                           'lower_population_per_abundance': lower_population_per_abundance,
                           'rho': rho,
                           'dN_bolic': dN_bolic,
                           'abundance_H2': abundance_H2,
                           'd_ab': d_ab,
                           'T': T,
                           'dT': dT,
                           'nu_ul': nu_ul,
                           'A_ul': A_ul,
                           'B_ul': B_ul,
                           'B_lu': B_lu,
                           'weight_upper': weight_upper,
                           'weight_lower': weight_lower,
                           'molecular_weight': data['molecular_weight']}
        population_grids.append(population_data)

    return population_grids, gpu

def make_observability_grids(hyper: hp.Hyper,
                             units: str,
                             node_comm: mpi4py.MPI.Intracomm | None = None) -> list[TracerData]:
    r"""
    Computes the emission and absorption coefficients of each tracer over a grid of
    total H number density, $\text{H}_2$ abundance, and temperature.

    Solves for the emission and absorption coefficients over a grid of
    gas density, $\text{H}_2$ abundance, and temperature,
    expressed per tracer abundance, where tracer abundance is expressed as a fraction
    of total H atom number density, as these quantities are linear with respect to this value.
    Solving per abundance allows the abundance factor to be applied as the final step of
    [observability determination][iris.observation.ObservabilityProcessor.forward],
    which reduces the backpropagation overhead of
    [`SyntheticObserver`][iris.observation.SyntheticObserver] in
    [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
    Grid dimensions are configured separately per spectral line as follows:

    * Gas mass density is constantly proportional to total number density of H atoms, which
    is bounded between `0` and
    [`interpolation_max_N_H_TOT[line]`][iris.hyper.ObserverHyper.interpolation_max_N_H_TOT],
    with [`N_H_TOT_steps[line]`][iris.hyper.ObserverHyper.N_H_TOT_steps] spaced linearly
    in `arcsinh(N_H_TOT)` space.
    * $\text{H}_2$ abundance, expressed as a fraction of total H atom number density, is bounded
    between the absolute extrema `0, 0.5`, with
    [`abundance_H2_steps[line]`][iris.hyper.ObserverHyper.abundance_H2_steps]
    linearly spaced steps.
    * Temperature is bounded between `0` and
    [`interpolation_max_T[line]`][iris.hyper.ObserverHyper.interpolation_max_T]
    with [`T_steps[line]`][iris.hyper.ObserverHyper.T_steps] linearly spaced steps.

    Note that all line emission and absorption will be linear (not constant) outside of these bounds.

    Args:
        hyper: A hyperparameters object.
        units: The units in which to compute population grids. One of `'iris', 'processing'`.
        node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support,
            if used during [dataset writing][iris.arepo_processing_write.Writer].

    Returns:
        A list of [`TracerData`][iris.chemistry.TracerData] dictionaries for each tracer, each containing:

            * `emission_factor`
            * `dj_drho`
            * `dj_d_ab`
            * `dj_dT`
            * `absorption_factor`
            * `d_alpha_drho`
            * `d_alpha_d_ab`
            * `d_alpha_dT`
            * `rho`
            * `dN_bolic`
            * `abundance_H2`
            * `d_ab`
            * `T`
            * `dT`
            * `nu_ul`
            * `molecular_weight`

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """
    if units == 'iris':
        mass = hyper.dataset_hyper._mass_iris_per_SI
        time = hyper.dataset_hyper._time_iris_per_SI
        length = hyper.dataset_hyper._length_iris_per_SI
        velocity = length / time
        acceleration = velocity / time
        force = mass * acceleration
        energy = force * length
        iris_number_unit = hyper.dataset_hyper.iris_number_unit
    elif units == 'processing':
        length = 100 / hyper.writer_hyper.length_cm_per_processing
        velocity = 100 / hyper.writer_hyper.velocity_cm_per_s_per_processing
        time = length / velocity
        mass = 1000 / hyper.writer_hyper.mass_g_per_processing
        acceleration = velocity / time
        force = mass * acceleration
        energy = force * length
        iris_number_unit = hyper.dataset_hyper.iris_number_unit
    else:
        raise ValueError("Invalid units provided to make_observability_grids. Must be 'iris' or 'processing'.")

    h = hyper.observer_hyper.h * (energy * time * iris_number_unit)
    
    population_grids, gpu = _make_population_grids(hyper=hyper, units=units, node_comm=node_comm)
    observability_grids = []

    # Compute emission and absorption coefficients for each spectral line.
    for i in range(hyper.observer_hyper.n_lines):
        # Get tracer data.
        population_data = population_grids[i]
        upper_population_per_abundance = population_data['upper_population_per_abundance']
        lower_population_per_abundance = population_data['lower_population_per_abundance']
        nu_ul = population_data['nu_ul']
        A_ul = population_data['A_ul']
        B_ul = population_data['B_ul']
        B_lu = population_data['B_lu']

        # Compute emission and absorption coefficients per tracer abundance.
        emission_factor = h * nu_ul / (4 * torch.pi) * upper_population_per_abundance * A_ul
        absorption_factor = h * nu_ul / (4 * torch.pi) * (
            lower_population_per_abundance * B_lu - upper_population_per_abundance * B_ul)

        # Clean up the GPU.
        del upper_population_per_abundance
        del lower_population_per_abundance
        del population_data['upper_population_per_abundance']
        del population_data['lower_population_per_abundance']

        # Compute the gradients of the emission and absorption grids for better interpolation/extrapolation.
        rho = population_data['rho']
        abundance_H2 = population_data['abundance_H2']
        T = population_data['T']
        if torch.cuda.is_available():
            coordinates = (rho.cuda(gpu), abundance_H2.cuda(gpu), T.cuda(gpu))
        else:
            coordinates = (rho, abundance_H2, T)
        dj_drho, dj_d_ab, dj_dT = torch.gradient(emission_factor, spacing=coordinates)
        d_alpha_drho, d_alpha_d_ab, d_alpha_dT = torch.gradient(absorption_factor, spacing=coordinates)
        # Clean up the GPU.
        del coordinates

        emission_factor = emission_factor.detach().cpu()
        dj_drho = dj_drho.detach().cpu()
        dj_d_ab = dj_d_ab.detach().cpu()
        dj_dT = dj_dT.detach().cpu()
        absorption_factor = absorption_factor.detach().cpu()
        d_alpha_drho = d_alpha_drho.detach().cpu()
        d_alpha_d_ab = d_alpha_d_ab.detach().cpu()
        d_alpha_dT = d_alpha_dT.detach().cpu()

        data = {'emission_factor': emission_factor,
                'dj_drho': dj_drho,
                'dj_d_ab': dj_d_ab,
                'dj_dT': dj_dT,
                'absorption_factor': absorption_factor,
                'd_alpha_drho': d_alpha_drho,
                'd_alpha_d_ab': d_alpha_d_ab,
                'd_alpha_dT': d_alpha_dT,
                'rho': rho,
                'dN_bolic': population_data['dN_bolic'],
                'abundance_H2': abundance_H2,
                'd_ab': population_data['d_ab'],
                'T': T,
                'dT': population_data['dT'],
                'nu_ul': population_data['nu_ul'],
                'molecular_weight': population_data['molecular_weight']}
        observability_grids.append(data)

    # Clean-up GPU and return key to the GPU manager with usage statistics.
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        if gpu is not None:
            gpu_manager = node_comm.Get_size() - 1
            memory_usage = int(torch.cuda.max_memory_allocated(gpu))
            node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)

    return observability_grids


class TracerData(typing.TypedDict, total=False):
    r"""
    A container for a variety of chemical and spectral data.

    Attributes:
        molecular_weight: The molecular weight of the species in g/mol.
        num_levels: The number of distinct quantum energy levels for which data is computed
            (however many levels are recorded in the LAMDA .dat file).
        energies: The energies of each level (`shape=[num_levels,]`, recorded in LAMDA in
            $\text{cm}^{-1}$, later converted to J).
        weights: The statistical weights (integer-valued, stored as `np.float32`)
            of each level (`shape=[num_levels,]`).
        A_ij: The spontaneous emission (Einstein $A$) coefficients (`np.float32`)
            of each transition, if applicable (`shape=[num_levels, num_levels]`).
        nu: The spectral frequencies (`np.float32`) of each level transition, if applicable
            (`shape=[num_levels, num_levels]`).
        K_ij_H: De-excitational collision rates of the tracer with atomic H,
            if recorded in the LAMDA file (per number density tracer, per number density H,
            `np.float32`, `shape=[num_levels, num_levels, len(T_H)]`).
        T_H: The temperatures over which `K_ij_H` is specified in the LAMDA .dat file.
        K_ij_H2: De-excitational collision rates of the tracer with molecular $\text{H}_2$
            (both ortho- and para-, if specified as a total average in the LAMDA .dat file,
            per number density tracer, per number density $\text{H}_2$,
            `np.float32`, `shape=[num_levels, num_levels, len(T_H2)]`).
        T_H2: The temperatures over which `K_ij_H2` is specified in the LAMDA .dat file.
        K_ij_para_H2: De-excitational collision rates of the tracer with para-$\text{H}_2$,
            if recorded in the LAMDA file (per number density tracer, per number density $\text{p-H}_2$,
            `np.float32`, `shape=[num_levels, num_levels, len(T_para_H2)]`).
        T_para_H2: The temperatures over which `K_ij_para_H2` is specified in the LAMDA .dat file.
        K_ij_ortho_H2: De-excitational collision rates of the tracer with ortho-$\text{H}_2$,
            if recorded in the LAMDA file (per number density tracer, per number density $\text{o-H}_2$,
            `np.float32`, `shape=[num_levels, num_levels, len(T_ortho_H2)]`).
        T_ortho_H2: The temperatures over which `K_ij_ortho_H2` is specified in the LAMDA .dat file.
        K_ij_He: De-excitational collision rates of the tracer with He,
            if recorded in the LAMDA file (per number density tracer, per number density He,
            `np.float32`, `shape=[num_levels, num_levels, len(T_He)]`).
        T_He: The temperatures over which `K_ij_He` is specified in the LAMDA .dat file.

        K: The collisional contribution to the level equilibrium system (per total H atom,
            i.e. all-partner, number density, per tracer number density,
            `torch.float32`, `shape=[num_levels, num_levels, len(abundance_H2), len(T)]`).
        A: The spontaneous emission contribution to the level equilibrium system 
            (per tracer number density, `torch.float32`, `shape=[num_levels, num_levels]`).
        abundance_H2_steps: Number of steps in the $\text{H}_2$ abundance grid.
        abundance_H2: The $\text{H}_2$ abundance grid values.
        d_ab: Step size of the $\text{H}_2$ abundance grid.
        interpolation_max_T: Peak temperature at which to interpolate collision rates.
            Emission and absorption are linear above this point.
        T_steps: Number of steps in the temperature grid.
        T: The temperature grid values.
        dT: Step size of the temperature grid.
        transition: A tuple of `upper_energy_level, lower_energy_level`. The ground state is
            indexed as 0.
        nu_ul: The spectral frequency of the line transition.
        A_ul: The spontaneous emission (Einstein $A$) coefficient of the line transition.
        B_ul: The stimulated emission (Einstein $B_{ul}$) coefficient of the line transition.
        B_lu: The absorption (Einstein $B_{lu}$) coefficient of the line transition.
        weight_upper: The statistical weight of the upper energy level of the line transition.
        weight_lower: The statistical weight of the lower energy level of the line transition.

        upper_population_per_abundance: The population of the upper level of the line transition,
            solved over a grid of total H atom number density, $\text{H}_2$ abundance, and temperature,
            expressed per tracer abundance, where tracer abundance is expressed as a fraction of
            total H atom number density, as the system is linear with respect to this value
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
            Solving per abundance allows the abundance factor to be applied as the final step of
            [observability determination][iris.observation.ObservabilityProcessor.forward],
            which reduces the backpropagation overhead of
            [`SyntheticObserver`][iris.observation.SyntheticObserver] in
            [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
        lower_population_per_abundance: The population of the lower level of the line transition,
            solved over a grid of total H atom number density, $\text{H}_2$ abundance, and temperature,
            expressed per tracer abundance, where tracer abundance is expressed as a fraction of
            total H atom number density, as the system is linear with respect to this value
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
            Solving per abundance allows the abundance factor to be applied as the final step of
            [observability determination][iris.observation.ObservabilityProcessor.forward],
            which reduces the backpropagation overhead of
            [`SyntheticObserver`][iris.observation.SyntheticObserver] in
            [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
        rho: The density grid values.
        dN_bolic: The step size of the total H atom number density grid, corresponding to the
            density grid, in the arc-hyperbolic-sine space in which this grid is linearly spaced.

        emission_factor: The emission coefficient of the line transition,
            solved over a grid of total H atom number density, $\text{H}_2$ abundance, and temperature,
            expressed per tracer abundance, where tracer abundance is expressed as a fraction of
            total H atom number density, as this quantity is linear with respect to this value
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
            Solving per abundance allows the abundance factor to be applied as the final step of
            [observability determination][iris.observation.ObservabilityProcessor.forward],
            which reduces the backpropagation overhead of
            [`SyntheticObserver`][iris.observation.SyntheticObserver] in
            [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
        dj_drho: The partial derivative of `self.emission_factor` with respect to `self.rho`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
        dj_d_ab: The partial derivative of `self.emission_factor` with respect to `self.abundance_H2`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
        dj_dT: The partial derivative of `self.emission_factor` with respect to `self.T`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
        absorption_factor: The absorption (and stimulated emission) coefficient of the line transition,
            solved over a grid of total H atom number density, $\text{H}_2$ abundance, and temperature,
            expressed per tracer abundance, where tracer abundance is expressed as a fraction of
            total H atom number density, as this quantity is linear with respect to this value
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
            Solving per abundance allows the abundance factor to be applied as the final step of
            [observability determination][iris.observation.ObservabilityProcessor.forward],
            which reduces the backpropagation overhead of
            [`SyntheticObserver`][iris.observation.SyntheticObserver] in
            [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance].
        d_alpha_drho: The partial derivative of `self.absorption_factor` with respect to `self.rho`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
        d_alpha_d_ab: The partial derivative of `self.absorption_factor` with respect to `self.abundance_H2`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
        d_alpha_dT: The partial derivative of `self.absorption_factor` with respect to `self.T`
            (`torch.float32`, `shape=[len(rho), len(abundance_H2), len(T)]`).
    """
    molecular_weight: float | torch.Tensor
    num_levels: int
    energies: np.ndarray
    weights: np.ndarray
    A_ij: np.ndarray
    nu: np.ndarray
    K_ij_H: np.ndarray | None
    T_H: np.ndarray | None
    K_ij_H2: np.ndarray | None
    T_H2: np.ndarray | None
    K_ij_para_H2: np.ndarray | None
    T_para_H2: np.ndarray | None
    K_ij_ortho_H2: np.ndarray | None
    T_ortho_H2: np.ndarray | None
    K_ij_He: np.ndarray | None
    T_He: np.ndarray | None

    K: torch.Tensor
    A: torch.Tensor
    abundance_H2_steps: int
    abundance_H2: torch.Tensor
    d_ab: torch.Tensor
    interpolation_max_T: float
    T_steps: int
    T: torch.Tensor
    dT: torch.Tensor
    transition: tuple[int, int]
    nu_ul: torch.Tensor
    A_ul: torch.Tensor
    B_ul: torch.Tensor
    B_lu: torch.Tensor
    weight_upper: torch.Tensor
    weight_lower: torch.Tensor

    upper_population_per_abundance: torch.Tensor
    lower_population_per_abundance: torch.Tensor
    rho: torch.Tensor
    dN_bolic: torch.Tensor

    emission_factor: torch.Tensor
    dj_drho: torch.Tensor
    dj_d_ab: torch.Tensor
    dj_dT: torch.Tensor
    absorption_factor: torch.Tensor
    d_alpha_drho: torch.Tensor
    d_alpha_d_ab: torch.Tensor
    d_alpha_dT: torch.Tensor
