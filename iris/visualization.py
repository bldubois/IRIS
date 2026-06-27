# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
All figure code for the IRIS paper (DuBois et al., 2026).

See `jobs/` for the actual figure executables.
"""

from __future__ import annotations

import os.path
import typing
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib import transforms
import matplotlib.patches as patches
import torch
import numpy as np
import json

if typing.TYPE_CHECKING:
    import hyper as hp
from . import arepo_processing as ap
from . import cube_processing
from . import observation as ob


ArrayLike = np.ndarray | torch.Tensor
PathLike = str | os.PathLike[str]

def cmz_overview(hyper: hp.Hyper, path: PathLike = 'cmz_overview.png') -> None:
    r_steps = hyper.coordinate_hyper.r_steps
    lon_steps = hyper.coordinate_hyper.lon_steps
    true_cube = cube_processing.make_default_cube(hyper, units='T K', numpy=True, verbose=True)[0][0]

    fig = plt.figure(figsize=(15, 3.5))
    fig.subplots_adjust(left=0.02,
                        right=0.92,
                        top=0.92,
                        bottom=0.15)
    gridspec = fig.add_gridspec(nrows=1,
                                ncols=7,
                                width_ratios=[0.2490, 0.0843, 0.3142, 0.0192, 0.3142, 0.0077, 0.0114],
                                wspace=0.0)
    observed_spec = gridspec[0, 2].subgridspec(nrows=2,
                                               ncols=1,
                                               height_ratios=[.43, .57],
                                               hspace=0.09)
    sedigism_spec = gridspec[0, 4].subgridspec(nrows=2,
                                               ncols=1,
                                               height_ratios=[.43, .57],
                                               hspace=0.09)
    sedigism_cbar_spec = gridspec[0, 6].subgridspec(nrows=1, ncols=1)

    top_down(density=np.zeros((r_steps, lon_steps), dtype=np.float32),
             ax=fig.add_subplot(gridspec[0, 0], projection='polar'),
             hyper=hyper,
             label='Top-Down Orbital Model\n(Prior Studies)',
             color_scale=1,
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=True,
             r_label=r'$r$ (kpc to sun)',
             orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
             orbit_opacity=1.0)

    lb(observed=true_cube,
       ax=fig.add_subplot(observed_spec[0, 0]),
       hyper=hyper,
       title='Observation-Orbit Overlay',
       color_bar=False,
       color_scale=5.0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=False,
       b_ticks=True,
       orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
       orbit_opacity=0.6)
    lv(observed=true_cube,
       ax=fig.add_subplot(observed_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=5.0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=True,
       v_ticks=True,
       orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
       orbit_opacity=0.6)

    lb(observed=true_cube,
       ax=fig.add_subplot(sedigism_spec[0, 0]),
       hyper=hyper,
       title=r'SEDIGISM $^{13}$CO(2-1) Data',
       color_bar=True,
       color_scale=5.0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=fig.add_subplot(sedigism_cbar_spec[0, 0]),
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='vertical',
       cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
       l_ticks=False,
       b_ticks=False)
    lv(observed=true_cube,
       ax=fig.add_subplot(sedigism_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=5.0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=True,
       v_ticks=False)

    fig.savefig(os.path.expanduser(path))
    return

def sims_overview(snapshot_path: PathLike,
                  dataset: ap.Dataset | ap.ConcatDataset,
                  hyper: hp.Hyper | None = None,
                  path: PathLike = 'sims_overview.png') -> None:
    if hyper is None:
        hyper = dataset.hyper

    simple = ob.IteratedSimpleObserver(hyper=hyper)
    simple.cuda()
    simple.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    simply_observed = simple(arepo, units='vrho SI')
    simply_observed *= density_conversion
    simply_observed = simply_observed.detach().cpu().numpy()[0][0]

    fig = plt.figure(figsize=(15.55, 3.5))
    fig.subplots_adjust(left=0.02,
                        right=0.92,
                        top=0.92,
                        bottom=0.15)
    gridspec = fig.add_gridspec(nrows=1,
                                ncols=11,
                                width_ratios=[
    0.0121, 0.0646, 0.2623, 0.0000, 0.0121, 0.0274, 0.2623, 0.0081, 0.3309, 0.0081, 0.0121],
                                wspace=0.0)
    wide_top_down_cbar_spec = gridspec[0, 0]
    wide_top_down_spec = gridspec[0, 2]
    top_down_cbar_spec = gridspec[0, 4]
    top_down_spec = gridspec[0, 6]
    simple_spec = gridspec[0, 8].subgridspec(nrows=2,
                                             ncols=1,
                                             height_ratios=[.43, .57],
                                             hspace=0.09)
    simple_cbar_spec = gridspec[0, 10]

    wide_top_down(snapshot_path=snapshot_path,
                  ax=fig.add_subplot(wide_top_down_spec),
                  hyper=hyper,
                  label=r'H$_2$ $z$-Column Density' + '\n(AREPO)',
                  color_bar=True,
                  cax=fig.add_subplot(wide_top_down_cbar_spec),
                  cbar_label=r'H$_2$ Column Density ($M_\odot / \text{pc}^2$)',
                  x_ticks=True,
                  y_ticks=True)

    top_down(density=arepo_top_down,
             ax=fig.add_subplot(top_down_spec, projection='polar'),
             hyper=hyper,
             label=r'H$_2$ Latitude-Mean Density' + '\n(AREPO CMZ)',
             color_bar=True,
             cax=fig.add_subplot(top_down_cbar_spec),
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             l_ticks=True,
             r_ticks=True)

    lb(observed=simply_observed,
       ax=fig.add_subplot(simple_spec[0, 0]),
       hyper=hyper,
       title='Density-Tracing (IRIS)',
       color_bar=True,
       color_scale=None,
       pin_color_scale_to=None,
       cax=fig.add_subplot(simple_cbar_spec),
       cbar_label='Mean Column Density\n' +  r'Per Unit Velocity ($M_\odot \, \text{s} / \text{pc}^3$)',
       cbar_orientation='vertical',
       l_ticks=False,
       b_ticks=False)
    lv(observed=simply_observed,
       ax=fig.add_subplot(simple_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=None,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=True,
       v_ticks=False)

    fig.savefig(os.path.expanduser(path))
    return

def side_by_side(arepo_top_down: ArrayLike,
                 left_cube: ArrayLike,
                 right_cube: ArrayLike,
                 left_name: str,
                 right_name: str,
                 hyper: hp.Hyper,
                 figsize: tuple[float, float] = (15.8355, 3.5),
                 subplots_adjust: tuple[float, float, float, float] = (.02, .92, .92, .15),
                 grid_spec_width_ratios: tuple[float, ...] = (
    0.0106, 0.0425, 0.2297, 0.0757, 0.3056, 0.0000, 0.0000, 0.0126, 0.3056, 0.0071, 0.0106),
                 grid_spec_height_ratios: tuple[float, float] = (.43, .57),
                 wspace: float = 0.0,
                 hspace: float = 0.09,
                 observer_arrow_offset: float = .20,
                 color_scale: float | None = None,
                 color_scale_center: float = 0,
                 color_norm_min: float | None = None,
                 color_norm_max: float | None = None,
                 cbar_ticks: typing.Sequence[float] | None = None,
                 T_threshold: float = 5e-2,
                 path: PathLike = 'side_by_side.png') -> None:
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=subplots_adjust[0],
                        right=subplots_adjust[1],
                        top=subplots_adjust[2],
                        bottom=subplots_adjust[3])
    gridspec = fig.add_gridspec(nrows=1,
                                ncols=11,
                                width_ratios=grid_spec_width_ratios,
                                wspace=wspace)
    top_down_cbar_spec = gridspec[0, 0]
    top_down_spec = gridspec[0, 2]
    left_spec = gridspec[0, 4].subgridspec(nrows=2,
                                           ncols=1,
                                           height_ratios=grid_spec_height_ratios,
                                           hspace=hspace)
    right_spec = gridspec[0, 8].subgridspec(nrows=2,
                                            ncols=1,
                                            height_ratios=grid_spec_height_ratios,
                                            hspace=hspace)
    right_cbar_spec = gridspec[0, 10]

    top_down(density=arepo_top_down,
             ax=fig.add_subplot(top_down_spec, projection='polar'),
             hyper=hyper,
             label='Top-Down Density\n(AREPO)',
             color_bar=True,
             cax=fig.add_subplot(top_down_cbar_spec),
             cbar_label=r'H$_2$ Density ($\text{kg} / \text{m}^3$)',
             observer_arrow_offset=observer_arrow_offset,
             l_ticks=True,
             r_ticks=True,
             r_label=r'$r$ (kpc to observer)')

    pin_color_scale_to = np.stack((left_cube, right_cube), axis=-1)
    lb(observed=left_cube,
       ax=fig.add_subplot(left_spec[0, 0]),
       hyper=hyper,
       title=r'Synthetic $^{13}$CO(2-1) ' + f'({left_name})',
       color_bar=False,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       pin_color_scale_to=pin_color_scale_to,
       cax=None,
       l_ticks=False,
       b_ticks=True)
    lv(observed=left_cube,
       ax=fig.add_subplot(left_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       pin_color_scale_to=pin_color_scale_to,
       cax=None,
       l_ticks=True,
       v_ticks=True)

    lb(observed=right_cube,
       ax=fig.add_subplot(right_spec[0, 0]),
       hyper=hyper,
       title=r'Synthetic $^{13}$CO(2-1) ' + f'({right_name})',
       color_bar=True,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       pin_color_scale_to=pin_color_scale_to,
       cax=fig.add_subplot(right_cbar_spec),
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='vertical',
       cbar_ticks=cbar_ticks,
       l_ticks=False,
       b_ticks=False)
    lv(observed=right_cube,
       ax=fig.add_subplot(right_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       pin_color_scale_to=pin_color_scale_to,
       cax=None,
       l_ticks=True,
       v_ticks=False)

    fig.savefig(os.path.expanduser(path))

    print(f'Threshold Value:\t{T_threshold:.4g} K')
    active_voxels = np.minimum(np.abs(left_cube), np.abs(right_cube)) > T_threshold
    error = 2 * np.abs(left_cube - right_cube) / (np.abs(left_cube) + np.abs(right_cube))
    error = error[active_voxels]

    print(f'Mean TSRE lvb:\t{error.mean():.4f}')
    print(f'{left_name} TAM lvb:\t{left_cube[left_cube > T_threshold].mean():.4f}\t\tMax:\t{left_cube.max():.4f}')
    print(f'{right_name} TAM lvb:\t{right_cube[right_cube > T_threshold].mean():.4f}\t\tMax:\t{right_cube.max():.4f}')

    reduction = hyper.cube_hyper.reduction
    if reduction == 'mean':
        left_lv = mean_intensity(left_cube, dim=1)
        right_lv = mean_intensity(right_cube, dim=1)
    elif reduction == 'max':
        left_lv = peak_intensity(left_cube, dim=1)
        right_lv = peak_intensity(right_cube, dim=1)
    else:
        raise ValueError("Invalid cube reduction specified; "
                         "hyper.cube_hyper.reduction must be one of 'mean' or 'max'.")

    active_pixels = np.minimum(np.abs(left_lv), np.abs(right_lv)) > T_threshold
    error = 2 * np.abs(left_lv - right_lv) / (np.abs(left_lv) + np.abs(right_lv))
    error = error[active_pixels]

    print(f'Mean TSRE lv:\t{error.mean():.4f}', flush=True)
    print(f'{left_name} TAM lv:\t{left_lv[left_lv > T_threshold].mean():.4f}\t\tMax:\t{left_lv.max():.4f}')
    print(f'{right_name} TAM lv:\t{right_lv[right_lv > T_threshold].mean():.4f}\t\tMax:\t{right_lv.max():.4f}')
    return

def iris_side_by_side(dataset: ap.Dataset | ap.ConcatDataset,
                      observer: ob.Observer,
                      hyper: hp.Hyper | None = None,
                      color_scale: float | None = 2e2,
                      color_scale_center: float = 0,
                      color_norm_min: float | None = 0,
                      color_norm_max: float | None = 1.2,
                      cbar_ticks: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0),
                      external_name: str = 'EXTERNAL',
                      path: PathLike = 'iris_side_by_side.png') -> None:
    if hyper is None:
        hyper = dataset.hyper
    observer.cuda()
    observer.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    external_observed = cube_processing.make_default_cube(hyper, units='T K')
    external_observed = external_observed.detach().cpu().numpy()[0][0]
    iris_observed = observer(arepo, units='Trj K')
    iris_observed = iris_observed.detach().cpu().numpy()[0][0]

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=external_observed,
                 right_cube=iris_observed,
                 left_name=external_name,
                 right_name='IRIS',
                 hyper=hyper,
                 figsize=(15, 3.8),
                 grid_spec_width_ratios=(
    0.0126, 0.0021, 0.2737, 0.0379, 0.3158, 0.0000, 0.0000, 0.0211, 0.3158, 0.0084, 0.0126),
                 grid_spec_height_ratios=(.5, .5),
                 observer_arrow_offset=0.25,
                 color_scale=color_scale,
                 color_scale_center=color_scale_center,
                 color_norm_min=color_norm_min,
                 color_norm_max=color_norm_max,
                 cbar_ticks=cbar_ticks,
                 path=path)
    return

def external_side_by_side(dataset: ap.Dataset | ap.ConcatDataset,
                          left_hyper: hp.Hyper,
                          right_hyper: hp.Hyper,
                          left_name: str,
                          right_name: str,
                          grid_spec_height_ratios: tuple[float, float] = (.43, .57),
                          color_scale: float = 2e2,
                          color_scale_center: float = 0,
                          color_norm_min: float | None = 0,
                          color_norm_max: float | None = 5.0,
                          cbar_ticks: typing.Sequence[float] | None = (0.0, 0.1, 1.0, 2.5, 5.0),
                          path: PathLike = 'external_side_by_side.png') -> None:
    arepo = dataset.sample(1, validation=False)
    mass = left_hyper.dataset_hyper._mass_iris_per_SI
    length = left_hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = left_hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, left_hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    left_cube = cube_processing.make_default_cube(left_hyper, units='T K')
    left_cube = left_cube.detach().cpu().numpy()[0][0]
    right_cube = cube_processing.make_default_cube(right_hyper, units='T K')
    right_cube = right_cube.detach().cpu().numpy()[0][0]

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=left_cube,
                 right_cube=right_cube,
                 left_name=left_name,
                 right_name=right_name,
                 hyper=left_hyper,
                 grid_spec_height_ratios=grid_spec_height_ratios,
                 color_scale=color_scale,
                 color_scale_center=color_scale_center,
                 color_norm_min=color_norm_min,
                 color_norm_max=color_norm_max,
                 cbar_ticks=cbar_ticks,
                 path=path)
    return

def optically_thin_vs_thick(dataset: ap.Dataset | ap.ConcatDataset,
                            hyper: hp.Hyper | None = None,
                            plus_dust: bool = False,
                            path: PathLike = 'optically_thin_vs_thick.png') -> None:
    if hyper is None:
        hyper = dataset.hyper
    if not plus_dust:
        hyper.observer_hyper.T_cmb = 0.
        hyper.observer_hyper.kappa_dust = [0.0]

    observer = ob.IteratedSyntheticObserver(hyper=hyper)
    observer.cuda()
    observer.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    thin_observed = observer(arepo, units='Trj K', transfer='optically thin')
    thin_observed = thin_observed.detach().cpu().numpy()[0][0]
    thick_observed = observer(arepo, units='Trj K', transfer='optically thick')
    thick_observed = thick_observed.detach().cpu().numpy()[0][0]

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=thin_observed,
                 right_cube=thick_observed,
                 left_name='Optically Thin',
                 right_name='Optically Thick + Dust' if plus_dust else 'Optically Thick',
                 hyper=hyper,
                 observer_arrow_offset=0.20,
                 color_scale=2e2,
                 color_scale_center=0,
                 cbar_ticks=(0.0, 0.1, 1.0, 10.0, 100.0),
                 path=path)
    return

def no_dust_vs_dust(dataset: ap.Dataset | ap.ConcatDataset,
                    hyper: hp.Hyper | None = None,
                    dust_name: str = 'With Dust',
                    color_scale=1e2,
                    color_scale_center=0,
                    color_norm_min=0,
                    color_norm_max=5.0,
                    cbar_ticks=(0.0, 0.1, 1.0, 2.5, 5.0),
                    path: PathLike = 'optically_thin_vs_thick.png') -> None:
    if hyper is None:
        hyper = dataset.hyper

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]

    observer_dust = ob.IteratedSyntheticObserver(hyper=hyper)
    observer_dust.cuda()
    observer_dust.eval()
    observed_dust = observer_dust(arepo, units='Trj K')
    observed_dust = observed_dust.detach().cpu().numpy()[0][0]
    observer_dust.cpu()
    del observer_dust

    hyper.observer_hyper.kappa_dust = [0.0]
    observer_no_dust = ob.IteratedSyntheticObserver(hyper=hyper)
    observer_no_dust.cuda()
    observer_no_dust.eval()
    observed_no_dust = observer_no_dust(arepo, units='Trj K')
    observed_no_dust = observed_no_dust.detach().cpu().numpy()[0][0]
    observer_no_dust.cpu()
    del observer_no_dust

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=observed_no_dust,
                 right_cube=observed_dust,
                 left_name='No Dust',
                 right_name=dust_name,
                 hyper=hyper,
                 observer_arrow_offset=0.20,
                 color_scale=color_scale,
                 color_scale_center=color_scale_center,
                 color_norm_min=color_norm_min,
                 color_norm_max=color_norm_max,
                 cbar_ticks=cbar_ticks,
                 path=path)
    return

def balance_OT_background(dataset: ap.Dataset | ap.ConcatDataset,
                          hyper: hp.Hyper | None = None,
                          left_background: float = 0,
                          right_background: float = 2.73,
                          plus_dust: bool = True,
                          path: PathLike = 'balance_OT_vs_FEP_CMB.png') -> None:
    if hyper is None:
        hyper = dataset.hyper
    if not plus_dust:
        hyper.observer_hyper.kappa_dust = [0.0]

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]

    hyper.observer_hyper.T_continuum = left_background
    observer_left = ob.IteratedSyntheticObserver(hyper=hyper)
    observer_left.cuda()
    observer_left.eval()
    observed_left = observer_left(arepo, units='Trj K')
    observed_left = observed_left.detach().cpu().numpy()[0][0]
    observer_left.cpu()
    del observer_left

    hyper.observer_hyper.T_continuum = right_background
    observer_right = ob.IteratedSyntheticObserver(hyper=hyper)
    observer_right.cuda()
    observer_right.eval()
    observed_right = observer_right(arepo, units='Trj K')
    observed_right = observed_right.detach().cpu().numpy()[0][0]
    observer_right.cpu()
    del observer_right

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=observed_left,
                 right_cube=observed_right,
                 left_name=f'OT-{left_background}K Balance',
                 right_name=f'OT-{right_background}K Balance',
                 hyper=hyper,
                 observer_arrow_offset=0.20,
                 color_scale=2e2,
                 color_scale_center=0,
                 color_norm_min=0,
                 color_norm_max=5.0,
                 cbar_ticks=(0.0, 0.1, 1.0, 2.5, 5.0),
                 path=path)
    return

def formal_vs_smooth(dataset: ap.Dataset | ap.ConcatDataset,
                     hyper: hp.Hyper | None = None,
                     plus_dust: bool = True,
                     path: PathLike = 'balance_OT_vs_FEP_CMB.png') -> None:
    if hyper is None:
        hyper = dataset.hyper
    if not plus_dust:
        hyper.observer_hyper.kappa_dust = [0.0]

    observer = ob.IteratedSyntheticObserver(hyper=hyper)
    observer.cuda()
    observer.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    formal_observed = observer(arepo, units='Trj K', integration='formal')
    formal_observed = formal_observed.detach().cpu().numpy()[0][0]
    smooth_observed = observer(arepo, units='Trj K', integration='smooth')
    smooth_observed = smooth_observed.detach().cpu().numpy()[0][0]

    side_by_side(arepo_top_down=arepo_top_down,
                 left_cube=formal_observed,
                 right_cube=smooth_observed,
                 left_name='Formal Integration',
                 right_name='Smooth Integration',
                 hyper=hyper,
                 observer_arrow_offset=0.20,
                 color_scale=2e2,
                 color_scale_center=0,
                 color_norm_min=0,
                 color_norm_max=5.0,
                 cbar_ticks=(0.0, 0.1, 1.0, 2.5, 5.0),
                 path=path)
    return

def continuum_temperature(dataset: ap.Dataset | ap.ConcatDataset,
                          hyper: hp.Hyper | None = None,
                          path: PathLike = 'continuum_temperature.png') -> None:
    if hyper is None:
        hyper = dataset.hyper

    observer = ob.IteratedSyntheticObserver(hyper=hyper)
    observer.cuda()
    observer.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    continuum = observer(arepo, units='Tb K', subtraction='continuum')
    continuum = continuum.detach().cpu().numpy()[0][0]

    fig = plt.figure(figsize=(18.0087, 3.5))
    fig.subplots_adjust(left=0.02,
                        right=0.92,
                        top=0.92,
                        bottom=0.15)
    gridspec = fig.add_gridspec(nrows=1,
                                ncols=7,
                                width_ratios=[
    0.0093, 0.0374, 0.2020, 0.0666, 0.6692, 0.0062, 0.0093],
                                wspace=0.0)
    top_down_cbar_spec = gridspec[0, 0]
    top_down_spec = gridspec[0, 2]
    continuum_spec = gridspec[0, 4]
    continuum_cbar_spec = gridspec[0, 6]

    top_down(density=arepo_top_down,
             ax=fig.add_subplot(top_down_spec, projection='polar'),
             hyper=hyper,
             label='Top-Down Density\n(AREPO)',
             color_bar=True,
             cax=fig.add_subplot(top_down_cbar_spec),
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             observer_arrow_offset=.20,
             l_ticks=True,
             r_ticks=True,
             r_label=r'$r$ (kpc to observer)')

    lb(observed=continuum,
       ax=fig.add_subplot(continuum_spec),
       hyper=hyper,
       title='Continuum Brightness Temperature (IRIS)',
       color_bar=True,
       color_scale=1e2,
       color_scale_center=2.7,
       color_norm_min=2.73,
       color_norm_max=3.2,
       pin_color_scale_to=None,
       cax=fig.add_subplot(continuum_cbar_spec),
       cbar_label='Brightness Temperature (K)',
       cbar_orientation='vertical',
       cbar_ticks=(2.73, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0),
       l_ticks=True,
       b_ticks=True)

    continuum_lb = continuum.mean(axis=2)
    T_threshold = 2.74
    active_pixels = continuum_lb > T_threshold
    print(f"Continuum Min:\t{continuum_lb.min():.4f}\tMax:\t{continuum_lb.max():.4f}")
    print(f"Continuum Mean:\t{continuum_lb.mean():.4f}\t"
          f"Threshold-Mean:\t{continuum_lb[active_pixels].mean():.4f}", flush=True)

    fig.savefig(os.path.expanduser(path))
    return

def simple_vs_synth(dataset: ap.Dataset | ap.ConcatDataset,
                    observer: ob.Observer,
                    hyper: hp.Hyper | None = None,
                    color_scale=2e2,
                    color_scale_center=0,
                    color_norm_min=0,
                    color_norm_max=5.0,
                    cbar_ticks=(0.0, 0.1, 1.0, 2.5, 5.0),
                    path: PathLike = 'simple_vs_synth.png') -> None:
    if hyper is None:
        hyper = dataset.hyper

    simple = ob.IteratedSimpleObserver(hyper=hyper)
    simple.cuda()
    simple.eval()
    observer.cuda()
    observer.eval()

    arepo = dataset.sample(1, validation=False).cuda()
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    arepo_top_down = ap.columnize_physical_tensor(arepo, hyper) / density
    arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0]
    simply_observed = simple(arepo, units='vrho SI')
    simply_observed *= density_conversion
    simply_observed = simply_observed.detach().cpu().numpy()[0][0]
    synth_observed = observer(arepo, units='Trj K')
    synth_observed = synth_observed.detach().cpu().numpy()[0][0]

    fig = plt.figure(figsize=(17.18, 3.5))
    fig.subplots_adjust(left=0.02,
                        right=0.92,
                        top=0.92,
                        bottom=0.15)
    gridspec = fig.add_gridspec(nrows=1,
                                ncols=11,
                                width_ratios=[
    0.0098, 0.0392, 0.2117, 0.0698, 0.2817, 0.0064, 0.0098, 0.0737, 0.2817, 0.0065, 0.0097],
                                wspace=0.0)
    top_down_cbar_spec = gridspec[0, 0]
    top_down_spec = gridspec[0, 2]
    simple_spec = gridspec[0, 4].subgridspec(nrows=2,
                                             ncols=1,
                                             height_ratios=[.43, .57],
                                             hspace=0.09)
    simple_cbar_spec = gridspec[0, 6]
    synth_spec = gridspec[0, 8].subgridspec(nrows=2,
                                            ncols=1,
                                            height_ratios=[.43, .57],
                                            hspace=0.09)
    synth_cbar_spec = gridspec[0, 10]

    top_down(density=arepo_top_down,
             ax=fig.add_subplot(top_down_spec, projection='polar'),
             hyper=hyper,
             label='Top-Down Density\n(AREPO)',
             color_bar=True,
             cax=fig.add_subplot(top_down_cbar_spec),
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             l_ticks=True,
             r_ticks=True,
             r_label=r'$r$ (kpc to observer)')

    lb(observed=simply_observed,
       ax=fig.add_subplot(simple_spec[0, 0]),
       hyper=hyper,
       title=r'Density-Tracing (IRIS)',
       color_bar=True,
       color_scale=None,
       pin_color_scale_to=None,
       cax=fig.add_subplot(simple_cbar_spec),
       cbar_label='Mean Column Density\n' + r'Per Unit Velocity ($M_\odot \, \text{s} / \text{pc}^3$)',
       cbar_orientation='vertical',
       l_ticks=False,
       b_ticks=True)
    lv(observed=simply_observed,
       ax=fig.add_subplot(simple_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=None,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=True,
       v_ticks=True)

    lb(observed=synth_observed,
       ax=fig.add_subplot(synth_spec[0, 0]),
       hyper=hyper,
       title=r'Synthetic $^{13}$CO(2-1) Observation (IRIS)',
       color_bar=True,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       cbar_ticks=cbar_ticks,
       pin_color_scale_to=None,
       cax=fig.add_subplot(synth_cbar_spec),
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='vertical',
       l_ticks=False,
       b_ticks=False)
    lv(observed=synth_observed,
       ax=fig.add_subplot(synth_spec[1, 0]),
       hyper=hyper,
       title='',
       color_bar=False,
       color_scale=color_scale,
       color_scale_center=color_scale_center,
       color_norm_min=color_norm_min,
       color_norm_max=color_norm_max,
       pin_color_scale_to=None,
       cax=None,
       l_ticks=True,
       v_ticks=False)

    fig.savefig(os.path.expanduser(path))
    return

def speed_test(speed_data_paths: PathLike | typing.Sequence[PathLike], path: PathLike = 'speed_test.png') -> None:
    if isinstance(speed_data_paths, (str, os.PathLike)):
        speed_data_paths = [speed_data_paths]

    speed_data = {}
    for speed_data_path in speed_data_paths:
        with open(os.path.expanduser(speed_data_path), 'r') as f:
            file_speed_data = json.load(f)
        for series_name, series_runs in file_speed_data.items():
            if isinstance(series_runs, list):
                speed_data.setdefault(series_name, []).extend(series_runs)
            else:
                speed_data[series_name] = series_runs

    def _load_runs() -> list[dict[str, typing.Any]]:
        runs = []
        for series_name, series_runs in speed_data.items():
            if not series_name.startswith('series_'):
                continue
            for run in series_runs:
                cleaned = dict(run)
                cleaned['series'] = series_name
                cleaned['pieces'] = float(cleaned.get('pieces', 1))
                cleaned['r_steps'] = int(cleaned['r_steps'])
                cleaned['lon_steps'] = int(cleaned['lon_steps'])
                cleaned['lat_steps'] = int(cleaned['lat_steps'])
                cleaned['v_steps'] = int(cleaned['v_steps'])
                cleaned['total_resolution'] = float(
                    cleaned.get(
                        'total_resolution',
                        cleaned['r_steps'] * cleaned['lon_steps'] * cleaned['lat_steps'] * cleaned['v_steps']))
                cleaned['process_resolution'] = float(
                    cleaned.get('process_resolution', cleaned['total_resolution'] / cleaned['pieces']))
                cleaned['radmc_total_time'] = float(cleaned['radmc_total_time'])
                if 'iris_time' in cleaned:
                    cleaned['iris_time'] = float(cleaned['iris_time'])
                else:
                    cleaned['iris_time'] = cleaned['radmc_total_time'] / float(cleaned['speedup'])
                if 'speedup' in cleaned:
                    cleaned['speedup'] = float(cleaned['speedup'])
                else:
                    cleaned['speedup'] = cleaned['radmc_total_time'] / cleaned['iris_time']
                runs.append(cleaned)
        return runs

    def _average_trials(runs: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
        grouped = {}
        for run in runs:
            key = (run['r_steps'], run['lon_steps'], run['lat_steps'], run['v_steps'])
            grouped.setdefault(key, []).append(run)

        averaged_runs = []
        for key, trials in grouped.items():
            averaged = dict(trials[0])
            averaged['series'] = ','.join(sorted({trial['series'] for trial in trials}))
            averaged['pieces'] = float(np.mean([trial['pieces'] for trial in trials]))
            averaged['radmc_line_time'] = float(np.mean([trial.get('radmc_line_time', np.nan) for trial in trials]))
            averaged['radmc_continuum_time'] = float(
                np.mean([trial.get('radmc_continuum_time', np.nan) for trial in trials]))
            averaged['radmc_total_time'] = float(np.mean([trial['radmc_total_time'] for trial in trials]))
            averaged['iris_time'] = float(np.mean([trial['iris_time'] for trial in trials]))
            averaged['speedup'] = float(np.mean([trial['speedup'] for trial in trials]))
            averaged['piece_scaled_speedup'] = float(np.mean(
                [trial['speedup'] * trial['pieces'] for trial in trials]))
            averaged['n_trials'] = len(trials)
            averaged_runs.append(averaged)
        return averaged_runs

    all_runs = _average_trials(_load_runs())
    if len(all_runs) == 0:
        raise ValueError(f'No speed-test runs found in {speed_data_paths}')

    def _run_sort_key(run: dict[str, typing.Any], x_getter: typing.Callable[[dict[str, typing.Any]], float]) -> tuple[float, float, str]:
        return x_getter(run), run['total_resolution'], run['series']

    def _finite_positive(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        return x[mask], y[mask]

    def _set_log_if_positive(ax: typing.Any, x: ArrayLike, y: ArrayLike) -> None:
        if np.all(np.asarray(x) > 0):
            ax.set_xscale('log')
        if np.all(np.asarray(y) > 0):
            ax.set_yscale('log')

    def _plot_runs(ax: typing.Any,
                   run_filter: typing.Callable[[dict[str, typing.Any]], bool],
                   x_getter: typing.Callable[[dict[str, typing.Any]], float],
                   xlabel: str,
                   title: str) -> None:
        runs = sorted([run for run in all_runs if run_filter(run)], key=lambda run: _run_sort_key(run, x_getter))
        if len(runs) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return

        x = np.array([x_getter(run) for run in runs], dtype=np.float64)
        raw = np.array([run['speedup'] for run in runs], dtype=np.float64)
        piece_scaled = np.array([run['piece_scaled_speedup'] for run in runs], dtype=np.float64)
        pieces = np.array([run['pieces'] for run in runs], dtype=np.float64)
        single_iteration = pieces <= 1
        forced_iterations = pieces > 1

        x_single, raw_single = _finite_positive(x[single_iteration], raw[single_iteration])
        x_forced, raw_forced = _finite_positive(x[forced_iterations], raw[forced_iterations])
        x_piece_scaled, piece_scaled = _finite_positive(x[forced_iterations], piece_scaled[forced_iterations])

        ax.scatter(x_single, raw_single, marker='o', color='#2a6fbb', alpha=0.85, s=30, edgecolors='none',
                   label='Raw (GPU with Single Iteration)')
        ax.scatter(x_forced, raw_forced, marker='^', color='#1b9e77', alpha=0.85, s=38, edgecolors='none',
                   label='Raw (GPU with Forced Multiple Iterations)')
        ax.scatter(x_piece_scaled, piece_scaled, marker='s', color='#d81b60', alpha=0.85, s=30, edgecolors='none',
                   label=r'Multiplied by GPU Iterations (When $>1$)')
        _set_log_if_positive(ax, x, np.concatenate([raw, piece_scaled]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Speedup factor')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)

    def _plot_scatter(ax: typing.Any,
                      x_getter: typing.Callable[[dict[str, typing.Any]], float],
                      xlabel: str,
                      title: str) -> None:
        x = np.array([x_getter(run) for run in all_runs], dtype=np.float64)
        raw = np.array([run['speedup'] for run in all_runs], dtype=np.float64)
        piece_scaled = np.array([run['piece_scaled_speedup'] for run in all_runs], dtype=np.float64)
        pieces = np.array([run['pieces'] for run in all_runs], dtype=np.float64)
        single_iteration = pieces <= 1
        forced_iterations = pieces > 1

        x_single, raw_single = _finite_positive(x[single_iteration], raw[single_iteration])
        x_forced, raw_forced = _finite_positive(x[forced_iterations], raw[forced_iterations])
        x_piece_scaled, piece_scaled = _finite_positive(x[forced_iterations], piece_scaled[forced_iterations])

        ax.scatter(x_single, raw_single, marker='o', color='#2a6fbb', alpha=0.75, s=26, edgecolors='none',
                   label='Raw (GPU with Single Iteration)')
        ax.scatter(x_forced, raw_forced, marker='^', color='#1b9e77', alpha=0.75, s=34, edgecolors='none',
                   label='Raw (GPU with Forced Multiple Iterations)')
        ax.scatter(x_piece_scaled, piece_scaled, marker='s', color='#d81b60', alpha=0.75, s=26, edgecolors='none',
                   label=r'Multiplied by GPU Iterations (When $>1$)')
        _set_log_if_positive(ax, x, np.concatenate([raw, piece_scaled]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Speedup factor')
        ax.set_title(title)
        ax.grid(True, which='both', alpha=0.25)

    fig = plt.figure(figsize=(12, 13))
    gridspec = fig.add_gridspec(nrows=3, ncols=2, hspace=0.35, wspace=0.25)
    axes = np.array([[fig.add_subplot(gridspec[row, col]) for col in range(2)] for row in range(3)])

    def _uses_standard_lon(run: dict[str, typing.Any]) -> bool:
        return run['lon_steps'] == 3 * run['lat_steps'] - 2

    _plot_runs(axes[0, 0],
               lambda run: run['lat_steps'] == 256 and run['lon_steps'] == 766 and run['v_steps'] == 256,
               lambda run: run['r_steps'],
               r'$n$ in $(r, \ell, b, v) = (n, 766, 256, 256)$',
               'Speedup vs. Radial Resolution')
    _plot_runs(axes[0, 1],
               lambda run: run['r_steps'] == 256 and _uses_standard_lon(run) and run['v_steps'] == 256,
               lambda run: run['lat_steps'],
               r'$n$ in $(r, \ell, b, v) = (256, 3n - 2, n, 256)$',
               'Speedup vs. Angular Resolution')
    _plot_runs(axes[1, 0],
               lambda run: run['r_steps'] == 256 and run['lat_steps'] == 256 and run['lon_steps'] == 766,
               lambda run: run['v_steps'],
               r'$n$ in $(r, \ell, b, v) = (256, 766, 256, n)$',
               'Speedup vs. Velocity Resolution')
    _plot_runs(axes[1, 1],
               lambda run: run['r_steps'] == run['lat_steps'] == run['v_steps'] and _uses_standard_lon(run),
               lambda run: run['r_steps'],
               r'$n$ in $(r, \ell, b, v) = (n, 3n - 2, n, n)$',
               'Speedup vs. Matched Resolution')

    _plot_scatter(axes[2, 0],
                  lambda run: run['total_resolution'],
                  r'$n = r{\ell}bv$',
                  'All-Run Speedup vs. Total Resolution')

    _plot_scatter(axes[2, 1],
                  lambda run: run['radmc_total_time'] / 60,
                  'RADMC3D total time (min)',
                  'All-Run Speedup vs. RADMC3D Time')

    handles = []
    labels = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle('Speed Comparison Between RADMC3D (1 CPU) and IRIS-SO (1 CPU + 1 GPU)',
                 y=0.995,
                 fontsize=17)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncols=3, frameon=False)
    fig.subplots_adjust(top=0.875)
    fig.savefig(os.path.expanduser(path))
    return

def loss_trajectory(checkpoint_directories: typing.Sequence[PathLike], path: PathLike = 'loss_trajectory.png') -> None:
    epochs = np.linspace(1, 32, 32)
    training_physical_losses = []
    validation_physical_losses = []
    for dir in checkpoint_directories:
        checkpoint_path = os.path.join(dir, 'stats.json')
        with open(os.path.expanduser(checkpoint_path), 'r') as f:
            stats = json.load(f)
        training_physical_losses.append(stats['training_physical_losses'])
        validation_physical_losses.append(stats['validation_physical_losses'])
    training_physical_losses = np.array(training_physical_losses)
    validation_physical_losses = np.array(validation_physical_losses)
    training_physical_losses = np.mean(training_physical_losses, axis=0)
    validation_physical_losses = np.mean(validation_physical_losses, axis=0)
    lr = np.array([1e-3,] * 16 + [5e-4, 2.5e-4] + [1.25e-4,] * 14)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, training_physical_losses, color='#2a6fbb', label='Averaged Training Loss')
    ax1.plot(epochs, validation_physical_losses, color='#d81b60', linestyle='--', label='Averaged Validation Loss')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.plot(epochs, lr, color='#1b9e77', linestyle='-.', label='Optimizer Learning Rate')
    ax2.set_ylabel('Learning Rate')
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax2.yaxis.set_major_formatter(formatter)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.expanduser(path))
    return

def synthetic_reversions(reverter: typing.Any,
                         pure_top_down_paths: typing.Sequence[PathLike],
                         pure_lv_paths: typing.Sequence[PathLike],
                         full_cone_top_down_paths: typing.Sequence[PathLike],
                         full_cone_lv_paths: typing.Sequence[PathLike],
                         noise_top_down_paths: typing.Sequence[PathLike],
                         noise_lv_paths: typing.Sequence[PathLike],
                         hyper: hp.Hyper,
                         full_cone_hyper: hp.Hyper,
                         noise_hyper: hp.Hyper,
                         path: PathLike = 'synthetic_reversions.png') -> None:
    fig = plt.figure(figsize=(16, 24))

    gridspec = fig.add_gridspec(nrows=3,
                                ncols=2,
                                width_ratios=[1, 1],
                                height_ratios=[1, 1, 1])
    pure1 = fig.add_subfigure(gridspec[0, 0])
    pure2 = fig.add_subfigure(gridspec[0, 1])
    fc1 = fig.add_subfigure(gridspec[1, 0])
    fc2 = fig.add_subfigure(gridspec[1, 1])
    noise1 = fig.add_subfigure(gridspec[2, 0])
    noise2 = fig.add_subfigure(gridspec[2, 1])

    reversion(fig=pure1,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=pure_top_down_paths[0],
              observed_path=pure_lv_paths[0],
              noise=None,
              litter=None,
              hyper=hyper)
    reversion(fig=pure2,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=pure_top_down_paths[1],
              observed_path=pure_lv_paths[1],
              noise=None,
              litter=None,
              hyper=hyper)

    reversion(fig=fc1,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=full_cone_top_down_paths[0],
              observed_path=full_cone_lv_paths[0],
              noise=None,
              litter=None,
              hyper=full_cone_hyper)
    reversion(fig=fc2,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=full_cone_top_down_paths[1],
              observed_path=full_cone_lv_paths[1],
              noise=None,
              litter=None,
              hyper=full_cone_hyper)

    reversion(fig=noise1,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=noise_top_down_paths[0],
              observed_path=noise_lv_paths[0],
              noise=None,
              litter=None,
              hyper=noise_hyper)
    reversion(fig=noise2,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=noise_top_down_paths[1],
              observed_path=noise_lv_paths[1],
              noise=None,
              litter=None,
              hyper=noise_hyper)

    shift_subfigure_contents(pure1, dx=.05, dy=0)
    shift_subfigure_contents(fc1, dx=.05, dy=0)
    shift_subfigure_contents(noise1, dx=.05, dy=0)
    shift_subfigure_contents(pure2, dx=-.05, dy=0)
    shift_subfigure_contents(fc2, dx=-.05, dy=0)
    shift_subfigure_contents(noise2, dx=-.05, dy=0)

    fig.text(0.5, 0.99,
             'Reversion of Cut-Out Observation',
             ha='center', va='top',
             fontsize=22, fontweight='bold')
    fig.text(0.029, 0.831,
             '(A)',
             ha='center', va='top',
             fontsize=22, fontweight='bold')
    fig.text(0.5, 0.66,
             'Reversion of Full-Cone Observation',
             ha='center', va='top',
             fontsize=22, fontweight='bold')
    fig.text(0.029, 0.501,
             '(B)',
             ha='center', va='top',
             fontsize=22, fontweight='bold')
    fig.text(0.5, 0.33,
             'Reversion of Noisy Full-Cone Observation',
             ha='center', va='top',
             fontsize=22, fontweight='bold')
    fig.text(0.029, 0.171,
             '(C)',
             ha='center', va='top',
             fontsize=22, fontweight='bold')

    fig.canvas.draw()
    fig.savefig(os.path.expanduser(path))
    return

def compute_synthetic_reversions(reverter: typing.Any,
                                 dataset: ap.Dataset | ap.ConcatDataset,
                                 full_cone_dataset: ap.Dataset | ap.ConcatDataset,
                                 hyper: hp.Hyper | None = None,
                                 full_cone_hyper: hp.Hyper | None = None) -> None:
    if hyper is None:
        hyper = dataset.hyper
    if full_cone_hyper is None:
        full_cone_hyper = full_cone_dataset.hyper
    hyper.observer_hyper.lon_pieces *= 2
    if full_cone_hyper is hyper:
        full_cone_hyper.observer_hyper.lon_pieces *= 2
    else:
        full_cone_hyper.observer_hyper.lon_pieces *= 4
    full_cone_hyper.observer_hyper.noise_sigma = 0.6
    reverter.cuda()
    reverter.eval()

    if isinstance(dataset, ap.SyntheticallyObservedDataset):
        pure_top_down, pure_lv = dataset.sample(n=2, numpy=False, validation=True)
        pure_lv = pure_lv.cuda()
    else:
        arepo = dataset.sample(n=2, validation=True).cuda()
        pure_top_down = ap.columnize_physical_tensor(arepo, hyper)

        observer = ob.IteratedSyntheticObserver(hyper=hyper)
        observer.cuda()
        observer.eval()
        observed = observer(arepo)
        pure_lv = reverter.reduction(observed)
        observer.cpu()
        del observer

    pure_reverted = reverter.multi_unit_call(pure_lv,
                                             in_units=hyper,
                                             out_units=hyper,
                                             reduce=False)

    pure_top_down = pure_top_down.detach().cpu().numpy()
    pure_lv = pure_lv.detach().cpu().numpy()
    pure_reverted = pure_reverted.detach().cpu().numpy()

    if isinstance(full_cone_dataset, ap.SyntheticallyObservedDataset):
        fc_top_down, fc_lv = full_cone_dataset.sample(n=4, numpy=False, validation=False)
        fc_lv = fc_lv.cuda()
    else:
        arepo = full_cone_dataset.sample(n=4, validation=False)
        fc_top_down = ap.columnize_physical_tensor(arepo, hyper)

        observer = ob.IteratedSyntheticObserver(hyper=full_cone_hyper, cpu_batch=True)
        observer.cuda()
        observer.eval()
        observed = observer(arepo)
        fc_lv = reverter.reduction(observed)
        observer.cpu()
        del observer

    noise_top_down = fc_top_down[2:]
    fc_top_down = fc_top_down[:2]
    noise = ob.Noise(hyper=full_cone_hyper)
    noise.cuda()
    noise_lv = noise(fc_lv[2:], mode='lv')
    fc_lv = fc_lv[:2]

    fc_reverted = reverter.multi_unit_call(fc_lv,
                                           in_units=full_cone_hyper,
                                           out_units=full_cone_hyper,
                                           reduce=False)
    noise_reverted = reverter.multi_unit_call(noise_lv,
                                              in_units=full_cone_hyper,
                                              out_units=full_cone_hyper,
                                              reduce=False)

    fc_top_down = fc_top_down.detach().cpu().numpy()
    fc_lv = fc_lv.detach().cpu().numpy()
    fc_reverted = fc_reverted.detach().cpu().numpy()

    noise_top_down = noise_top_down.detach().cpu().numpy()
    noise_lv = noise_lv.detach().cpu().numpy()
    noise_reverted = noise_reverted.detach().cpu().numpy()

    for i in (1, 2):
        with open(f'pure_top_down{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(pure_top_down[i - 1], axis=0))
        with open(f'pure_lv{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(pure_lv[i - 1], axis=0))
        with open(f'pure_reverted{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(pure_reverted[i - 1], axis=0))

        with open(f'fc_top_down{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(fc_top_down[i - 1], axis=0))
        with open(f'fc_lv{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(fc_lv[i - 1], axis=0))
        with open(f'fc_reverted{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(fc_reverted[i - 1], axis=0))

        with open(f'noise_top_down{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(noise_top_down[i - 1], axis=0))
        with open(f'noise_lv{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(noise_lv[i - 1], axis=0))
        with open(f'noise_reverted{i}.np', 'wb') as file:
            np.save(file, np.expand_dims(noise_reverted[i - 1], axis=0))
    return

def failure_modes(reverter: typing.Any,
                  wrong_dataset: ap.Dataset | ap.ConcatDataset,
                  observer: ob.Observer | None = None,
                  hyper: hp.Hyper | None = None,
                  path: PathLike = 'failure_modes.png') -> None:
    if hyper is None:
        hyper = wrong_dataset.hyper
    noise = ob.Noise(hyper=hyper)

    fig = plt.figure(figsize=(16, 8))

    gridspec = fig.add_gridspec(nrows=1,
                                ncols=2,
                                width_ratios=[1, 1])
    wrong_physics = fig.add_subfigure(gridspec[0, 0])
    just_noise = fig.add_subfigure(gridspec[0, 1])

    reversion(fig=wrong_physics,
              reverter=reverter,
              dataset=wrong_dataset,
              observer=observer,
              top_down_path=None,
              observed_path=None,
              noise=None,
              litter=None,
              hyper=hyper)
    reversion(fig=just_noise,
              reverter=reverter,
              dataset=None,
              observer=None,
              top_down_path=None,
              observed_path=None,
              noise=noise,
              litter=None,
              hyper=hyper)

    shift_subfigure_contents(wrong_physics, dx=.05, dy=0)
    shift_subfigure_contents(just_noise, dx=-.05, dy=0)

    fig.text(0.28, 0.95,
             'Alternate Physics Reversion',
             ha='center', va='top',
             fontsize=16, fontweight='bold')
    fig.text(0.73, 0.95,
             'Noise-Only Reversion',
             ha='center', va='top',
             fontsize=16, fontweight='bold')

    fig.canvas.draw()
    fig.savefig(os.path.expanduser(path))
    return

def true_reversions(reverters: typing.Sequence[typing.Any],
                    hyper: hp.Hyper,
                    path: PathLike = 'true_reversions.png') -> None:
    if len(reverters) != 4:
        raise ValueError('Expected reverters to be a list of length 4.')

    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    temperature = hyper.dataset_hyper._temperature_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    true_observation = cube_processing.make_default_cube(hyper)
    true_observation = true_observation.cuda()
    true_lv = reverters[0].reduction(true_observation)
    true_lv /= temperature
    true_lv = true_lv.detach().cpu().numpy()[0][0]
    reversions = []
    for reverter in reverters:
        reverter.cuda()
        reverter.eval()
        reverted_top_down = reverter.multi_unit_call(true_observation,
                                                     in_units=hyper,
                                                     out_units=hyper)
        reverted_top_down /= density
        reverted_top_down = reverted_top_down.detach().cpu().numpy()[0][0]
        reversions.append(reverted_top_down)
        reverter.cpu()

    fig = plt.figure(figsize=(8, 13))

    gridspec = fig.add_gridspec(nrows=9,
                                ncols=1,
                                height_ratios=[1.2, 0.1, 1.2, 0.1, 0.05, 0.1, 1, 0.1, 0.05])
    pair_1_spec = gridspec[0, 0].subgridspec(nrows=1,
                                             ncols=2,
                                             width_ratios=[1, 1])
    pair_2_spec = gridspec[2, 0].subgridspec(nrows=1,
                                             ncols=2,
                                             width_ratios=[1, 1])

    top_left_reversion_ax = fig.add_subplot(pair_1_spec[0, 0], projection='polar')
    top_right_reversion_ax = fig.add_subplot(pair_1_spec[0, 1], projection='polar')
    bottom_left_reversion_ax = fig.add_subplot(pair_2_spec[0, 0], projection='polar')
    bottom_right_reversion_ax = fig.add_subplot(pair_2_spec[0, 1], projection='polar')
    top_down_cax = fig.add_subplot(gridspec[4, 0])
    sedigism_ax = fig.add_subplot(gridspec[6, 0])
    sedigism_cax = fig.add_subplot(gridspec[8, 0])

    top_down(density=reversions[0],
             ax=top_left_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 1 Reversion)',
             color_bar=True,
             cax=top_down_cax,
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             cbar_orientation='horizontal',
             l_ticks=True,
             r_ticks=False)
    top_down(density=reversions[1],
             ax=top_right_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 2 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=True)
    top_down(density=reversions[2],
             ax=bottom_left_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 3 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=False)
    top_down(density=reversions[3],
             ax=bottom_right_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 4 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=True)

    lv(observed=true_lv,
       ax=sedigism_ax,
       hyper=hyper,
       title=r'SEDIGISM $^{13}$CO(2-1) Data',
       color_bar=True,
       color_scale=5e0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=sedigism_cax,
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='horizontal',
       cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    shift_axes(top_left_reversion_ax, dx=0.01, dy=0)
    shift_axes(bottom_left_reversion_ax, dx=0.01, dy=0)
    shift_axes(top_right_reversion_ax, dx=-0.02, dy=0)
    shift_axes(bottom_right_reversion_ax, dx=-0.02, dy=0)

    arrow_cmap = plt.get_cmap('inferno')
    purple_arrow_color = arrow_cmap(0.2)
    orange_arrow_color = arrow_cmap(0.8)
    arrow_style = 'simple,head_length=0.4,head_width=1.2,tail_width=0.6'
    arrow_1 = patches.FancyArrowPatch((0.321, 0.335),
                                      (0.321, 0.375),
                                      transform=fig.transFigure,
                                      facecolor=orange_arrow_color,
                                      edgecolor=purple_arrow_color,
                                      linewidth=2.5,
                                      arrowstyle=arrow_style,
                                      mutation_scale=40,
                                      zorder=10)
    fig.patches.append(arrow_1)
    arrow_2 = patches.FancyArrowPatch((0.716, 0.335),
                                      (0.716, 0.375),
                                      transform=fig.transFigure,
                                      facecolor=orange_arrow_color,
                                      edgecolor=purple_arrow_color,
                                      linewidth=2.5,
                                      arrowstyle=arrow_style,
                                      mutation_scale=40,
                                      zorder=10)
    fig.patches.append(arrow_2)

    fig.text(0.5, 0.95,
             'Reversion of True SEDIGISM Data',
             ha='center', va='top',
             fontsize=22, fontweight='bold')

    fig.canvas.draw()
    fig.savefig(os.path.expanduser(path))
    return

def true_reversions_orbits(reverters: typing.Sequence[typing.Any],
                           hyper: hp.Hyper,
                           path: PathLike = 'true_reversions.png') -> None:
    if len(reverters) != 4:
        raise ValueError('Expected reverters to be a list of length 4.')

    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    temperature = hyper.dataset_hyper._temperature_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    true_observation = cube_processing.make_default_cube(hyper)
    true_observation = true_observation.cuda()
    true_lv = reverters[0].reduction(true_observation)
    true_lv /= temperature
    true_lv = true_lv.detach().cpu().numpy()[0][0]
    reversions = []
    for reverter in reverters:
        reverter.cuda()
        reverter.eval()
        reverted_top_down = reverter.multi_unit_call(true_observation,
                                                     in_units=hyper,
                                                     out_units=hyper)
        reverted_top_down /= density
        reverted_top_down = reverted_top_down.detach().cpu().numpy()[0][0]
        reversions.append(reverted_top_down)
        reverter.cpu()

    fig = plt.figure(figsize=(8, 13))

    gridspec = fig.add_gridspec(nrows=9,
                                ncols=1,
                                height_ratios=[1.2, 0.1, 1.2, 0.1, 0.05, 0.1, 1, 0.1, 0.05])
    pair_1_spec = gridspec[0, 0].subgridspec(nrows=1,
                                             ncols=2,
                                             width_ratios=[1, 1])
    pair_2_spec = gridspec[2, 0].subgridspec(nrows=1,
                                             ncols=2,
                                             width_ratios=[1, 1])

    top_left_reversion_ax = fig.add_subplot(pair_1_spec[0, 0], projection='polar')
    top_right_reversion_ax = fig.add_subplot(pair_1_spec[0, 1], projection='polar')
    bottom_left_reversion_ax = fig.add_subplot(pair_2_spec[0, 0], projection='polar')
    bottom_right_reversion_ax = fig.add_subplot(pair_2_spec[0, 1], projection='polar')
    top_down_cax = fig.add_subplot(gridspec[4, 0])
    sedigism_ax = fig.add_subplot(gridspec[6, 0])
    sedigism_cax = fig.add_subplot(gridspec[8, 0])

    top_down(density=reversions[0],
             ax=top_left_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 1 Reversion)',
             color_bar=True,
             cax=top_down_cax,
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             cbar_orientation='horizontal',
             l_ticks=True,
             r_ticks=False,
             orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
             orbit_opacity=0.7,
             orbit_legend=True)
    top_down(density=reversions[1],
             ax=top_right_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 2 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=True,
             orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
             orbit_opacity=0.7,
             orbit_legend=False)
    top_down(density=reversions[2],
             ax=bottom_left_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 3 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=False,
             orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
             orbit_opacity=0.7,
             orbit_legend=False)
    top_down(density=reversions[3],
             ax=bottom_right_reversion_ax,
             hyper=hyper,
             label='Top-Down Density\n(Model 4 Reversion)',
             pin_color_scale_to=reversions[0],
             color_bar=False,
             cax=None,
             l_ticks=True,
             r_ticks=True,
             orbits=['walker', 'lipman min', 'lipman median', 'lipman max'],
             orbit_opacity=0.7,
             orbit_legend=False)

    lv(observed=true_lv,
       ax=sedigism_ax,
       hyper=hyper,
       title=r'SEDIGISM $^{13}$CO(2-1) Data',
       color_bar=True,
       color_scale=5e0,
       color_scale_center=0.5,
       pin_color_scale_to=None,
       cax=sedigism_cax,
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='horizontal',
       cbar_ticks=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    shift_axes(top_left_reversion_ax, dx=0.01, dy=0)
    shift_axes(bottom_left_reversion_ax, dx=0.01, dy=0)
    shift_axes(top_right_reversion_ax, dx=-0.02, dy=0)
    shift_axes(bottom_right_reversion_ax, dx=-0.02, dy=0)

    arrow_cmap = plt.get_cmap('inferno')
    purple_arrow_color = arrow_cmap(0.2)
    orange_arrow_color = arrow_cmap(0.8)
    arrow_style = 'simple,head_length=0.4,head_width=1.2,tail_width=0.6'
    arrow_1 = patches.FancyArrowPatch((0.321, 0.335),
                                      (0.321, 0.375),
                                      transform=fig.transFigure,
                                      facecolor=orange_arrow_color,
                                      edgecolor=purple_arrow_color,
                                      linewidth=2.5,
                                      arrowstyle=arrow_style,
                                      mutation_scale=40,
                                      zorder=10)
    fig.patches.append(arrow_1)
    arrow_2 = patches.FancyArrowPatch((0.716, 0.335),
                                      (0.716, 0.375),
                                      transform=fig.transFigure,
                                      facecolor=orange_arrow_color,
                                      edgecolor=purple_arrow_color,
                                      linewidth=2.5,
                                      arrowstyle=arrow_style,
                                      mutation_scale=40,
                                      zorder=10)
    fig.patches.append(arrow_2)

    fig.text(0.5, 0.95,
             'Reversion of True SEDIGISM Data',
             ha='center', va='top',
             fontsize=22, fontweight='bold')

    fig.canvas.draw()
    fig.savefig(os.path.expanduser(path))
    return

def shift_subfigure_contents(subfig: typing.Any, dx: float = 0.0, dy: float = 0.0) -> None:
    for ax in subfig.axes:
        pos = ax.get_position()
        ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])
    for artist in subfig.artists:
        artist.set_transform(
            transforms.Affine2D().translate(dx, dy) + artist.get_transform())
    return

def shift_axes(ax: typing.Any, dx: float = 0.0, dy: float = 0.0) -> None:
    pos = ax.get_position()
    ax.set_position([pos.x0 + dx, pos.y0 + dy, pos.width, pos.height])

def pull_edge_l_tick_labels_inward(ax: typing.Any, tolerance: float = 1.0) -> None:
    left_edge, right_edge = ax.bbox.x0, ax.bbox.x1
    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if not label.get_visible() or label.get_text() == '':
            continue

        tick_x = ax.transData.transform((tick, 0))[0]
        if abs(tick_x - left_edge) <= tolerance:
            label.set_ha('left')
        elif abs(tick_x - right_edge) <= tolerance:
            label.set_ha('right')

def reversion(fig: typing.Any,
              reverter: typing.Any,
              dataset: ap.Dataset | ap.ConcatDataset | None = None,
              observer: ob.Observer | None = None,
              top_down_path: PathLike | None = None,
              observed_path: PathLike | None = None,
              noise: typing.Any | None = None,
              litter: typing.Any | None = None,
              hyper: hp.Hyper | None = None) -> typing.Any:
    if hyper is None and dataset is None:
        raise ValueError('Must provide either hyper or dataset to visualize reversion.')
    elif hyper is None:
        hyper = dataset.hyper

    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    temperature = hyper.dataset_hyper._temperature_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    density *= density_conversion

    blank = False
    if top_down_path is not None and observed_path is not None:
        arepo_top_down = np.load(os.path.expanduser(top_down_path))[0][0] / density
        observed = np.load(os.path.expanduser(observed_path))
        observed = torch.tensor(observed, dtype=torch.float32).cuda()
        if len(observed.shape) == 5:
            observed_lv = reverter.reduction(observed)
        else:
            observed_lv = observed
    elif dataset is None:
        blank = True
        r_steps = hyper.coordinate_hyper.r_steps
        lon_steps = hyper.coordinate_hyper.lon_steps
        v_steps = hyper.cube_hyper.v_steps
        arepo_top_down = np.zeros((r_steps, lon_steps), dtype=np.float32)
        observed_lv = torch.zeros(1, 1, lon_steps, v_steps).cuda()
    elif isinstance(dataset, ap.SyntheticallyObservedDataset):
        arepo_top_down, observed_lv = dataset.sample(1, validation=True)
        arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0] / density
        observed_lv = observed_lv.cuda()
    elif observer is None:
        raise ValueError('Must provide observer or SyntheticallyObservedDataset or a pair of '
                         'top_down_path and observed_path to visualize reversion.')
    else:
        arepo_full = dataset.sample(1, validation=True)
        r_crop_min_index = hyper.coordinate_hyper.r_crop_min_index
        r_crop_max_index = hyper.coordinate_hyper.r_crop_max_index
        arepo = arepo_full[:, :, r_crop_min_index:r_crop_max_index, :, :].cuda()
        arepo_top_down = ap.columnize_physical_tensor(arepo_full, hyper)
        arepo_top_down = arepo_top_down.detach().cpu().numpy()[0][0] / density

        observer.cuda()
        observer.eval()
        if isinstance(observer, ob.IteratedSyntheticObserver):
            if not observer.cpu_batch:
                arepo_full = arepo_full.cuda()
            observed = observer(arepo_full)
        else:
            observed = observer(arepo)
        observed_lv = reverter.reduction(observed)
        observer.cpu()

    if noise is not None:
        noise.cuda()
        noise.eval()
        observed_lv = noise(observed_lv, inplace=True, mode='lv')
        noise.cpu()
    if litter is not None:
        litter_sample = litter.observed_sample()
        observed_lv += litter_sample

    reverter.cuda()
    reverter.eval()
    reverted_top_down = reverter.multi_unit_call(observed_lv,
                                                 in_units=hyper,
                                                 out_units=hyper,
                                                 reduce=False)
    reverted_top_down /= density
    observed_lv /= temperature

    reverted_top_down = reverted_top_down.detach().cpu().numpy()[0][0]
    observed_lv = observed_lv.detach().cpu().numpy()[0][0]

    gridspec = fig.add_gridspec(nrows=3,
                                ncols=1,
                                height_ratios=[1, 0.1, 1])
    top_down_spec = gridspec[0, 0].subgridspec(nrows=1,
                                               ncols=3,
                                               width_ratios=[0.05, 1, 1])
    observed_spec = gridspec[2, 0].subgridspec(nrows=2,
                                               ncols=1,
                                               height_ratios=[1, 0.05])

    top_down_cax = fig.add_subplot(top_down_spec[0, 0])
    arepo_top_down_ax = fig.add_subplot(top_down_spec[0, 1], projection='polar')
    reverted_top_down_ax = fig.add_subplot(top_down_spec[0, 2], projection='polar')
    observed_ax = fig.add_subplot(observed_spec[0, 0])
    observed_cbar_ax = fig.add_subplot(observed_spec[1, 0])

    top_down(density=arepo_top_down,
             ax=arepo_top_down_ax,
             hyper=hyper,
             label=r'Top-Down H$_2$ Density' + '\n(AREPO)',
             pin_color_scale_to=reverted_top_down if blank else None,
             color_bar=not blank,
             cax=top_down_cax,
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             l_ticks=True,
             r_ticks=False)
    top_down(density=reverted_top_down,
             ax=reverted_top_down_ax,
             hyper=hyper,
             label='Top-Down H$_2$ Density' + '\n(IRIS Reconstructed)',
             pin_color_scale_to=None if blank else arepo_top_down,
             color_bar=blank,
             cax=top_down_cax,
             cbar_label=r'H$_2$ Density ($M_\odot / \text{pc}^3$)',
             l_ticks=True,
             r_ticks=True)

    lv(observed=observed_lv,
       ax=observed_ax,
       hyper=hyper,
       title=r'Synthetic $^{13}$CO(2-1) Observation (IRIS)',
       color_bar=True,
       color_scale=1e2,
       color_scale_center=0.035,
       color_norm_min=-0.1,
       color_norm_max=2.0,
       pin_color_scale_to=None,
       cax=observed_cbar_ax,
       cbar_label='Mean Raleigh-Jeans Temperature (K)',
       cbar_orientation='horizontal',
       cbar_ticks=(-0.1, 0.0, 0.1, 0.25, 1.0, 2.0))

    shift_axes(top_down_cax, dx=-0.0750, dy=0.0063)
    shift_axes(arepo_top_down_ax, dx=-0.0163, dy=0.0063)
    shift_axes(reverted_top_down_ax, dx=-0.0500, dy=0.0063)
    shift_axes(observed_cbar_ax, dx=0, dy=-0.0313)

    arrow_cmap = plt.get_cmap('inferno')
    purple_arrow_color = arrow_cmap(0.2)
    orange_arrow_color = arrow_cmap(0.8)
    arrow_style = 'simple,head_length=0.4,head_width=1.2,tail_width=0.6'
    transform = getattr(fig, 'transSubfigure', fig.transFigure)
    arrow_down = patches.FancyArrowPatch((0.321, 0.511),
                                         (0.321, 0.431),
                                         transform=transform,
                                         facecolor=purple_arrow_color,
                                         edgecolor=orange_arrow_color,
                                         linewidth=2.5,
                                         arrowstyle=arrow_style,
                                         mutation_scale=40,
                                         zorder=10)
    fig.add_artist(arrow_down)
    arrow_up = patches.FancyArrowPatch((0.716, 0.431),
                                       (0.716, 0.511),
                                       transform=transform,
                                       facecolor=orange_arrow_color,
                                       edgecolor=purple_arrow_color,
                                       linewidth=2.5,
                                       arrowstyle=arrow_style,
                                       mutation_scale=40,
                                       zorder=10)
    fig.add_artist(arrow_up)

    return fig

def wide_top_down(snapshot_path: PathLike,
                  ax: typing.Any,
                  hyper: hp.Hyper,
                  resolution: tuple[int, int, int] = (1024, 1024, 1024),
                  box_size: float = 20000,
                  theta: float = -90,
                  label: str = r'$\rho$',
                  color_scale: float | None = None,
                  pin_color_scale_to: ArrayLike | None = None,
                  color_bar: bool = False,
                  cax: typing.Any | None = None,
                  cbar_label: str | None = None,
                  x_ticks: bool = True,
                  y_ticks: bool = True) -> None:
    snapshot = ap.Snapshot(snapshot_path, hyper)
    arepo_wide_top_down = snapshot.make_wide_top_down(resolution=resolution, box_size=box_size)
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density = parsec / parsec / solar_mass
    arepo_wide_top_down *= density
    #arepo_wide_top_down = np.zeros(resolution[:2], dtype=np.float32)
    box_size /= 1000
    observer_radius = hyper.coordinate_hyper.observer_radius / 1000
    theta *= np.pi / 180
    observer_position = observer_radius * np.array([np.cos(theta), np.sin(theta)])

    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    r_min = hyper.coordinate_hyper.r_min / 1000
    r_max = hyper.coordinate_hyper.r_max / 1000

    if pin_color_scale_to is None:
        density_color_scale = arepo_wide_top_down
    else:
        density_color_scale = pin_color_scale_to

    color_norm_min = density_color_scale.min()
    color_norm_max = density_color_scale.max()

    if color_scale is None:
        color_scale = 100 / np.mean(density_color_scale)

    forward = lambda x: np.asinh(color_scale * x)
    inverse = lambda y: np.sinh(y) / color_scale
    norm = colors.FuncNorm((forward, inverse), vmin=color_norm_min, vmax=color_norm_max)

    x_edges = np.linspace(-box_size / 2, box_size / 2, resolution[0])
    y_edges = np.linspace(-box_size / 2, box_size / 2, resolution[1])

    mesh = ax.pcolormesh(x_edges,
                         y_edges,
                         arepo_wide_top_down,
                         norm=norm,
                         cmap='inferno',
                         shading='auto')

    if color_bar:
        powers = np.ceil(np.log10(norm.inverse(np.linspace(0, norm(color_norm_max), 10)[1:])))
        powers = np.unique(powers.astype(np.int32)).astype(np.float32)
        ticks = np.insert(10.0 ** powers, 0, 0)

        def format(value: float, pos: int | None) -> str:
            if value > 0:
                exponent = int(np.log10(value))
                return f'$10^{{{exponent}}}$'
            else:
                return '0'

        formatter = ticker.FuncFormatter(format)

        if cax is None:
            cbar = plt.colorbar(mesh, ax=ax, location='left', ticks=ticks, format=formatter)
            if cbar_label is not None:
                cbar.set_label(cbar_label, loc='center', rotation=90)
        else:
            cbar = plt.colorbar(mesh, cax=cax, orientation='vertical', ticks=ticks, format=formatter)
            if cbar_label is not None:
                cbar.set_label(cbar_label, loc='center', rotation=90)

    ax.set_xlim(-box_size / 2, box_size / 2)
    ax.set_ylim(-box_size / 2, box_size / 2)
    ax.set_aspect('equal')

    obs_x, obs_y = observer_position

    theta_min = theta - lon_min * np.pi / 180
    theta_max = theta - lon_max * np.pi / 180

    angles = np.linspace(theta_min, theta_max, 200)

    x_outer = obs_x - r_max * np.cos(angles)
    y_outer = obs_y - r_max * np.sin(angles)

    x_inner = obs_x - r_min * np.cos(angles)
    y_inner = obs_y - r_min * np.sin(angles)

    x_side1 = obs_x - np.array([r_min, r_max]) * np.cos(theta_min)
    y_side1 = obs_y - np.array([r_min, r_max]) * np.sin(theta_min)

    x_side2 = obs_x - np.array([r_min, r_max]) * np.cos(theta_max)
    y_side2 = obs_y - np.array([r_min, r_max]) * np.sin(theta_max)

    x_dot1 = obs_x - np.array([0, r_min]) * np.cos(theta_min)
    y_dot1 = obs_y - np.array([0, r_min]) * np.sin(theta_min)

    x_dot2 = obs_x - np.array([0, r_min]) * np.cos(theta_max)
    y_dot2 = obs_y - np.array([0, r_min]) * np.sin(theta_max)

    ax.plot(x_outer, y_outer, color='white', linewidth=1.5)
    ax.plot(x_inner, y_inner, color='white', linewidth=1.5)

    ax.plot(x_side1, y_side1, color='white', linewidth=1.5)
    ax.plot(x_side2, y_side2, color='white', linewidth=1.5)

    ax.plot(x_dot1, y_dot1, color='white', linestyle=':', linewidth=1.2)
    ax.plot(x_dot2, y_dot2, color='white', linestyle=':', linewidth=1.2)

    ax.plot(obs_x,
            obs_y,
            marker='o',
            markersize=5,
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=0.8,
            zorder=10)
    ax.text(obs_x,
            obs_y - 0.028 * box_size,
            'observer',
            color='white',
            fontsize='medium',
            ha='center',
            va='top')

    locator = ticker.MaxNLocator(integer=True)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    if not x_ticks:
        ax.set_xticks([])
    else:
        ax.set_xlabel(r'$x$ (kpc)')
    if not y_ticks:
        ax.set_yticks([])
    else:
        ax.set_ylabel(r'$y$ (kpc)')
    if label is not None:
        ax.text(0.5,
                0.95,
                label,
                transform=ax.transAxes,
                color='white',
                fontsize='large',
                fontweight='bold',
                ha='center',
                va='top')
    return

def top_down(density: ArrayLike,
             ax: typing.Any,
             hyper: hp.Hyper,
             label: str = r'$\rho$',
             color_scale: float | None = None,
             color_scale_center: float = 0,
             pin_color_scale_to: ArrayLike | None = None,
             color_bar: bool = False,
             cax: typing.Any | None = None,
             cbar_label: str | None = None,
             cbar_orientation: str = 'vertical',
             l_ticks: bool = True,
             r_ticks: bool = True,
             r_label: str = r'$r$ (kpc to observer)',
             observer_arrow_offset: float = .20,
             orbits: typing.Sequence[str] = (),
             orbit_opacity: float = 0.8,
             orbit_nphi: int = 512,
             orbit_arrow_scale: float = 18,
             orbit_arrow_lw: float = 1.8,
             orbit_arrow_phase_shift: float = np.pi / 2,
             orbit_legend: bool = True) -> None:
    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    lon_edges = np.linspace(lon_min * np.pi / 180,
                            lon_max * np.pi / 180,
                            density.shape[1] + 1)
    r_min = hyper.coordinate_hyper.r_min / 1000
    r_max = hyper.coordinate_hyper.r_max / 1000
    r_crop_min_index = hyper.coordinate_hyper.r_crop_min_index
    r_crop_max_index = hyper.coordinate_hyper.r_crop_max_index
    r_steps = hyper.coordinate_hyper.r_steps
    r_crop_min = r_min + (r_max - r_min) * r_crop_min_index / r_steps
    r_crop_max = r_min + (r_max - r_min) * r_crop_max_index / r_steps
    r_edges = np.linspace(r_crop_min, r_crop_max, density.shape[0] + 1)

    if pin_color_scale_to is None:
        density_color_scale = density
    else:
        density_color_scale = pin_color_scale_to
    color_norm_min = density_color_scale.min()
    color_norm_max = density_color_scale.max()
    if color_scale is None:
        color_scale = 1 / np.mean(density_color_scale)
    forward = lambda x: np.asinh(color_scale * (x - color_scale_center))
    inverse = lambda y: np.sinh(y) / color_scale + color_scale_center
    norm = colors.FuncNorm((forward, inverse), vmin=color_norm_min, vmax=color_norm_max)

    mesh = ax.pcolormesh(lon_edges, r_edges, density, norm=norm, cmap='inferno', shading='auto')
    if color_bar:
        powers = np.ceil(np.log10(norm.inverse(np.linspace(0, norm(color_norm_max), 10)[1:])))
        powers = np.unique(powers.astype(np.int32)).astype(np.float32)
        ticks = np.insert(10. ** powers, 0, 0)

        def format(value: float, pos: int | None) -> str:
            if value > 0:
                exponent = int(np.log10(value))
                return f'$10^{{{exponent}}}$'
            else:
                return "0"

        formatter = ticker.FuncFormatter(format)

        if cax is None:
            if cbar_orientation == 'vertical':
                loc = 'left'
            else:
                loc = 'bottom'

            cbar = plt.colorbar(mesh, ax=ax, location=loc, ticks=ticks, format=formatter)
        else:
            cbar = plt.colorbar(mesh, cax=cax, orientation=cbar_orientation, ticks=ticks, format=formatter)

        if cbar_label is not None:
            if cbar_orientation == 'vertical':
                cbar.set_label(cbar_label, loc='center', rotation=90)
            else:
                cbar.set_label(cbar_label, loc='center')

    ax.set_thetamin(lon_min)
    ax.set_thetamax(lon_max)
    ax.set_rlim(r_crop_min, r_crop_max)
    ax.set_rorigin(0)
    ax.set_theta_zero_location('N')

    ax.xaxis.grid(False)
    if l_ticks:
        span = lon_max - lon_min
        if span <= 4.:
            step = .5
        else:
            step = 1.

        start_deg = np.ceil(lon_min / step) * step
        ticks_deg = np.arange(start_deg, lon_max + 1e-4, step)
        ticks_rad = np.deg2rad(ticks_deg)

        ax.set_xticks([])

        r_span = r_crop_max - r_crop_min
        tick_len = r_span * .015
        r_lbl = r_crop_min - (r_span * .03)

        for t_rad, t_deg in zip(ticks_rad, ticks_deg):
            deg = t_deg
            if abs(deg) < 1e-10:
                deg = 0
            if abs(deg - round(deg)) < 1e-5:
                lbl = f"{int(round(deg))}"
            else:
                lbl = f"{deg:.1f}"

            ax.plot([t_rad, t_rad], [r_crop_min, r_crop_min - tick_len],
                    color='black', linewidth=1, clip_on=False)

            ax.text(t_rad, r_lbl, lbl,
                    ha='center', va='top',
                    color='black', fontsize='medium')

        ax.text(.5,
                -.08,
                r'$\ell$ (GLON deg)',
                transform=ax.transAxes,
                color='black',
                fontsize='medium',
                ha='center',
                va='top')
    else:
        ax.set_xticks([])

    ax.yaxis.grid(False)
    if r_ticks:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.set_rlabel_position(lon_min - (lon_max - lon_min) / 10)

        offset_angle = (lon_max - lon_min) * observer_arrow_offset
        tick_angle_deg = lon_min - offset_angle
        tick_angle_rad = np.deg2rad(tick_angle_deg)
        text_rotation = lon_min + 90
        r_mid = (r_crop_min + r_crop_max) / 2
        r_span = r_crop_max - r_crop_min

        ax.text(tick_angle_rad, r_mid, r_label,
                color='black',
                fontsize='medium',
                rotation=text_rotation,
                ha='center',
                va='center')

        arrow_start_r = r_mid - (r_span * .0125 * len(r_label))
        ax.annotate('',
                    xy=(tick_angle_rad, r_crop_min),
                    xytext=(tick_angle_rad, arrow_start_r),
                    xycoords='data',
                    arrowprops=dict(facecolor='black',
                                    edgecolor='black',
                                    linestyle='--',
                                    linewidth=1.5,
                                    arrowstyle='->',
                                    mutation_scale=20),
                    annotation_clip=False)
    else:
        ax.set_yticks([])

    sagA_handle = None
    for orbit in orbits:
        phi = np.linspace(0.0, 2.0 * np.pi, orbit_nphi, endpoint=True)
        (r_orbit,
         lon_orbit,
         lat_orbit,
         v_orbit,
         sagA_r,
         sagA_lon,
         sagA_lat,
         color,
         legend_key,
         arrow_phi_shift) = cmz_model_orbit(phi,
                                            deg_out=False,
                                            orbit=orbit)
        r_orbit /= 1000
        sagA_r /= 1000

        if sagA_handle is None:
            sagA_handle = ax.scatter(sagA_lon, sagA_r,
                                     marker='*',
                                     s=120,
                                     color='white',
                                     edgecolors='cyan',
                                     linewidths=0.8,
                                     zorder=20,
                                     label=r'$\text{Sag A}^\ast$')

        if (r_orbit[0] != r_orbit[-1]) or (lon_orbit[0] != lon_orbit[-1]):
            r_orbit = np.append(r_orbit, r_orbit[0])
            lon_orbit = np.append(lon_orbit, lon_orbit[0])

        ax.plot(lon_orbit, r_orbit,
                color=color,
                alpha=orbit_opacity,
                linewidth=2.0,
                zorder=10,
                label=legend_key)

        n = len(phi)
        shift = int(n * (orbit_arrow_phase_shift + arrow_phi_shift) / 2 / np.pi)
        arrow_indices = [shift % n, (n // 2 + shift) % n]

        for i in arrow_indices:
            i0 = max(i - 2, 0)
            i1 = min(i + 2, orbit_nphi)
            ax.annotate('',
                        xy=(lon_orbit[i0], r_orbit[i0]),
                        xytext=(lon_orbit[i1], r_orbit[i1]),
                        xycoords='data',
                        arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        alpha=orbit_opacity,
                                        linewidth=orbit_arrow_lw,
                                        mutation_scale=orbit_arrow_scale),
                        annotation_clip=True)

    if orbits and orbit_legend:
        legend = ax.legend(loc='lower left',
                           bbox_to_anchor=(0.02, 0.02),
                           frameon=True,
                           facecolor='white',
                           edgecolor='black',
                           framealpha=0.6,
                           fontsize='x-small')

        legend.set_zorder(100)

    if label is not None:
        ax.text(.5, .95, label,
                transform=ax.transAxes,
                color='white',
                fontsize='large',
                fontweight='bold',
                ha='center',
                va='top')
    return

def lb(observed: ArrayLike,
       ax: typing.Any,
       hyper: hp.Hyper,
       title: str | None = None,
       color_bar: bool = False,
       color_scale: float | None = None,
       color_scale_center: float = 0,
       color_norm_min: float | None = None,
       color_norm_max: float | None = None,
       pin_color_scale_to: ArrayLike | None = None,
       cax: typing.Any | None = None,
       cbar_label: str | None = None,
       cbar_orientation: str = 'horizontal',
       cbar_ticks: typing.Sequence[float] | None = None,
       l_ticks: bool = True,
       b_ticks: bool = True,
       orbits: typing.Sequence[str] = (),
       orbit_opacity: float = 0.8,
       orbit_nphi: int = 512,
       orbit_arrow_scale: float = 18,
       orbit_arrow_lw: float = 1.8,
       orbit_arrow_phase_shift: float = -np.pi / 4) -> None:
    reduction = hyper.cube_hyper.reduction
    if len(observed.shape) == 2:
        im = observed.transpose()
    elif reduction == 'max':
        im = peak_intensity(observed, dim=2).transpose()
    elif reduction == 'mean':
        im = mean_intensity(observed, dim=2).transpose()
    else:
        raise ValueError(
            "Invalid cube reduction specified; "
            "hyper.cube_hyper.reduction must be one of 'mean' or 'max' "
            "or cube must already be a reduced lb image.")

    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    lon_edges = np.linspace(lon_min, lon_max, im.shape[1] + 1)
    lat_min = hyper.coordinate_hyper.lat_min
    lat_max = hyper.coordinate_hyper.lat_max
    lat_edges = np.linspace(lat_min, lat_max, im.shape[0] + 1)

    if pin_color_scale_to is not None:
        observed_color_scale = pin_color_scale_to
    else:
        observed_color_scale = observed
    if color_norm_min is not None and color_norm_max is not None:
        pass
    elif len(observed.shape) == 2:
        color_norm_min = observed_color_scale.min()
        color_norm_max = observed_color_scale.max()
    elif reduction == 'max':
        color_scale_lb = peak_intensity(observed_color_scale, dim=2)
        color_scale_lv = peak_intensity(observed_color_scale, dim=1)
        color_norm_min = np.minimum(color_scale_lb.min(), color_scale_lv.min())
        color_norm_max = np.maximum(color_scale_lb.max(), color_scale_lv.max())
    elif reduction == 'mean':
        color_scale_lb = mean_intensity(observed_color_scale, dim=2)
        color_scale_lv = mean_intensity(observed_color_scale, dim=1)
        color_norm_min = np.minimum(color_scale_lb.min(), color_scale_lv.min())
        color_norm_max = np.maximum(color_scale_lb.max(), color_scale_lv.max())
    else:
        raise ValueError(
            "Invalid cube reduction specified; "
            "hyper.cube_hyper.reduction must be one of 'mean' or 'max' "
            "or cube/pin_color_scale_to must already be a reduced lb/lv image.")

    if color_scale is None:
        color_scale = 1 / np.mean(observed_color_scale)
    forward = lambda x: np.asinh(color_scale * (x - color_scale_center))
    inverse = lambda y: np.sinh(y) / color_scale + color_scale_center
    norm = colors.FuncNorm((forward, inverse), vmin=color_norm_min, vmax=color_norm_max)
    mesh = ax.pcolormesh(lon_edges, lat_edges, im, cmap='gray' if orbits else 'inferno', norm=norm)

    if color_bar:
        if cbar_ticks is not None:
            ticks = np.array(cbar_ticks)
            format = lambda value, pos: '2.73' if np.abs(value - 2.73) < 1e-3 else f'{value:.1f}'
        else:
            values = norm.inverse(np.linspace(0, norm(color_norm_max), 8))
            values = values[values > 1e-10]
            powers = np.ceil(np.log10(values))
            powers = np.unique(powers.astype(np.int32)).astype(np.float32)
            ticks = np.insert(10. ** powers, 0, 0)
            if color_norm_min < 0:
                values = -norm.inverse(np.linspace(0, norm(color_norm_min), 8))
                values = values[values > 1e-10]
                neg_powers = np.ceil(np.log10(values))
                neg_ticks = -(10. ** neg_powers)
                ticks = np.concatenate((neg_ticks, ticks), axis=0)
            def format(value: float, pos: int | None) -> str:
                if value > 1e-10:
                    exponent = int(np.log10(value))
                    return rf'$10^{{{exponent}}}$'
                elif value < -1e-10:
                    exponent = int(np.log10(-value))
                    return rf'$-1 \times 10^{{{exponent}}}$'
                else:
                    return '0'

        formatter = ticker.FuncFormatter(format)

        if cax is None:
            if cbar_orientation == 'vertical':
                loc = 'right'
            else:
                loc = 'bottom'

            cbar = plt.colorbar(mesh, ax=ax, location=loc, ticks=ticks, format=formatter)
        else:
            cbar = plt.colorbar(mesh, cax=cax, orientation=cbar_orientation, ticks=ticks, format=formatter)

        if cbar_label is not None:
            if cbar_orientation == 'vertical':
                cbar.set_label(cbar_label, loc='center', rotation=90, labelpad=8)
            else:
                cbar.set_label(cbar_label, loc='center')

    ax.invert_xaxis()
    ax.set_aspect('equal')

    sagA_handle = None
    for orbit in orbits:
        phi = np.linspace(0.0, 2.0 * np.pi, orbit_nphi, endpoint=True)
        (r_orbit,
         lon_orbit,
         lat_orbit,
         v_orbit,
         sagA_r,
         sagA_lon,
         sagA_lat,
         color,
         legend_key,
         arrow_phi_shift) = cmz_model_orbit(phi,
                                            deg_out=True,
                                            orbit=orbit)

        lon_orbit = np.asarray(lon_orbit)
        lat_orbit = np.asarray(lat_orbit)

        if (lon_orbit[0] != lon_orbit[-1]) or (lat_orbit[0] != lat_orbit[-1]):
            lon_orbit = np.append(lon_orbit, lon_orbit[0])
            lat_orbit = np.append(lat_orbit, lat_orbit[0])

        ax.plot(lon_orbit, lat_orbit,
                color=color,
                alpha=orbit_opacity,
                linewidth=2.0,
                zorder=10,
                label=legend_key)

        n = len(phi)
        shift = int(n * (orbit_arrow_phase_shift + arrow_phi_shift) / 2 / np.pi)
        arrow_indices = [shift % n, (n // 2 + shift) % n]

        for i in arrow_indices:
            i0 = max(i - 2, 0)
            i1 = min(i + 2, orbit_nphi)
            ax.annotate('',
                        xy=(lon_orbit[i0], lat_orbit[i0]),
                        xytext=(lon_orbit[i1], lat_orbit[i1]),
                        xycoords='data',
                        arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        alpha=orbit_opacity,
                                        linewidth=orbit_arrow_lw,
                                        mutation_scale=orbit_arrow_scale),
                        annotation_clip=True)

        if sagA_handle is None:
            sagA_handle = ax.scatter(sagA_lon, sagA_lat,
                                     marker='*',
                                     s=120,
                                     color='white',
                                     edgecolors='cyan',
                                     linewidths=0.8,
                                     zorder=20)

    if title is not None:
        ax.text(0.5, 0.95, title,
                transform=ax.transAxes,
                color='white',
                fontsize='large',
                fontweight='bold',
                ha='center',
                va='top')

    if l_ticks:
        ax.set_xlabel(r'$\ell$ (GLON deg)')
        pull_edge_l_tick_labels_inward(ax)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])
    if b_ticks:
        ax.set_ylabel(r'$b$ (GLAT deg)')
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
    return


def lv(observed: ArrayLike,
       ax: typing.Any,
       hyper: hp.Hyper,
       title: str | None = None,
       color_bar: bool = False,
       color_scale: float | None = None,
       color_scale_center: float = 0,
       color_norm_min: float | None = None,
       color_norm_max: float | None = None,
       pin_color_scale_to: ArrayLike | None = None,
       cax: typing.Any | None = None,
       cbar_label: str | None = None,
       cbar_orientation: str = 'horizontal',
       cbar_ticks: typing.Sequence[float] | None = None,
       l_ticks: bool = True,
       v_ticks: bool = True,
       orbits: typing.Sequence[str] = (),
       orbit_opacity: float = 0.8,
       orbit_nphi: int = 512,
       orbit_arrow_scale: float = 18,
       orbit_arrow_lw: float = 1.8,
       orbit_arrow_phase_shift: float = np.pi / 2) -> None:
    reduction = hyper.cube_hyper.reduction
    if len(observed.shape) == 2:
        im = observed.transpose()
    elif reduction == 'max':
        im = peak_intensity(observed, dim=1).transpose()
    elif reduction == 'mean':
        im = mean_intensity(observed, dim=1).transpose()
    else:
        raise ValueError(
            "Invalid cube reduction specified; "
            "hyper.cube_hyper.reduction must be one of 'mean' or 'max' "
            "or cube must already be a reduced lv image.")

    lon_min = hyper.coordinate_hyper.lon_min
    lon_max = hyper.coordinate_hyper.lon_max
    lon_edges = np.linspace(lon_min, lon_max, im.shape[1] + 1)
    v_min = hyper.cube_hyper.v_min
    v_max = hyper.cube_hyper.v_max
    v_edges = np.linspace(v_min, v_max, im.shape[0] + 1)

    if pin_color_scale_to is not None:
        observed_color_scale = pin_color_scale_to
    else:
        observed_color_scale = observed
    if color_norm_min is not None and color_norm_max is not None:
        pass
    elif len(observed.shape) == 2:
        color_norm_min = observed_color_scale.min()
        color_norm_max = observed_color_scale.max()
    elif reduction == 'max':
        color_scale_lb = peak_intensity(observed_color_scale, dim=2)
        color_scale_lv = peak_intensity(observed_color_scale, dim=1)
        color_norm_min = np.minimum(color_scale_lb.min(), color_scale_lv.min())
        color_norm_max = np.maximum(color_scale_lb.max(), color_scale_lv.max())
    elif reduction == 'mean':
        color_scale_lb = mean_intensity(observed_color_scale, dim=2)
        color_scale_lv = mean_intensity(observed_color_scale, dim=1)
        color_norm_min = np.minimum(color_scale_lb.min(), color_scale_lv.min())
        color_norm_max = np.maximum(color_scale_lb.max(), color_scale_lv.max())
    else:
        raise ValueError(
            "Invalid cube reduction specified; "
            "hyper.cube_hyper.reduction must be one of 'mean' or 'max' "
            "or cube/pin_color_scale_to must already be a reduced lb/lv image.")

    if color_scale is None:
        color_scale = 1 / np.mean(observed_color_scale)
    forward = lambda x: np.asinh(color_scale * (x - color_scale_center))
    inverse = lambda y: np.sinh(y) / color_scale + color_scale_center
    norm = colors.FuncNorm((forward, inverse), vmin=color_norm_min, vmax=color_norm_max)
    mesh = ax.pcolormesh(lon_edges, v_edges, im, cmap='gray' if orbits else 'inferno', norm=norm)

    if color_bar:
        if cbar_ticks is not None:
            ticks = np.array(cbar_ticks)
            format = lambda value, pos: '2.73' if np.abs(value - 2.73) < 1e-3 else f'{value:.1f}'
        else:
            values = norm.inverse(np.linspace(0, norm(color_norm_max), 8))
            values = values[values > 1e-10]
            powers = np.ceil(np.log10(values))
            powers = np.unique(powers.astype(np.int32)).astype(np.float32)
            ticks = np.insert(10. ** powers, 0, 0)
            if color_norm_min < 0:
                values = -norm.inverse(np.linspace(0, norm(color_norm_min), 8))
                values = values[values > 1e-10]
                neg_powers = np.ceil(np.log10(values))
                neg_ticks = -(10. ** neg_powers)
                ticks = np.concatenate((neg_ticks, ticks), axis=0)
            def format(value: float, pos: int | None) -> str:
                if value > 1e-10:
                    exponent = int(np.log10(value))
                    return rf'$10^{{{exponent}}}$'
                elif value < -1e-10:
                    exponent = int(np.log10(-value))
                    return rf'$-1 \times 10^{{{exponent}}}$'
                else:
                    return '0'

        formatter = ticker.FuncFormatter(format)

        if cax is None:
            if cbar_orientation == 'vertical':
                loc = 'right'
            else:
                loc = 'bottom'

            cbar = plt.colorbar(mesh, ax=ax, location=loc, ticks=ticks, format=formatter)
        else:
            cbar = plt.colorbar(mesh, cax=cax, orientation=cbar_orientation, ticks=ticks, format=formatter)

        if cbar_label is not None:
            if cbar_orientation == 'vertical':
                cbar.set_label(cbar_label, loc='center', rotation=90, labelpad=8)
            else:
                cbar.set_label(cbar_label, loc='center')

    ax.invert_xaxis()
    ax.set_aspect((lon_max - lon_min) / (v_max - v_min) / 3)

    sagA_handle = None
    for orbit in orbits:
        phi = np.linspace(0.0, 2.0 * np.pi, orbit_nphi, endpoint=True)
        (r_orbit,
         lon_orbit,
         lat_orbit,
         v_orbit,
         sagA_r,
         sagA_lon,
         sagA_lat,
         color,
         legend_key,
         arrow_phi_shift) = cmz_model_orbit(phi,
                                            deg_out=True,
                                            orbit=orbit)

        lon_orbit = np.asarray(lon_orbit)
        v_orbit = np.asarray(v_orbit)

        if (lon_orbit[0] != lon_orbit[-1]) or (v_orbit[0] != v_orbit[-1]):
            lon_orbit = np.append(lon_orbit, lon_orbit[0])
            v_orbit = np.append(v_orbit, v_orbit[0])

        ax.plot(lon_orbit, v_orbit,
                color=color,
                alpha=orbit_opacity,
                linewidth=2.0,
                zorder=10,
                label=legend_key)

        n = len(phi)
        shift = int(n * (orbit_arrow_phase_shift + arrow_phi_shift) / 2 / np.pi)
        arrow_indices = [shift % n, (n // 2 + shift) % n]

        for i in arrow_indices:
            i0 = max(i - 2, 0)
            i1 = min(i + 2, orbit_nphi)
            ax.annotate('',
                        xy=(lon_orbit[i0], v_orbit[i0]),
                        xytext=(lon_orbit[i1], v_orbit[i1]),
                        xycoords='data',
                        arrowprops=dict(arrowstyle='->',
                                        color=color,
                                        alpha=orbit_opacity,
                                        linewidth=orbit_arrow_lw,
                                        mutation_scale=orbit_arrow_scale),
                        annotation_clip=True)

        if sagA_handle is None:
            sagA_handle = ax.scatter(sagA_lon, 0.0,
                                     marker='*',
                                     s=120,
                                     color='white',
                                     edgecolors='cyan',
                                     linewidths=0.8,
                                     zorder=20)

    if title is not None:
        ax.text(0.5, 0.95, title,
                transform=ax.transAxes,
                color='white',
                fontsize='large',
                fontweight='bold',
                ha='center',
                va='top')

    if l_ticks:
        ax.set_xlabel(r'$\ell$ (GLON deg)')
        pull_edge_l_tick_labels_inward(ax)
    else:
        ax.set_xlabel("")
        ax.set_xticks([])

    if v_ticks:
        ax.set_ylabel(r'$v$ (km/s)')
    else:
        ax.set_ylabel("")
        ax.set_yticks([])
    return

def peak_intensity(cube: ArrayLike, dim: int) -> np.ndarray:
    return np.max(cube, axis=dim)

def mean_intensity(cube: ArrayLike, dim: int) -> np.ndarray:
    return np.mean(cube, axis=dim)

def cmz_model_orbit(phi: ArrayLike,
                    deg_out: bool = True,
                    orbit: str = 'lipman median') -> tuple[ArrayLike,
                                                           ArrayLike,
                                                           ArrayLike,
                                                           ArrayLike,
                                                           float,
                                                           float,
                                                           float,
                                                           str,
                                                           str,
                                                           float]:
    r_0_adapted = 8277.0                # pc
    sagA_r = 8277.0                     # pc
    sagA_lon = -0.056 / 180 * np.pi     # rad
    sagA_lat = -0.046 / 180 * np.pi     # rad
    if orbit == 'walker':
        a = 90.0                        # pc
        b = 55.0                        # pc
        z_0 = 12.5                      # pc
        alpha = 0.4                     # rad
        theta = 25.0 / 180 * np.pi      # rad
        v_0 = 130.0                     # km/s
        r_0 = 8100.0                    # pc
        lon_center = 0.05 / 180 * np.pi      # rad
        lat_center = sagA_lat
        x_center = 0.0
        y_center = r_0
        z_center = 0.0
        v_x_center = 2.2                # km/s
        v_y_center = 0.0                # km/s
        v_z_center = 0.0                # km/s
        color = '#FFB000'
        legend_key = 'Walker et al. (2025)'
        arrow_phi_shift = np.pi / 8
    elif orbit == 'lipman min':
        a = 72.0                        # pc
        b = 26.0                        # pc
        z_0 = 14.6                      # pc
        alpha = 0.04                    # rad
        theta = 15.8 / 180 * np.pi      # rad
        v_0 = 130.9                     # km/s
        r_0 = 8100.0                    # pc
        lon_center = sagA_lon
        lat_center = sagA_lat
        x_center = 0.0
        y_center = r_0
        z_center = 0.0
        v_x_center = 0.0                # km/s
        v_y_center = 0.0                # km/s
        v_z_center = 0.0                # km/s
        color = '#FF2DAA'
        legend_key = 'Lipman et al. (2026) Min'
        arrow_phi_shift = -np.pi / 8
    elif orbit == 'lipman median':
        a = 83.0                        # pc
        b = 34.0                        # pc
        z_0 = 14.0                      # pc
        alpha = 0.12                    # rad
        theta = 16.6 / 180 * np.pi      # rad
        v_0 = 129.6                     # km/s
        r_0 = 8100.0                    # pc
        lon_center = sagA_lon
        lat_center = sagA_lat
        x_center = 0.0
        y_center = r_0
        z_center = 0.0
        v_x_center = 0.0                # km/s
        v_y_center = 0.0                # km/s
        v_z_center = 0.0                # km/s
        color = '#0072FF'
        legend_key = 'Lipman et al. (2026) Median'
        arrow_phi_shift = 0
    elif orbit == 'lipman max':
        a = 146.0                       # pc
        b = 58.0                        # pc
        z_0 = 14.4                      # pc
        alpha = 0.11                    # rad
        theta = 42.9 / 180 * np.pi      # rad
        v_0 = 109.0                     # km/s
        r_0 = 8100.0                    # pc
        lon_center = sagA_lon
        lat_center = sagA_lat
        x_center = 0.0
        y_center = r_0
        z_center = 0.0
        v_x_center = 0.0                # km/s
        v_y_center = 0.0                # km/s
        v_z_center = 0.0                # km/s
        color = '#00993f'
        legend_key = 'Lipman et al. (2026) Max'
        arrow_phi_shift = np.pi / 4
    else:
        raise ValueError('Invalid orbit model.')
    L_0 = v_0 * b

    x = a * np.cos(phi)
    y = -b * np.sin(phi)
    z = z_0 * np.sin(alpha - 2 * phi)

    dx_dphi = -a * np.sin(phi)
    dy_dphi = -b * np.cos(phi)
    norm = np.sqrt(dx_dphi * dx_dphi + dy_dphi * dy_dphi)

    R = np.sqrt(x * x + y * y)
    v_rot = L_0 / R
    beta = np.atan2(y, x)
    cos_psi = (dx_dphi * np.sin(beta) - dy_dphi * np.cos(beta)) / norm
    v = v_rot / cos_psi
    v_x = v * dx_dphi / norm
    v_y = v * dy_dphi / norm

    xy = np.expand_dims(np.stack((x, y), axis=-1), axis=-1)
    rot = np.array([[[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]]], dtype=np.float32)
    xy = np.matmul(rot, xy).squeeze(axis=-1)
    x = xy[:, 0]
    y = xy[:, 1]

    v_xy = np.expand_dims(np.stack((v_x, v_y), axis=-1), axis=-1)
    v_xy = np.matmul(rot, v_xy).squeeze(axis=-1)
    v_x = v_xy[:, 0]
    v_y = v_xy[:, 1]
    v_z = np.zeros_like(v_x)

    x += x_center
    y += y_center
    z += z_center
    v_x += v_x_center
    v_y += v_y_center
    v_z += v_z_center

    xy = x * x + y * y
    r = np.sqrt(xy + z * z)
    xy = np.sqrt(xy)
    lon = np.atan(-x / y) + lon_center
    lat = np.atan(z / xy) + lat_center
    v_r = (x * v_x + y * v_y + z * v_z) / r
    r *= r_0_adapted / r_0
    if deg_out:
        lon *= 180 / np.pi
        lat *= 180 / np.pi
        sagA_lon *= 180 / np.pi
        sagA_lat *= 180 / np.pi
    return r, lon, lat, v_r, sagA_r, sagA_lon, sagA_lat, color, legend_key, arrow_phi_shift
