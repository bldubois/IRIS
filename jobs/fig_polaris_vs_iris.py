# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois, Jonah Baade, Jack Sullivan, and Stefan Reissl
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the POLARIS-vs-IRIS comparison figure from the IRIS paper (DuBois et al., 2026).
"""

from __future__ import annotations

from collections import defaultdict
import math
import os
from pathlib import Path
import struct
import subprocess
import time as pytime

from mpi4py import MPI
import numpy as np
import torch
from astropy.io import fits
from scipy.spatial import ConvexHull, Delaunay, Voronoi

from iris import hyper as hp
from iris import cube_processing
from iris import arepo_processing
from iris import arepo_processing_write as ap
from iris import observation
from iris import visualization


def read_positive_int_env(*names: str, default: int = 1) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except ValueError:
            continue
    return default


SNAPSHOT_PATH = '/path/to/snapshot.hdf5'
IRIS_DATASET_DIR = '~/IRIS/data/polaris_sidebyside'
POLARIS_FILES_DIR = '~/IRIS/data/polaris_files'
POLARIS_EXECUTABLE = '~/POLARIS/bin/polaris'
POLARIS_DUST_PATH = '~/IRIS/chem/polaris_dust.nk'
POLARIS_THREADS = read_positive_int_env('POLARIS_THREADS', 'SLURM_CPUS_PER_TASK', 'OMP_NUM_THREADS')
OBSERVE_IRIS = True
OBSERVE_POLARIS = True
PEAK_13CO_ABUNDANCE = 4e-2
MU_POLARIS = 2.0
DUST_TO_GAS_RATIO = 0.01
LAMDA_PEAK_COLLISION_TEMPERATURE = np.inf #2995.0
SIDE_BY_SIDE_PATH = '~/IRIS/output/polaris_vs_iris.png'


class TestConfig(hp.Hyper):
    def __init__(self):
        super().__init__()

        self.writer_hyper.points_per_snapshot = 1
        self.writer_hyper.unit_calculation_sample_size = 1

        self.dataset_hyper.CMZ_scale_factor = None
        self.dataset_hyper.CMZ_scale_range = None
        self.dataset_hyper.CMZ_density_factor = None
        self.dataset_hyper.CMZ_density_range = None
        self.dataset_hyper.CMZ_skew_factor = None
        self.dataset_hyper.CMZ_skew_range = None

        self.coordinate_hyper.theta_zero = 270.
        self.coordinate_hyper.spin_orientation = -1
        self.coordinate_hyper.observer_radius = 8277.
        self.coordinate_hyper.r_steps = 256
        self.coordinate_hyper.r_crop_min_index = 0
        self.coordinate_hyper.r_crop_max_index = 256
        self.coordinate_hyper.r_min = 7677.
        self.coordinate_hyper.r_max = 8877.
        self.coordinate_hyper.lon_steps = 382
        self.coordinate_hyper.lon_min = -3.
        self.coordinate_hyper.lon_max = 3.
        self.coordinate_hyper.lat_steps = 128
        self.coordinate_hyper.lat_min = -1.
        self.coordinate_hyper.lat_max = 1.

        self.observer_hyper.lon_pieces = 1
        self.observer_hyper.lat_pieces = 16
        self.observer_hyper.v_subsamples = 2
        self.observer_hyper.blur_inputs = True
        self.observer_hyper.blur_kernel_r = 3
        self.observer_hyper.blur_kernel_lon = 5
        self.observer_hyper.blur_kernel_lat = 3
        self.observer_hyper.out_blur_fwhm = None
        self.observer_hyper.T_cmb = 0.
        self.observer_hyper.n_lines = 1
        self.observer_hyper.chem_path = ['~/IRIS/chem/13C16O_no_H.dat']
        self.observer_hyper.transition = [(2, 1)]
        self.observer_hyper.kappa_dust = [1e-3]
        self.observer_hyper.N_H_TOT_steps = [128]
        self.observer_hyper.interpolation_max_N_H_TOT = [1.0e12]
        self.observer_hyper.bolic_normalization = [1.]
        self.observer_hyper.abundance_H2_steps = [64]
        self.observer_hyper.T_steps = [64]
        self.observer_hyper.interpolation_max_T = [3000.]
        self.observer_hyper.T_inf = 5e4
        self.observer_hyper.T_continuum = 2.75

        self.cube_hyper.fits_map = self.from_fits
        self.cube_hyper.conversion_raw_to_T_K = [self.intensity_to_raleigh_jeans_temperature]
        self.cube_hyper.clean_noise = [None]
        self.cube_hyper.reduction = 'mean'
        self.cube_hyper.v_min = -400.
        self.cube_hyper.v_max = 400.
        self.cube_hyper.v_steps = 128
        return

    def from_fits(self) -> list:
        total_path = Path(os.path.expanduser(POLARIS_FILES_DIR)) / 'total.fits'
        total = fits.getdata(total_path)
        continuum_path = Path(os.path.expanduser(POLARIS_FILES_DIR)) / 'continuum.fits'
        continuum = fits.getdata(continuum_path)
        polaris = total - continuum
        return [polaris.astype(np.float32).copy()]

    def intensity_to_raleigh_jeans_temperature(self, I, hyper):
        return cube_processing.intensity_to_raleigh_jeans_temperature(I=I, hyper=hyper, nu_ul=220.3986841281e9)


def processing_units(hyper: hp.Hyper) -> tuple[float, float, float, float]:
    length_m = hyper.writer_hyper.length_cm_per_processing / 100
    velocity_m_s = hyper.writer_hyper.velocity_cm_per_s_per_processing / 100
    mass_kg = hyper.writer_hyper.mass_g_per_processing / 1000
    density_kg_m3 = mass_kg / length_m / length_m / length_m
    return length_m, velocity_m_s, mass_kg, density_kg_m3


def velocity_grid(hyper: hp.Hyper, fine: bool) -> np.ndarray:
    v_min = hyper.cube_hyper.v_min * 1000
    v_max = hyper.cube_hyper.v_max * 1000
    v_steps = hyper.cube_hyper.v_steps
    if not fine or hyper.observer_hyper.v_subsamples <= 0:
        return np.linspace(v_min, v_max, v_steps, dtype=np.float64)

    fine_steps = 2 * v_steps * hyper.observer_hyper.v_subsamples + 1
    dv = (v_max - v_min) / (v_steps - 1)
    return np.linspace(v_min - dv / 2, v_max + dv / 2, fine_steps, dtype=np.float64)


def polaris_line_velocity_grid(hyper: hp.Hyper) -> np.ndarray:
    target_velocity = velocity_grid(hyper, fine=True)
    max_velocity = max(abs(target_velocity[0]), abs(target_velocity[-1]))
    if target_velocity.size == 1 or max_velocity == 0:
        return np.array([0.], dtype=np.float64)

    target_dv = target_velocity[1] - target_velocity[0]
    intervals_float = 2 * max_velocity / target_dv
    intervals = int(round(intervals_float))
    if not np.isclose(intervals_float, intervals, rtol=0, atol=1e-8):
        raise ValueError(
            'Cannot construct a symmetric POLARIS velocity grid that contains '
            'the IRIS fine velocity grid as an exact contiguous subset. Adjust '
            'v_min/v_max/v_steps/v_subsamples so max(|v_fine|) is an integer '
            'multiple of the fine velocity spacing.')
    return np.linspace(-max_velocity, max_velocity, intervals + 1, dtype=np.float64)


def wavelength_grid(hyper: hp.Hyper, fine: bool) -> np.ndarray:
    nu_ul = 220.3986841281e9
    c = hyper.observer_hyper.c
    v = velocity_grid(hyper, fine=fine)
    return c / ((1 + v / c) * nu_ul)


def line_center_wavelength(hyper: hp.Hyper) -> float:
    nu_ul = 220.3986841281e9
    return hyper.observer_hyper.c / nu_ul


def oriented_bounds(lo: float, hi: float, orientation: int) -> tuple[float, float]:
    oriented = (lo * orientation, hi * orientation)
    return min(oriented), max(oriented)


def observer_geometry(snapshot: arepo_processing.Snapshot | None,
                      hyper: hp.Hyper) -> tuple[float, float, float, float, float, float, float]:
    if hyper.writer_hyper._length_parsec_per_processing is None:
        cm_per_parsec = 100.0 * hyper.dataset_hyper.meters_per_parsec
        hyper.writer_hyper._length_parsec_per_processing = (
            hyper.writer_hyper.length_cm_per_processing / cm_per_parsec)

    length_parsec_per_processing = hyper.writer_hyper._length_parsec_per_processing
    theta = hyper.coordinate_hyper.theta_zero * np.pi / 180
    observer_r = hyper.coordinate_hyper.observer_radius / length_parsec_per_processing
    orientation = hyper.coordinate_hyper.spin_orientation
    lon_min, lon_max = oriented_bounds(hyper.coordinate_hyper.lon_min * np.pi / 180,
                                       hyper.coordinate_hyper.lon_max * np.pi / 180,
                                       orientation)
    lat_min, lat_max = oriented_bounds(hyper.coordinate_hyper.lat_min * np.pi / 180,
                                       hyper.coordinate_hyper.lat_max * np.pi / 180,
                                       orientation)
    r_min = hyper.coordinate_hyper.r_min / length_parsec_per_processing
    r_max = hyper.coordinate_hyper.r_max / length_parsec_per_processing
    return observer_r, theta, r_min, r_max, lon_min, lon_max, lat_min, lat_max


def plane_detector_geometry(hyper: hp.Hyper) -> tuple[float, float, float, float, float]:
    length_m, _, _, _ = processing_units(hyper)
    observer_r, theta, _, _, _, _, _, _ = observer_geometry(None, hyper)
    distance_m = observer_r * length_m

    lon_min = math.radians(hyper.coordinate_hyper.lon_min)
    lon_max = math.radians(hyper.coordinate_hyper.lon_max)
    lat_min = math.radians(hyper.coordinate_hyper.lat_min)
    lat_max = math.radians(hyper.coordinate_hyper.lat_max)
    lon_center = 0.5 * (lon_min + lon_max)
    dlon = (lon_max - lon_min) / (hyper.coordinate_hyper.lon_steps - 1)
    dlat = (lat_max - lat_min) / (hyper.coordinate_hyper.lat_steps - 1)
    sidelength_x = distance_m * abs(lon_max - lon_min + dlon)
    sidelength_y = distance_m * abs(lat_max - lat_min + dlat)

    # POLARIS plane rays initially propagate along +z. With the default IRIS
    # midplane target, rotate them to the observer-to-center direction. The
    # detector x/y axes then track increasing internal lon/lat respectively.
    rot_angle_1 = 90.0
    internal_lon_center = hyper.coordinate_hyper.spin_orientation * lon_center
    rot_angle_2 = (math.degrees(theta + internal_lon_center) - 90.0) % 360.0
    return rot_angle_1, rot_angle_2, distance_m, sidelength_x, sidelength_y


def prune_snapshot(snapshot: arepo_processing.Snapshot, hyper: hp.Hyper) -> tuple[np.ndarray, ...]:
    observer_r, theta, r_min, r_max, lon_min, lon_max, lat_min, lat_max = observer_geometry(snapshot, hyper)
    return snapshot._prune_particles(
        r_min,
        r_max,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        observer_r,
        theta,
        snapshot.positions,
        snapshot.velocities,
        snapshot.densities,
        snapshot.temperatures,
        snapshot.abundances_H2,
        snapshot.abundances_CO,
        snapshot.dust_temperatures)


def find_neighbors(tri: Delaunay) -> dict[int, set[int]]:
    neighbors = defaultdict(set)
    for simplex in tri.simplices:
        for idx in simplex:
            others = set(simplex)
            others.remove(idx)
            neighbors[idx] = neighbors[idx].union(others)
    return neighbors


def voronoi_volumes(points: np.ndarray,
                    min_x: float,
                    max_x: float,
                    min_y: float,
                    max_y: float,
                    min_z: float,
                    max_z: float) -> np.ndarray:
    voronoi = Voronoi(points)
    volumes = np.zeros(voronoi.npoints, dtype=np.float64)
    for i, reg_num in enumerate(voronoi.point_region):
        indices = voronoi.regions[reg_num]
        if -1 in indices or len(indices) == 0:
            continue
        vertices = voronoi.vertices[indices]
        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        if x.min() <= min_x or x.max() >= max_x:
            continue
        if y.min() <= min_y or y.max() >= max_y:
            continue
        if z.min() <= min_z or z.max() >= max_z:
            continue
        volumes[i] = ConvexHull(vertices).volume
    return volumes


def write_polaris_grid(snapshot: arepo_processing.Snapshot,
                       hyper: hp.Hyper,
                       path: Path,
                       peak_13CO_abundance: float,
                       mu_polaris: float = MU_POLARIS) -> None:
    (positions,
     velocities,
     densities,
     temperatures,
     abundances_H2,
     abundances_CO,
     dust_temperatures) = prune_snapshot(snapshot, hyper)

    if len(positions) < 4:
        raise RuntimeError('POLARIS Voronoi grid requires at least four cells after pruning.')

    length_m, velocity_m_s, _, density_kg_m3 = processing_units(hyper)
    positions_m = positions.astype(np.float64) * length_m
    velocities_m_s = velocities.astype(np.float64) * velocity_m_s
    densities_kg_m3 = densities.astype(np.float64) * density_kg_m3
    temperatures_K = temperatures.astype(np.float64) * hyper.writer_hyper.temperature_K_per_processing
    dust_temperatures_K = dust_temperatures.astype(np.float64) * hyper.writer_hyper.temperature_K_per_processing

    gas_mass_per_H_nucleus = (
        hyper.observer_hyper.m_H
        + hyper.observer_hyper.abundance_He * hyper.observer_hyper.m_He)
    n_H_total = densities_kg_m3 / gas_mass_per_H_nucleus
    n_H2 = n_H_total * abundances_H2.astype(np.float64)
    n_13co = n_H_total * abundances_CO.astype(np.float64) * peak_13CO_abundance
    h2_mass_density_kg_m3 = n_H2 * mu_polaris * hyper.observer_hyper.m_H
    dust_mass_density_kg_m3 = densities_kg_m3 * DUST_TO_GAS_RATIO
    n_H2 = np.where(np.isfinite(n_H2) & (n_H2 > 0), n_H2, 0.)
    n_13co = np.where(np.isfinite(n_13co) & (n_13co > 0), n_13co, 0.)
    h2_mass_density_kg_m3 = np.where(
        np.isfinite(h2_mass_density_kg_m3) & (h2_mass_density_kg_m3 > 0),
        h2_mass_density_kg_m3,
        0.)
    dust_mass_density_kg_m3 = np.where(
        np.isfinite(dust_mass_density_kg_m3) & (dust_mass_density_kg_m3 > 0),
        dust_mass_density_kg_m3,
        0.)
    # The POLARIS Grid ID 17 (GRIDratio) id defined as n_species / n_gas
    # Here the POLARIS gas density is the H2 collider density, so ID 17 is n_13CO / n_H2.
    grid_ratio_13co = np.divide(
        n_13co,
        n_H2,
        out=np.zeros_like(n_13co),
        where=n_H2 > 0)

    grid_ratio_13co = np.where(np.isfinite(temperatures_K), grid_ratio_13co, 0.)
    temperatures_K = np.where(np.isfinite(temperatures_K), temperatures_K, 0.)
    temperatures_K = np.minimum(temperatures_K, LAMDA_PEAK_COLLISION_TEMPERATURE)

    points = positions.astype(np.float64)
    l_max = 1.005 * 2 * np.max(np.abs(points))
    lo = -l_max / 2
    hi = l_max / 2

    print(f'Creating POLARIS Voronoi mesh with {len(points)} cells...', flush=True)
    tri = Delaunay(points)
    neighbors = find_neighbors(tri)
    hull = set(int(i) for i in tri.convex_hull.ravel())
    tri.close()

    volumes = voronoi_volumes(points, lo, hi, lo, hi, lo, hi)
    missing = volumes == 0
    if np.any(missing):
        dv = l_max**3 - volumes.sum()
        volumes[missing] = dv / np.count_nonzero(missing)

    volumes_m3 = volumes * length_m**3
    l_max_m = l_max * length_m
    data_ids = [28, 29, 2, 3, 7, 8, 9, 17]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(struct.pack('H', 50))
        f.write(struct.pack('H', len(data_ids)))
        for data_id in data_ids:
            f.write(struct.pack('H', data_id))
        f.write(struct.pack('d', float(len(points))))
        f.write(struct.pack('d', float(l_max_m)))

        for i in range(len(points)):
            f.write(struct.pack('f', float(positions_m[i, 0])))
            f.write(struct.pack('f', float(positions_m[i, 1])))
            f.write(struct.pack('f', float(positions_m[i, 2])))
            f.write(struct.pack('d', float(volumes_m3[i])))

            f.write(struct.pack('f', float(h2_mass_density_kg_m3[i])))
            f.write(struct.pack('f', float(dust_mass_density_kg_m3[i])))
            f.write(struct.pack('f', float(dust_temperatures_K[i])))
            f.write(struct.pack('f', float(temperatures_K[i])))
            f.write(struct.pack('f', float(velocities_m_s[i, 0])))
            f.write(struct.pack('f', float(velocities_m_s[i, 1])))
            f.write(struct.pack('f', float(velocities_m_s[i, 2])))
            f.write(struct.pack('f', float(grid_ratio_13co[i])))

            neighbor_list = sorted(neighbors[i])
            neighbor_count = len(neighbor_list)
            if i in hull:
                neighbor_count *= -1
            f.write(struct.pack('i', int(neighbor_count)))
            for neighbor in neighbor_list:
                f.write(struct.pack('i', int(neighbor)))
    print(f'Wrote POLARIS grid to {path}.', flush=True)


def write_polaris_command(hyper: hp.Hyper,
                          path: Path,
                          grid_path: Path,
                          line_out: Path,
                          cont_out: Path,
                          nr_threads: int = POLARIS_THREADS,
                          mu_polaris: float = MU_POLARIS) -> None:
    line_velocity = polaris_line_velocity_grid(hyper)
    continuum_wavelength = line_center_wavelength(hyper)
    rot_angle_1, rot_angle_2, distance_m, sidelength_x, sidelength_y = plane_detector_geometry(hyper)

    line_out.mkdir(parents=True, exist_ok=True)
    cont_out.mkdir(parents=True, exist_ok=True)
    gas_path = os.path.expanduser(hyper.observer_hyper.chem_path[0])
    dust_path = os.path.expanduser(POLARIS_DUST_PATH)
    line_channels = line_velocity.size
    max_velocity = np.max(np.abs(line_velocity))
    lon_steps = hyper.coordinate_hyper.lon_steps
    lat_steps = hyper.coordinate_hyper.lat_steps
    text = f"""# Generated by jobs/fig_polaris_vs_iris.py
<common>
    <dust_component> "{dust_path}" "plaw" 1.0 920.0 5e-09 2.5e-07 -3.5
    # The grid contains explicit GRIDdust_mdens values; this is only POLARIS' fallback if that field is absent.
    <mass_fraction> {DUST_TO_GAS_RATIO:.12g}
    <mu> {mu_polaris:.12g}
    <nr_threads> {nr_threads}
    <axis1> 1 0 0
    <axis2> 0 0 1
</common>

<task> 1
    <cmd> CMD_LINE_EMISSION
    <vel_maps> 1
    <max_subpixel_lvl> 1
    <gas_species> "{gas_path}" 2 -1
    <detector_line nr_pixel = "{lon_steps}*{lat_steps}" vel_channels = "{line_channels}"> 1 2 1 {max_velocity:.12e} {rot_angle_1:.12e} {rot_angle_2:.12e} {distance_m:.12e} {sidelength_x:.12e} {sidelength_y:.12e}
    <path_grid> "{grid_path}"
    <path_out> "{line_out}/"
</task>

<task> 1
    <cmd> CMD_DUST_EMISSION
    <max_subpixel_lvl> 1
    <detector_dust nr_pixel = "{lon_steps}*{lat_steps}"> {continuum_wavelength:.12e} {continuum_wavelength:.12e} 1 1 {rot_angle_1:.12e} {rot_angle_2:.12e} {distance_m:.12e} {sidelength_x:.12e} {sidelength_y:.12e}
    <path_grid> "{grid_path}"
    <path_out> "{cont_out}/"
</task>
"""
    path.write_text(text)
    return


def find_polaris_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f'No POLARIS output matching {pattern} under {directory}.')
    return matches[0]


def plane_sr_per_pixel(header: fits.Header) -> float:
    return abs(float(header['CDELT1']) * float(header['CDELT2'])) / float(header['DISTANCE'])**2


def orient_plane_image(image: np.ndarray, hyper: hp.Hyper) -> np.ndarray:
    image = image.T
    if hyper.coordinate_hyper.spin_orientation == -1:
        image = np.flip(image, axis=(0, 1))
    return image


def read_polaris_plane_line_image(path: Path, hyper: hp.Hyper) -> np.ndarray:
    with fits.open(path) as hdul:
        sr_per_pixel = plane_sr_per_pixel(hdul[0].header)
        image = np.array(hdul[0].data[0], dtype=np.float64)
    return (orient_plane_image(image, hyper) / sr_per_pixel).astype(np.float32)


def read_polaris_line_cube(line_out: Path, hyper: hp.Hyper) -> np.ndarray:
    pattern = 'vel_channel_maps_species_0001_line_0001_vel_*.fits*'
    files = sorted(line_out.rglob(pattern))
    if not files:
        raise FileNotFoundError(f'No POLARIS line channel maps under {line_out}.')
    cube = np.empty((hyper.coordinate_hyper.lon_steps,
                     hyper.coordinate_hyper.lat_steps,
                     len(files)), dtype=np.float32)
    for i, path in enumerate(files):
        cube[..., i] = read_polaris_plane_line_image(path, hyper)
    # POLARIS labels velocity channels along the detector ray direction used
    # above, opposite to IRIS' positive-toward-observer radial convention.
    return cube[..., ::-1].copy()


def read_polaris_continuum_image(cont_out: Path, hyper: hp.Hyper) -> np.ndarray:
    path = find_polaris_file(cont_out, 'polaris_detector_nr0001.fits*')
    with fits.open(path) as hdul:
        sr_per_pixel = plane_sr_per_pixel(hdul[0].header)
        data = np.array(hdul[0].data[0], dtype=np.float64)
    return (orient_plane_image(data[0], hyper) / sr_per_pixel).astype(np.float32)


def crop_velocity_cube(cube: np.ndarray,
                       source_velocity: np.ndarray,
                       target_velocity: np.ndarray) -> np.ndarray:
    if cube.shape[-1] != source_velocity.size:
        raise ValueError(
            f'Expected {source_velocity.size} POLARIS velocity channels, got {cube.shape[-1]}.')

    tolerance = max(1e-6, 1e-12 * np.max(np.abs(source_velocity)))
    if target_velocity[0] < source_velocity[0] - tolerance or target_velocity[-1] > source_velocity[-1] + tolerance:
        raise ValueError('POLARIS velocity grid does not cover the requested IRIS velocity grid.')

    start = int(np.argmin(np.abs(source_velocity - target_velocity[0])))
    stop = start + target_velocity.size
    if stop > source_velocity.size:
        raise ValueError('POLARIS velocity grid does not contain enough channels for the requested crop.')

    cropped_velocity = source_velocity[start:stop]
    if not np.allclose(cropped_velocity, target_velocity, rtol=0, atol=tolerance):
        raise ValueError(
            'POLARIS velocity channels do not contain the IRIS fine velocity grid '
            'as an exact contiguous subset; refusing to interpolate.')
    return cube[..., start:stop]


def integrate_velocity_cube(cube: np.ndarray, hyper: hp.Hyper) -> np.ndarray:
    v_subsamples = hyper.observer_hyper.v_subsamples
    if v_subsamples <= 0:
        return cube

    v_steps = hyper.cube_hyper.v_steps
    fine_steps = 2 * v_steps * v_subsamples + 1
    if cube.shape[-1] != fine_steps:
        raise ValueError(
            f'Expected {fine_steps} fine velocity channels, got {cube.shape[-1]}.')

    # Simpson frequency integration from TransferProcessor._optically_thick_transfer_bdf2
    I_d_nu_normed = cube[..., 1:-1:2].copy()
    I_d_nu_normed *= 4
    I_d_nu_normed += cube[..., :-2:2]
    I_d_nu_normed += cube[..., 2::2]
    I_d_nu_normed /= 6 * v_subsamples
    return I_d_nu_normed.reshape(cube.shape[:-1] + (v_steps, v_subsamples)).sum(axis=-1)


def write_polaris_fits(files_dir: Path, hyper: hp.Hyper) -> None:
    native_velocity = polaris_line_velocity_grid(hyper)
    target_velocity = velocity_grid(hyper, fine=True)
    line_cube_native = read_polaris_line_cube(files_dir / 'line', hyper)
    line_cube_fine = crop_velocity_cube(line_cube_native, native_velocity, target_velocity)
    line_cube = integrate_velocity_cube(line_cube_fine, hyper)
    continuum_image = read_polaris_continuum_image(files_dir / 'cont', hyper)
    continuum_cube = np.broadcast_to(continuum_image[..., None], line_cube.shape).copy()
    fits.writeto(files_dir / 'total.fits', line_cube.astype(np.float32), overwrite=True)
    fits.writeto(files_dir / 'continuum.fits', continuum_cube.astype(np.float32), overwrite=True)
    return


def clear_polaris_outputs(files_dir: Path) -> None:
    for pattern in [
        'line/**/*.fits',
        'line/**/*.fits.gz',
        'cont/**/*.fits',
        'cont/**/*.fits.gz',
        'total.fits',
        'continuum.fits',
        'polaris.fits',
    ]:
        for path in files_dir.glob(pattern):
            path.unlink()
    return


def run_polaris(command_path: Path, nr_threads: int = POLARIS_THREADS) -> float:
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(nr_threads)
    print(f'Running POLARIS with {nr_threads} OpenMP thread(s).', flush=True)
    start = pytime.time()
    subprocess.run([os.path.expanduser(POLARIS_EXECUTABLE), str(command_path)], check=True, env=env)
    end = pytime.time()
    return end - start


def observe_polaris(hyper: hp.Hyper) -> float:
    files_dir = Path(os.path.expanduser(POLARIS_FILES_DIR))
    files_dir.mkdir(parents=True, exist_ok=True)
    grid_path = files_dir / 'grid_voronoi.dat'
    command_path = files_dir / 'POLARIS.cmd'
    clear_polaris_outputs(files_dir)

    snapshot = arepo_processing.Snapshot(SNAPSHOT_PATH, hyper, gpu_interpolate=False)
    write_polaris_grid(snapshot, hyper, grid_path, PEAK_13CO_ABUNDANCE)
    write_polaris_command(hyper,
                          command_path,
                          grid_path,
                          files_dir / 'line',
                          files_dir / 'cont',
                          nr_threads=POLARIS_THREADS)

    polaris_time = run_polaris(command_path, POLARIS_THREADS)
    write_polaris_fits(files_dir, hyper)
    return polaris_time


def observe_iris(writer: ap.Writer, hyper: hp.Hyper) -> tuple[observation.IteratedSyntheticObserver, torch.Tensor, float]:
    abundance = observation.Constant_CO_13C16O(hyper, peak=PEAK_13CO_ABUNDANCE)
    observer = observation.IteratedSyntheticObserver(hyper, abundance=abundance)
    observer.cuda()
    observer.eval()
    arepo = writer.dataset.sample(1, validation=False).cuda()
    start = pytime.time()
    observed = observer(arepo, units='Trj K').detach().cpu()
    end = pytime.time()
    return observer, observed, end - start


if not OBSERVE_IRIS and not OBSERVE_POLARIS:
    raise SystemExit

hyper = TestConfig()
hyper.validate()

if OBSERVE_IRIS:
    writer = ap.Writer(path=IRIS_DATASET_DIR,
                       snapshot_paths=[SNAPSHOT_PATH],
                       hyper=hyper,
                       dataset_type=ap.StandardDataset,
                       gpu_interpolate=True,
                       gpu_normalize=False,
                       verbose=True)
    rank = writer.rank
else:
    world_comm = MPI.COMM_WORLD
    rank = world_comm.Get_rank()

if rank == 0:
    if OBSERVE_POLARIS:
        polaris_time = observe_polaris(hyper)
        print(f'POLARIS Total Time:\t\t{polaris_time / 60:.2f} min', flush=True)

    if OBSERVE_IRIS:
        hyper.observer_hyper.v_subsamples = 8
        observer, observed, iris_time = observe_iris(writer, hyper)
        print(f'IRIS-SO Total Time:\t\t{iris_time:.2f} s', flush=True)
        if OBSERVE_POLARIS:
            speedup = polaris_time / iris_time
            print(f'Speed-Up Factor:\t\t{speedup:.2f}', flush=True)
        visualization.iris_side_by_side(dataset=writer.dataset,
                                        observer=observer,
                                        hyper=hyper,
                                        external_name='POLARIS',
                                        path=SIDE_BY_SIDE_PATH)
