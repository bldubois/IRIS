# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Generate the RADMC3D OT vs LVG Balance figure from the IRIS paper (DuBois et al., 2026).
"""

import sys
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.interpolate import NearestNDInterpolator

from iris import hyper as hp
from iris import cube_processing
from iris import arepo_processing as ap_core
from iris import arepo_processing_write as ap
from iris import visualization

ALT_BALANCE = 'LVG'
assert ALT_BALANCE == 'LVG' or ALT_BALANCE == 'LTE'
UNITS_DATASET_DIR = '/path/to/training_data_1'
IRIS_DATASET_DIR = f'~/IRIS/data/OT_vs_{ALT_BALANCE}'
SNAPSHOT_PATH = '/path/to/snapshot.hdf5'
BASE_FILES_DIR = '~/IRIS/data'
RADMC3D_EXECUTABLE = '/path/to/radmc3d'
OBSERVE_RADMC = True
PEAK_13CO_ABUNDANCE = 4e-2
FIG_PATH = f'~/IRIS/output/balance_OT_vs_{ALT_BALANCE}.png'
LVG_ESC_PROB_LENGTH_SCALE_PC = 0.05
LVG_NONLTE_MAXITER = 1000
LVG_NONLTE_CONVCRIT = 1e-2


class TestConfig(hp.SEDIGISM_13C16O):
    def __init__(self, files_dir):
        super().__init__()
        self.files_dir = Path(os.path.expanduser(files_dir))

        self.writer_hyper.points_per_snapshot = 1
        self.dataset_hyper.CMZ_scale_factor = None
        self.dataset_hyper.CMZ_scale_range = None
        self.coordinate_hyper.theta_zero = 270.

        self.coordinate_hyper.r_steps = 256
        self.coordinate_hyper.r_crop_min_index = 0
        self.coordinate_hyper.r_crop_max_index = 256
        self.coordinate_hyper.lon_steps = 509 # (b - 1) * 4 + 1 = 4b - 3
        self.coordinate_hyper.lat_steps = 128

        self.observer_hyper.lon_pieces = 1
        self.observer_hyper.lat_pieces = 1
        self.observer_hyper.v_subsamples = 2
        self.observer_hyper.blur_inputs = True
        self.observer_hyper.blur_kernel_r = 3
        self.observer_hyper.blur_kernel_lon = 5
        self.observer_hyper.blur_kernel_lat = 3
        self.observer_hyper.out_blur_fwhm = None
        self.observer_hyper.chem_path = ['~/IRIS/chem/13C16O_no_H.dat']
        self.observer_hyper.kappa_dust = [1e-3]
        self.observer_hyper.T_continuum = 2.73
        self.observer_hyper.T_cmb = 2.73

        self.cube_hyper.v_steps = 512
        self.cube_hyper.fits_map = self.from_fits
        self.cube_hyper.conversion_raw_to_T_K = [self.intensity_to_raleigh_jeans_temperature]
        return

    def from_fits(self) -> list:
        total = fits.getdata(self.files_dir / 'total.fits').astype(np.float32)
        total = np.flip(total.transpose(2, 1, 0), axis=0)
        continuum = fits.getdata(self.files_dir / 'continuum.fits').astype(np.float32)
        continuum = np.flip(continuum.transpose(2, 1, 0), axis=0)
        radmc = total - continuum
        return [radmc.copy()]

    def intensity_to_raleigh_jeans_temperature(self, I, hyper):
        return cube_processing.intensity_to_raleigh_jeans_temperature(I=I, hyper=hyper, nu_ul=220.3986841281e9)


def interpolate_snapshot_velocity(snapshot_path: str,
                                  hyper,
                                  theta: float = 0.) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    snapshot = ap_core.Snapshot(snapshot_path, hyper, gpu_interpolate=False)
    theta += np.deg2rad(hyper.coordinate_hyper.theta_zero)

    length_parsec_per_processing = hyper.writer_hyper._length_parsec_per_processing
    r_min = hyper.coordinate_hyper.r_min / length_parsec_per_processing
    r_max = hyper.coordinate_hyper.r_max / length_parsec_per_processing
    r_steps = hyper.coordinate_hyper.r_steps
    observer_r = hyper.coordinate_hyper.observer_radius / length_parsec_per_processing

    lon_min = np.deg2rad(hyper.coordinate_hyper.lon_min) * hyper.coordinate_hyper.spin_orientation
    lon_max = np.deg2rad(hyper.coordinate_hyper.lon_max) * hyper.coordinate_hyper.spin_orientation
    lon_steps = hyper.coordinate_hyper.lon_steps
    lat_min = np.deg2rad(hyper.coordinate_hyper.lat_min) * hyper.coordinate_hyper.spin_orientation
    lat_max = np.deg2rad(hyper.coordinate_hyper.lat_max) * hyper.coordinate_hyper.spin_orientation
    lat_steps = hyper.coordinate_hyper.lat_steps

    (positions,
     velocities,
     _,
     _,
     _,
     _,
     _) = snapshot._prune_particles(r_min,
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

    r_values = np.linspace(r_min, r_max, r_steps, dtype=np.float32)
    lon_values = np.linspace(lon_min, lon_max, lon_steps, dtype=np.float32)
    lat_values = np.linspace(lat_min, lat_max, lat_steps, dtype=np.float32)

    velocity_components = np.empty((3, r_steps, lon_steps, lat_steps), dtype=np.float32)
    if len(positions) > 0:
        interpolator = NearestNDInterpolator(positions, velocities)
        r_pieces = max(hyper.coordinate_hyper.r_pieces, 8)
        r_piece_edges = np.linspace(0, r_steps, r_pieces + 1, dtype=int)
        for r_lo_index, r_hi_index in zip(r_piece_edges[:-1], r_piece_edges[1:]):
            r, lon, lat = np.meshgrid(r_values[r_lo_index:r_hi_index],
                                      lon_values,
                                      lat_values,
                                      indexing='ij')
            x, y, z = snapshot._map_spherical_to_arepo(r,
                                                       lon,
                                                       lat,
                                                       observer_r,
                                                       theta,
                                                       cupy=False)
            grid = np.stack((x, y, z), axis=-1)
            velocity_components[:, r_lo_index:r_hi_index] = interpolator(grid).transpose(3, 0, 1, 2)
    else:
        velocity_components.fill(0.)

    r, lon, lat = np.meshgrid(r_values, lon_values, lat_values, indexing='ij')
    phi = np.pi + theta + lon
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    r_hat = np.stack((cos_lat * cos_phi, cos_lat * sin_phi, sin_lat), axis=0)
    lon_hat = np.stack((-sin_phi, cos_phi, np.zeros_like(phi)), axis=0)
    lat_hat = np.stack((-sin_lat * cos_phi, -sin_lat * sin_phi, cos_lat), axis=0)

    velocity_cm_per_s = hyper.writer_hyper.velocity_cm_per_s_per_processing
    v_r = np.einsum('i...,i...->...', velocity_components, r_hat) * velocity_cm_per_s
    v_lon = np.einsum('i...,i...->...', velocity_components, lon_hat) * velocity_cm_per_s
    v_lat = np.einsum('i...,i...->...', velocity_components, lat_hat) * velocity_cm_per_s
    snapshot.file.close()
    return v_r.astype(np.float64), v_lon.astype(np.float64), v_lat.astype(np.float64)


def radmc3d_observation(dataset, hyper, balance_code, snapshot_path: str | None = None):
    mass = hyper.dataset_hyper._mass_iris_per_SI / 1000
    length = hyper.dataset_hyper._length_iris_per_SI / 100
    cm_per_parsec = hyper.dataset_hyper._length_iris_per_parsec / length
    time = hyper.dataset_hyper._time_iris_per_SI
    velocity = length / time
    temperature = hyper.dataset_hyper._temperature_iris_per_SI
    volume = length * length * length
    density = mass / volume

    arepo = dataset.sample(1, validation=False)
    arepo = arepo.detach().numpy().astype(np.float64)
    v_r, rho, T, abundance_H2, abundance_CO, T_dust = arepo[0]
    rho /= density
    if snapshot_path is None:
        v_r /= -velocity
        v_lon = np.zeros_like(v_r, dtype=np.float64)
        v_lat = np.zeros_like(v_r, dtype=np.float64)
    else:
        v_r, v_lon, v_lat = interpolate_snapshot_velocity(snapshot_path, hyper)
    v_turb = np.zeros_like(v_r, dtype=np.float64)
    T /= temperature
    T = np.where(np.isfinite(T), T, 0)
    lamda_peak_collision_temperature = 2995.
    T = np.minimum(T, lamda_peak_collision_temperature)
    T_dust /= temperature

    nu_ul = 220.3986841281e9
    c = hyper.observer_hyper.c * 100
    lambda_ul_micron = (c * 1e4) / nu_ul
    ism_molecular_mass = (hyper.observer_hyper.m_H
                          + hyper.observer_hyper.abundance_He
                          * hyper.observer_hyper.m_He) * 1000
    N_H_TOT = rho / ism_molecular_mass
    N_H2 = N_H_TOT * abundance_H2
    ortho_to_para_H2_ratio = hyper.observer_hyper.ortho_to_para_H2_ratio
    N_oH2 = N_H2 * ortho_to_para_H2_ratio / (1 + ortho_to_para_H2_ratio)
    N_pH2 = N_H2 / (1 + ortho_to_para_H2_ratio)
    N_H = N_H_TOT - 2 * N_H2
    N_13CO = N_H_TOT * abundance_CO * PEAK_13CO_ABUNDANCE

    dust_to_gas_ratio = .01
    rho_dust = rho * dust_to_gas_ratio
    kappa_dust = hyper.observer_hyper.kappa_dust[0] * dust_to_gas_ratio

    hyper.files_dir.mkdir(parents=True, exist_ok=True)
    radmc_exe = os.path.expanduser(RADMC3D_EXECUTABLE)

    v_min = hyper.cube_hyper.v_min * 1e5
    v_max = hyper.cube_hyper.v_max * 1e5
    v_steps = hyper.cube_hyper.v_steps
    v = np.linspace(v_min, v_max, v_steps, dtype=np.float64)
    v_subsamples = hyper.observer_hyper.v_subsamples
    if v_subsamples > 0:
        fine_steps = 2 * v_steps * v_subsamples + 1
        dv = (v_max - v_min) / (v_steps - 1)
        v_fine = np.linspace(v_min - dv / 2, v_max + dv / 2, fine_steps, dtype=np.float64)
    else:
        v_fine = v
    lambda_micron = lambda_ul_micron / (1 + v / c)
    lambda_fine_micron = lambda_ul_micron / (1 + v_fine / c)

    r_min = hyper.coordinate_hyper.r_min * cm_per_parsec
    r_max = hyper.coordinate_hyper.r_max * cm_per_parsec
    r_steps = hyper.coordinate_hyper.r_steps
    dr = (r_max - r_min) / (r_steps - 1)
    r_edges = np.linspace(r_min - dr / 2, r_max + dr / 2, r_steps + 1, dtype=np.float64)

    lon_min = np.deg2rad(hyper.coordinate_hyper.lon_min)
    lon_max = np.deg2rad(hyper.coordinate_hyper.lon_max)
    lon_steps = hyper.coordinate_hyper.lon_steps
    dlon = (lon_max - lon_min) / (lon_steps - 1)
    lon_edges = np.linspace(lon_min - dlon / 2, lon_max + dlon / 2, lon_steps + 1, dtype=np.float64)
    phi_edges = (lon_edges + np.pi).astype(np.float64)

    lat_min = np.deg2rad(hyper.coordinate_hyper.lat_min)
    lat_max = np.deg2rad(hyper.coordinate_hyper.lat_max)
    lat_steps = hyper.coordinate_hyper.lat_steps
    dlat = (lat_max - lat_min) / (lat_steps - 1)
    lat_edges = np.linspace(lat_min - dlat / 2, lat_max + dlat / 2, lat_steps + 1, dtype=np.float64)
    theta_edges = np.pi / 2 - lat_edges[::-1]

    def to_radmc_scalar(scalar_array: np.ndarray) -> np.ndarray:
        return np.flip(np.transpose(scalar_array, (0, 2, 1)), axis=1)

    def to_radmc_velocity(vr: np.ndarray,
                          vlon: np.ndarray,
                          vlat: np.ndarray) -> np.ndarray:
        vr_radmc = to_radmc_scalar(vr)
        vtheta_radmc = to_radmc_scalar(-vlat)
        vphi_radmc = to_radmc_scalar(vlon)
        return np.stack([vr_radmc, vtheta_radmc, vphi_radmc], axis=0)

    def write_radmc3d(path: Path | str) -> None:
        lines_tbg = hyper.observer_hyper.T_continuum
        if lines_tbg is None:
            lines_tbg = 0.
        with open(path, 'w') as f:
            f.write('scattering_mode_max=0\n')  # No scattering
            f.write(f'lines_mode={balance_code}\n')  # 1 = LTE, 3 = LVG, 4 = OT
            f.write(f'lines_tbg={lines_tbg}\n')
            if balance_code == 3:
                f.write(f'lines_nonlte_maxiter={LVG_NONLTE_MAXITER}\n')
                f.write(f'lines_nonlte_convcrit={LVG_NONLTE_CONVCRIT:.9e}\n')
            f.write('rto_style=1\n')  # Output in ASCII
            f.write('camera_localobs_projection=2\n')  # Spherical projection
        return

    def write_amr_grid(path: Path | str) -> None:
        with open(path, 'w') as f:
            f.write('1\n')  # iformat
            f.write('0\n')  # regular grid
            f.write('100\n')  # spherical coordinates
            f.write('0\n')  # gridinfo
            f.write('1 1 1\n')  # include r, theta, phi
            f.write(f'{r_steps} {lat_steps} {lon_steps}\n')  # grid dimensions
            np.savetxt(f, r_edges, fmt='%.9e', newline=' ')  # r edges
            f.write('\n')
            np.savetxt(f, theta_edges, fmt='%.9e', newline=' ')  # theta edges
            f.write('\n')
            np.savetxt(f, phi_edges, fmt='%.9e', newline=' ')  # phi edges
        return

    def write_dust_scalar(path: Path | str, scalar_array: np.ndarray) -> None:
        with open(path, 'w') as f:
            f.write('1\n')  # iformat
            f.write(f'{scalar_array.size}\n')  # number of cells
            f.write('1\n')  # number of dust species
            flat = np.ravel(scalar_array, order='F')
            np.savetxt(f, flat, fmt='%.9e')  # dust densities
        return

    def write_wavelengths(path: Path | str, lamda: np.ndarray) -> None:
        with open(path, 'w') as f:
            f.write(f'{lamda.size}\n')  # number of wavelengths
            np.savetxt(f, lamda, fmt='%.12e')  # wavelengths
        return

    def write_dust_opacity(path: Path | str) -> None:
        with open(path, 'w') as f:
            f.write('2\n')  # iformat
            f.write('1\n')  # number of dust species
            f.write('-----------------------------\n')
            f.write('1\n')  # read opacity from dustkappa_*.inp
            f.write('0\n')  # thermal grain
            f.write('const\n')  # species name
        return

    def write_dust_kappa(path: Path | str, kappa_abs_cgs: float) -> None:
        # Constant absorption-only opacity over a broad wavelength range.
        lamda = np.logspace(-1, 5, 400)
        with open(path, 'w') as f:
            f.write('1\n')  # iformat
            f.write(f'{lamda.size}\n')  # number of wavelengths
            for l in lamda:
                f.write(f'{l:.9e} {kappa_abs_cgs:.9e}\n')  # wavelength, absorption opacity
        return

    def write_lines(path: Path | str) -> None:
        with open(path, 'w') as f:
            f.write('2\n')  # iformat
            f.write('1\n')  # number of lines
            # f.write('13C16O leiden 0 0 3\n')    # molecule name, data format, iduma, idumb, n collision partners
            f.write('13C16O leiden 0 0 2\n')
            f.write('p-h2\n')  # collisions with para-H2
            f.write('o-h2\n')  # collisions with ortho-H2
            # f.write('h\n')                      # collisions with H
        return

    def write_scalar(path: Path | str, scalar_array: np.ndarray) -> None:
        with open(path, 'w') as f:
            f.write('1\n')  # iformat
            f.write(f'{scalar_array.size}\n')  # number of cells
            flat = np.ravel(scalar_array, order='F')
            np.savetxt(f, flat, fmt='%.9e')  # cell values
        return

    def write_velocities(path: Path | str, vector_component_arrays: np.ndarray) -> None:
        with open(path, 'w') as f:
            f.write('1\n')  # iformat
            f.write(f'{vector_component_arrays[0].size}\n')  # number of cells
            data = np.column_stack(
                [np.ravel(component_array, order='F') for component_array in vector_component_arrays])
            np.savetxt(f, data, fmt='%.9e')  # cell values
        return

    write_radmc3d(os.path.join(hyper.files_dir, 'radmc3d.inp'))
    write_amr_grid(os.path.join(hyper.files_dir, 'amr_grid.inp'))

    write_dust_scalar(os.path.join(hyper.files_dir, 'dust_density.inp'), to_radmc_scalar(rho_dust))
    write_dust_scalar(os.path.join(hyper.files_dir, 'dust_temperature.dat'), to_radmc_scalar(T_dust))
    write_wavelengths(os.path.join(hyper.files_dir, 'wavelength_micron.inp'), lambda_fine_micron)
    write_dust_opacity(os.path.join(hyper.files_dir, 'dustopac.inp'))
    write_dust_kappa(os.path.join(hyper.files_dir, 'dustkappa_const.inp'), kappa_dust)

    write_lines(os.path.join(hyper.files_dir, 'lines.inp'))
    chem_path = Path(os.path.expanduser(hyper.observer_hyper.chem_path[0]))
    shutil.copyfile(chem_path, os.path.join(hyper.files_dir, 'molecule_13C16O.inp'))
    write_scalar(os.path.join(hyper.files_dir, 'numberdens_13C16O.inp'), to_radmc_scalar(N_13CO))
    write_scalar(os.path.join(hyper.files_dir, 'gas_temperature.inp'), to_radmc_scalar(T))
    write_velocities(os.path.join(hyper.files_dir, 'gas_velocity.inp'), to_radmc_velocity(v_r, v_lon, v_lat))
    write_scalar(os.path.join(hyper.files_dir, 'microturbulence.inp'), to_radmc_scalar(v_turb))
    if balance_code == 3:
        escprob_lengthscale = np.full_like(v_r, LVG_ESC_PROB_LENGTH_SCALE_PC * cm_per_parsec)
        write_scalar(os.path.join(hyper.files_dir, 'escprob_lengthscale.inp'), to_radmc_scalar(escprob_lengthscale))
    write_scalar(os.path.join(hyper.files_dir, 'numberdens_p-h2.inp'), to_radmc_scalar(N_pH2))
    write_scalar(os.path.join(hyper.files_dir, 'numberdens_o-h2.inp'), to_radmc_scalar(N_oH2))
    write_scalar(os.path.join(hyper.files_dir, 'numberdens_h.inp'), to_radmc_scalar(N_H))
    write_wavelengths(os.path.join(hyper.files_dir, 'camera_wavelength_micron.inp'), lambda_fine_micron)

    def read_radmc_image(fname: Path | str, integrate_velocity: bool = True) -> np.ndarray:
        with open(fname, 'r') as f:
            f.readline()
            image_lon_steps, image_lat_steps = map(int, f.readline().split())
            image_v_steps = int(f.readline())
            f.readline()
            for _ in range(image_v_steps):
                f.readline()
            I_v = np.loadtxt(f, dtype=np.float64)
            I_v = I_v.reshape((image_v_steps, image_lat_steps, image_lon_steps))
            # Simpson frequency integration from TransferProcessor._optically_thick_transfer_bdf2
            if v_subsamples > 0 and integrate_velocity:
                I_d_nu_normed = I_v[1:-1:2, :, :].copy()
                I_d_nu_normed *= 4
                I_d_nu_normed += I_v[:-2:2, :, :]
                I_d_nu_normed += I_v[2::2, :, :]
                I_d_nu_normed /= 6 * v_subsamples
                I = I_d_nu_normed.reshape(
                    (v_steps, v_subsamples, image_lat_steps, image_lon_steps)).sum(axis=1)
            else:
                I = I_v
        return I

    lon_center = 0.5 * (lon_min + lon_max)
    lat_center = 0.5 * (lat_min + lat_max)
    phi_center = lon_center + np.pi
    pointpc = (
        np.cos(lat_center) * np.cos(phi_center),
        np.cos(lat_center) * np.sin(phi_center),
        np.sin(lat_center))

    cmd_total = [
        radmc_exe,
        'image',
        'loadlambda',
        'noscat',
        'norefine',
        'nofluxcons',
        'npixx', str(lon_steps),
        'npixy', str(lat_steps),
        'zoomradian', str(lon_edges[0]), str(lon_edges[-1]), str(lat_edges[0]), str(lat_edges[-1]),
        'truepix',
        'locobspc', '0', '0', '0',
        'pointpc', str(pointpc[0]), str(pointpc[1]), str(pointpc[2])]
    subprocess.run(cmd_total, cwd=hyper.files_dir, check=True)

    total_cgs = read_radmc_image(os.path.join(hyper.files_dir, 'image.out'), integrate_velocity=True)
    total_jy_per_sr = total_cgs * 1e23  # erg/s/cm^2/Hz/sr -> Jy/sr
    fits.writeto(os.path.join(hyper.files_dir, 'total.fits'),
                 total_jy_per_sr.astype(np.float64),
                 overwrite=True)

    os.remove(os.path.join(hyper.files_dir, 'wavelength_micron.inp'))
    os.remove(os.path.join(hyper.files_dir, 'camera_wavelength_micron.inp'))
    write_wavelengths(os.path.join(hyper.files_dir, 'wavelength_micron.inp'), lambda_micron)
    write_wavelengths(os.path.join(hyper.files_dir, 'camera_wavelength_micron.inp'), lambda_micron)

    cmd_cont = [
        radmc_exe,
        'image',
        'loadlambda',
        'noline',
        'noscat',
        'norefine',
        'nofluxcons',
        'npixx', str(lon_steps),
        'npixy', str(lat_steps),
        'zoomradian', str(lon_edges[0]), str(lon_edges[-1]), str(lat_edges[0]), str(lat_edges[-1]),
        'truepix',
        'locobspc', '0', '0', '0',
        'pointpc', str(pointpc[0]), str(pointpc[1]), str(pointpc[2])]
    subprocess.run(cmd_cont, cwd=hyper.files_dir, check=True)

    cont_cgs = read_radmc_image(os.path.join(hyper.files_dir, 'image.out'), integrate_velocity=False)
    cont_jy_per_sr = cont_cgs * 1e23
    fits.writeto(os.path.join(hyper.files_dir, 'continuum.fits'),
                 cont_jy_per_sr.astype(np.float64),
                 overwrite=True)
    return

local_cache = sys.argv[1]
hyper_ALT = TestConfig(files_dir=os.path.join(os.path.expanduser(BASE_FILES_DIR), f'radmc3d_{ALT_BALANCE}'))
hyper_ALT.validate()
reader = ap.Reader(path=UNITS_DATASET_DIR,
                   dataset_type=ap.SyntheticallyObservedDataset)

writer = ap.Writer(path=IRIS_DATASET_DIR,
                   snapshot_paths=[SNAPSHOT_PATH],
                   hyper=hyper_ALT,
                   dataset_type=ap.StandardDataset,
                   units_from=reader.dataset,
                   gpu_interpolate=True,
                   gpu_normalize=False,
                   verbose=True)

if writer.rank == 0:
    hyper_OT = TestConfig(files_dir=os.path.join(os.path.expanduser(BASE_FILES_DIR), 'radmc3d_OT'))
    hyper_OT.validate()
    ap.Reader(path=UNITS_DATASET_DIR,
              dataset_type=ap.SyntheticallyObservedDataset,
              hyper=hyper_OT)
    if OBSERVE_RADMC:
        radmc3d_observation(dataset=writer.dataset, hyper=hyper_OT, balance_code=4)
    writer.world_comm.Barrier()

    visualization.external_side_by_side(dataset=writer.dataset,
                                        left_hyper=hyper_OT,
                                        right_hyper=hyper_ALT,
                                        left_name='OT Balance',
                                        right_name=f'{ALT_BALANCE} Balance',
                                        path=FIG_PATH)
elif writer.rank == 1 and OBSERVE_RADMC:
    ap.Reader(path=UNITS_DATASET_DIR,
              dataset_type=ap.SyntheticallyObservedDataset,
              hyper=hyper_ALT)
    radmc3d_observation(dataset=writer.dataset,
                        hyper=hyper_ALT,
                        balance_code=3 if ALT_BALANCE == 'LVG' else 1,
                        snapshot_path=SNAPSHOT_PATH)
    writer.world_comm.Barrier()
else:
    writer.world_comm.Barrier()
