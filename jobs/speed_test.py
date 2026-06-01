# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details

"""
Run IRIS-vs-RADMC-3D speed-test comparisons.
Generates the data for the Speed Test figure from the IRIS paper (DuBois et al., 2026).
"""

import os
import shutil
import subprocess
import time as pytime
import gc
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
import torch

from iris import hyper as hp
from iris import arepo_processing_write as ap
from iris import observation


class TestConfig(hp.Hyper):
    def __init__(self, r_steps: int, lon_steps: int, lat_steps: int, v_steps: int) -> None:
        super().__init__()

        resolution = r_steps * lon_steps * lat_steps * v_steps
        max_resolution = 2.8 * 2 ** 29
        reduction_factor = resolution / max_resolution
        pieces = int(np.ceil(reduction_factor))

        self.writer_hyper.points_per_snapshot = 1
        self.writer_hyper.unit_calculation_sample_size = 1

        self.dataset_hyper.CMZ_scale_factor = None
        self.dataset_hyper.CMZ_scale_range = None
        self.dataset_hyper.CMZ_density_factor = None
        self.dataset_hyper.CMZ_density_range = None
        self.dataset_hyper.CMZ_skew_factor = None
        self.dataset_hyper.CMZ_skew_range = None

        self.coordinate_hyper.observer_radius = 8277.
        self.coordinate_hyper.r_steps = r_steps
        self.coordinate_hyper.r_crop_min_index = 0
        self.coordinate_hyper.r_crop_max_index = r_steps
        self.coordinate_hyper.r_min = 7600.
        self.coordinate_hyper.r_max = 8800.
        self.coordinate_hyper.lon_steps = lon_steps
        self.coordinate_hyper.lon_min = -3.
        self.coordinate_hyper.lon_max = 3.
        self.coordinate_hyper.lat_steps = lat_steps
        self.coordinate_hyper.lat_min = -1.
        self.coordinate_hyper.lat_max = 1.

        self.observer_hyper.lon_pieces = 1
        self.observer_hyper.lat_pieces = pieces
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

        self.cube_hyper.reduction = 'mean'
        self.cube_hyper.v_min = -400.
        self.cube_hyper.v_max = 400.
        self.cube_hyper.v_steps = v_steps
        return

def test_iteration(r_steps: int,
                   lon_steps: int,
                   lat_steps: int,
                   v_steps: int,
                   n: int) -> dict | None:
    hyper = TestConfig(r_steps=r_steps, lon_steps=lon_steps, lat_steps=lat_steps, v_steps=v_steps)
    writer = ap.Writer(path=f'~/IRIS/speed_test_{n}',
                       snapshot_paths=['~/IRIS/data/test.hdf5'],
                       hyper=hyper,
                       dataset_type=ap.StandardDataset,
                       gpu_interpolate=True,
                       gpu_normalize=False,
                       verbose=True)

    result = None
    if writer.rank == 0:
        mass = hyper.dataset_hyper._mass_iris_per_SI / 1000
        length = hyper.dataset_hyper._length_iris_per_SI / 100
        cm_per_parsec = hyper.dataset_hyper._length_iris_per_parsec / length
        time = hyper.dataset_hyper._time_iris_per_SI
        velocity = length / time
        temperature = hyper.dataset_hyper._temperature_iris_per_SI
        volume = length * length * length
        density = mass / volume

        arepo = writer.dataset.sample(1, validation=False, numpy=True).astype(np.float64)

        v_r, rho, T, abundance_H2, abundance_CO, T_dust = arepo[0]
        rho /= density
        v_r /= -velocity
        v_lon = np.zeros_like(v_r, dtype=np.float64)
        v_lat = np.zeros_like(v_r, dtype=np.float64)
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
        peak_13CO_abundance = 4e-2
        N_13CO = N_H_TOT * abundance_CO * peak_13CO_abundance

        dust_to_gas_ratio = .01
        rho_dust = rho * dust_to_gas_ratio
        kappa_dust = hyper.observer_hyper.kappa_dust[0] * dust_to_gas_ratio

        files_dir = Path(os.path.expanduser(f'~/IRIS/data/radmc3d_files_speed_test_{n}'))
        files_dir.mkdir(parents=True, exist_ok=True)
        radmc_exe = os.path.expanduser('~/radmc3d-2.0/src/radmc3d')

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
        dr = (r_max - r_min) / (r_steps - 1)
        r_edges = np.linspace(r_min - dr / 2, r_max + dr / 2, r_steps + 1, dtype=np.float64)

        lon_min = np.deg2rad(hyper.coordinate_hyper.lon_min)
        lon_max = np.deg2rad(hyper.coordinate_hyper.lon_max)
        dlon = (lon_max - lon_min) / (lon_steps - 1)
        lon_edges = np.linspace(lon_min - dlon / 2, lon_max + dlon / 2, lon_steps + 1, dtype=np.float64)
        phi_edges = (lon_edges + np.pi).astype(np.float64)

        lat_min = np.deg2rad(hyper.coordinate_hyper.lat_min)
        lat_max = np.deg2rad(hyper.coordinate_hyper.lat_max)
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
            with open(path, 'w') as f:
                f.write('scattering_mode_max=0\n')  # No scattering
                f.write('lines_mode=4\n')  # OT level balance, or 3 for LVG level balance
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
                f.write('13C16O leiden 0 0 3\n')  # molecule name, data format, iduma, idumb, n collision partners
                f.write('p-h2\n')  # collisions with para-H2
                f.write('o-h2\n')  # collisions with ortho-H2
                f.write('h\n')  # collisions with H
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

        start = pytime.time()
        write_radmc3d(os.path.join(files_dir, 'radmc3d.inp'))
        write_amr_grid(os.path.join(files_dir, 'amr_grid.inp'))

        write_dust_scalar(os.path.join(files_dir, 'dust_density.inp'), to_radmc_scalar(rho_dust))
        write_dust_scalar(os.path.join(files_dir, 'dust_temperature.dat'), to_radmc_scalar(T_dust))
        write_wavelengths(os.path.join(files_dir, 'wavelength_micron.inp'), lambda_fine_micron)
        write_dust_opacity(os.path.join(files_dir, 'dustopac.inp'))
        write_dust_kappa(os.path.join(files_dir, 'dustkappa_const.inp'), kappa_dust)

        write_lines(os.path.join(files_dir, 'lines.inp'))
        chem_path = Path(os.path.expanduser(hyper.observer_hyper.chem_path[0]))
        shutil.copyfile(chem_path, os.path.join(files_dir, 'molecule_13C16O.inp'))
        write_scalar(os.path.join(files_dir, 'numberdens_13C16O.inp'), to_radmc_scalar(N_13CO))
        write_scalar(os.path.join(files_dir, 'gas_temperature.inp'), to_radmc_scalar(T))
        write_velocities(os.path.join(files_dir, 'gas_velocity.inp'), to_radmc_velocity(v_r, v_lon, v_lat))
        write_scalar(os.path.join(files_dir, 'microturbulence.inp'), to_radmc_scalar(v_turb))
        write_scalar(os.path.join(files_dir, 'numberdens_p-h2.inp'), to_radmc_scalar(N_pH2))
        write_scalar(os.path.join(files_dir, 'numberdens_o-h2.inp'), to_radmc_scalar(N_oH2))
        write_scalar(os.path.join(files_dir, 'numberdens_h.inp'), to_radmc_scalar(N_H))
        write_wavelengths(os.path.join(files_dir, 'camera_wavelength_micron.inp'), lambda_fine_micron)
        end = pytime.time()
        radmc_write_time = end - start

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
        start = pytime.time()
        subprocess.run(cmd_total, cwd=files_dir, check=True)
        end = pytime.time()
        radmc_line_time = end - start

        total_cgs = read_radmc_image(os.path.join(files_dir, 'image.out'), integrate_velocity=True)
        total_jy_per_sr = total_cgs * 1e23  # erg/s/cm^2/Hz/sr -> Jy/sr
        fits.writeto(os.path.join(files_dir, 'total.fits'),
                     total_jy_per_sr.astype(np.float64),
                     overwrite=True)

        os.remove(os.path.join(files_dir, 'wavelength_micron.inp'))
        os.remove(os.path.join(files_dir, 'camera_wavelength_micron.inp'))
        write_wavelengths(os.path.join(files_dir, 'wavelength_micron.inp'), lambda_micron)
        write_wavelengths(os.path.join(files_dir, 'camera_wavelength_micron.inp'), lambda_micron)

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
        start = pytime.time()
        subprocess.run(cmd_cont, cwd=files_dir, check=True)
        end = pytime.time()
        radmc_continuum_time = end - start

        cont_cgs = read_radmc_image(os.path.join(files_dir, 'image.out'), integrate_velocity=False)
        cont_jy_per_sr = cont_cgs * 1e23
        fits.writeto(os.path.join(files_dir, 'continuum.fits'),
                     cont_jy_per_sr.astype(np.float64),
                     overwrite=True)

        abundance = observation.Constant_CO_13C16O(hyper, peak=peak_13CO_abundance)
        observer = observation.IteratedSyntheticObserver(hyper, abundance=abundance)

        observer.cuda()
        observer.eval()
        arepo = writer.dataset.sample(1, validation=False).cuda()
        start = pytime.time()
        observed = observer(arepo).detach().cpu()
        end = pytime.time()
        iris_time = end - start

        del observer, arepo, observed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        radmc_raw_time = radmc_line_time + radmc_continuum_time
        radmc_total_time = radmc_write_time + radmc_raw_time
        raw_speedup = radmc_raw_time / iris_time
        speedup = radmc_total_time / iris_time
        print(f'RADMC3D Write Time:\t\t{radmc_write_time / 60:.2f} min\t'
              f'RADMC3D Line Time:\t\t{radmc_line_time / 60:.2f} min\t'
              f'RADMC3D Continuum Time:\t\t{radmc_continuum_time / 60:.2f} min')
        print(f'RADMC3D Raw Time:\t\t{radmc_raw_time / 60:.2f} min\t'
              f'RADMC3D Total Time:\t\t{radmc_total_time / 60:.2f} min')
        print(f'IRIS-SO Total Time:\t\t{iris_time:.2f} s')
        print(f'Raw Speed-Up Factor:\t\t{raw_speedup:.2f}\t'
              f'Total Speed-Up Factor:\t\t{speedup:.2f}', flush=True)

        shutil.rmtree(os.path.expanduser(f'~/IRIS/speed_test_{n}'))
        shutil.rmtree(os.path.expanduser(f'~/IRIS/data/radmc3d_files_speed_test_{n}'))
        pieces = hyper.observer_hyper.lat_pieces
        result = {'r_steps': r_steps,
                  'lon_steps': lon_steps,
                  'lat_steps': lat_steps,
                  'v_steps': v_steps,
                  'pieces': pieces,
                  'image_resolution': lon_steps * lat_steps,
                  'process_resolution': r_steps * lon_steps * lat_steps * v_steps / pieces,
                  'total_resolution': r_steps * lon_steps * lat_steps * v_steps,
                  'radmc_write_time': radmc_write_time,
                  'radmc_line_time': radmc_line_time,
                  'radmc_continuum_time': radmc_continuum_time,
                  'radmc_raw_time': radmc_raw_time,
                  'radmc_total_time': radmc_total_time,
                  'iris_time': iris_time,
                  'raw_speedup': raw_speedup,
                  'speedup': speedup}
    writer.world_comm.Barrier()
    return result

battery = []

battery.append([(256, 256, 256)])
battery.append([(256, 256, 256)])

battery.append([(steps, 256, 256) for steps in (8, 10, 12, 14, 16)])
battery.append([(steps, 256, 256) for steps in (8, 10, 12, 14, 16)])
battery.append([(steps, 256, 256) for steps in (20, 24, 28, 32)])
battery.append([(steps, 256, 256) for steps in (20, 24, 28, 32)])
battery.append([(steps, 256, 256) for steps in (40, 48, 56)])
battery.append([(steps, 256, 256) for steps in (40, 48, 56)])
battery.append([(steps, 256, 256) for steps in (64, 80)])
battery.append([(steps, 256, 256) for steps in (64, 80)])
battery.append([(96, 256, 256)])
battery.append([(96, 256, 256)])
battery.append([(112, 256, 256)])
battery.append([(112, 256, 256)])
battery.append([(128, 256, 256)])
battery.append([(128, 256, 256)])
battery.append([(160, 256, 256)])
battery.append([(160, 256, 256)])
battery.append([(192, 256, 256)])
battery.append([(192, 256, 256)])
battery.append([(224, 256, 256)])
battery.append([(224, 256, 256)])

battery.append([(256, steps, 256) for steps in (8, 10, 12, 14, 16)])
battery.append([(256, steps, 256) for steps in (8, 10, 12, 14, 16)])
battery.append([(256, steps, 256) for steps in (20, 24, 28, 32)])
battery.append([(256, steps, 256) for steps in (20, 24, 28, 32)])
battery.append([(256, steps, 256) for steps in (40, 48, 56)])
battery.append([(256, steps, 256) for steps in (40, 48, 56)])
battery.append([(256, steps, 256) for steps in (64, 80)])
battery.append([(256, steps, 256) for steps in (64, 80)])
battery.append([(256, 96, 256)])
battery.append([(256, 96, 256)])
battery.append([(256, 112, 256)])
battery.append([(256, 112, 256)])
battery.append([(256, 128, 256)])
battery.append([(256, 128, 256)])
battery.append([(256, 160, 256)])
battery.append([(256, 160, 256)])
battery.append([(256, 192, 256)])
battery.append([(256, 192, 256)])
battery.append([(256, 224, 256)])
battery.append([(256, 224, 256)])

battery.append([(256, 256, steps) for steps in (8, 10, 12, 14, 16)])
battery.append([(256, 256, steps) for steps in (8, 10, 12, 14, 16)])
battery.append([(256, 256, steps) for steps in (20, 24, 28, 32)])
battery.append([(256, 256, steps) for steps in (20, 24, 28, 32)])
battery.append([(256, 256, steps) for steps in (40, 48, 56)])
battery.append([(256, 256, steps) for steps in (40, 48, 56)])
battery.append([(256, 256, steps) for steps in (64, 80)])
battery.append([(256, 256, steps) for steps in (64, 80)])
battery.append([(256, 256, 96)])
battery.append([(256, 256, 96)])
battery.append([(256, 256, 112)])
battery.append([(256, 256, 112)])
battery.append([(256, 256, 128)])
battery.append([(256, 256, 128)])
battery.append([(256, 256, 160)])
battery.append([(256, 256, 160)])
battery.append([(256, 256, 192)])
battery.append([(256, 256, 192)])
battery.append([(256, 256, 224)])
battery.append([(256, 256, 224)])

battery.append([(steps, steps, steps) for steps in (8, 10, 12, 14, 16)])
battery.append([(steps, steps, steps) for steps in (8, 10, 12, 14, 16)])
battery.append([(steps, steps, steps) for steps in (20, 24, 28, 32)])
battery.append([(steps, steps, steps) for steps in (20, 24, 28, 32)])
battery.append([(steps, steps, steps) for steps in (40, 48, 56)])
battery.append([(steps, steps, steps) for steps in (40, 48, 56)])
battery.append([(steps, steps, steps) for steps in (64, 80)])
battery.append([(steps, steps, steps) for steps in (64, 80)])
battery.append([(96, 96, 96)])
battery.append([(96, 96, 96)])
battery.append([(112, 112, 112)])
battery.append([(112, 112, 112)])
battery.append([(128, 128, 128)])
battery.append([(128, 128, 128)])
battery.append([(160, 160, 160)])
battery.append([(160, 160, 160)])
battery.append([(192, 192, 192)])
battery.append([(192, 192, 192)])
battery.append([(224, 224, 224)])
battery.append([(224, 224, 224)])

n = int(os.getenv('SLURM_ARRAY_TASK_ID'))
trial = 'A' if n % 2 == 1 else 'B'
series = battery[n - 1]

results = []
for r_steps, lat_steps, v_steps in series:
    lon_steps = 3 * lat_steps - 2
    data = test_iteration(r_steps, lon_steps, lat_steps, v_steps, n)
    if data is not None:
        results.append(data)

if len(results) > 0:
    path = os.path.expanduser(f'~/IRIS/data/speed_test_{trial}.json')
    all_results = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            all_results = json.load(f)
    all_results[f'series_{n}'] = results
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=4)
