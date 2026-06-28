# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Convergence testing for radial resolution and velocity subsampling in synthetic observation.
"""

from __future__ import annotations

import copy
import gc
import os
from pathlib import Path

import numpy as np
import torch

from iris import arepo_processing as ap
from iris import arepo_processing_write as apw
from iris import hyper as hp
from iris import observation
from iris import visualization


SNAPSHOT_PATH = '/path/to/test/snapshot.hdf5'
ROOT_DIR = '~/IRIS/data/resolution_convergence'
R_STEPS_SERIES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)
V_SUBSAMPLE_SERIES = tuple(range(8))


def make_hyper(r_steps: int) -> hp.Hyper:
    hyper = hp.SEDIGISM_13C16O(r_steps=r_steps)
    hyper.writer_hyper.total_snapshots = 1
    hyper.writer_hyper.points_per_snapshot = 1
    hyper.writer_hyper.unit_calculation_sample_size = 1
    hyper.dataset_hyper.CMZ_scale_factor = None
    hyper.dataset_hyper.CMZ_scale_range = None
    hyper.coordinate_hyper.theta_zero = 270.
    hyper.validate()
    return hyper

def write_standard_dataset(snapshot_path: str,
                           dataset_root: Path,
                           r_steps: int,
                           rank: int) -> apw.Writer:
    if rank == 0:
        print(f'\nMaking r_steps={r_steps} dataset...\n', flush=True)
    hyper = make_hyper(r_steps)
    return apw.Writer(path=str(dataset_root / f'r_steps_{r_steps}'),
                     snapshot_paths=[snapshot_path],
                     hyper=hyper,
                     dataset_type=ap.StandardDataset,
                     gpu_interpolate=True,
                     gpu_normalize=False,
                     verbose=True)

def density_scale(hyper: hp.Hyper) -> float:
    mass = hyper.dataset_hyper._mass_iris_per_SI
    length = hyper.dataset_hyper._length_iris_per_SI
    volume = length * length * length
    density = mass / volume
    solar_mass = 1.988e30
    parsec = hyper.dataset_hyper.meters_per_parsec
    density_conversion = solar_mass / parsec / parsec / parsec
    return density * density_conversion

def make_top_down(arepo: torch.Tensor, hyper: hp.Hyper) -> np.ndarray:
    top_down = ap.columnize_physical_tensor(arepo, hyper) / density_scale(hyper)
    return top_down.detach().cpu().numpy()[0][0]

def observe_arepo(arepo: torch.Tensor, hyper: hp.Hyper) -> np.ndarray:
    observer = observation.IteratedSyntheticObserver(hyper, cpu_batch=True)
    observer.eval()
    try:
        if torch.cuda.is_available():
            observer.cuda()
        with torch.no_grad():
            observed = observer(arepo, units='Trj K').detach().cpu()
        return observed.numpy()[0][0]
    finally:
        observer.cpu()
        del observer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def side_by_side_pair(top_down: np.ndarray,
                      left_cube: np.ndarray,
                      right_cube: np.ndarray,
                      left_name: str,
                      right_name: str,
                      hyper: hp.Hyper,
                      path: Path) -> None:
    print(f'\nside_by_side pair: {left_name} vs {right_name}', flush=True)
    visualization.side_by_side(arepo_top_down=top_down,
                               left_cube=left_cube,
                               right_cube=right_cube,
                               left_name=left_name,
                               right_name=right_name,
                               color_scale=2e2,
                               color_scale_center=0,
                               color_norm_min=0,
                               color_norm_max=5.0,
                               cbar_ticks=(0.0, 0.1, 1.0, 2.5, 5.0),
                               hyper=hyper,
                               path=str(path))

def run_test_a(datasets: dict[int, ap.StandardDataset],
               hypers: dict[int, hp.Hyper],
               output_root: Path) -> tuple[torch.Tensor, np.ndarray]:
    observations = {}
    top_downs = {}
    arepos = {}

    for r_steps in R_STEPS_SERIES:
        print(f'\nTest A observing r_steps={r_steps}', flush=True)
        arepo = datasets[r_steps].sample(1, validation=False)
        arepos[r_steps] = arepo
        top_downs[r_steps] = make_top_down(arepo, hypers[r_steps])
        observations[r_steps] = observe_arepo(arepo, hypers[r_steps])

    test_a_dir = output_root / 'test_a_r_steps'
    test_a_dir.mkdir(parents=True, exist_ok=True)
    for low, high in zip(R_STEPS_SERIES[:-1], R_STEPS_SERIES[1:]):
        low_cube = observations[low]
        high_cube = observations[high]
        side_by_side_pair(top_down=top_downs[low],
                          left_cube=low_cube,
                          right_cube=high_cube,
                          left_name=f'r_steps={low}',
                          right_name=f'r_steps={high}',
                          hyper=hypers[low],
                          path=test_a_dir / f'r_steps_{low}_vs_{high}.png')

    return arepos[512], top_downs[512]

def run_test_b(default_arepo: torch.Tensor,
               default_top_down: np.ndarray,
               default_hyper: hp.Hyper,
               output_root: Path) -> None:
    observations = {}
    hypers = {}

    for v_subsamples in V_SUBSAMPLE_SERIES:
        print(f'\nTest B observing v_subsamples={v_subsamples}', flush=True)
        hyper = copy.deepcopy(default_hyper)
        hyper.cube_hyper.v_steps = 128
        hyper.observer_hyper.lat_pieces = 2
        hyper.observer_hyper.lon_pieces = 2
        hyper.observer_hyper.v_subsamples = v_subsamples
        hyper.validate()
        hypers[v_subsamples] = hyper
        observations[v_subsamples] = observe_arepo(default_arepo, hyper)

    test_b_dir = output_root / 'test_b_v_subsamples'
    test_b_dir.mkdir(parents=True, exist_ok=True)
    for low, high in zip(V_SUBSAMPLE_SERIES[:-1], V_SUBSAMPLE_SERIES[1:]):
        low_cube = observations[low]
        high_cube = observations[high]
        side_by_side_pair(top_down=default_top_down,
                          left_cube=low_cube,
                          right_cube=high_cube,
                          left_name=f'v_subsamples={low}',
                          right_name=f'v_subsamples={high}',
                          hyper=hypers[low],
                          path=test_b_dir / f'v_subsamples_{low}_vs_{high}.png')

root_dir = Path(os.path.expanduser(ROOT_DIR))
dataset_root = root_dir / 'datasets'
output_root = root_dir / 'plots'
rank = apw.MPI.COMM_WORLD.Get_rank()
if rank == 0:
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
apw.MPI.COMM_WORLD.Barrier()

datasets = {}
hypers = {}
for r_steps in R_STEPS_SERIES:
    writer = write_standard_dataset(snapshot_path=SNAPSHOT_PATH,
                                    dataset_root=dataset_root,
                                    r_steps=r_steps,
                                    rank=rank)
    if rank == 0:
        datasets[r_steps] = writer.dataset
        hypers[r_steps] = writer.hyper

if rank == 0:
    default_arepo, default_top_down = run_test_a(datasets, hypers, output_root)
    run_test_b(default_arepo, default_top_down, hypers[512], output_root)
    print('\nDiagnostic complete.', flush=True)
