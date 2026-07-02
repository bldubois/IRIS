# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
AREPO processing framework and data pipeline.

Manages conversion of AREPO simulation snapshots into data tensors
and associated datasets for model training, testing, and visualization.
Contains all core data pipeline logic, but MPI process coordination
for dataset writing is separated into [`arepo_processing_write`][iris.arepo_processing_write]
so that read-only features can be used without loading MPI modules on an HPC environment.
Import this module if you only need to [read][iris.arepo_processing.Reader] datasets
(e.g. for model training). Import [`arepo_processing_write`][iris.arepo_processing_write]
if [writing][iris.arepo_processing_write.Writer] is required, which will trigger an import
of mpi4py.

Attributes:
    CUPY_ENABLED: If `True`, will use CuPy for GPU support when interpolating the AREPO
        Voronoi mesh over the IRIS spherical coordinate grid. Otherwise, will use SciPy on CPU only.
        CuPy-enabled GPU interpolation is a critical performance optimization, but is a fragile
        installation, since, as of December 2025, the CuPy interpolator is pre-release. CuPy
        installation is therefore treated as an optional add-on by the iris package toml.
        IRIS will automatically detect if CuPy is available when setting this attribute.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import gc
import json
import copy
import random
import typing
import warnings
import subprocess

if typing.TYPE_CHECKING:
    import mpi4py
from contextlib import nullcontext

import h5py
import numpy as np
from scipy.interpolate import NearestNDInterpolator as CPUNearestNDInterpolator
try:
    import cupy as cp
    from cupyx.scipy.interpolate import NearestNDInterpolator as GPUNearestNDInterpolator
    CUPY_ENABLED: bool = True
except ImportError:
    CUPY_ENABLED: bool = False
import torch

from . import observation
from . import hyper as hp


def gauge_cpu_memory() -> float:
    """
    Computes the total CPU memory usage of the current SLURM job (across all nodes and processes).

    Returns:
        Total CPU memory usage (in GiB).
    """
    job_id = os.environ.get('SLURM_JOB_ID')
    if not job_id:
        return 0.

    try:
        # Query memory usage of each SLURM subprocess via sstat
        cmd = ['sstat', '-j', job_id, '-o', 'MaxRSS', '-n', '-p', '-a']
        result = subprocess.check_output(cmd, text=True)

        total_mem_gib = 0.

        for line in result.strip().split('\n'):
            if not line:
                continue
            raw_val = line.strip().replace('|', '')
            if not raw_val:
                continue

            multiplier = 1.
            if raw_val.endswith('K'):
                multiplier = 1 / (1024 ** 2)
            elif raw_val.endswith('M'):
                multiplier = 1 / 1024
            elif raw_val.endswith('G'):
                multiplier = 1.

            try:
                numeric_part = float(raw_val[:-1]) if raw_val[-1].isalpha() else float(raw_val)
                total_mem_gib += numeric_part * multiplier
            except ValueError:
                continue

        return total_mem_gib

    except subprocess.CalledProcessError:
        return 0.0

def columnize_physical_tensor(physical_tensor: torch.Tensor,
                              hyper: hp.Hyper,
                              requires_grad: bool = False) -> torch.Tensor:
    r"""
    Computes the "top-down" (latitude-meaned) $\text{H}_2$ density
    of a physical tensor in the radial viewing range.

    First extracts the density of the physical tensor within the radial bounds
    `hyper.coordinate_hyper.r_crop_min_index` and `hyper.coordinate_hyper.r_crop_max_index`
    specified within a [`Hyper`][iris.hyper.Hyper] object.
    Note that these radial bounds are tensor indices, not radial coordinates in physical units.
    This configurable radial viewing range allows computation of a constrained,
    top-down density tensor from an extended observational cone.
    Once extracted, converts this raw density into mass-density of $\text{H}_2$.
    Means the resulting density tensor over the galactic latitude dimension
    to generate a top-down view.

    Args:
        physical_tensor: The physical tensor
            (see [`StandardDataset`][iris.arepo_processing.StandardDataset]).
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        requires_grad: Set to True if gradients are required; False, otherwise.

    Returns:
        The top-down density tensor.
    """
    with nullcontext() if requires_grad else torch.no_grad():
        mw_H2 = hyper.observer_hyper.mw_H2
        mw_H = hyper.observer_hyper.mw_H
        abundance_He = hyper.observer_hyper.abundance_He
        mw_He = hyper.observer_hyper.mw_He
        r_crop_min_index = hyper.coordinate_hyper.r_crop_min_index
        r_crop_max_index = hyper.coordinate_hyper.r_crop_max_index

        rho = physical_tensor[:, 1, r_crop_min_index:r_crop_max_index, :, :]
        abundance_H2 = physical_tensor[:, 3, r_crop_min_index:r_crop_max_index, :, :]

        rho_H2 = rho * abundance_H2 * mw_H2 / (mw_H + abundance_He * mw_He)
        columnized = torch.mean(rho_H2, dim=3).unsqueeze(dim=1)
    return columnized

class Dataset(torch.utils.data.Dataset):
    """
    The base class for all finite, non-concatenated datasets.

    Extended by [`StandardDataset`][iris.arepo_processing.StandardDataset]
    and [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset].
    Use pre-observed datasets for all model training, as they store
    only top-down density and lv observations on disk.
    Use standard datasets only when retaining other tensor information is required.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors (or pre-observed pairs) in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """

    path: str
    hyper: hp.Hyper | None
    read_only: bool
    index: list
    abnormal: list
    cardinality: int
    training: torch.utils.data.Subset
    validation: torch.utils.data.Subset

    def __init__(self,
                 path: str,
                 hyper: hp.Hyper | None,
                 *args: any,
                 read_only: bool = False,
                 seed: int = 1216,
                 **kwargs: any) -> None:
        super().__init__()
        self.path = os.path.expanduser(path)
        self.hyper = hyper
        self.read_only = read_only

        self.index = []
        self.abnormal = []
        self.cardinality = 0

        self._load()

        generator = torch.Generator().manual_seed(seed)
        num_validation = int(self.hyper.training_hyper.validation_data_fraction * len(self))
        num_training = len(self) - num_validation
        self.training, self.validation = torch.utils.data.random_split(self,
                                                                       [num_training, num_validation],
                                                                       generator=generator)
        return

    def __len__(self) -> int:
        """
        Gets the cardinality of the dataset.

        Returns:
            The [`Dataset`][iris.arepo_processing.Dataset] `cardinality`.
        """
        return self.cardinality

    def __getitem__(self, item: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Gets an indexed entry of the [`Dataset`][iris.arepo_processing.Dataset].

        Args:
            item: The index of the item to fetch.

        Returns:
            A physical tensor or pre-observed pair.
        """
        return self.get_entry(item)

    def _load(self) -> None:
        """
        Attempts to load the [`Dataset`][iris.arepo_processing.Dataset] from `self.path`.

        If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
        will open the existing dataset for extension. If no [`Hyper`][iris.hyper.Hyper]
        object was provided to [`Dataset`][iris.arepo_processing.Dataset] during instantiation,
        will adopt the [`Hyper`][iris.hyper.Hyper] read from the disk. Otherwise, will retain
        [`Hyper`][iris.hyper.Hyper] provided during instantiation, but will
        [`_copy_units`][iris.arepo_processing.Dataset._copy_units] from the [`Hyper`][iris.hyper.Hyper]
        on disk. If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
        will call [`_make`][iris.arepo_processing.Dataset._make] if not
        `self.read_only` or raise a `RuntimeError` otherwise.

        Raises:
            RuntimeError: If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at
                `self.path` and `self.read_only`.
        """
        try:
            hyper = hp.Hyper()
            hyper.from_json(os.path.join(self.path, 'hyper.json'))
            with open(os.path.join(self.path, 'attributes.json'), 'r') as f:
                attributes = json.load(f)
                cardinality = attributes['cardinality']
                index = attributes['index']
        except Exception:
            if self.read_only:
                raise RuntimeError('Unreadable dataset.')
            self._make()
            return

        if self.hyper is None:
            self.hyper = hyper
        else:
            self._copy_units(hyper)
        self.cardinality = cardinality
        self.index = index
        return

    def save(self) -> None:
        """
        Saves [`Dataset`][iris.arepo_processing.Dataset] attributes to the disk.

        Writes `self.hyper`, `self.index`, and `cardinality` to human-readable JSON files at
        `self.path`. The `index` and `cardinality` attributes are combined into a single
        `attributes.json`, while `hyper` is stored as its own `hyper.json`.

        Raises:
            RuntimeError: If `self.read_only`.
        """
        if self.read_only:
            raise RuntimeError('Cannot write dataset in read-only mode.')
        self.hyper.to_json(os.path.join(self.path, 'hyper.json'))
        with open(os.path.join(self.path, 'attributes.json'), 'w') as f:
            json.dump({'cardinality': self.cardinality, 'index': self.index}, f, indent=4)
        return

    def _copy_units(self, hyper: hp.Hyper) -> None:
        """
        Copies units to `self.hyper` attribute from an external [`Hyper`][iris.hyper.Hyper] object.

        Args:
            hyper: The hyperparameters object from which to copy units.
        """
        self.hyper.dataset_hyper.iris_number_unit = hyper.dataset_hyper.iris_number_unit
        self.hyper.dataset_hyper._length_iris_per_SI = hyper.dataset_hyper._length_iris_per_SI
        self.hyper.dataset_hyper._length_iris_per_parsec = hyper.dataset_hyper._length_iris_per_parsec
        self.hyper.dataset_hyper._time_iris_per_SI = hyper.dataset_hyper._time_iris_per_SI
        self.hyper.dataset_hyper._mass_iris_per_SI = hyper.dataset_hyper._mass_iris_per_SI
        self.hyper.dataset_hyper._temperature_iris_per_SI = hyper.dataset_hyper._temperature_iris_per_SI
        return

    def take_units(self, hyper: hp.Hyper) -> bool:
        """
        In addition to [copying units][iris.arepo_processing.Dataset._copy_units] from an external
        [`Hyper`][iris.hyper.Hyper] object, will also compute conversions from processing units
        into IRIS units for
        [units normalization post-processing][iris.arepo_processing_write.Writer._normalize].

        Args:
            hyper: The hyperparameters object from which to take units.

        Returns:
            A bool value indicating whether [normalization][iris.arepo_processing_write.Writer._normalize]
                is required since IRIS units are different from processing units.
        """
        self._copy_units(hyper)

        length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
        mass_g_per_processing = self.hyper.writer_hyper.mass_g_per_processing
        velocity_cm_per_s_per_processing = self.hyper.writer_hyper.velocity_cm_per_s_per_processing

        temperature_processing_per_SI = 1.0
        velocity_processing_per_SI = 100.0 / velocity_cm_per_s_per_processing
        mass_processing_per_SI = 1000.0 / mass_g_per_processing
        length_processing_per_SI = 100.0 / length_cm_per_processing
        volume_processing_per_SI = length_processing_per_SI * length_processing_per_SI * length_processing_per_SI
        density_processing_per_SI = mass_processing_per_SI / volume_processing_per_SI

        length_iris_per_SI = self.hyper.dataset_hyper._length_iris_per_SI
        volume_iris_per_SI = length_iris_per_SI * length_iris_per_SI * length_iris_per_SI
        time_iris_per_SI = self.hyper.dataset_hyper._time_iris_per_SI
        velocity_iris_per_SI = length_iris_per_SI / time_iris_per_SI
        mass_iris_per_SI = self.hyper.dataset_hyper._mass_iris_per_SI
        density_iris_per_SI = mass_iris_per_SI / volume_iris_per_SI
        temperature_iris_per_SI = self.hyper.dataset_hyper._temperature_iris_per_SI

        velocity_iris_per_processing = velocity_iris_per_SI / velocity_processing_per_SI
        density_iris_per_processing = density_iris_per_SI / density_processing_per_SI
        temperature_iris_per_processing = temperature_iris_per_SI / temperature_processing_per_SI

        self.hyper.dataset_hyper._velocity_iris_per_processing = velocity_iris_per_processing
        self.hyper.dataset_hyper._density_iris_per_processing = density_iris_per_processing
        self.hyper.dataset_hyper._temperature_iris_per_processing = temperature_iris_per_processing

        epsilon = 1e-6
        iris_processing_units_different = (
            abs(velocity_iris_per_processing - 1.0) > epsilon or
            abs(density_iris_per_processing - 1.0) > epsilon or
            abs(temperature_iris_per_processing - 1.0) > epsilon)
        return iris_processing_units_different

    def _make(self) -> None:
        """
        Attempts to make a new [`Dataset`][iris.arepo_processing.Dataset] directory at `self.path`.

        If no directory exists at `self.path`, will make the new
        directory. Otherwise, will search all paths `path + f'_{n} for n in range(1, 99)` for
        an available directory path. Will update `self.path` to the new path.

        Raises:
            RuntimeError: If `self.hyper` is `None`.
            RuntimeError: If no available path is found.
        """
        if self.hyper is None:
            raise RuntimeError('No hyper provided to dataset.')
        if os.path.exists(self.path):
            path = None
            for i in range(1, 99):
                p = self.path + '_{}'.format(i)
                if not os.path.exists(p):
                    path = p
                    break
            if path is None:
                raise RuntimeError('Unusable dataset directory path.')
            self.path = path

        os.mkdir(self.path)
        return

    def shuffle(self) -> None:
        """
        Permanently shuffles `self.index`.
        """
        random.shuffle(self.index)
        return

    def make_training_and_validation_dataloaders(self, cpus_per_gpu: int) -> tuple[
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader,
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader]:
        """
        Make `DataLoader` objects for training and validation datasets.

        Makes `torch.utils.data.DataLoader` and `torch.utils.data.distributed.DistributedSampler`
        objects for `self.training` and `self.validation`. The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main training loop, to include forward
        and backward passes, step computations, and stats logging. In practice, loading time
        from disk to CPU is the primary compute bottleneck during training--not forwards/backwards
        pass computation, so allocating extra worker processes via SLURM is highly recommended
        even if training on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            A tuple of:
                `training_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `training_dataloader (torch.utils.data.DataLoader)`,
                `validation_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `validation_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        training_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.training,
                                                                           shuffle=True,
                                                                           drop_last=True)
        training_dataloader = torch.utils.data.DataLoader(dataset=self.training,
                                                          batch_size=batch_size,
                                                          sampler=training_sampler,
                                                          num_workers=num_workers,
                                                          prefetch_factor=2,
                                                          persistent_workers=True,
                                                          pin_memory=True,
                                                          drop_last=True)
        if len(self.validation) >= batch_size:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.validation,
                                                                                 shuffle=True,
                                                                                 drop_last=True)
            validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation,
                                                                batch_size=batch_size,
                                                                sampler=validation_sampler,
                                                                num_workers=num_workers,
                                                                prefetch_factor=2,
                                                                persistent_workers=True,
                                                                pin_memory=True,
                                                                drop_last=True)
        else:
            validation_sampler = None
            validation_dataloader = None
        return training_sampler, training_dataloader, validation_sampler, validation_dataloader

    def make_test_dataloader(self, cpus_per_gpu: int) -> tuple[
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader]:
        """
        Make `DataLoader` objects for test dataset.

        Makes a single `torch.utils.data.DataLoader` and `torch.utils.data.distributed.DistributedSampler`
        objects for the complete [`Dataset`][iris.arepo_processing.Dataset]. The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main test computations, to include forward pass,
        loss computation, and stats logging. In practice, loading time from disk to CPU is the
        primary compute bottleneck during testing--not forwards pass computation, so allocating
        extra worker processes via SLURM is highly recommended even if testing on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            A tuple of:
                `test_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `test_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self,
                                                                       shuffle=True,
                                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=self,
                                                      batch_size=batch_size,
                                                      sampler=test_sampler,
                                                      num_workers=num_workers,
                                                      prefetch_factor=2,
                                                      persistent_workers=True,
                                                      pin_memory=True,
                                                      drop_last=True)
        return test_sampler, test_dataloader


class ConcatDataset(torch.utils.data.ConcatDataset):
    """
    Allows concatenation of [`Dataset`][iris.arepo_processing.Dataset] objects.

    Will automatically unify multiple [`Dataset`][iris.arepo_processing.Dataset] objects
    into a single `torch.utils.data.ConcatDataset` for training or testing. Works with
    either [`StandardDataset`][iris.arepo_processing.StandardDataset] objects or
    [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] objects,
    but assumes all constituent datasets are of the same type. Automatically handles
    units conversion of all datasets into the units of the primary (index-0) dataset.

    Attributes:
        from_type: Type of the primary (index-0) dataset
            (e.g. [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]).
        hyper: Hyperparameters object.
        training: Random subset used for
            [`Reverter`][iris.reversion.Reverter] training. (Deterministically sampled via `seed`.
            The set complement of `self.validation`.)
        validation: Random subset used for
            [`Reverter`][iris.reversion.Reverter] validation. (Deterministically sampled via `seed`.
            The set complement of `self.training`.)

    Args:
        datasets: The [`Dataset`][iris.arepo_processing.Dataset] objects to be concatenated.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.

    Raises:
        TypeError: If `type(datasets[0])` not one of
            [`StandardDataset`][iris.arepo_processing.StandardDataset],
            [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset],
            [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].
    """

    from_type: type[Dataset]
    hyper: hp.Hyper
    training: torch.utils.data.Subset
    validation: torch.utils.data.Subset

    def __init__(self, datasets: typing.Sequence[Dataset], seed: int = 1216) -> None:
        self.from_type = type(datasets[0])
        self.hyper = datasets[0].hyper
        generator = torch.Generator().manual_seed(seed)
        units_corrected_datasets = [datasets[0]]
        if issubclass(self.from_type, StandardDataset):
            units_corrected_datasets += [UnitsCorrectedStandardDataset(d, datasets[0]) for d in datasets[1:]]
        elif issubclass(self.from_type, SyntheticallyObservedDataset):
            units_corrected_datasets += [UnitsCorrectedSyntheticallyObservedDataset(d, datasets[0]) for d in datasets[1:]]
        elif issubclass(self.from_type, SimplyObservedDataset):
            units_corrected_datasets += [UnitsCorrectedSimplyObservedDataset(d, datasets[0]) for d in datasets[1:]]
        else:
            raise TypeError('Unexpected Dataset type in ConcatDataset. '
                            'Must be one of StandardDataset, SyntheticallyObservedDataset, SimplyObservedDataset.')

        super().__init__(units_corrected_datasets)

        num_validation = int(self.hyper.training_hyper.validation_data_fraction * len(self))
        num_training = len(self) - num_validation
        self.training, self.validation = torch.utils.data.random_split(self,
                                                                       [num_training, num_validation],
                                                                       generator=generator)
        return

    def make_training_and_validation_dataloaders(self, cpus_per_gpu: int) -> tuple[
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader,
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader]:
        """
        Make `DataLoader` objects for training and validation datasets.

        Makes `torch.utils.data.DataLoader` and `torch.utils.data.distributed.DistributedSampler`
        objects for `self.training` and `self.validation`. The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main training loop, to include forward
        and backward passes, step computations, and stats logging. In practice, loading time
        from disk to CPU is the primary compute bottleneck during training--not forwards/backwards
        pass computation, so allocating extra worker processes via SLURM is highly recommended
        even if training on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            A tuple of:
                `training_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `training_dataloader (torch.utils.data.DataLoader)`,
                `validation_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `validation_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        training_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.training,
                                                                           shuffle=True,
                                                                           drop_last=True)
        training_dataloader = torch.utils.data.DataLoader(dataset=self.training,
                                                          batch_size=batch_size,
                                                          sampler=training_sampler,
                                                          num_workers=num_workers,
                                                          prefetch_factor=2,
                                                          persistent_workers=True,
                                                          pin_memory=True,
                                                          drop_last=True)
        if len(self.validation) >= batch_size:
            validation_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self.validation,
                                                                                 shuffle=True,
                                                                                 drop_last=True)
            validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation,
                                                                batch_size=batch_size,
                                                                sampler=validation_sampler,
                                                                num_workers=num_workers,
                                                                prefetch_factor=2,
                                                                persistent_workers=True,
                                                                pin_memory=True,
                                                                drop_last=True)
        else:
            validation_sampler = None
            validation_dataloader = None
        return training_sampler, training_dataloader, validation_sampler, validation_dataloader

    def make_test_dataloader(self, cpus_per_gpu: int) -> tuple[
        torch.utils.data.distributed.DistributedSampler,
        torch.utils.data.DataLoader]:
        """
        Make `DataLoader` objects for test dataset.

        Makes a single `torch.utils.data.DataLoader` and `torch.utils.data.distributed.DistributedSampler`
        objects for the complete [`ConcatDataset`][iris.arepo_processing.ConcatDataset]. The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main test computations, to include forward pass,
        loss computation, and stats logging. In practice, loading time from disk to CPU is the
        primary compute bottleneck during testing--not forwards pass computation, so allocating
        extra worker processes via SLURM is highly recommended even if testing on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            A tuple of:
                `test_sampler (torch.utils.data.distributed.DistributedSampler)`,
                `test_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=self,
                                                                       shuffle=True,
                                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(dataset=self,
                                                      batch_size=batch_size,
                                                      sampler=test_sampler,
                                                      num_workers=num_workers,
                                                      prefetch_factor=2,
                                                      persistent_workers=True,
                                                      pin_memory=True,
                                                      drop_last=True)
        return test_sampler, test_dataloader


class InfiniteSet(torch.utils.data.IterableDataset):
    """
    An infinite iterable over a [`Dataset`][iris.arepo_processing.Dataset]
    or [`ConcatDataset`][iris.arepo_processing.ConcatDataset].

    Will randomly and infinitely iterate over a base [`Dataset`][iris.arepo_processing.Dataset]
    or [`ConcatDataset`][iris.arepo_processing.ConcatDataset]. Used as the raw
    `torch.utils.data.IterableDataset` for [`InfiniteDataset`][iris.arepo_processing.InfiniteDataset]
    as well as `InfiniteDataset.training` and `InfiniteDataset.validation` subsets.

    Attributes:
        finite_set: The base [`Dataset`][iris.arepo_processing.Dataset]
            or [`ConcatDataset`][iris.arepo_processing.ConcatDataset] over which to iterate.

    Args:
        finite_set: Sets `self.finite_set`.
    """

    finite_set: Dataset | ConcatDataset

    def __init__(self, finite_set : Dataset | ConcatDataset) -> None:
        super().__init__()
        self.finite_set = finite_set
        return

    def __iter__(self) -> typing.Iterator[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        """
        Infinitely yields random entries from the `self.finite_set`.

        Automatically handles worker seeding using `torch.utils.data.get_worker_info()`
        to ensure unique randomness when used with multiple `DataLoader` workers.

        Yields:
            A single physical tensor or pre-observed pair.
        """
        info = torch.utils.data.get_worker_info()
        random.seed(info.seed)
        while True:
            i = random.randint(0, len(self.finite_set) - 1)
            yield self.finite_set[i]


class InfiniteDataset(InfiniteSet):
    """
    An infinitely iterable dataset used for adding litter during training or testing.

    Constructed via [`Reader`][iris.arepo_processing.Reader]. See 
    [`train_reverter`][iris.training.train_reverter] for notes on using litter.

    Attributes:
        finite_set: The base [`Dataset`][iris.arepo_processing.Dataset]
            or [`ConcatDataset`][iris.arepo_processing.ConcatDataset] over which to iterate.
        from_type: Type or `from_type` attribute of the `finite_dataset` specified.
        hyper: Hyperparameters object.
        units_corrected_finite_dataset: The `finite_dataset` corrected to the units
            specified in `units_dataset.hyper`.
        training: Random subset of `self.units_corrected_finite_dataset`
            used for [`Reverter`][iris.reversion.Reverter] training. (Deterministically sampled via `seed`.
            The set complement of `self.validation`.)
        validation: Random subset of `self.units_corrected_finite_dataset`
            used for [`Reverter`][iris.reversion.Reverter] training. (Deterministically sampled via `seed`.
            The set complement of `self.training`.)

    Args:
        finite_dataset: The finite [`Dataset`][iris.arepo_processing.Dataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset] over which to infinitely iterate.
        units_dataset: Yields all data in these units.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.

    Raises:
        TypeError: If `finite_dataset` is not one of or is not a concatenation of one of
            [`StandardDataset`][iris.arepo_processing.StandardDataset],
            [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset],
            [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].
    """

    from_type: type[Dataset]
    hyper: hp.Hyper
    units_corrected_finite_dataset: (UnitsCorrectedStandardDataset | 
                                     UnitsCorrectedSyntheticallyObservedDataset |
                                     UnitsCorrectedSimplyObservedDataset)
    training: InfiniteSet
    validation: InfiniteSet | None
    
    def __init__(self,
                 finite_dataset : Dataset | ConcatDataset,
                 units_dataset: Dataset | ConcatDataset,
                 seed: int = 1216) -> None:
        if isinstance(finite_dataset, ConcatDataset):
            self.from_type = finite_dataset.from_type
        else:
            self.from_type = type(finite_dataset)
        self.hyper = units_dataset.hyper
        if issubclass(self.from_type, StandardDataset):
            self.units_corrected_finite_dataset = UnitsCorrectedStandardDataset(finite_dataset, units_dataset)
        elif issubclass(self.from_type, SyntheticallyObservedDataset):
            self.units_corrected_finite_dataset = UnitsCorrectedSyntheticallyObservedDataset(finite_dataset, units_dataset)
        elif issubclass(self.from_type, SimplyObservedDataset):
            self.units_corrected_finite_dataset = UnitsCorrectedSimplyObservedDataset(finite_dataset, units_dataset)
        else:
            raise TypeError(
                'Unexpected Dataset type in InfiniteSet. '
                'Must be one of StandardDataset, SyntheticallyObservedDataset, SimplyObservedDataset.')

        super().__init__(finite_set=self.units_corrected_finite_dataset)

        generator = torch.Generator().manual_seed(seed)
        num_validation = int(self.hyper.training_hyper.validation_data_fraction * len(self.units_corrected_finite_dataset))
        num_training = len(self.units_corrected_finite_dataset) - num_validation
        training, validation = torch.utils.data.random_split(self.units_corrected_finite_dataset,
                                                             [num_training, num_validation],
                                                             generator=generator)
        self.training = InfiniteSet(finite_set=training)
        if len(validation) > 0:
            self.validation = InfiniteSet(finite_set=validation)
        else:
            self.validation = None
        return

    def make_infinite_training_validation_dataloaders(self, cpus_per_gpu: int) -> tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader]:
        """
        Make `DataLoader` objects for training and validation datasets.

        Makes `torch.utils.data.DataLoader` objects for
        `self.training` and `self.validation`. Does not make
        `torch.utils.data.distributed.DistributedSampler` objects, since data sampling is
        infinite and random (uniquely per distributed process, as guaranteed by the seeding
        strategy in [`__iter__`][iris.arepo_processing.InfiniteSet.__iter__]). The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire finite dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main training loop, to include forward
        and backward passes, step computations, and stats logging. In practice, loading time
        from disk to CPU is the primary compute bottleneck during training--not forwards/backwards
        pass computation, so allocating extra worker processes via SLURM is highly recommended
        even if training on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            A tuple of:
                `training_dataloader (torch.utils.data.DataLoader)`,
                `validation_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        training_dataloader = torch.utils.data.DataLoader(dataset=self.training,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          prefetch_factor=2,
                                                          persistent_workers=True,
                                                          pin_memory=True,
                                                          drop_last=False)
        if self.validation is not None:
            validation_dataloader = torch.utils.data.DataLoader(dataset=self.validation,
                                                                batch_size=batch_size,
                                                                num_workers=num_workers,
                                                                prefetch_factor=2,
                                                                persistent_workers=True,
                                                                pin_memory=True,
                                                                drop_last=False)
        else:
            validation_dataloader = None
        return training_dataloader, validation_dataloader

    def make_infinite_test_dataloader(self, cpus_per_gpu: int) -> torch.utils.data.DataLoader:
        """
        Make `DataLoader` objects for test dataset.

        Makes a single `torch.utils.data.DataLoader` object for infinite iteration over the complete
        `self.units_corrected_finite_dataset`. Does not make a `torch.utils.data.distributed.DistributedSampler`
        object, since data sampling is infinite and random (uniquely per distributed process, as guaranteed by
        the seeding strategy in [`__iter__`][iris.arepo_processing.InfiniteSet.__iter__]). The `DataLoader` class
        allows data streaming such that only a manageable collection of data batches is loaded
        into memory at any given time, as opposed to loading the entire dataset into memory at once.
        (Even memory-optimized [pre-observed datasets][iris.arepo_processing.PreObservedDataset]
        may occupy 100s of GiBs at scale.) Assigns all excess CPU processes as `DataLoader` workers.
        These workers load tensors from the disk onto the CPU asynchronously while each primary
        `torchrun` GPU manager process executes the main test computations, to include forward pass,
        loss computation, and stats logging. In practice, loading time from disk to CPU is the
        primary compute bottleneck during testing--not forwards pass computation, so allocating
        extra worker processes via SLURM is highly recommended even if testing on only one GPU.

        Args:
            cpus_per_gpu: The number of CPU processes per GPU. Used to compute
                `num_workers = cpus_per_gpu - 1`.

        Returns:
            The `test_dataloader (torch.utils.data.DataLoader)`.
        """
        batch_size = self.hyper.training_hyper.batch_size
        num_workers = cpus_per_gpu - 1

        test_dataloader = torch.utils.data.DataLoader(dataset=self,
                                                      batch_size=batch_size,
                                                      num_workers=num_workers,
                                                      prefetch_factor=2,
                                                      persistent_workers=True,
                                                      pin_memory=True,
                                                      drop_last=False)
        return test_dataloader

    def observed_sample(self, observer: observation.Observer | None = None) -> torch.Tensor:
        """
        Returns a sample of just observed litter.

        Args:
            observer: Used for computing an observation if `self.from_type` is
                [`StandardDataset`][iris.arepo_processing.StandardDataset].
                Set `None` if `self.from_type` is
                [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset].

        Raises:
            RuntimeError: If `self.from_type` is
                [`StandardDataset`][iris.arepo_processing.StandardDataset] and `observer` is `None`.
        """
        i = random.randint(0, len(self.units_corrected_finite_dataset) - 1)
        item = self.units_corrected_finite_dataset[i]
        if issubclass(self.from_type, PreObservedDataset):
            _, litter = item
            litter = litter.unsqueeze(dim=0).cuda()
        elif observer is not None:
            item = item.unsqueeze(dim=0).cuda()
            observer.eval()
            observer.cuda()
            litter = observer(item)
        else:
            raise RuntimeError('For observed_sample, must provide observer or from_type must be PreObservedDataset.')
        return litter


class UnitsCorrectedDataset(torch.utils.data.Dataset):
    """
    Wraps a [`Dataset`][iris.arepo_processing.Dataset] or
    [`ConcatDataset`][iris.arepo_processing.ConcatDataset] and corrects units of fetched
    tensors to those of `units_dataset`.

    Attributes:
        dataset: The wrapped [`Dataset`][iris.arepo_processing.Dataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset].
        velocity_to_units: The conversion factor into `units_dataset` units.
        density_to_units: The conversion factor into `units_dataset` units.
        temperature_to_units: The conversion factor into `units_dataset` units.

    Args:
        dataset: Sets `self.dataset`.
        units_dataset: The dataset from which to compute conversion factors.
    """

    dataset: Dataset | ConcatDataset
    velocity_to_units: float
    density_to_units: float
    temperature_to_units: float

    def __init__(self,
                 dataset: Dataset | ConcatDataset,
                 units_dataset: Dataset | ConcatDataset) -> None:
        super().__init__()
        self.dataset = dataset

        time = dataset.hyper.dataset_hyper._time_iris_per_SI
        length = dataset.hyper.dataset_hyper._length_iris_per_SI
        mass = dataset.hyper.dataset_hyper._mass_iris_per_SI
        temperature = dataset.hyper.dataset_hyper._temperature_iris_per_SI
        velocity = length / time
        density = mass / length / length / length
        v_density = density * time

        units_time = units_dataset.hyper.dataset_hyper._time_iris_per_SI
        units_length = units_dataset.hyper.dataset_hyper._length_iris_per_SI
        units_mass = units_dataset.hyper.dataset_hyper._mass_iris_per_SI
        units_temperature = units_dataset.hyper.dataset_hyper._temperature_iris_per_SI
        units_velocity = units_length / units_time
        units_density = units_mass / units_length / units_length / units_length
        units_v_density = units_density * units_time

        self.velocity_to_units = units_velocity / velocity
        self.density_to_units = units_density / density
        self.v_density_to_units = units_v_density / v_density
        self.temperature_to_units = units_temperature / temperature
        return

    def __len__(self) -> int:
        """
        Returns the length of the wrapped dataset.

        Returns: `len(self.dataset)`.
        """
        return len(self.dataset)

class UnitsCorrectedStandardDataset(UnitsCorrectedDataset):
    """
    Wraps a [`StandardDataset`][iris.arepo_processing.StandardDataset] or
    [`ConcatDataset`][iris.arepo_processing.ConcatDataset] with `from_type` of
    [`StandardDataset`][iris.arepo_processing.StandardDataset] and corrects units
    of fetched tensors to those of `units_dataset`.

    Attributes:
        dataset: The wrapped [`StandardDataset`][iris.arepo_processing.StandardDataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset].
        velocity_to_units: The conversion factor into `units_dataset` units.
        density_to_units: The conversion factor into `units_dataset` units.
        temperature_to_units: The conversion factor into `units_dataset` units.

    Args:
        dataset: Sets `self.dataset`.
        units_dataset: The dataset from which to compute conversion factors.
    """

    dataset: StandardDataset | ConcatDataset

    def __init__(self,
                 dataset: StandardDataset | ConcatDataset,
                 units_dataset: StandardDataset | ConcatDataset) -> None:
        super().__init__(dataset=dataset, units_dataset=units_dataset)
        return

    def __getitem__(self, item: int) -> torch.Tensor:
        """
        Fetches an indexed physical tensor in the target units.

        See [`StandardDataset`][iris.arepo_processing.StandardDataset] for details.

        Args:
            item: The tensor index.

        Returns:
            A physical tensor.
        """
        physical_tensor = self.dataset[item]
        v_r = physical_tensor[0]
        rho = physical_tensor[1]
        T = physical_tensor[2]
        T_dust = physical_tensor[5]
        v_r *= self.velocity_to_units
        rho *= self.density_to_units
        T *= self.temperature_to_units
        T_dust *= self.temperature_to_units
        return physical_tensor

class UnitsCorrectedSyntheticallyObservedDataset(UnitsCorrectedDataset):
    """
    Wraps a [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset] or
    [`ConcatDataset`][iris.arepo_processing.ConcatDataset] with `from_type` of
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
    and corrects units of fetched tensors to those of `units_dataset`.

    Attributes:
        dataset: The wrapped
            [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset].
        velocity_to_units: The conversion factor into `units_dataset` units.
        density_to_units: The conversion factor into `units_dataset` units.
        temperature_to_units: The conversion factor into `units_dataset` units.

    Args:
        dataset: Sets `self.dataset`.
        units_dataset: The dataset from which to compute conversion factors.
    """

    dataset: SyntheticallyObservedDataset | ConcatDataset

    def __init__(self,
                 dataset: SyntheticallyObservedDataset | ConcatDataset,
                 units_dataset: SyntheticallyObservedDataset | ConcatDataset) -> None:
        super().__init__(dataset=dataset, units_dataset=units_dataset)
        return

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches an indexed, synthetically observed pair in the target units.

        See [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset] for details.

        Args:
            item: The pair index.

        Returns:
            A synthetically observed pair.
        """
        columnized, observed = self.dataset[item]
        columnized *= self.density_to_units
        observed *= self.temperature_to_units
        return columnized, observed


class UnitsCorrectedSimplyObservedDataset(UnitsCorrectedDataset):
    """
    Wraps a [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset] or
    [`ConcatDataset`][iris.arepo_processing.ConcatDataset] with `from_type` of
    [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset]
    and corrects units of fetched tensors to those of `units_dataset`.

    Attributes:
        dataset: The wrapped
            [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset].
        velocity_to_units: The conversion factor into `units_dataset` units.
        density_to_units: The conversion factor into `units_dataset` units.
        temperature_to_units: The conversion factor into `units_dataset` units.

    Args:
        dataset: Sets `self.dataset`.
        units_dataset: The dataset from which to compute conversion factors.
    """

    dataset: SimplyObservedDataset | ConcatDataset

    def __init__(self,
                 dataset: SimplyObservedDataset | ConcatDataset,
                 units_dataset: SimplyObservedDataset | ConcatDataset) -> None:
        super().__init__(dataset=dataset, units_dataset=units_dataset)
        return

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches an indexed, simply observed pair in the target units.

        See [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset] for details.

        Args:
            item: The pair index.

        Returns:
            A simply observed pair.
        """
        columnized, observed = self.dataset[item]
        columnized *= self.density_to_units
        observed *= self.v_density_to_units
        return columnized, observed


class DatasetParent(Dataset):
    """
    A parent dataset established by the manager process of a
    [`Writer`][iris.arepo_processing_write.Writer] into which
    [children][iris.arepo_processing.DatasetChild] instantiated by worker processes of a
    [`Writer`][iris.arepo_processing_write.Writer] can be merged.

    Exists for type-inheritance only. Used only in [`Writer`][iris.arepo_processing_write.Writer] mode.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors (or pre-observed pairs) in the dataset.

    Args:
        path: Directory on disk at which to establish the DatasetParent.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """
    pass


class DatasetChild(Dataset):
    """
    A child dataset established by a worker process of a
    [`Writer`][iris.arepo_processing_write.Writer] that can be merged into the
    [parent][iris.arepo_processing.DatasetParent] instantiated by the manager process of a
    [`Writer`][iris.arepo_processing_write.Writer].

    Only for merging into a [`DatasetParent`][iris.arepo_processing.DatasetParent]
    in [`Writer`][iris.arepo_processing_write.Writer] mode. Not saved to the disk as a readable
    [`Dataset`][iris.arepo_processing.Dataset] itself.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory belonging to [`DatasetParent`][iris.arepo_processing.DatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to [`DatasetParent`][iris.arepo_processing.DatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors (or pre-observed pairs) in the dataset.
        
    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """

    name: str
    parent_path: str
    
    def __init__(self,
                 name: str,
                 parent_path: str,
                 *args: any,
                 **kwargs: any) -> None:
        self.name = name
        self.parent_path = parent_path
        super().__init__(*args,
                         path=(os.path.join(parent_path, name)),
                         **kwargs)
        return

    def _make(self) -> None:
        """
        Attempts to make a new [`DatasetChild`][iris.arepo_processing.DatasetChild] directory at `self.path`.

        If no directory exists at `self.path`, will make the new
        directory. Otherwise, will search all paths `path + f'_{n} for n in range(1, 99)` for
        an available directory path. Will update `self.path`
        to the new path. Overrides [`Dataset._make`][iris.arepo_processing.Dataset._make] because
        [`DatasetChild`][iris.arepo_processing.DatasetChild] also tracks
        `self.name` attribute, which must be updated if `self.path` is updated.

        Raises:
            RuntimeError: If `self.hyper` is `None`.
            RuntimeError: If no available path is found.
        """
        if self.hyper is None:
            raise RuntimeError('No hyper provided to dataset.')
        if os.path.exists(self.path):
            path = None
            for i in range(1, 99):
                p = self.path + '_{}'.format(i)
                if not os.path.exists(p):
                    path = p
                    name = self.name + '_{}'.format(i)
                    break
            if path is None:
                raise RuntimeError('Unusable dataset directory path.')
            self.path = path
            self.name = name

        os.mkdir(self.path)
        return


class StandardDataset(Dataset):
    """
    A [`Dataset`][iris.arepo_processing.Dataset] of physical tensors.

    Stores physical tensors as NumPy (.np) files on disk and converts to PyTorch
    tensors during fetching. Each tensor has element type `np.float32` or `torch.float32`
    (single-precision float). Storing the full physical tensor demands substantial disk space
    and creates a substantial latency when loading a tensor from the disk into memory.
    Use a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] instead
    for [`Reverter`][iris.reversion.Reverter] training. See
    [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor]
    for details regarding the definition of a physical tensor.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """
    def add_tensor(self,
                   tensor: torch.Tensor | np.ndarray,
                   node_comm: mpi4py.MPI.Intracomm | None = None) -> None:
        """
        Writes a physical tensor to the disk, adds its path to
        `self.index`, and increments `self.cardinality`.

        Args:
            tensor: The physical tensor to be added.
            node_comm: A node intracomm. Not used. Only included for type-agnostic call with
                [`add_tensor`][iris.arepo_processing.SyntheticallyObservedDataset.add_tensor] and
                [`add_tensor`][iris.arepo_processing.SimplyObservedDataset.add_tensor].

        Raises:
            RuntimeError: If attempting to write a tensor to the disk when `self.read_only`.
        """
        if self.read_only:
            raise RuntimeError('Cannot write dataset in read-only mode.')
        tensor_index = 'tensor_{}.npy'.format(self.cardinality)
        self.index.append(tensor_index)
        with open(os.path.join(self.path, tensor_index), 'wb') as file:
            np.save(file, tensor)
        self.cardinality += 1
        return

    def update_tensor(self, tensor: torch.Tensor | np.ndarray, num: int) -> None:
        """
        Overwrites a physical tensor on the disk.

        Args:
            tensor: The physical tensor to be added.
            num: The tensor index to overwrite.

        Raises:
            RuntimeError: If attempting to write a tensor to the disk when `self.read_only`.
        """
        if self.read_only:
            raise Exception('Cannot write dataset in read-only mode.')
        tensor_index = self.index[num]
        with open(os.path.join(self.path, tensor_index), 'wb') as file:
            np.save(file, tensor)
        return

    def get_entry(self,
                  num: int | None = None,
                  tensor_index: str | None = None,
                  numpy: bool = False) -> torch.Tensor | np.ndarray:
        """
        Gets a physical tensor from the disk.

        A method specific to [`StandardDataset`][iris.arepo_processing.StandardDataset],
        the helper called by [`__getitem__`][iris.arepo_processing.Dataset.__getitem__],
        but with more options.

        Args:
            num: The integer index of the tensor to get. Not required if `tensor_index` is provided.
            tensor_index: The path on disk of the tensor to get. Not required if `num` is provided.
            numpy: If `True`, returns the physical tensor as the original NumPy array loaded
                from the disk. Otherwise, converts to a PyTorch tensor.

        Returns:
            The physical tensor.

        Raises:
            RuntimeError: If neither `num` nor `tensor_index` is provided.
        """
        if num is not None:
            tensor_index = self.index[num]
        elif tensor_index is None:
            raise RuntimeError('Must provide num (int) or tensor_index (str) to get_entry.')
        tensor = np.load(os.path.join(self.path, tensor_index))
        if not numpy:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        return tensor

    def sample(self,
               n: int,
               numpy: bool = False,
               validation: bool = False,
               abnormal: bool = False) -> torch.Tensor | np.ndarray:
        """
        Gets a sample batch of physical tensors.

        Randomly samples a specified number of physical tensors and returns them stacked along `dim=0`.

        Args:
            n: The number of tensors to get.
            numpy: If `True`, returns the sample as a NumPy array.
                Otherwise, returns a PyTorch tensor.
            validation: If `True`, samples only from `self.validation`.
            abnormal: If `True`, samples only from physical tensors not yet normalized.

        Returns:
            The sample batch.

        Raises:
            ValueError: If `n < 1` or `n > self.cardinality`.
        """
        if n < 1 or n > self.cardinality:
            raise ValueError('Invalid sample size for dataset.')
        if abnormal:
            indices = random.sample(self.abnormal, n)
            if numpy:
                tensors = np.stack([self.get_entry(tensor_index=path, numpy=True) for path in indices])
            else:
                tensors = torch.stack([self.get_entry(tensor_index=path) for path in indices])
        else:
            if validation:
                num_validation = int(self.hyper.training_hyper.validation_data_fraction * self.cardinality)
                indices = random.sample(range(num_validation), n)
            else:
                indices = random.sample(range(self.cardinality), n)
            if numpy:
                tensors = np.stack([self.get_entry(i, numpy=True) for i in indices])
            else:
                tensors = torch.stack([self.get_entry(i) for i in indices])
        return tensors

    def calculate_iris_units(self) -> bool:
        """
        Calculates standard conversion factors from SI units to IRIS units.

        If no IRIS units are found in `self.hyper`, adopts the processing units specified in
        `self.hyper.writer_hyper` as IRIS units.
        Computes conversion factors from SI units to IRIS units. Unlike
        [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units]
        or [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units],
        does not apply any normalization of units based on dataset statistics.

        Returns:
            A `bool`, `True` if IRIS units (found) are different from processing units, `False` otherwise.
        """
        if (self.hyper.dataset_hyper._velocity_iris_per_processing is None
            or self.hyper.dataset_hyper._density_iris_per_processing is None
            or self.hyper.dataset_hyper._temperature_iris_per_processing is None
            or self.hyper.dataset_hyper._length_iris_per_SI is None
            or self.hyper.dataset_hyper._length_iris_per_parsec is None
            or self.hyper.dataset_hyper._time_iris_per_SI is None
            or self.hyper.dataset_hyper._mass_iris_per_SI is None
            or self.hyper.dataset_hyper._temperature_iris_per_SI is None):

            length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
            mass_g_per_processing = self.hyper.writer_hyper.mass_g_per_processing
            velocity_cm_per_s_per_processing = self.hyper.writer_hyper.velocity_cm_per_s_per_processing
            temperature_K_per_processing = self.hyper.writer_hyper.temperature_K_per_processing

            length_iris_per_SI = 100 / length_cm_per_processing
            m_per_parsec = self.hyper.dataset_hyper.meters_per_parsec
            length_iris_per_parsec = length_iris_per_SI * m_per_parsec
            velocity_iris_per_SI = 100 / velocity_cm_per_s_per_processing
            time_iris_per_SI = length_iris_per_SI / velocity_iris_per_SI
            mass_iris_per_SI = 1000 / mass_g_per_processing
            temperature_iris_per_SI = 1 / temperature_K_per_processing

            self.hyper.dataset_hyper._velocity_iris_per_processing = 1.
            self.hyper.dataset_hyper._density_iris_per_processing = 1.
            self.hyper.dataset_hyper._temperature_iris_per_processing = 1.

            self.hyper.dataset_hyper._length_iris_per_SI = float(length_iris_per_SI)
            self.hyper.dataset_hyper._length_iris_per_parsec = float(length_iris_per_parsec)
            self.hyper.dataset_hyper._time_iris_per_SI = float(time_iris_per_SI)
            self.hyper.dataset_hyper._mass_iris_per_SI = float(mass_iris_per_SI)
            self.hyper.dataset_hyper._temperature_iris_per_SI = float(temperature_iris_per_SI)

            iris_processing_units_different = False
        else:
            epsilon = 1e-6
            iris_processing_units_different = (
                abs(self.hyper.dataset_hyper._velocity_iris_per_processing - 1) > epsilon or
                abs(self.hyper.dataset_hyper._density_iris_per_processing - 1) > epsilon or
                abs(self.hyper.dataset_hyper._temperature_iris_per_processing - 1) > epsilon)
        return iris_processing_units_different

    @staticmethod
    def spawn_parent(*args: any, **kwargs: any) -> StandardDatasetParent:
        """
        Spawns a new [`StandardDatasetParent`][iris.arepo_processing.StandardDatasetParent].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`StandardDatasetParent`][iris.arepo_processing.StandardDatasetParent]
                constructors.
            **kwargs: Passed to [`StandardDatasetParent`][iris.arepo_processing.StandardDatasetParent]
                constructors.

        Returns:
            The new [`StandardDatasetParent`][iris.arepo_processing.StandardDatasetParent] object.
        """
        return StandardDatasetParent(*args, **kwargs)

    @staticmethod
    def spawn_child(*args: any, **kwargs: any) -> StandardDatasetChild:
        """
        Spawns a new [`StandardDatasetChild`][iris.arepo_processing.StandardDatasetChild].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`StandardDatasetChild`][iris.arepo_processing.StandardDatasetChild]
                constructors.
            **kwargs: Passed to [`StandardDatasetChild`][iris.arepo_processing.StandardDatasetChild]
                constructors.

        Returns:
            The new [`StandardDatasetChild`][iris.arepo_processing.StandardDatasetChild] object.
        """
        return StandardDatasetChild(*args, **kwargs)


class StandardDatasetParent(DatasetParent, StandardDataset):
    """
    Extends both [`DatasetParent`][iris.arepo_processing.DatasetParent] and 
    [`StandardDataset`][iris.arepo_processing.StandardDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors (or pre-observed pairs) in the dataset.

    Args:
        path: Directory on disk at which to establish the dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """
    def merge(self, child: StandardDatasetChild) -> None:
        """
        Merges a child dataset (owned by a [`Writer`][iris.arepo_processing_write.Writer] worker)
        into `self` (owned by the [`Writer`][iris.arepo_processing_write.Writer] manager).

        Args:
            child: The dataset to merge into `self`.

        Raises:
            RuntimeError: If `child` is not a
                [`StandardDatasetChild`][iris.arepo_processing.StandardDatasetChild]
                sharing the same root path as `self`.
        """
        if isinstance(child, StandardDatasetChild):
            if child.parent_path == self.path:
                child_indices = [os.path.join(child.name, tensor_index) for tensor_index in child.index]
                self.index.extend(child_indices)
                self.abnormal.extend(child_indices)
                self.cardinality += child.cardinality
                return
        raise RuntimeError('Error merging datasets.')


class StandardDatasetChild(DatasetChild, StandardDataset):
    """
    Extends both [`DatasetChild`][iris.arepo_processing.DatasetChild] and
    [`StandardDataset`][iris.arepo_processing.StandardDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory within the [`StandardDatasetParent`][iris.arepo_processing.StandardDatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to [`DatasetParent`][iris.arepo_processing.DatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of tensors (or pre-observed pairs) in the dataset.

    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """
    def normalize(self,
                  hyper: hp.Hyper,
                  node_comm: mpi4py.MPI.Intracomm,
                  gpu_normalize: bool) -> None:
        """
        Converts the dataset into IRIS units.

        Args:
            hyper: The dataset from which to pull IRIS units.
            node_comm: A node-wide MPI intracomm used to communicate with the GPU manager for that node.
            gpu_normalize: If `True`, will move the physical tensors to the GPU before normalization.
                (A legacy feature; not a measurable performance gain.)
        """
        # If GPU support is available and normalizing on GPU,
        # attempt to query the GPU manager and request the access key.
        if torch.cuda.is_available() and gpu_normalize:
            node_size = node_comm.Get_size()
            gpu = None
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)

        for i in range(self.cardinality):
            # If GPU normalizing, move the tensor to the GPU.
            if torch.cuda.is_available() and gpu_normalize:
                tensor = self.get_entry(i, numpy=False)
                tensor = tensor.cuda(gpu)
            else:
                tensor = self.get_entry(i, numpy=True)
            # Normalize tensors.
            with torch.no_grad():
                tensor[0] *= hyper.dataset_hyper._velocity_iris_per_processing
                tensor[1] *= hyper.dataset_hyper._density_iris_per_processing
                tensor[2] *= hyper.dataset_hyper._temperature_iris_per_processing
                tensor[5] *= hyper.dataset_hyper._temperature_iris_per_processing
            if torch.cuda.is_available() and gpu_normalize:
                tensor = tensor.detach().cpu().numpy()
            self.update_tensor(tensor, i)

        # Clean-up GPU and return key to the GPU manager with usage statistics.
        if torch.cuda.is_available() and gpu_normalize:
            if node_size > 1:
                gc.collect()
                torch.cuda.empty_cache()
                memory_usage = int(torch.cuda.max_memory_allocated(gpu))
                node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)
        return


class PreObservedDataset(Dataset):
    """
    A [`Dataset`][iris.arepo_processing.Dataset] of observed pairs.

    Rather than storing the entire physical tensor on disk as in
    [`StandardDataset`][iris.arepo_processing.StandardDataset], stores only a
    [top-down density image][iris.arepo_processing.columnize_physical_tensor] and an
    [observation][iris.observation.Observer] computed from the original physical tensor.
    Produces an order-of-magnitude savings in storage space on disk as well as in latency
    loading from disk into memory. As such, is the primary dataset type that should be used
    for storing large datasets and [`Reverter`][iris.reversion.Reverter] training. Is the
    base class for
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset] and
    [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].
    Top-down density images and observations are stored as separate NumPy (.np) files
    on disk, and are fetched as `columnized, observed` tuples of PyTorch tensors.
    Each tensor has element type `np.float32` or `torch.float32` (single-precision float).

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        observer_kwargs: A `dict` of extra keyword args to be passed to the `observer` forward call.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        observer_kwargs: A `dict` of extra keyword args to be passed to the `observer` forward call.
            Sets `self.observer_kwargs`. If `None`, sets `self.observer_kwargs = {}`,
            i.e. no extra keyword args are passed to the observer.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """

    observer: observation.Observer | None
    reduction: typing.Callable
    observer_kwargs: dict

    def __init__(self,
                 *args: any,
                 node_comm: mpi4py.MPI.Intracomm | None = None,
                 observer_kwargs: dict | None = None,
                 **kwargs: any) -> None:
        super().__init__(*args, **kwargs)
        reduction = self.hyper.cube_hyper.reduction
        if reduction == 'mean':
            self.reduction = self._reduce_mean
        elif reduction == 'max':
            self.reduction = self._reduce_max
        else:
            raise ValueError("Cube reduction must be one of: 'mean', 'max'.")
        if self.read_only:
            self.observer = None
            self.observer_kwargs = None
        else:
            self._init_observer(node_comm)
            if observer_kwargs is None:
                observer_kwargs = {}
            self.observer_kwargs = observer_kwargs
        return

    def _init_observer(self, node_comm: mpi4py.MPI.Intracomm) -> None:
        """
        Abstract method definition for initializing an observer. MPI node intracomm required
            for communication with GPU manager to enable GPU support.
        """
        pass

    def _reduce_mean(self, ppv: torch.Tensor) -> torch.Tensor:
        """
        Applies a mean reduction of a PPV cube over the latitude dimension.

        Args:
            ppv: The PPV cube to reduce.

        Returns:
            A latitude-velocity (PV) mean observation.
        """
        return torch.mean(ppv, dim=3)

    def _reduce_max(self, ppv: torch.Tensor) -> torch.Tensor:
        """
        Applies a max reduction of a PPV cube over the latitude dimension.

        Args:
            ppv: The PPV cube to reduce.

        Returns:
            A latitude-velocity (PV) max observation.
        """
        return torch.max(ppv, dim=3)[0]

    def add_tensor(self,
                   tensor: torch.Tensor | np.ndarray,
                   node_comm: mpi4py.MPI.Intracomm) -> None:
        """
        Computes an observed pair from a physical tensor, write the pair to the disk,
        adds the pair of paths to `self.index`, and increments `self.cardinality`.

        Args:
            tensor: The physical tensor to be columnized/observed.
            node_comm: An MPI node intracomm used to communicate with the GPU manager for
                GPU support during observation.

        Raises:
            RuntimeError: If attempting to write a tensor to the disk when `self.read_only`.
            RuntimeError: If `self.observer` not set.
        """
        if self.read_only:
            raise RuntimeError('Cannot write dataset in read-only mode.')

        if self.observer is None:
            raise RuntimeError('No observer set in PreObservedDataset. Do not instantiate this class directly.')
        tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(dim=0)

        # If GPU support is available, attempt to query the GPU manager and request the access key.
        # If successful, move the tensor and observer to the GPU.
        if torch.cuda.is_available():
            node_size = node_comm.Get_size()
            gpu = None
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)
            tensor = tensor.cuda(gpu)
            self.observer.cuda(gpu)

        # Compute the observed pair.
        if self.observer.in_blur is not None:
            with torch.no_grad():
                tensor = self.observer.in_blur(tensor, inplace=True)
        with torch.no_grad():
            columnized = columnize_physical_tensor(tensor, self.hyper).squeeze(dim=0).cpu().numpy()
            observed = self.observer(tensor, bypass_blur_in=True, **self.observer_kwargs)
            observed = self.reduction(observed).squeeze(dim=0).cpu().numpy()

        # Clean-up GPU and return key to the GPU manager with usage statistics.
        del tensor
        if torch.cuda.is_available():
            if node_size > 1:
                self.observer = self.observer.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                memory_usage = int(torch.cuda.max_memory_allocated(gpu))
                node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)

        # Write the observed pair to the disk.
        columnized_index = 'columnized_{}.npy'.format(self.cardinality)
        observed_index = 'observed_{}.npy'.format(self.cardinality)
        self.index.append([columnized_index, observed_index])
        with open(os.path.join(self.path, columnized_index), 'wb') as file:
            np.save(file, columnized)
        with open(os.path.join(self.path, observed_index), 'wb') as file:
            np.save(file, observed)
        self.cardinality += 1
        return

    def update_entry(self,
                     columnized: torch.Tensor | np.ndarray,
                     observed: torch.Tensor | np.ndarray,
                     num: int) -> None:
        """
        Overwrites an observed pair on the disk.

        Args:
            columnized: The [top-down density image][iris.arepo_processing.columnize_physical_tensor] to add.
            observed: The [observation][iris.observation.Observer] to add.
            num: The pair index to overwrite.

        Raises:
            RuntimeError: If attempting to write an observed pair to the disk when `self.read_only`.
        """
        if self.read_only:
            raise Exception('Cannot write dataset in read-only mode.')
        columnized_index, observed_index = self.index[num]
        with open(os.path.join(self.path, columnized_index), 'wb') as file:
            np.save(file, columnized)
        with open(os.path.join(self.path, observed_index), 'wb') as file:
            np.save(file, observed)
        return

    def get_entry(self, num: int | None = None,
                  tensor_indices: typing.Sequence[str] | None = None,
                  numpy: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets an observed pair from the disk.

        A method specific to [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset],
        the helper called by [`__getitem__`][iris.arepo_processing.Dataset.__getitem__],
        but with more options.

        Args:
            num: The integer index of the observed pair to get. Not required if `tensor_indices` is provided.
            tensor_indices: The paths on disk of the observed pair to get. Not required if `num` is provided.
            numpy: If `True`, returns the observed pair as a tuple of the original NumPy arrays loaded
                from the disk. Otherwise, returns a tuple of converted PyTorch tensors.

        Returns:
            The observed pair, a tuple of `columnized, observed`.

        Raises:
            RuntimeError: If neither `num` nor `tensor_indices` is provided.
        """
        if num is not None:
            columnized_index, observed_index = self.index[num]
        elif tensor_indices is not None:
            columnized_index, observed_index = tensor_indices
        else:
            raise RuntimeError('Must provide num (int) or tensor_indices ([str, str]) to get_entry.')
        columnized = np.load(os.path.join(self.path, columnized_index))
        observed = np.load(os.path.join(self.path, observed_index))
        if not numpy:
            columnized = torch.tensor(columnized, dtype=torch.float32)
            observed = torch.tensor(observed, dtype=torch.float32)
        return columnized, observed

    def sample(self,
               n: int,
               numpy: bool = False,
               validation: bool = False,
               abnormal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets sample batches of observed pairs.

        Randomly samples a specified number of observed pairs. Returns a tuple `columnized, observed`,
        where `columnized` and `observed` are each the respective samples stacked along `dim=0`.

        Args:
            n: The number of tensors to get.
            numpy: If `True`, returns the sample as a tuple of NumPy arrays.
                Otherwise, returns a tuple of PyTorch tensors.
            validation: If `True`, samples only from `self.validation`.
            abnormal: If `True`, samples only from physical tensors not yet normalized.

        Returns:
            The tuple of sample batches.

        Raises:
            ValueError: If `n < 1` or `n > self.cardinality`.
        """
        if n < 1 or n > self.cardinality:
            raise ValueError('Invalid sample size for dataset.')
        if abnormal:
            indices = random.sample(self.abnormal, n)
            columnized, observed = zip(*[self.get_entry(tensor_indices=paths, numpy=numpy) for paths in indices])
            if numpy:
                columnized = np.stack(columnized)
                observed = np.stack(observed)
            else:
                columnized = torch.stack(columnized)
                observed = torch.stack(observed)
        else:
            if validation:
                num_validation = int(self.hyper.training_hyper.validation_data_fraction * self.cardinality)
                indices = random.sample(range(num_validation), n)
            else:
                indices = random.sample(range(self.cardinality), n)
            columnized, observed = zip(*[self.get_entry(i, numpy=numpy) for i in indices])
            if numpy:
                columnized = np.stack(columnized)
                observed = np.stack(observed)
            else:
                columnized = torch.stack(columnized)
                observed = torch.stack(observed)
        return columnized, observed


class PreObservedDatasetParent(DatasetParent, PreObservedDataset):
    """
    Extends both [`DatasetParent`][iris.arepo_processing.DatasetParent] and
    [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def merge(self, child: PreObservedDatasetChild) -> None:
        """
        Merges a child dataset (owned by a [`Writer`][iris.arepo_processing_write.Writer] worker)
        into `self` (owned by the [`Writer`][iris.arepo_processing_write.Writer] manager).

        Args:
            child: The dataset to merge into `self`.

        Raises:
            RuntimeError: If `child` is not a
                [`PreObservedDatasetChild`][iris.arepo_processing.PreObservedDatasetChild]
                sharing the same root path as `self`.
        """
        if isinstance(child, PreObservedDatasetChild):
            if child.parent_path == self.path:
                child_indices = [[os.path.join(child.name, columnized_index),
                                  os.path.join(child.name, observed_index)]
                                   for columnized_index, observed_index in child.index]
                self.index.extend(child_indices)
                self.abnormal.extend(child_indices)
                self.cardinality += child.cardinality
                return
        raise RuntimeError('Error merging datasets.')


class PreObservedDatasetChild(DatasetChild, PreObservedDataset):
    """
    Extends both [`DatasetChild`][iris.arepo_processing.DatasetChild] and
    [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory within the [`PreObservedDatasetParent`][iris.arepo_processing.PreObservedDatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to [`DatasetParent`][iris.arepo_processing.DatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    pass


class SyntheticallyObservedDataset(PreObservedDataset):
    """
    A [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] of synthetically observed pairs.

    Uses [`SyntheticObserver`][iris.observation.SyntheticObserver] to compute true synthetic
    observations of each physical tensor in the dataset. Contrast with
    [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset], which only computes
    [simple observations][iris.observation.SimpleObserver] (density projections).
    The primary dataset type for [`Reverter`][iris.reversion.Reverter] training.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def __init__(self,
                 *args: any,
                 abundance: observation.Abundance | None = None,
                 **kwargs: any) -> None:
        self.abundance = abundance
        super().__init__(*args, **kwargs)
        return

    def _init_observer(self, node_comm: mpi4py.MPI.Intracomm | None = None) -> None:
        """
        Creates an [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver]
        for processing of physical tensors.

        Args:
            node_comm: The node-wide MPI intracomm used to communicate with the GPU manager for GPU support.
        """
        self.observer = observation.IteratedSyntheticObserver(self.hyper,
                                                              abundance=self.abundance,
                                                              units='processing',
                                                              node_comm=node_comm)
        self.observer.eval()
        return

    def calculate_iris_units(self) -> bool:
        """
        Calculates standard conversion factors from SI units to IRIS units.

        If no IRIS units are found in `self.hyper`,
        defines IRIS units by normalizing density and (brightness) temperature in processing units
        according to a sample estimate (using Bessel's correction) of the global
        standard deviations of these variables in the dataset. Normalization is
        to make values optimal for [`Reverter`][iris.reversion.Reverter] training.
        Standard deviations are computed from a random sample of
        `self.hyper.writer_hyper.unit_calculation_sample_size`
        observed pairs. Computes conversion factors from SI units to IRIS units.

        Returns:
            A `bool`, `True` if IRIS units (found or computed) are different from processing units,
                `False` otherwise.
        """
        if (self.hyper.dataset_hyper._density_iris_per_processing is None
            or self.hyper.dataset_hyper._length_iris_per_SI is None
            or self.hyper.dataset_hyper._length_iris_per_parsec is None
            or self.hyper.dataset_hyper._time_iris_per_SI is None
            or self.hyper.dataset_hyper._mass_iris_per_SI is None
            or self.hyper.dataset_hyper._temperature_iris_per_SI is None):

            num = self.hyper.writer_hyper.unit_calculation_sample_size
            columnized, observed = self.sample(num, numpy=True, abnormal=True)

            density_iris_per_processing = 1. / np.std(columnized.flatten(), ddof=1)
            temperature_iris_per_processing = 1. / np.std(observed.flatten(), ddof=1)

            length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
            mass_g_per_processing = self.hyper.writer_hyper.mass_g_per_processing
            velocity_cm_per_s_per_processing = self.hyper.writer_hyper.velocity_cm_per_s_per_processing
            temperature_K_per_processing = self.hyper.writer_hyper.temperature_K_per_processing

            time_processing_per_SI = velocity_cm_per_s_per_processing / length_cm_per_processing
            length_processing_per_SI = 100 / length_cm_per_processing
            volume_processing_per_SI = length_processing_per_SI * length_processing_per_SI * length_processing_per_SI
            mass_processing_per_SI = 1000 / mass_g_per_processing
            density_processing_per_SI = mass_processing_per_SI / volume_processing_per_SI
            temperature_processing_per_SI = 1 / temperature_K_per_processing

            density_iris_per_SI = density_iris_per_processing * density_processing_per_SI
            temperature_iris_per_SI = temperature_iris_per_processing * temperature_processing_per_SI
            time_iris_per_SI = time_processing_per_SI
            mass_iris_per_SI = mass_processing_per_SI
            volume_iris_per_SI = mass_iris_per_SI / density_iris_per_SI
            length_iris_per_SI = np.pow(volume_iris_per_SI, 1 / 3)
            m_per_parsec = self.hyper.dataset_hyper.meters_per_parsec
            length_iris_per_parsec = length_iris_per_SI * m_per_parsec

            self.hyper.dataset_hyper._density_iris_per_processing = float(density_iris_per_processing)
            self.hyper.dataset_hyper._temperature_iris_per_processing = float(temperature_iris_per_processing)
            self.hyper.dataset_hyper._length_iris_per_SI = float(length_iris_per_SI)
            self.hyper.dataset_hyper._length_iris_per_parsec = float(length_iris_per_parsec)
            self.hyper.dataset_hyper._time_iris_per_SI = float(time_iris_per_SI)
            self.hyper.dataset_hyper._mass_iris_per_SI = float(mass_iris_per_SI)
            self.hyper.dataset_hyper._temperature_iris_per_SI = float(temperature_iris_per_SI)

            iris_processing_units_different = True
        else:
            epsilon = 1e-6
            iris_processing_units_different = (
                abs(self.hyper.dataset_hyper._density_iris_per_processing - 1) > epsilon or
                abs(self.hyper.dataset_hyper._temperature_iris_per_processing - 1) > epsilon)
        return iris_processing_units_different

    @staticmethod
    def spawn_parent(*args: any, **kwargs: any) -> SyntheticallyObservedDatasetParent:
        """
        Spawns a new [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent]
                constructors.
            **kwargs: Passed to [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent]
                constructors.

        Returns:
            The new [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent] object.
        """
        return SyntheticallyObservedDatasetParent(*args, **kwargs)

    @staticmethod
    def spawn_child(*args: any, **kwargs: any) -> SyntheticallyObservedDatasetChild:
        """
        Spawns a new [`SyntheticallyObservedDatasetChild`][iris.arepo_processing.SyntheticallyObservedDatasetChild].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`SyntheticallyObservedDatasetChild`][iris.arepo_processing.SyntheticallyObservedDatasetChild]
                constructors.
            **kwargs: Passed to [`SyntheticallyObservedDatasetChild`][iris.arepo_processing.SyntheticallyObservedDatasetChild]
                constructors.

        Returns:
            The new [`SyntheticallyObservedDatasetChild`][iris.arepo_processing.SyntheticallyObservedDatasetChild] object.
        """
        return SyntheticallyObservedDatasetChild(*args, **kwargs)


class SyntheticallyObservedDatasetParent(PreObservedDatasetParent, SyntheticallyObservedDataset):
    """
    Extends both [`PreObservedDatasetParent`][iris.arepo_processing.PreObservedDatasetParent] and
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or
            raise an Exception otherwise. If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    pass


class SyntheticallyObservedDatasetChild(PreObservedDatasetChild, SyntheticallyObservedDataset):
    """
    Extends both [`PreObservedDatasetChild`][iris.arepo_processing.PreObservedDatasetChild] and
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory within the
            [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to
            [`SyntheticallyObservedDatasetParent`][iris.arepo_processing.SyntheticallyObservedDatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def normalize(self,
                  hyper: hp.Hyper,
                  node_comm: mpi4py.MPI.Intracomm,
                  gpu_normalize: bool) -> None:
        """
        Converts the dataset into IRIS units.

        Args:
            hyper: The dataset from which to pull IRIS units.
            node_comm: A node-wide MPI intracomm used to communicate with the GPU manager for that node.
            gpu_normalize: If `True`, will move the physical tensors to the GPU before normalization.
                (A legacy feature; not a measurable performance gain.)
        """
        # If GPU support is available and normalizing on GPU,
        # attempt to query the GPU manager and request the access key.
        if torch.cuda.is_available() and gpu_normalize:
            node_size = node_comm.Get_size()
            gpu = None
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)

        for i in range(self.cardinality):
            # If GPU normalizing, move observed pairs to the GPU.
            if torch.cuda.is_available() and gpu_normalize:
                columnized, observed = self.get_entry(i, numpy=False)
                columnized = columnized.cuda(gpu)
                observed = observed.cuda(gpu)
            else:
                columnized, observed = self.get_entry(i, numpy=True)
            # Normalize observed pairs.
            with torch.no_grad():
                columnized *= hyper.dataset_hyper._density_iris_per_processing
                observed *= hyper.dataset_hyper._temperature_iris_per_processing
            if torch.cuda.is_available() and gpu_normalize:
                columnized = columnized.detach().cpu().numpy()
                observed = observed.detach().cpu().numpy()
            self.update_entry(columnized, observed, i)

        # Clean-up GPU and return key to the GPU manager with usage statistics.
        if torch.cuda.is_available() and gpu_normalize:
            if node_size > 1:
                gc.collect()
                torch.cuda.empty_cache()
                memory_usage = int(torch.cuda.max_memory_allocated(gpu))
                node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)
        return


class CPUBatchObservedDataset(SyntheticallyObservedDataset):
    """
    A [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
    that uses CPU batching during observation.

    Uses [`SyntheticObserver`][iris.observation.SyntheticObserver] with `cpu_batch=True`
    to compute synthetic observations of each physical tensor in the dataset.
    Necessary if physical tensors are too large to fit on the GPU, e.g. for
    full-cone observation.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self.read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def _init_observer(self, node_comm: mpi4py.MPI.Intracomm | None = None) -> None:
        """
        Creates an [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver]
        with `cpu_batch=True` for processing of physical tensors.

        Args:
            node_comm: The node-wide MPI intracomm used to communicate with the GPU manager for GPU support.
        """
        self.observer = observation.IteratedSyntheticObserver(self.hyper,
                                                              abundance=self.abundance,
                                                              units='processing',
                                                              node_comm=node_comm,
                                                              cpu_batch=True)
        self.observer.eval()
        return

    def add_tensor(self,
                   tensor: torch.Tensor | np.ndarray,
                   node_comm: mpi4py.MPI.Intracomm) -> None:
        """
        Computes an observed pair from a physical tensor, write the pair to the disk,
        adds the pair of paths to `self.index`, and increments `self.cardinality`.

        Differs from [`PreObservedDataset.add_tensor`][iris.arepo_processing.PreObservedDataset.add_tensor]
        by not moving physical tensors to the GPU prior to calling `self.observer`
        and by not precomputing [input blur][iris.observation.VelocityBlur].

        Args:
            tensor: The physical tensor to be columnized/observed.
            node_comm: An MPI node intracomm used to communicate with the GPU manager for
                GPU support during observation.

        Raises:
            RuntimeError: If attempting to write a tensor to the disk when `self.read_only`.
            RuntimeError: If `self.observer` not set.
        """
        if self.read_only:
            raise RuntimeError('Cannot write dataset in read-only mode.')

        if self.observer is None:
            raise RuntimeError('No observer set in CPUBatchObservedDataset.')
        tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(dim=0)

        # If GPU support is available, attempt to query the GPU manager and request the access key.
        # If successful, move the observer to the GPU, but not he physical tensor.
        gpu = None
        if torch.cuda.is_available():
            node_size = node_comm.Get_size()
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)
            self.observer.cuda(gpu)

        # Compute the observed pair.
        with torch.no_grad():
            columnized = columnize_physical_tensor(tensor, self.hyper).squeeze(dim=0).cpu().numpy()
            observed = self.observer(tensor, gpu=gpu)
            observed = self.reduction(observed).squeeze(dim=0).cpu().numpy()

        # Clean-up GPU and return key to the GPU manager with usage statistics.
        if torch.cuda.is_available():
            if node_size > 1:
                self.observer = self.observer.cpu()
                gc.collect()
                torch.cuda.empty_cache()
                memory_usage = int(torch.cuda.max_memory_allocated(gpu))
                node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)

        # Write the observed pair to the disk.
        columnized_index = 'columnized_{}.npy'.format(self.cardinality)
        observed_index = 'observed_{}.npy'.format(self.cardinality)
        self.index.append([columnized_index, observed_index])
        with open(os.path.join(self.path, columnized_index), 'wb') as file:
            np.save(file, columnized)
        with open(os.path.join(self.path, observed_index), 'wb') as file:
            np.save(file, observed)
        self.cardinality += 1
        return

    @staticmethod
    def spawn_parent(*args: any, **kwargs: any) -> CPUBatchObservedDatasetParent:
        """
        Spawns a new [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent]
                constructors.
            **kwargs: Passed to [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent]
                constructors.

        Returns:
            The new [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent] object.
        """
        return CPUBatchObservedDatasetParent(*args, **kwargs)

    @staticmethod
    def spawn_child(*args: any, **kwargs: any) -> CPUBatchObservedDatasetChild:
        """
        Spawns a new [`CPUBatchObservedDatasetChild`][iris.arepo_processing.CPUBatchObservedDatasetChild].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`CPUBatchObservedDatasetChild`][iris.arepo_processing.CPUBatchObservedDatasetChild]
                constructors.
            **kwargs: Passed to [`CPUBatchObservedDatasetChild`][iris.arepo_processing.CPUBatchObservedDatasetChild]
                constructors.

        Returns:
            The new [`CPUBatchObservedDatasetChild`][iris.arepo_processing.CPUBatchObservedDatasetChild] object.
        """
        return CPUBatchObservedDatasetChild(*args, **kwargs)


class CPUBatchObservedDatasetParent(PreObservedDatasetParent, CPUBatchObservedDataset):
    """
    Extends both [`PreObservedDatasetParent`][iris.arepo_processing.PreObservedDatasetParent] and
    [`CPUBatchObservedDataset`][iris.arepo_processing.CPUBatchObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not `self.read_only` or
            raise an Exception otherwise. If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    pass


class CPUBatchObservedDatasetChild(CPUBatchObservedDataset, SyntheticallyObservedDatasetChild):
    """
    Extends both [`CPUBatchObservedDataset`][iris.arepo_processing.CPUBatchObservedDataset] and
    [`SyntheticallyObservedDatasetChild`][iris.arepo_processing.SyntheticallyObservedDatasetChild].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory within the
            [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to
            [`CPUBatchObservedDatasetParent`][iris.arepo_processing.CPUBatchObservedDatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    pass


class SimplyObservedDataset(PreObservedDataset):
    """
    A [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] of simply observed pairs.

    Uses [`SimpleObserver`][iris.observation.SimpleObserver] to compute simple
    observations (density projections) of each physical tensor in the dataset. Contrast with
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset],
    which computes full synthetic observations. Use to investigate the theoretical
    information limit in predicting top-down density images from lv-reduced observations,
    without the confounding effects of a true synthetic observation.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        read_only: If True, prevents writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.
        training: Random subset used for [`Reverter`][iris.reversion.Reverter] training.
            (Deterministically sampled via `seed`. The set complement of `self.validation`.)
        validation: Random subset used for [`Reverter`][iris.reversion.Reverter] validation.
            (Deterministically sampled via `seed`. The set complement of `self.training`.)

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not
            `self,read_only` or raise an Exception otherwise.
            If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        read_only: Sets `self.read_only`.
        seed: Seed for the random number generator used for deterministically random sampling of
            `self.training` and `self.validation` subsets.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def _init_observer(self, node_comm: mpi4py.MPI.Intracomm | None = None) -> None:
        """
        Creates an [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver]
        for processing of physical tensors.

        Args:
            node_comm: The node-wide MPI intracomm used to communicate with the GPU manager for GPU support.
        """
        self.observer = observation.IteratedSimpleObserver(self.hyper, units='processing')
        self.observer.eval()
        return

    def calculate_iris_units(self) -> bool:
        """
        Calculates standard conversion factors from SI units to IRIS units.

        If no IRIS units are found in `self.hyper`,
        defines IRIS units by normalizing density and velocity-density in processing units
        according to a sample estimate (using Bessel's correction) of the global
        standard deviation in the dataset. Normalization is to make values optimal for
        [`Reverter`][iris.reversion.Reverter] training. Standard deviations are computed
        from a random sample of
        `self.hyper.writer_hyper.unit_calculation_sample_size`
        observed pairs. Computes conversion factors from SI units to IRIS units.

        Returns:
            A `bool`, `True` if IRIS units (found or computed) are different from processing units,
                `False` otherwise.
        """
        if (self.hyper.dataset_hyper._density_iris_per_processing is None
            or self.hyper.dataset_hyper._v_density_iris_per_processing is None
            or self.hyper.dataset_hyper._length_iris_per_SI is None
            or self.hyper.dataset_hyper._length_iris_per_parsec is None
            or self.hyper.dataset_hyper._time_iris_per_SI is None
            or self.hyper.dataset_hyper._mass_iris_per_SI is None
            or self.hyper.dataset_hyper._temperature_iris_per_SI is None):

            num = self.hyper.writer_hyper.unit_calculation_sample_size
            columnized, observed = self.sample(num, numpy=True, abnormal=True)

            density_iris_per_processing = 1 / np.std(columnized.flatten(), ddof=1)
            v_density_iris_per_processing = 1 / np.std(observed.flatten(), ddof=1)

            length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
            mass_g_per_processing = self.hyper.writer_hyper.mass_g_per_processing
            velocity_cm_per_s_per_processing = self.hyper.writer_hyper.velocity_cm_per_s_per_processing
            temperature_K_per_processing = self.hyper.writer_hyper.temperature_K_per_processing

            time_processing_per_SI = velocity_cm_per_s_per_processing / length_cm_per_processing
            length_processing_per_SI = 100 / length_cm_per_processing
            volume_processing_per_SI = length_processing_per_SI * length_processing_per_SI * length_processing_per_SI
            mass_processing_per_SI = 1000 / mass_g_per_processing
            density_processing_per_SI = mass_processing_per_SI / volume_processing_per_SI
            v_density_processing_per_SI = density_processing_per_SI * time_processing_per_SI
            temperature_processing_per_SI = 1 / temperature_K_per_processing

            density_iris_per_SI = density_iris_per_processing * density_processing_per_SI
            v_density_iris_per_SI = v_density_iris_per_processing * v_density_processing_per_SI
            time_iris_per_SI = v_density_iris_per_SI / density_iris_per_SI
            mass_iris_per_SI = mass_processing_per_SI
            volume_iris_per_SI = mass_iris_per_SI / density_iris_per_SI
            length_iris_per_SI = np.pow(volume_iris_per_SI, 1 / 3)
            m_per_parsec = self.hyper.dataset_hyper.meters_per_parsec
            length_iris_per_parsec = length_iris_per_SI * m_per_parsec
            temperature_iris_per_SI = temperature_processing_per_SI

            self.hyper.dataset_hyper._density_iris_per_processing = float(density_iris_per_processing)
            self.hyper.dataset_hyper._v_density_iris_per_processing = float(v_density_iris_per_processing)
            self.hyper.dataset_hyper._length_iris_per_SI = float(length_iris_per_SI)
            self.hyper.dataset_hyper._length_iris_per_parsec = float(length_iris_per_parsec)
            self.hyper.dataset_hyper._time_iris_per_SI = float(time_iris_per_SI)
            self.hyper.dataset_hyper._mass_iris_per_SI = float(mass_iris_per_SI)
            self.hyper.dataset_hyper._temperature_iris_per_SI = float(temperature_iris_per_SI)

            iris_processing_units_different = True
        else:
            epsilon = 1e-6
            iris_processing_units_different = (
                abs(self.hyper.dataset_hyper._density_iris_per_processing - 1) > epsilon or
                abs(self.hyper.dataset_hyper._v_density_iris_per_processing - 1) > epsilon)
        return iris_processing_units_different

    @staticmethod
    def spawn_parent(*args: any, **kwargs: any) -> SimplyObservedDatasetParent:
        """
        Spawns a new [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent]
                constructors.
            **kwargs: Passed to [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent]
                constructors.

        Returns:
            The new [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent] object.
        """
        return SimplyObservedDatasetParent(*args, **kwargs)

    @staticmethod
    def spawn_child(*args: any, **kwargs: any) -> SimplyObservedDatasetChild:
        """
        Spawns a new [`SimplyObservedDatasetChild`][iris.arepo_processing.SimplyObservedDatasetChild].

        Used in [`Writer`][iris.arepo_processing_write.Writer] to allow type-agnostic calls
        from the base class type.

        Args:
            *args: Passed to [`SimplyObservedDatasetChild`][iris.arepo_processing.SimplyObservedDatasetChild]
                constructors.
            **kwargs: Passed to [`SimplyObservedDatasetChild`][iris.arepo_processing.SimplyObservedDatasetChild]
                constructors.

        Returns:
            The new [`SimplyObservedDatasetChild`][iris.arepo_processing.SimplyObservedDatasetChild] object.
        """
        return SimplyObservedDatasetChild(*args, **kwargs)


class SimplyObservedDatasetParent(PreObservedDatasetParent, SimplyObservedDataset):
    """
    Extends both [`PreObservedDatasetParent`][iris.arepo_processing.PreObservedDatasetParent] and
    [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        path: Directory on disk at which to establish the Dataset.
            If a readable [`Dataset`][iris.arepo_processing.Dataset] already exists at the directory,
            will open and extend the existing dataset.
            If no readable [`Dataset`][iris.arepo_processing.Dataset] exists at the directory,
            will attempt to write a new directory if not `self.read_only` or
            raise an Exception otherwise. If an unreadable directory exists at this path,
            will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    pass


class SimplyObservedDatasetChild(PreObservedDatasetChild, SimplyObservedDataset):
    """
    Extends both [`PreObservedDatasetChild`][iris.arepo_processing.PreObservedDatasetChild] and
    [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].

    Used during dataset [writing][iris.arepo_processing_write.Writer] only.

    Attributes:
        path: Path to the dataset location on disk.
        name: Name of the subdirectory within the
            [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent]
            in which the child is hosted. Enables merging.
        parent_path: Path of the subdirectory belonging to
            [`SimplyObservedDatasetParent`][iris.arepo_processing.SimplyObservedDatasetParent]
            in which the child is hosted. Enables merging.
        hyper: Hyperparameters object.
        observer: The observer applied to each physical tensor during processing.
        reduction: The reduction function applied to a full observation (PPV cube) before writing to disk.
        index: List of individual tensor paths.
        abnormal: List of tensor paths that have yet to be normalized (units-corrected).
        cardinality: Number of observed pairs in the dataset.

    Args:
        name: Sets `self.name`.
        parent_path: Directory on disk of the parent dataset. Attempts to establish
            the dataset at `path='parent_path/name'`.
            If the directory `path` is not free, will check if an amended path is available for writing.
            Final (amended) value sets `self.path`.
            (See [`_make`][iris.arepo_processing.DatasetChild._make] for details.)
        hyper: A [`Hyper`][iris.hyper.Hyper] hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        node_comm: An MPI intracomm used to communicate with the GPU manager if available.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If, when attempting to set `self.reduction`,
            `self.hyper.cube_hyper.reduction` is not one of 'mean' or 'max'.
    """
    def normalize(self,
                  hyper: hp.Hyper,
                  node_comm: mpi4py.MPI.Intracomm,
                  gpu_normalize: bool) -> None:
        """
        Converts the dataset into IRIS units.

        Args:
            hyper: The dataset from which to pull IRIS units.
            node_comm: A node-wide MPI intracomm used to communicate with the GPU manager for that node.
            gpu_normalize: If `True`, will move the physical tensors to the GPU before normalization.
                (A legacy feature; not a measurable performance gain.)
        """
        # If GPU support is available and normalizing on GPU,
        # attempt to query the GPU manager and request the access key.
        if torch.cuda.is_available() and gpu_normalize:
            node_size = node_comm.Get_size()
            gpu = None
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)

        for i in range(self.cardinality):
            # If GPU normalizing, move observed pairs to the GPU.
            if torch.cuda.is_available() and gpu_normalize:
                columnized, observed = self.get_entry(i, numpy=False)
                columnized = columnized.cuda(gpu)
                observed = observed.cuda(gpu)
            else:
                columnized, observed = self.get_entry(i, numpy=True)
            # Normalize observed pairs.
            with torch.no_grad():
                columnized *= hyper.dataset_hyper._density_iris_per_processing
                observed *= hyper.dataset_hyper._v_density_iris_per_processing
            if torch.cuda.is_available() and gpu_normalize:
                columnized = columnized.detach().cpu().numpy()
                observed = observed.detach().cpu().numpy()
            self.update_entry(columnized, observed, i)

        # Clean-up GPU and return key to the GPU manager with usage statistics.
        if torch.cuda.is_available() and gpu_normalize:
            if node_size > 1:
                gc.collect()
                torch.cuda.empty_cache()
                memory_usage = int(torch.cuda.max_memory_allocated(gpu))
                node_comm.Isend(np.array([gpu, memory_usage, 0], dtype=np.int64), dest=gpu_manager, tag=9)
        return


class Snapshot:
    r"""
    An AREPO simulation snapshot.

    Reads an AREPO simulation snapshot (in HDF5 file format).
    Automatically handles all value parsing/computation and units conversions.
    Applies any desired perturbations specified in hyperparameters. Computes
    [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
    from the snapshot by interpolating the cell values of the AREPO Voronoi mesh
    over the IRIS spherical coordinate grid defined according to the observer origin.

    Attributes:
        hyper: A hyperparameters object.
        path: Path to the AREPO snapshot (HDF5 file) on disk.
        file: The `h5py.File` object for the AREPO snapshot loaded into memory.
        gpu_interpolate: Whether to use CuPy-enabled GPU support during interpolation of the
            AREPO Voronoi mesh onto the IRIS spherical coordinate grid.
            Is only set to true if `gpu_interpolate=True` is specified in args,
            a GPU is detected by the Python process, and `iris.arepo_processing.CUPY_ENABLED`,
            i.e. CuPy is installed and successfully imported.
        galactic_center: The center of the snapshot in AREPO coordinates/AREPO units.
            Set by [`_get_arepo_center`][iris.arepo_processing.Snapshot._get_arepo_center].
        num_particles: The number of cell particles in the AREPO snapshot.
        positions: The positions of each cell particle in centered AREPO coordinates
            (converted to processing units).
        velocities: The velocities of each cell particle (converted to processing units).
        densities: The densities of each cell particle (converted to processing units).
        temperatures: The temperatures of each cell particle (converted to processing units).
        abundances_H2: The $\text{H}_2$ abundances of each cell particle,
            expressed as a fraction of total number density of H atoms.
        abundances_CO: The CO abundances of each cell particle,
            expressed as a fraction of total number density of H atoms.
        dust_temperatures: The dust temperatures of each cell particle (converted to processing units).
        
    Args:
        path: Sets `self.path`.
        hyper: Sets `self.hyper`.
        gpu_interpolate: If `True`, and a GPU is detected by the Python process, and
            `iris.arepo_processing.CUPY_ENABLED`, sets `self.gpu_interpolate` to `True`.
    """

    hyper: hp.Hyper
    path: str
    file: h5py.File
    gpu_interpolate: bool
    galactic_center: np.ndarray
    num_particles: int
    positions: np.ndarray
    velocities: np.ndarray
    densities: np.ndarray
    temperatures: np.ndarray
    abundances_H2: np.ndarray
    abundances_CO: np.ndarray
    dust_temperatures: np.ndarray

    def __init__(self, path: str, hyper: hp.Hyper, gpu_interpolate: bool = True) -> None:
        self.hyper = hyper
        self.path = os.path.expanduser(path)
        self.gpu_interpolate = gpu_interpolate and torch.cuda.is_available() and CUPY_ENABLED
        if gpu_interpolate and not torch.cuda.is_available():
            warnings.warn('Writer initialized with gpu_interpolate=True, but no GPU found. '
                          'Reverting to CPU interpolation.')
        if gpu_interpolate and not CUPY_ENABLED:
            warnings.warn('Writer initialized with gpu_interpolate=True, but no CuPy distribution found. '
                          'Reverting to CPU interpolation.')
        self._parse()
        return

    def _parse(self) -> None:
        """
        Parses the AREPO snapshot HDF5 file from the disk.
        """
        self.file = h5py.File(self.path, 'r')
        self._get_arepo_center()
        self._get_particle_values()
        self._apply_units_and_perturbations()
        self.num_particles = len(self.densities)
        return

    def _get_arepo_center(self) -> None:
        """
        Finds the rotational center of the galactic disk in the AREPO snapshot
        by computing the center of the simulation box.
        """
        arepo_box_size = float(self.file['Header'].attrs['BoxSize'])
        length = arepo_box_size / 2
        self.galactic_center = np.array([length, length, length], dtype=np.float32)
        return

    def _get_particle_values(self) -> None:
        r"""
        Gets values of all cell particle attributes.

        Sets `self.positions`, `self.velocities`, `self.densities`, `self.temperatures`,
        `self.abundances_H2`, `self.abundances_CO`, and `dust_temperatures`.

        If `self.hyper.dataset_hyper.use_AREPO_abundances`,
        $\text{H}_2$, $\text{H}^+$, and CO abundances are adopted from AREPO. Otherwise, all H
        is assumed to be $\text{H}_2$ and CO abundance is set to 0.

        Approximates gas temperatures from the internal energies recorded per cell
        in the AREPO snapshot via application of the Equipartition Theorem.
        For this calculation, treats the ISM as an ideal gas of atomic H, He,
        and molecular $\text{H}_2$. Note that by default, the AREPO cell variable
        InternalEnergy is a function of translational kinetic energy only. This value
        does not model a separate rotational or vibrational kinetic energy. Therefore,
        this internal energy is distributed over only 3 (translational) degrees of freedom,
        even for diatomic $\text{H}_2$.

        Computation of the temperature by application of
        the Equipartition Theorem over this partial internal energy assuming a uniform
        3 degrees of freedom thus yields precisely the kinetic temperature relevant in
        [computing level balances][iris.chemistry._make_population_grids] via standard lookup
        tables and in
        [computing thermal broadening][iris.observation.ObservabilityProcessor._line_profile].
        These are the two primary applications of gas temperature in IRIS. The only other application
        of temperature is if temperature is used in computing the
        [molecular abundance][iris.observation.Abundance] of a particular spectral tracer. In this
        last case, it should simply be noted that the gas temperature recorded is the kinetic temperature.

        Above `self.hyper.observer_hyper.T_inf` (if this value is not `None`),
        temperature is set to `np.inf`, and emission/absorption are later ignored for
        any such cell during observation. This prevents unreliable modeling of extremely hot,
        bright gas cells in which tracers are expected to have thermally decomposed.
        Additionally, because $\text{H}^+$ collisions are not modeled during
        [computation of the level balance][iris.chemistry._make_population_grids].
        """
        length_m_per_arepo = self.file['Header'].attrs['UnitLength_in_cm'] / 100
        velocity_m_per_s_per_arepo = self.file['Header'].attrs['UnitVelocity_in_cm_per_s'] / 100
        mass_kg_per_arepo = self.file['Header'].attrs['UnitMass_in_g'] / 1000
        time_s_per_arepo = length_m_per_arepo / velocity_m_per_s_per_arepo
        energy_joule_per_arepo = mass_kg_per_arepo * length_m_per_arepo * length_m_per_arepo / time_s_per_arepo / time_s_per_arepo

        H2_id = self.hyper.dataset_hyper.AREPO_H2_abundance_id
        H_plus_id = self.hyper.dataset_hyper.AREPO_H_plus_abundance_id
        CO_id = self.hyper.dataset_hyper.AREPO_CO_abundance_id
        abundance_He = self.hyper.observer_hyper.abundance_He
        m_He = self.hyper.observer_hyper.m_He
        m_H = self.hyper.observer_hyper.m_H
        k = self.hyper.observer_hyper.k / energy_joule_per_arepo     # energy_arepo / Kelvin
        T_inf = self.hyper.observer_hyper.T_inf
        if T_inf is None:
            T_inf = np.inf
        iris_number_unit = self.hyper.dataset_hyper.iris_number_unit
        ism_mass_per_iris_number_H_atom = (m_H + abundance_He * m_He) * (iris_number_unit / mass_kg_per_arepo)

        self.positions = np.array(self.file['PartType0']['Coordinates'], dtype=np.float32)
        self.velocities = np.array(self.file['PartType0']['Velocities'], dtype=np.float32)
        self.densities = np.array(self.file['PartType0']['Density'], dtype=np.float32)
        self.dust_temperatures = np.array(self.file['PartType0']['DustTemperature'], dtype=np.float32)

        if self.hyper.dataset_hyper.use_AREPO_abundances:
            abundance_H2 = np.array(self.file['PartType0']['ChemicalAbundances'][:, H2_id], dtype=np.float32)
            abundance_H_plus = np.array(self.file['PartType0']['ChemicalAbundances'][:, H_plus_id], dtype=np.float32)
            abundance_CO = np.array(self.file['PartType0']['ChemicalAbundances'][:, CO_id], dtype=np.float32)
        else:
            abundance_H2 = np.full(self.densities.shape, .5, dtype=np.float32)
            abundance_H_plus = np.full(self.densities.shape, 0, dtype=np.float32)
            abundance_CO = np.full(self.densities.shape, 0, dtype=np.float32)
        self.abundances_H2 = abundance_H2
        self.abundances_CO = abundance_CO

        # Compute gas temperatures using the Equipartition Theorem.
        # Note that AREPO InternalEnergy incorporates only translational kinetic energy.
        # Rotational and vibrational kinetic energy are not modeled.
        # Therefore, the temperature computed via the application of the Equipartition Theorem
        # to this partial internal energy, and assuming only three (translational) degrees of freedom
        # is the exact kinetic temperature.
        internal_energies = np.array(self.file['PartType0']['InternalEnergy'], dtype=np.float32)
        temperature_factor = internal_energies * (ism_mass_per_iris_number_H_atom / k) / iris_number_unit
        # abundance_term = abundance_H + abundance_H_plus + abundance_e_minus + abundance_H2 + abundance_He
        abundance_term = 1 + abundance_H_plus - abundance_H2 + abundance_He
        kinetic_temperature = (2 / 3) * temperature_factor / abundance_term
        too_hot_for_tracers = kinetic_temperature > T_inf
        self.temperatures = np.where(too_hot_for_tracers, np.inf, kinetic_temperature)
        return

    def _apply_units_and_perturbations(self) -> None:
        """
        Converts cell particle values into processing units and applies perturbations
        specified in `self.hyper`.

        Processing units are defined in [`WriterHyper`][iris.hyper.WriterHyper].
        Perturbations include:

        * Scale perturbations: Density-conserving length scaling according to a factor
        set randomly within `self.hyper.writer_hyper.CMZ_scale_range`
        or set to `self.hyper.writer_hyper.CMZ_scale_factor`
        if `CMZ_scale_range` is `None`. Not applied if both are `None`.
        * Density perturbations: Density-scaling according to a factor
        set randomly within `self.hyper.writer_hyper.CMZ_density_range`
        or set to `self.hyper.writer_hyper.CMZ_density_factor`
        if `CMZ_density_range` is `None`. Not applied if both are `None`.
        * Skew perturbations (experimental, not recommended): Random coordinate skews according to a factor
        set randomly within `self.hyper.writer_hyper.CMZ_skew_range`
        or set to `self.hyper.writer_hyper.CMZ_skew_factor`
        if `CMZ_skew_range` is `None`. Not applied if both are `None`.
        """
        length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
        velocity_cm_per_s_per_processing = self.hyper.writer_hyper.velocity_cm_per_s_per_processing
        mass_g_per_processing = self.hyper.writer_hyper.mass_g_per_processing
        temperature_K_per_processing = self.hyper.writer_hyper.temperature_K_per_processing

        length_cm_per_arepo = self.file['Header'].attrs['UnitLength_in_cm']
        velocity_cm_per_s_per_arepo = self.file['Header'].attrs['UnitVelocity_in_cm_per_s']
        mass_g_per_arepo = self.file['Header'].attrs['UnitMass_in_g']
        temperature_K_per_arepo = 1.

        # Compute conversions from AREPO units to processing units.
        length_conversion = length_cm_per_arepo / length_cm_per_processing
        velocity_conversion = velocity_cm_per_s_per_arepo / velocity_cm_per_s_per_processing
        mass_conversion = mass_g_per_arepo / mass_g_per_processing
        volume_conversion = length_conversion * length_conversion * length_conversion
        density_conversion = mass_conversion / volume_conversion
        temperature_conversion = temperature_K_per_arepo / temperature_K_per_processing

        # Center the Cartesian coordinate system on the galactic center.
        self.positions -= np.expand_dims(self.galactic_center, axis=0)

        # Compute density-conserving scale perturbations.
        if self.hyper.dataset_hyper.CMZ_scale_range is not None:
            CMZ_scale_factor = random.uniform(self.hyper.dataset_hyper.CMZ_scale_range[0],
                                              self.hyper.dataset_hyper.CMZ_scale_range[1])
        else:
            CMZ_scale_factor = self.hyper.dataset_hyper.CMZ_scale_factor
        if CMZ_scale_factor is not None:
            length_conversion *= CMZ_scale_factor
            velocity_conversion *= CMZ_scale_factor

        # Compute skew perturbations.
        if self.hyper.dataset_hyper.CMZ_skew_range is not None:
            CMZ_skew_factor = random.uniform(self.hyper.dataset_hyper.CMZ_skew_range[0],
                                             self.hyper.dataset_hyper.CMZ_skew_range[1])
        else:
            CMZ_skew_factor = self.hyper.dataset_hyper.CMZ_skew_factor
        if CMZ_skew_factor is not None:
            skew_matrix = np.expand_dims(self._random_skew_matrix(CMZ_skew_factor), axis=0)
            length_conversion = length_conversion * skew_matrix
            velocity_conversion = velocity_conversion * skew_matrix

        # Compute density perturbations.
        if self.hyper.dataset_hyper.CMZ_density_range is not None:
            CMZ_density_factor = random.uniform(self.hyper.dataset_hyper.CMZ_density_range[0],
                                                self.hyper.dataset_hyper.CMZ_density_range[1])
        else:
            CMZ_density_factor = self.hyper.dataset_hyper.CMZ_density_factor
        if CMZ_density_factor is not None:
            density_conversion *= CMZ_density_factor

        # Apply all transformations.
        if CMZ_skew_factor is not None:
            self.galactic_center = np.expand_dims(self.galactic_center, axis=-1)
            self.positions = np.expand_dims(self.positions, axis=-1)
            self.velocities = np.expand_dims(self.velocities, axis=-1)
            self.galactic_center = np.matmul(length_conversion, self.galactic_center)
            self.positions = np.matmul(length_conversion, self.positions)
            self.velocities = np.matmul(velocity_conversion, self.velocities)
            self.galactic_center = np.squeeze(self.galactic_center, axis=-1)
            self.positions = np.squeeze(self.positions, axis=-1)
            self.velocities = np.squeeze(self.velocities, axis=-1)
        else:
            self.galactic_center *= length_conversion
            self.positions *= length_conversion
            self.velocities *= velocity_conversion
        self.densities *= density_conversion
        self.temperatures *= temperature_conversion
        self.dust_temperatures *= temperature_conversion
        return

    def _random_skew_matrix(self, CMZ_skew_factor: float) -> np.ndarray:
        """
        Generates a random coordinate skew transformation matrix.

        Args:
            CMZ_skew_factor: The skew factor.

        Returns:
            The transformation matrix.
        """
        theta = random.uniform(0, 2 * np.pi)
        a = CMZ_skew_factor
        b = 1 / CMZ_skew_factor
        cos = np.cos(theta)
        sin = np.sin(theta)
        skew = np.array([[a * cos * cos + b * sin * sin, (a - b) * cos * sin, 0],
                         [(a - b) * cos * sin, b * cos * cos + a * sin * sin, 0],
                         [0, 0, 1]], dtype=np.float32)
        return skew

    def make_physical_tensor(self,
                             dataset: Dataset,
                             node_comm: mpi4py.MPI.Intracomm,
                             theta: float) -> None:
        r"""
        Computes a physical tensor and adds it to a [`Dataset`][iris.arepo_processing.Dataset].

        A "physical tensor" distills all information from an
        [AREPO snapshot][iris.arepo_processing.Snapshot] necessary to compute a
        [top-down density image][iris.arepo_processing.columnize_physical_tensor] and either a full
        [synthetic observation][iris.observation.SyntheticObserver] or a
        [simple observation][iris.observation.SimpleObserver]. Has dimensions
        `channel, r, lon, lat`, where:
        `channel` includes radial velocity, gas mass density, gas temperature,
        $\text{H}_2$ abundance per total number density of H atoms,
        CO abundance per total number density of H atoms, and dust temperature;
        `r` (radial distance from observer)
        has size `self.hyper.coordinate_hyper.r_steps`;
        `lon` (galactic longitude in the observer plane of sky)
        has size `self.hyper.coordinate_hyper.lon_steps`;
        and `lat` (galactic latitude in the observer plane of sky)
        has size `self.hyper.coordinate_hyper.lat_steps`.

        The IRIS spherical coordinate grid (`r, lon, lat`), over which the AREPO cell values
        (specified on a Voronoi mesh) are interpolated, is computed over a spherical coordinate
        system defined according to the observer origin. The angular coordinates `lon, lat`
        are defined according to the same convention as galactic longitude and latitude.
        The observer is located at `self.hyper.coordinate_hyper.observer_radius` (in parsecs, within the
        [perturbed][iris.arepo_processing.Snapshot._apply_units_and_perturbations] units of length)
        from `self.galactic_center`, at an
        angle of `theta +` `self.hyper.coordinate_hyper.theta_zero`, as measured
        counter-clockwise from the AREPO positive $x$-axis. The bounds of the grid are defined by
        `self.hyper.coordinate_hyper.r_min`, `self.hyper.coordinate_hyper.r_max`,
        `self.hyper.coordinate_hyper.lon_min`, `self.hyper.coordinate_hyper.lon_max`,
        `self.hyper.coordinate_hyper.lat_min`, `self.hyper.coordinate_hyper.lat_max`.

        If `self.hyper.coordinate_hyper.jitter_r`, a random deviation within
        `self.hyper.coordinate_hyper.jitter_r_min` and
        `self.hyper.coordinate_hyper.jitter_r_max` is added to `observer_radius`.
        If `self.hyper.coordinate_hyper.jitter_lon`, a random deviation within
        `self.hyper.coordinate_hyper.jitter_lon_min` and
        `self.hyper.coordinate_hyper.jitter_lon_max` is added to the
        longitude bounds `lon_min, lon_max`.

        Longitude is also oriented by `self.hyper.coordinate_hyper.spin_orientation`.
        If `spin_orientation == 1`, the snapshot is viewed right-side-up.
        If `spin_orientation == -1`, the snapshot is viewed upside-down. This option is applied
        because the Milky Way Galaxy rotates counter-clockwise with
        respect to the standard galactic coordinate system, but AREPO simulations may
        rotate clockwise, and so must be flipped upside down (not rotationally reversed)
        to produce a [`Dataset`][iris.arepo_processing.Dataset] of consistently defined
        physical tensors, on which the [`Reverter`][iris.reversion.Reverter] can learn
        inference behaviors that generalize to true observations.

        The [interpolation][iris.arepo_processing.Snapshot._interpolate] step
        is computed on GPU with CuPy if `self.gpu_interpolate`,
        or on CPU with SciPy otherwise, using a nearest-neighbor interpolation algorithm
        in either case. If GPU interpolating, the interpolation will be
        chunked into `self.hyper.coordinate_hyper.r_pieces` separate chunks
        that each fit onto GPU memory. After computing the physical tensor,
        this method calls the appropriate `add_tensor` method in the
        [`Dataset`][iris.arepo_processing.Dataset].

        Args:
             dataset: The dataset into which to add the new physical tensor.
             node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support.
             theta: The angle, measured counter-clockwise from the AREPO $x$-axis,
                of the ray pointing from the galactic center to the observer position
                (up to an addition of `self.hyper.coordinate_hyper.theta_zero`).
        """
        theta += self.hyper.coordinate_hyper.theta_zero / 180 * np.pi
        r_steps = self.hyper.coordinate_hyper.r_steps
        lon_steps = self.hyper.coordinate_hyper.lon_steps
        lat_steps = self.hyper.coordinate_hyper.lat_steps
        length_parsec_per_processing = self.hyper.writer_hyper._length_parsec_per_processing
        r_min = self.hyper.coordinate_hyper.r_min / length_parsec_per_processing
        r_max = self.hyper.coordinate_hyper.r_max / length_parsec_per_processing
        observer_r = self.hyper.coordinate_hyper.observer_radius / length_parsec_per_processing
        if self.hyper.coordinate_hyper.jitter_r:
            jitter = np.random.uniform(low=self.hyper.coordinate_hyper.jitter_r_min / length_parsec_per_processing,
                                       high=self.hyper.coordinate_hyper.jitter_r_max / length_parsec_per_processing,
                                       size=None)
            observer_r += jitter
        lon_min = self.hyper.coordinate_hyper.lon_min * np.pi / 180
        lon_max = self.hyper.coordinate_hyper.lon_max * np.pi / 180
        lon_min *= self.hyper.coordinate_hyper.spin_orientation
        lon_max *= self.hyper.coordinate_hyper.spin_orientation
        if self.hyper.coordinate_hyper.jitter_lon:
            jitter = np.random.uniform(low=self.hyper.coordinate_hyper.jitter_lon_min * np.pi / 180,
                                       high=self.hyper.coordinate_hyper.jitter_lon_max * np.pi / 180,
                                       size=None)
            lon_min += jitter
            lon_max += jitter
        lat_min = self.hyper.coordinate_hyper.lat_min * np.pi / 180
        lat_max = self.hyper.coordinate_hyper.lat_max * np.pi / 180
        lat_min *= self.hyper.coordinate_hyper.spin_orientation
        lat_max *= self.hyper.coordinate_hyper.spin_orientation

        # Prune particles outside the spherical coordinate grid prior to interpolation.
        (positions,
         velocities,
         densities,
         temperatures,
         abundances_H2,
         abundances_CO,
         dust_temperatures) = self._prune_particles(r_min,
                                                    r_max,
                                                    lon_min,
                                                    lon_max,
                                                    lat_min,
                                                    lat_max,
                                                    observer_r,
                                                    theta,
                                                    self.positions,
                                                    self.velocities,
                                                    self.densities,
                                                    self.temperatures,
                                                    self.abundances_H2,
                                                    self.abundances_CO,
                                                    self.dust_temperatures)

        if self.gpu_interpolate:
            # Interpolate on the GPU with CuPy.
            # Attempt to query the GPU manager and request the access key.
            node_size = node_comm.Get_size()
            gpu = None
            if node_size > 1:
                gpu_manager = node_size - 1
                node_comm.Send(np.array([node_comm.Get_rank()], dtype=np.int32), dest=gpu_manager, tag=7)
                gpu = node_comm.recv(source=gpu_manager, tag=8)
            device = cp.cuda.Device(gpu)

            # Preallocate contiguous space for the physical tensor on the CPU.
            physical_tensor = np.empty((6, r_steps, lon_steps, lat_steps), dtype=np.float32)
            r_lo_index = 0

            # Chunk the interpolation into pieces that fit into GPU memory.
            r_pieces = self.hyper.coordinate_hyper.r_pieces
            dr = (r_max - r_min) / (r_steps - 1)
            piece_length = (r_max - r_min) / r_pieces
            piece_length = int(piece_length / dr) * dr
            r_hi = r_min - dr
            for _ in range(r_pieces):
                if r_pieces > 1:
                    r_lo = r_hi + dr
                    r_hi = min(r_max, r_lo + piece_length)
                    r_st = round((r_hi - r_lo) / dr) + 1

                    # Prune particles outside this chunk.
                    (positions_piece,
                     velocities_piece,
                     densities_piece,
                     temperatures_piece,
                     abundances_H2_piece,
                     abundances_CO_piece,
                     dust_temperatures_piece) = self._prune_particles(r_lo - dr / 2,
                                                                      r_hi + dr / 2,
                                                                      lon_min,
                                                                      lon_max,
                                                                      lat_min,
                                                                      lat_max,
                                                                      observer_r,
                                                                      theta,
                                                                      positions,
                                                                      velocities,
                                                                      densities,
                                                                      temperatures,
                                                                      abundances_H2,
                                                                      abundances_CO,
                                                                      dust_temperatures)
                else:
                    r_lo = r_min
                    r_hi = r_max
                    r_st = r_steps

                    (positions_piece,
                     velocities_piece,
                     densities_piece,
                     temperatures_piece,
                     abundances_H2_piece,
                     abundances_CO_piece,
                     dust_temperatures_piece) = (positions,
                                                 velocities,
                                                 densities,
                                                 temperatures,
                                                 abundances_H2,
                                                 abundances_CO,
                                                 dust_temperatures)

                with device:
                    # Send chunk to the GPU as CuPy arrays.
                    positions_piece = cp.asarray(positions_piece)
                    velocities_piece = cp.asarray(velocities_piece)
                    densities_piece = cp.asarray(densities_piece)
                    temperatures_piece = cp.asarray(temperatures_piece)
                    abundances_H2_piece = cp.asarray(abundances_H2_piece)
                    abundances_CO_piece = cp.asarray(abundances_CO_piece)
                    dust_temperatures_piece = cp.asarray(dust_temperatures_piece)

                    # Interpolate the chunk.
                    physical_tensor[:, r_lo_index:r_lo_index + r_st] = self._interpolate(
                        positions=positions_piece,
                        velocities=velocities_piece,
                        densities=densities_piece,
                        temperatures=temperatures_piece,
                        abundances_H2=abundances_H2_piece,
                        abundances_CO=abundances_CO_piece,
                        dust_temperatures=dust_temperatures_piece,
                        observer_r=observer_r,
                        theta=theta,
                        r_min=r_lo,
                        r_max=r_hi,
                        r_steps=r_st,
                        lon_min=lon_min,
                        lon_max=lon_max,
                        lon_steps=lon_steps,
                        lat_min=lat_min,
                        lat_max=lat_max,
                        lat_steps=lat_steps,
                        cupy=True).get().astype(np.float32)

                    r_lo_index += r_st

                    # Free memory on the GPU.
                    del positions_piece
                    del velocities_piece
                    del densities_piece
                    del temperatures_piece
                    del abundances_H2_piece
                    del abundances_CO_piece
                    del dust_temperatures_piece

            # Clean-up GPU and return key to the GPU manager with usage statistics.
            if node_size > 1:
                gc.collect()
                with device:
                    memory_pool = cp.get_default_memory_pool()
                    memory_usage = memory_pool.total_bytes()
                    memory_pool.free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    node_comm.Isend(np.array([gpu, 0, memory_usage], dtype=np.int64), dest=gpu_manager, tag=9)

        else:
            # Interpolate on the CPU with SciPy.
            physical_tensor = self._interpolate(positions=positions,
                                                velocities=velocities,
                                                densities=densities,
                                                temperatures=temperatures,
                                                abundances_H2=abundances_H2,
                                                abundances_CO=abundances_CO,
                                                dust_temperatures=dust_temperatures,
                                                observer_r=observer_r,
                                                theta=theta,
                                                r_min=r_min,
                                                r_max=r_max,
                                                r_steps=r_steps,
                                                lon_min=lon_min,
                                                lon_max=lon_max,
                                                lon_steps=lon_steps,
                                                lat_min=lat_min,
                                                lat_max=lat_max,
                                                lat_steps=lat_steps,
                                                cupy=False)
            physical_tensor = np.ascontiguousarray(physical_tensor.astype(np.float32))

        # Send the physical tensor to the dataset for writing onto disk
        # or computation of an observed pair.
        dataset.add_tensor(physical_tensor, node_comm)
        return

    def _prune_particles(self,
                         r_min: float,
                         r_max: float,
                         lon_min: float,
                         lon_max: float,
                         lat_min: float,
                         lat_max: float,
                         observer_r: float,
                         theta: float,
                         positions: np.ndarray,
                         velocities: np.ndarray,
                         densities: np.ndarray,
                         temperatures: np.ndarray,
                         abundances_H2: np.ndarray,
                         abundances_CO: np.ndarray,
                         dust_temperatures: np.ndarray) -> tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
        r"""
        Prunes particles outside specified bounds.

        Does not convert all particle positions into spherical coordinates and prune exactly.
        Instead, inscribes the observational frustum in an AREPO coordinate prism
        and prunes all particles outside this prism.

        Args:
            r_min: The minimal `r` bound of the observational frustum.
            r_max: The maximal `r` bound of the observational frustum.
            lon_min: The minimal `lon` bound of the observational frustum.
            lon_max: The maximal `lon` bound of the observational frustum.
            lat_min: The minimal `lat` bound of the observational frustum.
            lat_max: The maximal `lat` bound of the observational frustum.
            observer_r: The distance of the observer from the galactic center.
            theta: The angle, measured counter-clockwise from the AREPO $x$-axis,
                of the ray pointing from the galactic center to the observer position.
            positions: The particle positions in translated AREPO coordinates (in processing units).
            velocities: The particle velocities.
            densities: The particle densities.
            temperatures: The particle temperatures.
            abundances_H2: The particle $\text{H}_2$ abundances.
            abundances_CO: The particle CO abundances.
            dust_temperatures: The particle dust temperatures.

        Returns:
            A tuple `positions, velocities, densities, temperatures,
                abundances_H2, abundances_CO, dust_temperatures` of pruned values.
        """
        vertices = np.array([[r_min, lon_min, lat_min],
                             [r_min, lon_min, lat_max],
                             [r_min, lon_max, lat_min],
                             [r_min, lon_max, lat_max],
                             [r_max, lon_min, lat_min],
                             [r_max, lon_min, lat_max],
                             [r_max, lon_max, lat_min],
                             [r_max, lon_max, lat_max]], dtype=np.float32)
        x, y, z = self._map_spherical_to_arepo(vertices[:, 0],
                                               vertices[:, 1],
                                               vertices[:, 2],
                                               observer_r,
                                               theta)
        x_min = np.min(x)
        y_min = np.min(y)
        z_min = np.min(z)
        x_max = np.max(x)
        y_max = np.max(y)
        z_max = np.max(z)

        indices = np.where((positions[:, 0] >= x_min)
                           & (positions[:, 0] <= x_max)
                           & (positions[:, 1] >= y_min)
                           & (positions[:, 1] <= y_max)
                           & (positions[:, 2] >= z_min)
                           & (positions[:, 2] <= z_max))[0]
        positions = positions[indices]
        velocities = velocities[indices]
        densities = densities[indices]
        temperatures = temperatures[indices]
        abundances_H2 = abundances_H2[indices]
        abundances_CO = abundances_CO[indices]
        dust_temperatures = dust_temperatures[indices]
        return (positions,
                velocities,
                densities,
                temperatures,
                abundances_H2,
                abundances_CO,
                dust_temperatures)

    def _map_spherical_to_arepo(self,
                                r: np.ndarray | cp.ndarray,
                                lon: np.ndarray | cp.ndarray,
                                lat: np.ndarray | cp.ndarray,
                                observer_r: float,
                                theta: float,
                                cupy: bool = False) -> tuple[np.ndarray | cp.ndarray,
                                                             np.ndarray | cp.ndarray,
                                                             np.ndarray | cp.ndarray]:
        r"""
        Maps a set of points in the IRIS spherical coordinate system to the translated AREPO
        coordinate system (galactic-center origin, in processing units of distance).

        Args:
            r: The `r` values of the points in IRIS spherical coordinates.
            lon: The `lon` values of the points in IRIS spherical coordinates.
            lat: The `lat` values of the points in IRIS spherical coordinates.
            observer_r: The distance of the observer from the galactic center.
            theta: The angle, measured counter-clockwise from the AREPO $x$-axis,
                of the ray pointing from the galactic center to the observer position.
            cupy: If `True`, uses CuPy functions (on GPU), otherwise uses NumPy (on CPU).

        Returns:
            A tuple `x, y, z` of coordinates in the translated AREPO coordinate system
                (galactic-center origin, in processing units of distance).
        """
        if cupy:
            lib = cp
        else:
            lib = np

        observer_x = observer_r * lib.cos(theta)
        observer_y = observer_r * lib.sin(theta)
        observer_z = 0.0
        xy = r * lib.cos(lat)
        z = r * lib.sin(lat) + observer_z
        phi = lib.pi + theta + lon
        x = observer_x + xy * lib.cos(phi)
        y = observer_y + xy * lib.sin(phi)
        return x, y, z

    def _project_velocities(self,
                            positions: np.ndarray | cp.ndarray,
                            velocities: np.ndarray | cp.ndarray,
                            observer_r: float,
                            theta: float,
                            cupy: bool = False) -> np.ndarray | cp.ndarray:
        r"""
        Computes radial velocities with respect to the observer.

        A radial velocity is the projection of the particle velocity onto the vector
        pointing from the particle position towards the observer, i.e. is positive
        when the particle is moving towards the observer.

        Args:
            positions: The particle positions in translated AREPO coordinates (in processing units).
            velocities: The particle velocities in AREPO coordinates (in processing units).
            observer_r: The distance of the observer from the galactic center.
            theta: The angle, measured counter-clockwise from the AREPO $x$-axis,
                of the ray pointing from the galactic center to the observer position.
            cupy: If `True`, uses CuPy functions (on GPU), otherwise uses NumPy (on CPU).

        Returns:
            An array of particle radial velocities.
        """
        if cupy:
            lib = cp
        else:
            lib = np

        observer_x = observer_r * lib.cos(theta)
        observer_y = observer_r * lib.sin(theta)
        observer_z = lib.array(0.0, dtype=lib.float32)
        observer_xyz = lib.stack((observer_x,
                                  observer_y,
                                  observer_z)).astype(lib.float32)
        r = observer_xyz - positions
        r_norm = lib.linalg.norm(r, axis=1, keepdims=True)
        r_hat = r / r_norm
        v_r = lib.einsum('...i,...i', r_hat, velocities)
        return v_r

    def _interpolate(self,
                     positions: np.ndarray | cp.ndarray,
                     velocities: np.ndarray | cp.ndarray,
                     densities: np.ndarray | cp.ndarray,
                     temperatures: np.ndarray | cp.ndarray,
                     abundances_H2: np.ndarray | cp.ndarray,
                     abundances_CO: np.ndarray | cp.ndarray,
                     dust_temperatures: np.ndarray | cp.ndarray,
                     observer_r: float,
                     theta: float,
                     r_min: float,
                     r_max: float,
                     r_steps: int,
                     lon_min: float,
                     lon_max: float,
                     lon_steps: int,
                     lat_min: float,
                     lat_max: float,
                     lat_steps: int,
                     cupy: bool = False) -> np.ndarray | cp.ndarray:
        r"""
        Interpolates an unstructured mesh of particle values over the IRIS spherical coordinate grid
        to produce a physical tensor (or physical tensor chunk).

        Args:
            positions: The particle positions in translated AREPO coordinates (in processing units).
            velocities: The particle velocities.
            densities: The particle densities.
            temperatures: The particle temperatures.
            abundances_H2: The particle $\text{H}_2$ abundances.
            abundances_CO: The particle CO abundances.
            dust_temperatures: The particle dust temperatures.
            observer_r: The distance of the observer from the galactic center.
            theta: The angle, measured counter-clockwise from the AREPO $x$-axis,
                of the ray pointing from the galactic center to the observer position.
            r_min: The minimal `r` bound of the observational frustum.
            r_max: The maximal `r` bound of the observational frustum.
            r_steps: The number of `r` points in the spherical grid.
            lon_min: The minimal `lon` bound of the observational frustum.
            lon_max: The maximal `lon` bound of the observational frustum.
            lon_steps: The number of `lon` points in the spherical grid.
            lat_min: The minimal `lat` bound of the observational frustum.
            lat_max: The maximal `lat` bound of the observational frustum.
            lat_steps: The number of `lat` points in the spherical grid.
            cupy: If `True`, uses CuPy functions (on GPU), otherwise uses NumPy (on CPU).

        Returns:
            The interpolated physical tensor.
        """
        if cupy:
            lib = cp
            Interpolator = GPUNearestNDInterpolator
        else:
            lib = np
            Interpolator = CPUNearestNDInterpolator

        velocities = self._project_velocities(positions, velocities, observer_r, theta, cupy=cupy)
        values = lib.stack((velocities,
                            densities,
                            temperatures,
                            abundances_H2,
                            abundances_CO,
                            dust_temperatures), axis=-1)

        r, lon, lat = lib.meshgrid(lib.linspace(r_min, r_max, r_steps, dtype=lib.float32),
                                   lib.linspace(lon_min, lon_max, lon_steps, dtype=lib.float32),
                                   lib.linspace(lat_min, lat_max, lat_steps, dtype=lib.float32),
                                   indexing='ij')
        x, y, z = self._map_spherical_to_arepo(r, lon, lat, observer_r, theta, cupy=cupy)
        grid = lib.stack((x, y, z), axis=-1)

        if len(positions) > 0:
            interpolator = Interpolator(positions, values)
            physical_tensor = interpolator(grid).transpose(3, 0, 1, 2)
        else:
            physical_tensor = lib.zeros((6, r_steps, lon_steps, lat_steps), dtype=lib.float32)
        return physical_tensor

    def make_wide_top_down(self,
                           resolution: tuple[int, int, int] = (1024, 1024, 1024),
                           box_size: float = 25000,
                           SI: bool = True) -> np.ndarray:
        """
        Computes a wide top-down H2 column-density projection of the snapshot.

        For use in making the publication [sims overview figure][iris.visualization.sims_overview].

        Args:
            resolution: Interpolation resolution along the x, y, and z axes.
            box_size: Width of the cubic interpolation volume in parsecs.
            SI: If `True`, converts the returned column density to SI units.

        Returns:
            The projected H2 column density.
        """
        if self.hyper.writer_hyper._length_parsec_per_processing is None:
            cm_per_parsec = 100.0 * self.hyper.dataset_hyper.meters_per_parsec
            length_cm_per_processing = self.hyper.writer_hyper.length_cm_per_processing
            self.hyper.writer_hyper._length_parsec_per_processing = length_cm_per_processing / cm_per_parsec
        box_size /= self.hyper.writer_hyper._length_parsec_per_processing
        x, y, z = np.meshgrid(np.linspace(-box_size / 2,
                                          box_size / 2,
                                          resolution[0],
                                          dtype=np.float32),
                              np.linspace(-box_size / 2,
                                          box_size / 2,
                                          resolution[1],
                                          dtype=np.float32),
                              np.linspace(-box_size / 2,
                                          box_size / 2,
                                          resolution[2],
                                          dtype=np.float32),
                              indexing='ij')
        grid = np.stack((x, y, z), axis=-1)

        mw_H2 = self.hyper.observer_hyper.mw_H2
        mw_H = self.hyper.observer_hyper.mw_H
        abundance_He = self.hyper.observer_hyper.abundance_He
        mw_He = self.hyper.observer_hyper.mw_He
        rho_H2 = self.densities * self.abundances_H2 * mw_H2 / (mw_H + abundance_He * mw_He)

        interpolator = CPUNearestNDInterpolator(self.positions, rho_H2)
        densities = interpolator(grid)
        densities = np.mean(densities, axis=2) * box_size
        if SI:
            length_SI_per_processing = self.hyper.writer_hyper.length_cm_per_processing / 100
            mass_SI_per_processing = self.hyper.writer_hyper.mass_g_per_processing / 1000
            column_density_SI_per_processing = mass_SI_per_processing / length_SI_per_processing / length_SI_per_processing
            densities *= column_density_SI_per_processing
        return densities


class Processor:
    """
    The base class extended by [`Reader`][iris.arepo_processing.Reader] and
    [`Writer`][iris.arepo_processing_write.Writer] for coordinating dataset processing tasks.

    Attributes:
        hyper: A hyperparameters object.
        path: A path to/list of paths to a [`Dataset`][iris.arepo_processing.Dataset]
            directory/list of [`Dataset`][iris.arepo_processing.Dataset] directories on the disk.
        dataset: A dataset or concatenated dataset.

    Args:
        path: Sets `self.path`.
        hyper: Sets `self.hyper`.
        dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to
            [read][iris.arepo_processing.Reader] or [write][iris.arepo_processing_write.Writer].
    """

    hyper: hp.Hyper
    path: str | typing.Sequence[str]
    dataset: Dataset | ConcatDataset | None

    def __init__(self,
                 path: str | typing.Sequence[str],
                 hyper: hp.Hyper | None = None,
                 dataset_type: type[Dataset] = PreObservedDataset) -> None:
        self.hyper = hyper
        self.path = path
        self.dataset = None
        self._load_dataset(path, dataset_type)
        return

    def _load_dataset(self, path: str | typing.Sequence[str], dataset_type: type[Dataset]) -> None:
        """
        Loads a dataset or list of datasets. An abstract method extending classes must override.

        Args:
            path: A path/list of paths to a dataset directory/list of dataset directories.
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] at `path`.
        """
        pass


class Reader(Processor):
    """
    A class for reading [`Dataset`][iris.arepo_processing.Dataset] objects from the disk.

    Automatically handles [dataset concatenation][iris.arepo_processing.ConcatDataset]
    and reading of litter datasets as [`InfiniteDataset`][iris.arepo_processing.InfiniteDataset]
    objects. See [`train_reverter`][iris.training.train_reverter] for notes on using litter.
    Sets all `self.dataset` and `self.litter` `read_only` attributes as `True` to prevent dataset corruption.

    Attributes:
        hyper: A hyperparameters object.
        path: A path to/list of paths to a [`Dataset`][iris.arepo_processing.Dataset]
            directory/list of [`Dataset`][iris.arepo_processing.Dataset] directories on the disk.
        dataset: A dataset or concatenated dataset.
        litter: A litter dataset.

    Args:
        path: Sets `self.path`.
        hyper: Sets `self.hyper`.
        dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to read.
    """

    dataset: Dataset | ConcatDataset
    litter: InfiniteDataset | None

    def __init__(self,
                 path: str | typing.Sequence[str],
                 hyper: hp.Hyper | None = None,
                 dataset_type: type[Dataset] = PreObservedDataset,
                 litter_path: str | typing.Sequence[str] | None = None,
                 litter_type: type[Dataset] = PreObservedDataset) -> None:
        super().__init__(path=path,
                         hyper=hyper,
                         dataset_type=dataset_type)
        if litter_path is None:
            self.litter = None
        else:
            if isinstance(litter_path, (list, tuple)):
                d = litter_type(litter_path[0], copy.deepcopy(self.hyper), read_only=True)
                litter_datasets = [d]
                if len(litter_path) > 1:
                    for p in litter_path[1:]:
                        d = litter_type(p, copy.deepcopy(self.hyper), read_only=True)
                        litter_datasets.append(d)
                litter_dataset = ConcatDataset(litter_datasets)
            else:
                litter_dataset = litter_type(litter_path, copy.deepcopy(self.hyper), read_only=True)
            self.litter = InfiniteDataset(finite_dataset=litter_dataset, units_dataset=self.dataset)
        return

    def _load_dataset(self, path: str | typing.Sequence[str], dataset_type: type[Dataset]) -> None:
        """
        Loads a dataset or list of datasets in read-only mode.

        Args:
            path: A path/list of paths to a dataset directory/list of dataset directories.
                If a list or tuple, will read each [`Dataset`][iris.arepo_processing.Dataset]
                separately and produce a [`ConcatDataset`][iris.arepo_processing.ConcatDataset].
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] at `path`.

        Raises:
            RuntimeError: If no [hyperparameters][iris.hyper.Hyper] were provided to the `Reader`.
        """
        if isinstance(path, (list, tuple)):
            d = dataset_type(path[0], self.hyper, read_only=True)
            if self.hyper is None:
                if d.hyper is None:
                    raise RuntimeError('No hyper provided to Processor object.')
                self.hyper = d.hyper
            datasets = [d]

            if len(path) > 1:
                for p in path[1:]:
                    d = dataset_type(p, copy.deepcopy(self.hyper), read_only=True)
                    datasets.append(d)

            self.dataset = ConcatDataset(datasets)
        else:
            self.dataset = dataset_type(path, self.hyper, read_only=True)
            if self.hyper is None:
                if self.dataset.hyper is None:
                    raise Exception('No hyper provided to Processor object.')
                self.hyper = self.dataset.hyper
        return
