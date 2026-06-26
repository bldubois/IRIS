# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
A consolidation point for all project hyperparameters.

The class [`Hyper`][iris.hyper.Hyper] consolidates all project hyperparameters into a Python
object for convenient access during code execution. This includes dataset configurations,
coordinate definitions, physical units, constants, and training tunables. Also includes
a JSON serialization and deserialization routine for human-readable storage on disk.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import json
import typing

import torch
import numpy as np

from . import training
from . import cube_processing


class DataClass:
    """
    An abstract type used to specify behavior during JSON serialization and deserialization of a
    [`Hyper`][iris.hyper.Hyper] object.
    """
    pass


class Hyper(DataClass):
    """
    The primary consolidation point for all project hyperparameters.

    A `Hyper` object contains a set of high-level attributes that each group together a broad class
    of hyperparameters. The methods [`from_json`][iris.hyper.Hyper.from_json] and
    [`to_json`][iris.hyper.Hyper.to_json] read and write the `Hyper` object from or to
    a human-readable JSON file for disk storage. The default values in this base class are
    incomplete. It is intended that this class be extended by a complete configuration,
    or a complete JSON configuration be loaded from the disk. Note, however, that certain
    callable attributes are not saved to the JSON file. The method [`validate`][iris.hyper.Hyper.validate]
    performs a check to ensure that the extended or loaded hyperparameters object is complete
    and consistent. It enforces strict type accuracy, raising an `AssertionError` when encountering
    any type mismatch (such as between `float` and `int`) rather than attempting to cast
    attributes to the correct type. Note that certain characteristics, such as those relating
    to callable attributes, are not checked by `validate`, and may produce runtime errors
    even with a validated `Hyper` object.

    Attributes:
        writer_hyper: Contains all hyperparameters pertaining to
            [dataset writing][iris.arepo_processing_write.Writer].
        dataset_hyper: Contains all hyperparameters pertaining to
            [datasets][iris.arepo_processing.Dataset], e.g. units.
        coordinate_hyper: Contains all hyperparameters pertaining to coordinate definitions for an
            [AREPO snapshot][iris.arepo_processing.Snapshot].
        observer_hyper: Contains all hyperparameters pertaining to [observation][iris.observation].
        cube_hyper: Contains all hyperparameters pertaining to [cube processing][iris.cube_processing].
        training_hyper: Contains all hyperparameters pertaining to [Reverter training][iris.training].
    """

    writer_hyper: WriterHyper
    dataset_hyper: DatasetHyper
    coordinate_hyper: CoordinateHyper
    observer_hyper: ObserverHyper
    cube_hyper: CubeHyper
    training_hyper: TrainingHyper

    def __init__(self) -> None:
        self.writer_hyper = WriterHyper()
        self.dataset_hyper = DatasetHyper()
        self.coordinate_hyper = CoordinateHyper()
        self.observer_hyper = ObserverHyper()
        self.cube_hyper = CubeHyper()
        self.training_hyper = TrainingHyper()
        return

    def validate(self) -> None:
        """
        Ensures whether a `Hyper` object is complete and consistent.

        Enforces strict type accuracy, raising an `AssertionError` when encountering
        any type mismatch (such as between `float` and `int`) rather than attempting to cast
        attributes to the correct type. Note that certain characteristics, such as those relating
        to callable attributes, are not checked by `validate`, and may produce runtime errors
        even with a validated `Hyper` object. Also note that this method uses Python
        `assert` statements, which will be disabled if the script is run in optimized mode.
        Do not run the code in optimized mode if making a call to `validate`.
        """
        assert type(self.writer_hyper.total_snapshots) == int
        assert self.writer_hyper.total_snapshots > 0
        if self.writer_hyper.min_snapshot_index is not None:
            assert type(self.writer_hyper.min_snapshot_index) == int
        if self.writer_hyper.max_snapshot_index is not None:
            assert type(self.writer_hyper.max_snapshot_index) == int
            if self.writer_hyper.min_snapshot_index is not None:
                assert self.writer_hyper.min_snapshot_index <= self.writer_hyper.max_snapshot_index
        assert type(self.writer_hyper.points_per_snapshot) == int
        assert self.writer_hyper.points_per_snapshot > 0
        assert type(self.writer_hyper.unit_calculation_sample_size) == int
        assert self.writer_hyper.unit_calculation_sample_size > 0
        assert self.writer_hyper.unit_calculation_sample_size <= self.writer_hyper.total_snapshots * self.writer_hyper.points_per_snapshot

        assert type(self.writer_hyper.length_cm_per_processing) == float
        assert self.writer_hyper.length_cm_per_processing > 0
        assert type(self.writer_hyper.mass_g_per_processing) == float
        assert self.writer_hyper.mass_g_per_processing > 0
        assert type(self.writer_hyper.velocity_cm_per_s_per_processing) == float
        assert self.writer_hyper.velocity_cm_per_s_per_processing > 0
        assert type(self.writer_hyper.temperature_K_per_processing) == float
        assert self.writer_hyper.temperature_K_per_processing > 0

        assert type(self.dataset_hyper.meters_per_parsec) == float
        assert self.dataset_hyper.meters_per_parsec > 0
        assert type(self.dataset_hyper.use_AREPO_abundances) == bool
        if self.dataset_hyper.use_AREPO_abundances:
            assert type(self.dataset_hyper.AREPO_H2_abundance_id) == int
            assert self.dataset_hyper.AREPO_H2_abundance_id >= 0
            assert type(self.dataset_hyper.AREPO_H_plus_abundance_id) == int
            assert self.dataset_hyper.AREPO_H_plus_abundance_id >= 0
            assert type(self.dataset_hyper.AREPO_CO_abundance_id) == int
            assert self.dataset_hyper.AREPO_CO_abundance_id >= 0
            assert self.dataset_hyper.AREPO_H2_abundance_id != self.dataset_hyper.AREPO_H_plus_abundance_id
            assert self.dataset_hyper.AREPO_H2_abundance_id != self.dataset_hyper.AREPO_CO_abundance_id
            assert self.dataset_hyper.AREPO_H_plus_abundance_id != self.dataset_hyper.AREPO_CO_abundance_id
        assert type(self.dataset_hyper.iris_number_unit) == float
        assert self.dataset_hyper.iris_number_unit > 0

        assert (self.dataset_hyper.CMZ_scale_factor is None or
                type(self.dataset_hyper.CMZ_scale_factor) == float)
        if type(self.dataset_hyper.CMZ_scale_factor) == float:
            assert self.dataset_hyper.CMZ_scale_factor > 0
        assert (self.dataset_hyper.CMZ_scale_range is None or
                isinstance(self.dataset_hyper.CMZ_scale_range, (tuple, list)))
        if type(self.dataset_hyper.CMZ_scale_range) == list or type(self.dataset_hyper.CMZ_scale_range) == tuple:
            assert len(self.dataset_hyper.CMZ_scale_range) == 2
            assert (type(self.dataset_hyper.CMZ_scale_range[0]) == float and
                    type(self.dataset_hyper.CMZ_scale_range[1]) == float)
            assert self.dataset_hyper.CMZ_scale_range[0] > 0
            assert self.dataset_hyper.CMZ_scale_range[0] < self.dataset_hyper.CMZ_scale_range[1]

        assert (self.dataset_hyper.CMZ_skew_factor is None or
                type(self.dataset_hyper.CMZ_skew_factor) == float)
        if type(self.dataset_hyper.CMZ_skew_factor) == float:
            assert self.dataset_hyper.CMZ_skew_factor > 0
        assert (self.dataset_hyper.CMZ_skew_range is None or
                isinstance(self.dataset_hyper.CMZ_skew_range, (tuple, list)))
        if type(self.dataset_hyper.CMZ_skew_range) == list or type(self.dataset_hyper.CMZ_skew_range) == tuple:
            assert len(self.dataset_hyper.CMZ_skew_range) == 2
            assert (type(self.dataset_hyper.CMZ_skew_range[0]) == float and
                    type(self.dataset_hyper.CMZ_skew_range[1]) == float)
            assert self.dataset_hyper.CMZ_skew_range[0] > 0
            assert self.dataset_hyper.CMZ_skew_range[0] < self.dataset_hyper.CMZ_skew_range[1]

        assert (self.dataset_hyper.CMZ_density_factor is None or
                type(self.dataset_hyper.CMZ_density_factor) == float)
        if type(self.dataset_hyper.CMZ_density_factor) == float:
            assert self.dataset_hyper.CMZ_density_factor > 0
        assert (self.dataset_hyper.CMZ_density_range is None or
                isinstance(self.dataset_hyper.CMZ_density_range, (tuple, list)))
        if type(self.dataset_hyper.CMZ_density_range) == list or type(self.dataset_hyper.CMZ_density_range) == tuple:
            assert len(self.dataset_hyper.CMZ_density_range) == 2
            assert (type(self.dataset_hyper.CMZ_density_range[0]) == float and
                    type(self.dataset_hyper.CMZ_density_range[1]) == float)
            assert self.dataset_hyper.CMZ_density_range[0] > 0
            assert self.dataset_hyper.CMZ_density_range[0] < self.dataset_hyper.CMZ_density_range[1]

        assert type(self.coordinate_hyper.theta_zero) == float
        assert self.coordinate_hyper.theta_zero >= 0
        assert self.coordinate_hyper.theta_zero < 360
        assert (self.coordinate_hyper.spin_orientation == 1 or
                self.coordinate_hyper.spin_orientation == -1)
        assert type(self.coordinate_hyper.observer_radius) == float
        assert self.coordinate_hyper.observer_radius > 0
        assert type(self.coordinate_hyper.jitter_r) == bool
        if self.coordinate_hyper.jitter_r:
            assert type(self.coordinate_hyper.jitter_r_min) == float
            assert type(self.coordinate_hyper.jitter_r_max) == float
        if (type(self.coordinate_hyper.jitter_r_min) == float and
            type(self.coordinate_hyper.jitter_r_max) == float):
            assert self.coordinate_hyper.jitter_r_min < self.coordinate_hyper.jitter_r_max
        else:
            assert self.coordinate_hyper.jitter_r_min is None
            assert self.coordinate_hyper.jitter_r_max is None

        assert type(self.coordinate_hyper.r_steps) == int
        assert self.coordinate_hyper.r_steps > 0
        assert type(self.coordinate_hyper.r_pieces) == int
        assert self.coordinate_hyper.r_pieces > 0
        assert self.coordinate_hyper.r_pieces <= self.coordinate_hyper.r_steps
        assert type(self.coordinate_hyper.r_min) == float
        assert self.coordinate_hyper.r_min > 0
        assert type(self.coordinate_hyper.r_max) == float
        assert self.coordinate_hyper.r_min < self.coordinate_hyper.r_max
        assert type(self.coordinate_hyper.r_crop_min_index) == int
        assert self.coordinate_hyper.r_crop_min_index >= 0
        assert self.coordinate_hyper.r_crop_min_index < self.coordinate_hyper.r_steps
        assert type(self.coordinate_hyper.r_crop_max_index) == int
        assert self.coordinate_hyper.r_crop_max_index > self.coordinate_hyper.r_crop_min_index
        assert self.coordinate_hyper.r_crop_max_index <= self.coordinate_hyper.r_steps

        assert type(self.coordinate_hyper.lon_steps) == int
        assert self.coordinate_hyper.lon_steps > 0
        assert type(self.coordinate_hyper.lon_min) == float
        assert self.coordinate_hyper.lon_min >= -180
        assert type(self.coordinate_hyper.lon_max) == float
        assert self.coordinate_hyper.lon_min < self.coordinate_hyper.lon_max
        assert self.coordinate_hyper.lon_max <= 180
        assert type(self.coordinate_hyper.jitter_lon) == bool
        if self.coordinate_hyper.jitter_lon:
            assert type(self.coordinate_hyper.jitter_lon_min) == float
            assert type(self.coordinate_hyper.jitter_lon_max) == float
        if (type(self.coordinate_hyper.jitter_lon_min) == float and
                type(self.coordinate_hyper.jitter_lon_max) == float):
            assert self.coordinate_hyper.jitter_lon_min < self.coordinate_hyper.jitter_lon_max
        else:
            assert self.coordinate_hyper.jitter_lon_min is None
            assert self.coordinate_hyper.jitter_lon_max is None
        assert type(self.coordinate_hyper.lat_steps) == int
        assert self.coordinate_hyper.lat_steps > 0
        assert type(self.coordinate_hyper.lat_min) == float
        assert self.coordinate_hyper.lat_min >= -90
        assert type(self.coordinate_hyper.lat_max) == float
        assert self.coordinate_hyper.lat_min < self.coordinate_hyper.lat_max
        assert self.coordinate_hyper.lat_max <= 90

        assert type(self.observer_hyper.lon_pieces) == int
        assert self.observer_hyper.lon_pieces > 0
        assert type(self.observer_hyper.lat_pieces) == int
        assert self.observer_hyper.lat_pieces > 0
        assert self.observer_hyper.lat_pieces <= self.coordinate_hyper.lat_steps
        assert type(self.observer_hyper.v_subsamples) == int
        assert self.observer_hyper.v_subsamples >= 0
        if self.observer_hyper.v_subsamples == 0:
            assert self.observer_hyper.lon_pieces <= self.coordinate_hyper.lon_steps
        else:
            assert (self.observer_hyper.lon_pieces *
                    self.observer_hyper.v_subsamples * 2) <= self.coordinate_hyper.lon_steps
        assert type(self.observer_hyper.blur_inputs) == bool
        if self.observer_hyper.blur_inputs:
            assert type(self.observer_hyper.in_blur_kernel_r) == int
            assert type(self.observer_hyper.in_blur_kernel_lon) == int
            assert type(self.observer_hyper.in_blur_kernel_lat) == int
            assert type(self.observer_hyper.in_blur_sigma) == float
        if (type(self.observer_hyper.in_blur_kernel_r) == int and
            type(self.observer_hyper.in_blur_kernel_lon) == int and
            type(self.observer_hyper.in_blur_kernel_lat) == int and
            type(self.observer_hyper.in_blur_sigma) == float):
            assert self.observer_hyper.in_blur_kernel_r > 0
            assert self.observer_hyper.in_blur_kernel_r <= self.coordinate_hyper.r_steps
            assert self.observer_hyper.in_blur_kernel_lon > 0
            assert self.observer_hyper.in_blur_kernel_lon <= self.coordinate_hyper.lon_steps
            assert self.observer_hyper.in_blur_kernel_lat > 0
            assert self.observer_hyper.in_blur_kernel_lat <= self.coordinate_hyper.lat_steps
            assert self.observer_hyper.in_blur_sigma > 0
        else:
            assert self.observer_hyper.in_blur_kernel_r is None
            assert self.observer_hyper.in_blur_kernel_lon is None
            assert self.observer_hyper.in_blur_kernel_lat is None
            assert self.observer_hyper.in_blur_sigma is None
        assert self.observer_hyper.out_blur_fwhm is None or type(self.observer_hyper.out_blur_fwhm) == float
        if type(self.observer_hyper.out_blur_fwhm) == float:
            assert self.observer_hyper.out_blur_fwhm > 0
        assert ((self.observer_hyper.noise_mean is None and
                self.observer_hyper.noise_sigma is None) or
                (type(self.observer_hyper.noise_mean) == float and
                 type(self.observer_hyper.noise_sigma) == float))
        if type(self.observer_hyper.noise_sigma) == float:
            assert self.observer_hyper.noise_sigma > 0

        assert type(self.observer_hyper.k) == float
        assert self.observer_hyper.k > 0
        assert type(self.observer_hyper.h) == float
        assert self.observer_hyper.h > 0
        assert type(self.observer_hyper.c) == float
        assert self.observer_hyper.c > 0
        assert type(self.observer_hyper.L) == float
        assert self.observer_hyper.L > 0

        epsilon = 1e-8
        assert type(self.observer_hyper.mw_He) == float
        assert self.observer_hyper.mw_He > 0
        assert type(self.observer_hyper.m_He) == float
        assert abs(self.observer_hyper.m_He -
                   self.observer_hyper.mw_He / self.observer_hyper.L / 1000.) < epsilon
        assert type(self.observer_hyper.abundance_He) == float
        assert self.observer_hyper.abundance_He >= 0
        assert type(self.observer_hyper.mw_H2) == float
        assert self.observer_hyper.mw_H2 > 0
        assert type(self.observer_hyper.m_H2) == float
        assert abs(self.observer_hyper.m_H2 -
                   self.observer_hyper.mw_H2 / self.observer_hyper.L / 1000.) < epsilon
        assert type(self.observer_hyper.ortho_to_para_H2_ratio) == float
        assert self.observer_hyper.ortho_to_para_H2_ratio >= 0
        assert type(self.observer_hyper.mw_H) == float
        assert self.observer_hyper.mw_H > 0
        assert type(self.observer_hyper.m_H) == float
        assert abs(self.observer_hyper.m_H -
                   self.observer_hyper.mw_H / self.observer_hyper.L / 1000.) < epsilon

        assert type(self.observer_hyper.n_lines) == int
        assert self.observer_hyper.n_lines > 0
        assert isinstance(self.observer_hyper.chem_path, (tuple, list))
        assert len(self.observer_hyper.chem_path) == self.observer_hyper.n_lines
        for p in self.observer_hyper.chem_path:
            assert type(p) == str
        assert isinstance(self.observer_hyper.transition, (tuple, list))
        assert len(self.observer_hyper.transition) == self.observer_hyper.n_lines
        for t in self.observer_hyper.transition:
            assert isinstance(t, (tuple, list))
            assert len(t) == 2
            assert type(t[0]) == int
            assert type(t[1]) == int
            assert t[1] >= 0
            assert t[0] > t[1]
        assert isinstance(self.observer_hyper.kappa_dust, (tuple, list))
        assert len(self.observer_hyper.kappa_dust) == self.observer_hyper.n_lines
        for k in self.observer_hyper.kappa_dust:
            assert type(k) == float
            assert k >= 0
        assert isinstance(self.observer_hyper.N_H_TOT_steps, (tuple, list))
        assert len(self.observer_hyper.N_H_TOT_steps) == self.observer_hyper.n_lines
        for n in self.observer_hyper.N_H_TOT_steps:
            assert type(n) == int
            assert n > 0
        assert isinstance(self.observer_hyper.interpolation_max_N_H_TOT, (tuple, list))
        assert len(self.observer_hyper.interpolation_max_N_H_TOT) == self.observer_hyper.n_lines
        for n in self.observer_hyper.interpolation_max_N_H_TOT:
            assert type(n) == float
            assert n > 0
        assert isinstance(self.observer_hyper.bolic_normalization, (tuple, list))
        assert len(self.observer_hyper.bolic_normalization) == self.observer_hyper.n_lines
        for n in self.observer_hyper.bolic_normalization:
            assert type(n) == float
            assert n > 0
        assert isinstance(self.observer_hyper.abundance_H2_steps, (tuple, list))
        assert len(self.observer_hyper.abundance_H2_steps) == self.observer_hyper.n_lines
        for a in self.observer_hyper.abundance_H2_steps:
            assert type(a) == int
            assert a > 0
        assert type(self.observer_hyper.max_level_throughput_resolution) == int
        assert self.observer_hyper.max_level_throughput_resolution > 0
        assert isinstance(self.observer_hyper.T_steps, (tuple, list))
        assert len(self.observer_hyper.T_steps) == self.observer_hyper.n_lines
        for t in self.observer_hyper.T_steps:
            assert type(t) == int
            assert t > 0
        assert isinstance(self.observer_hyper.interpolation_max_T, (tuple, list))
        assert len(self.observer_hyper.interpolation_max_T) == self.observer_hyper.n_lines
        for t in self.observer_hyper.interpolation_max_T:
            assert type(t) == float
            assert t > 0
        if self.observer_hyper.T_inf is not None:
            assert type(self.observer_hyper.T_inf) == float
            assert self.observer_hyper.T_inf > 0
        assert (self.observer_hyper.T_continuum is None or
                type(self.observer_hyper.T_continuum) == float)
        if type(self.observer_hyper.T_continuum) == float:
            assert self.observer_hyper.T_continuum >= 0
        assert type(self.observer_hyper.T_cmb) == float
        assert self.observer_hyper.T_cmb >= 0

        assert ((self.cube_hyper.data_path is not None and
                self.cube_hyper.fits_map is None) or
                (self.cube_hyper.data_path is None and
                 self.cube_hyper.fits_map is not None))
        if self.cube_hyper.data_path is not None:
            assert isinstance(self.cube_hyper.data_path, (tuple, list))
            assert len(self.cube_hyper.data_path) == self.observer_hyper.n_lines
            for p in self.cube_hyper.data_path:
                assert type(p) == str
        if self.cube_hyper.conversion_raw_to_T_K is not None:
            assert isinstance(self.cube_hyper.conversion_raw_to_T_K, (tuple, list))
            assert len(self.cube_hyper.conversion_raw_to_T_K) == self.observer_hyper.n_lines
        if self.cube_hyper.beam_efficiency is not None:
            assert isinstance(self.cube_hyper.beam_efficiency, (tuple, list))
            assert len(self.cube_hyper.beam_efficiency) == self.observer_hyper.n_lines
            for b in self.cube_hyper.beam_efficiency:
                assert type(b) == float
                assert b > 0
                assert b <= 1
        if self.cube_hyper.clean_noise is not None:
            assert isinstance(self.cube_hyper.clean_noise, (tuple, list))
            assert len(self.cube_hyper.clean_noise) == self.observer_hyper.n_lines
        assert type(self.cube_hyper.v_min) == float
        assert type(self.cube_hyper.v_max) == float
        assert self.cube_hyper.v_min < self.cube_hyper.v_max
        assert type(self.cube_hyper.v_steps) == int
        assert self.cube_hyper.v_steps > 0
        assert type(self.cube_hyper.reduction) == str
        assert self.cube_hyper.reduction == 'mean' or self.cube_hyper.reduction == 'max'

        assert type(self.training_hyper.validation_data_fraction) == float
        assert self.training_hyper.validation_data_fraction >= 0
        assert self.training_hyper.validation_data_fraction < 1
        assert type(self.training_hyper.epochs) == int
        assert self.training_hyper.epochs > 0
        assert type(self.training_hyper.batch_size) == int
        assert self.training_hyper.batch_size > 0
        assert type(self.training_hyper.batches_per_update) == int
        assert self.training_hyper.batches_per_update > 0
        assert type(self.training_hyper.density_normalization) == float
        assert self.training_hyper.density_normalization > 0
        return

    def to_json(self, path: str) -> None:
        """
        Writes the `Hyper` object to the disk as a human-readable JSON file.

        Non-serializable attributes such as types or callables are written to the disk as `None`.

        Args:
            path: The path on disk at which to write the JSON file.
        """
        with open(os.path.expanduser(path), 'w') as f:
            data = self._serialize(self)
            json.dump(data, f, indent=4)
        return

    def _serialize(self, obj: any) -> any:
        """
        A recursive utility that turns the `Hyper` object into a Python `dict` that can be saved as a
        JSON file.

        Args:
            obj: The object to serialize.

        Returns:
            The object as a primitive, `list` of primitives, or `dict` of primitives, primitives lists,
            or `dict` objects. If obj is a nonserializable object such as a type or callable, returns
            `None`.
        """
        if isinstance(obj, DataClass):
            data = {}
            for key, value in obj.__dict__.items():
                serialized_val = self._serialize(value)
                data[key] = serialized_val
            return data
        if isinstance(obj, (list, tuple)):
            return [self._serialize(item) for item in obj]
        if callable(obj):
            return None
        if type(obj).__module__ == 'builtins':
            return obj
        return None

    def from_json(self, path: str) -> None:
        """
        Reads a `Hyper` object stored on the disk as a human-readable JSON file and imports
        all attributes.

        Non-serializable attributes such as types or callables are not imported.

        Args:
            path: The path on disk at which to read the JSON file.
        """
        with open(os.path.expanduser(path), 'r') as f:
            data = json.load(f)
        self._deserialize(self, data)
        return

    def _deserialize(self, target_obj: any, data_dict: any) -> None:
        """
        A recursive utility that imports attributes from a Python `dict` loaded from a JSON file
        into to a `Hyper` object.

        Does not overwrite callable attributes.

        Args:
            target_obj: The object into which to import attributes.
            data_dict: The `dict` object from which to import attributes.
        """
        for key, value in data_dict.items():
            if hasattr(target_obj, key):
                attr = getattr(target_obj, key)
                if isinstance(attr, DataClass):
                    self._deserialize(attr, value)
                elif callable(attr):
                    continue
                elif type(attr).__module__ == 'builtins':
                    setattr(target_obj, key, value)
        return


class WriterHyper(DataClass):
    """
    Contains all hyperparameters pertaining to [dataset writing][iris.arepo_processing_write.Writer].

    Attributes:
        total_snapshots: If calling `Writer` with `snapshot_directory`, the maximum number of AREPO
            snapshots that will be sampled. Snapshots will be sampled one at a time, without replacement,
            until either there are no snapshots left or `total_snapshots` is reached.
            See [`_issue_generation_tasks`][iris.arepo_processing_write.Writer._issue_generation_tasks].
        min_snapshot_index: If not `None`, `Writer` will only sample snapshots whose filenames end
            with an integer index greater than or equal to this value.
            See [`_issue_generation_tasks`][iris.arepo_processing_write.Writer._issue_generation_tasks].
        max_snapshot_index: If not `None`, `Writer` will only sample snapshots whose filenames end
            with an integer index less than or equal to this value.
            See [`_issue_generation_tasks`][iris.arepo_processing_write.Writer._issue_generation_tasks].
        points_per_snapshot: The number of points `Writer` will construct from each AREPO snapshot.
            The observations will be made from the points of a regular `n`-gon of
            `n=points_per_snapshot` vertices around the galactic center.
            See [`_issue_generation_tasks`][iris.arepo_processing_write.Writer._issue_generation_tasks].
        unit_calculation_sample_size: The number of preobserved pairs that will be randomly sampled
            when computing the units of a `SyntheticallyObservedDataset` or `SimplyObservedDataset`.
            See [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units] and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        length_cm_per_processing: A centimeter in the units of length used during AREPO snapshot processing.
            See [`Snapshot`][iris.arepo_processing.Snapshot].
        mass_g_per_processing: A gram in the units of mass used during AREPO snapshot processing.
            See [`Snapshot`][iris.arepo_processing.Snapshot].
        velocity_cm_per_s_per_processing: A centimeter per second in the units of velocity used during
            AREPO snapshot processing. See [`Snapshot`][iris.arepo_processing.Snapshot].
        temperature_K_per_processing: A Kelvin in the units of temperature used during AREPO
            snapshot processing. See [`Snapshot`][iris.arepo_processing.Snapshot].
        _length_parsec_per_processing: A parsec in the units of length used during AREPO snapshot processing.
            Computed automatically from `length_cm_per_processing`.
            See [`Snapshot`][iris.arepo_processing.Snapshot].
    """

    def __init__(self) -> None:
        self.total_snapshots: int | None = 20
        self.min_snapshot_index: int | None = None
        self.max_snapshot_index: int | None = None
        self.points_per_snapshot: int | None = 64
        self.unit_calculation_sample_size: int | None = 4

        self.length_cm_per_processing: float | None = 1e7
        self.mass_g_per_processing: float | None = 1.
        self.velocity_cm_per_s_per_processing: float | None = 1e7
        self.temperature_K_per_processing: float | None = 1.

        # Computed automatically:
        self._length_parsec_per_processing: float | None = None
        return


class DatasetHyper(DataClass):
    r"""
    Contains all hyperparameters pertaining to [datasets][iris.arepo_processing.Dataset].

    Attributes:
        meters_per_parsec: Conversion from parsecs to meters.
        use_AREPO_abundances: If `True`, uses $\text{H}_2$ and CO abundances from an AREPO snapshot.
            Otherwise, assumes all H is $\text{H}_2$ and CO abundance is zero.
            See [`_get_particle_values`][iris.arepo_processing.Snapshot._get_particle_values].
        AREPO_H2_abundance_id: In an AREPO HDF5 snapshot, chemical abundances are stored together in
            an array of inner dimension equal to the number of abundances tracked. The index of
            $\text{H}_2$ abundance in this array.
            See [`_get_particle_values`][iris.arepo_processing.Snapshot._get_particle_values].
        AREPO_H_plus_abundance_id: In an AREPO HDF5 snapshot, chemical abundances are stored together in
            an array of inner dimension equal to the number of abundances tracked. The index of
            $\text{H}^+$ abundance in this array.
            See [`_get_particle_values`][iris.arepo_processing.Snapshot._get_particle_values].
        AREPO_CO_abundance_id: In an AREPO HDF5 snapshot, chemical abundances are stored together in
            an array of inner dimension equal to the number of abundances tracked. The index of
            CO abundance in this array.
            See [`_get_particle_values`][iris.arepo_processing.Snapshot._get_particle_values].
        iris_number_unit: A unitless constant applied to number-density variables in order to stabilize
            numerical calculations. Interpolating physical tensors and making synthetic observations
            requires an extraordinary dynamic range in numerical expression and is difficult to
            stabilize in single precision. Applying a number-unit helps ensure single-precision stability.
        CMZ_scale_factor: If not `None`, a constant by which to adjust the snapshot length (and velocity).
            Is density conserving, i.e. is not mass conserving.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        CMZ_scale_range: If not `None`, an interval in which a constant by which to adjust the snapshot
            length (and velocity) will be randomly determined according to a uniform distribution.
            Is density conserving, i.e. is not mass conserving.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        CMZ_skew_factor: If not `None`, a constant by which to skew a snapshot about the galactic center.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        CMZ_skew_range: If not `None`, an interval in which a constant by which to skew a snapshot
            about the galactic center will be randomly determined according to a uniform distribution.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        CMZ_density_factor: If not `None`, a constant by which to adjust the snapshot density.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        CMZ_density_range: If not `None`, an interval in which a constant by which to adjust the snapshot
            density will be randomly determined according to a uniform distribution.
            See [`_apply_units_and_perturbations`][iris.arepo_processing.Snapshot._apply_units_and_perturbations].
        _velocity_iris_per_processing: The velocity conversion from processing units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _density_iris_per_processing: The density conversion from processing units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _v_density_iris_per_processing: The velocity-density (mass per unit area-velocity) conversion
            from processing units to IRIS units. Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _temperature_iris_per_processing: The temperature conversion from processing units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].

        _length_iris_per_SI: The length conversion from SI units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _length_iris_per_parsec: The length conversion from parsecs to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _time_iris_per_SI: The time conversion from SI units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _mass_iris_per_SI: The mass conversion from SI units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
        _temperature_iris_per_SI: The temperature conversion from SI units to IRIS units.
            Computed automatically, do not set.
            See [`StandardDataset.calculate_iris_units`][iris.arepo_processing.StandardDataset.calculate_iris_units],
            [`SyntheticallyObservedDataset.calculate_iris_units`][iris.arepo_processing.SyntheticallyObservedDataset.calculate_iris_units], and
            [`SimplyObservedDataset.calculate_iris_units`][iris.arepo_processing.SimplyObservedDataset.calculate_iris_units].
    """
    def __init__(self) -> None:
        self.meters_per_parsec: float | None = 3.08567758128e16
        self.use_AREPO_abundances: bool | None = True
        self.AREPO_H2_abundance_id: int | None = 0
        self.AREPO_H_plus_abundance_id: int | None = 1
        self.AREPO_CO_abundance_id: int | None = 2
        self.iris_number_unit: float | None = 1e15
        self.CMZ_scale_factor: float | None = None
        self.CMZ_scale_range: typing.Sequence[float] | None = [.25, 1.]
        self.CMZ_skew_factor: float | None = None
        self.CMZ_skew_range: typing.Sequence[float] | None = None
        self.CMZ_density_factor: float | None = None
        self.CMZ_density_range: typing.Sequence[float] | None = None

        # Do not specify:
        self._velocity_iris_per_processing: float | None = None
        self._density_iris_per_processing: float | None = None
        self._v_density_iris_per_processing: float | None = None
        self._temperature_iris_per_processing: float | None = None

        self._length_iris_per_SI: float | None = None
        self._length_iris_per_parsec: float | None = None
        self._time_iris_per_SI: float | None = None
        self._mass_iris_per_SI: float | None = None
        self._temperature_iris_per_SI: float | None = None
        return


class CoordinateHyper(DataClass):
    r"""
    Contains all hyperparameters pertaining to coordinate definitions for an
    [AREPO snapshot][iris.arepo_processing.Snapshot].

    Attributes:
        theta_zero: The angle at which to set `theta=0` when defining the $n$-gon for physical
            tensor generation. In degrees, counterclockwise from the positive $x$-axis in AREPO.
            See [`_issue_generation_tasks`][iris.arepo_processing_write.Writer._issue_generation_tasks].
        spin_orientation: An orientation variable that ensures all snapshots are oriented
            counterclockwise, as is the Milky Way in the standard galactic coordinate system.
            Use `spin_orientation=1` for counterclockwise AREPO simulations and
            `spin_orientation=-1` for clockwise AREPO simulations.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        observer_radius: The distance of the observer from the galactic center in parsecs.
            Note that this distance is applied after any CMZ scaling perturbations are applied
            (see [`DatasetHyper.CMZ_scale_factor`][iris.hyper.DatasetHyper]).
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_r: If `True`, will deviate the distance of the observer from the galactic center
            by a uniform random variable in the interval `jitter_r_min, jitter_r_max`.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_r_min: Use with `jitter_r`. In parsecs from the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_r_max: Use with `jitter_r`. In parsecs from the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        r_steps: The number of radial steps in a physical tensor.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        r_pieces: The number of chunks into which to divide a physical tensor during GPU interpolation,
            if enabled. Used to avoid a GPU OOM error.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        r_min: The minimal bound of the radial dimension of a physical tensor.
            In parsecs from the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        r_max: The maximal bound of the radial dimension of a physical tensor.
            In parsecs from the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        r_crop_min_index: The integer index in the radial dimension of a physical tensor forming the
            minimal bound of a top-down density image.
            See [`columnize_physical_tensor`][iris.arepo_processing.columnize_physical_tensor].
        r_crop_max_index: The index in the radial dimension of a physical tensor forming the
            maximal bound of a top-down density image.
            See [`columnize_physical_tensor`][iris.arepo_processing.columnize_physical_tensor].
        lon_steps: The number of longitude steps in a physical tensor.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        lon_min: The minimal bound of the longitude dimension of a physical tensor.
            In degrees of galactic longitude with respect to the observer position.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        lon_max: The maximal bound of the longitude dimension of a physical tensor.
            In degrees of galactic longitude with respect to the observer position.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_lon: If `True`, will deviate the longitudinal centerline of the observational plane
            by a uniform random variable in the interval `jitter_lon_min, jitter_lon_max`.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_lon_min: Use with `jitter_lon`.
            In degrees of galactic longitude with respect to the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        jitter_lon_max: Use with `jitter_lon`.
            In degrees of galactic longitude with respect to the observer.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        lat_steps: The number of latitude steps in a physical tensor.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        lat_min: The minimal bound of the latitude dimension of a physical tensor.
            In degrees of galactic latitude with respect to the observer position.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        lat_max: The maximal bound of the latitude dimension of a physical tensor.
            In degrees of galactic latitude with respect to the observer position.
            See [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
    """
    def __init__(self) -> None:
        self.theta_zero: float | None = 0.
        self.spin_orientation: int | None = -1
        self.observer_radius: float | None = 8277.
        self.jitter_r: bool | None = False
        self.jitter_r_min: float | None = None
        self.jitter_r_max: float | None = None
        self.r_steps: int | None = 512
        self.r_pieces: int | None = 1
        self.r_min: float | None = None
        self.r_max: float | None = None
        self.r_crop_min_index: int | None = 0
        self.r_crop_max_index: int | None = 512
        self.lon_steps: int | None = 512
        self.lon_min: float | None = None
        self.lon_max: float | None = None
        self.jitter_lon: bool | None = False
        self.jitter_lon_min: float | None = None
        self.jitter_lon_max: float | None = None
        self.lat_steps: int | None = 128
        self.lat_min: float | None = None
        self.lat_max: float | None = None
        return


class ObserverHyper(DataClass):
    r"""
    Contains all hyperparameters pertaining to [observation][iris.observation].

    Attributes:
        lon_pieces: The number of longitude sections into which to divide the skyplane
            in the iterative ray-batching scheme employed by an `IteratedObserver`.
            Must satisfy `2 * v_subsamples * lon_pieces <= lon_steps` or
            `lon_pieces <= lon_steps` if `v_subsamples == 0`.
            See [`IteratedObserver`][iris.observation.IteratedObserver].
        lat_pieces: The number of latitude sections into which to divide the skyplane
            in the iterative ray-batching scheme employed by an `IteratedObserver`.
            Must satisfy `lat_pieces <= lat_steps`.
            See [`IteratedObserver`][iris.observation.IteratedObserver].
        v_subsamples: The number of fine velocity steps per velocity channel during optically thick
            transfer. Each fine step represents the midpoint of a step in the post-transfer velocity
            integration via Simpson's Rule. The total number of points in the fine velocity grid is
            `2 * v_steps * v_subsamples + 1`. See
            [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor] and
            [`_optically_thick_transfer`][iris.observation.TransferProcessor._optically_thick_transfer].
        blur_inputs: If `True`, applies an input-blur over the velocity channel of a physical tensor
            prior to [observation][iris.observation.Observer] and
            [top-down density image reduction][iris.arepo_processing.columnize_physical_tensor].
            See [`VelocityBlur`][iris.observation.VelocityBlur].
        in_blur_kernel_r: The radial size in pixels of the velocity blurring kernel.
            See [`VelocityBlur`][iris.observation.VelocityBlur].
        in_blur_kernel_lon: The longitude size in pixels of the velocity blurring kernel.
            See [`VelocityBlur`][iris.observation.VelocityBlur].
        in_blur_kernel_lat: The latitude size in pixels of the velocity blurring kernel.
            See [`VelocityBlur`][iris.observation.VelocityBlur].
        in_blur_sigma: The size in fractional pixels of the dimensionally symmetrical
            standard deviation of the Gaussian weight distribution according to which
            the velocity blurring kernel is weighted.
            See [`VelocityBlur`][iris.observation.VelocityBlur].
        out_blur_fwhm: The full-width-half-max (FWHM) of the Gaussian blurring convolution
            (point-spread function) applied over the latitude and longitude dimensions of a
            [synthetically observed][iris.observation.SyntheticObserver] PPV cube to mimic
            the nonzero angular resolution of a real antenna. If `None`, no convolution is
            applied, yielding an ideal observation at the theoretical limit of angular resolution.
            See [`BeamBlur`][iris.observation.BeamBlur].
        noise_mean: The mean of the Gaussian noise optionally added to an
            [observation][iris.observation.Observer] post-process.
            See [`Noise`][iris.observation.Noise].
        noise_sigma: The standard deviation of the Gaussian noise optionally added to an
            [observation][iris.observation.Observer] post-process.
            See [`Noise`][iris.observation.Noise].
        k: The Boltzmann constant in SI units.
        h: The Planck constant in SI units.
        c: The vacuum speed of light in SI units.
        L: The Avagadro constant in $\text{mol}^{-1}$.
        mw_He: The molar weight of He in $\text{g}/\text{mol}$.
        m_He: The mass of an He atom in kg.
        abundance_He: The abundance of He, as a fraction of the total H atom number density.
        mw_H2: The molar weight of $\text{H}_2$ in $\text{g}/\text{mol}$.
        m_H2: The mass of an $\text{H}_2$ molecule in kg.
        ortho_to_para_H2_ratio: The abundance ratio of orthohydrogen to parahydrogen,
            assumed constant throughout the ISM.
            See [`_compute_single_molecule`][iris.chemistry._compute_single_molecule].
        mw_H: The molar weight of H in $\text{g}/\text{mol}$.
        m_H: The mass of an H atom in kg.
        n_lines: The number of spectral lines to be synthetically observed.
            See [`SyntheticObserver`][iris.observation.SyntheticObserver].
        chem_path: A list of length `n_lines` of paths to chemical data files for each synthetically
            observed tracer molecule. Each data file must be formatted in the LAMDA .dat convention.
            All chemical data is parsed automatically from these files.
            See [`SyntheticObserver`][iris.observation.SyntheticObserver].
        transition: A list of length `n_lines` containing the specific energy transition
            to be synthetically observed for each specific tracer molecule.
            E.g., [(2, 1), (3, 2)] would denote the 2-1 transition of the first tracer molecule
            and the 3-2 transition of the second tracer molecule. The ground state is enumerated
            as 0 rather than as 1.
            See [`SyntheticObserver`][iris.observation.SyntheticObserver].
        kappa_dust: A list of length `n_lines` of dust opacities in $\text{m}^2/\text{kg}$ at each
            synthetically observed spectral line. Dust is coarsely assumed to be of a single species,
            uniformly distributed throughout the ISM according to a standard mass ratio.
            That is, supposing, for example, a standard ratio of 1:100, then the ratio of gas mass
            to dust mass in the ISM is assumed to be 100. The dust opacity is assumed to be constant
            over the small frequency window of a PPV cube in a single spectral line, but not across
            separate spectral lines that may differ drastically in transition frequency. Thus, `
            n_lines` separate opacities are expected. Each opacity should be specified per unit of
            gas mass incorporating the standard (e.g. 1:100) fraction, as opposed to per unit dust mass.
            See [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor].
        N_H_TOT_steps: A list of length `n_lines` of total steps in the mass-density/total-H-atom
            number-density dimension of the observability grids and grid gradients associated with
            each synthetically observed spectral line. Is a list because different grid dimensions
            are allowed for each separate spectral line (although all lines must be observed on the
            same PPV cube dimensions).
            See [`make_observability_grids`][iris.chemistry.make_observability_grids].
        interpolation_max_N_H_TOT: A list of length `n_lines` of maximum values in $\text{m}^{-3}$
            in the total-H-atom number-density dimension of the observability grids and grid gradients
            associated with each synthetically observed spectral line. Is a list because different grid
            dimensions are allowed for each separate spectral line (although all lines must be observed
            on the same PPV cube dimensions). Note that emission is linear (not constant) above these
            values. See [`make_observability_grids`][iris.chemistry.make_observability_grids].
        bolic_normalization: A list of length `n_lines` of normalization constants in $\text{m}^{-3}$
            for mapping to the hyperbolic space of the total-H-atom number-density dimension in the
            observability grids and grid gradients associated with each synthetically observed spectral
            line. Is a list because different grid configurations are allowed for each separate spectral
            line (although all lines must be observed on the same PPV cube dimensions).
        abundance_H2_steps: A list of length `n_lines` of total steps in the $\text{H}_2$ abundance
            dimension of the observability grids and grid gradients associated with
            each synthetically observed spectral line. Is a list because different grid dimensions
            are allowed for each separate spectral line (although all lines must be observed on the
            same PPV cube dimensions).
            See [`make_observability_grids`][iris.chemistry.make_observability_grids].
        max_level_throughput_resolution: The maximum total throughput size permitted when solving
            the level-balance systems in
            [`_make_population_grids`][iris.chemistry._make_population_grids], measured as
            `N_H_TOT_steps * abundance_H2_steps * T_steps * levels * levels`.
            If a line would exceed this limit, the level-balance solve is automatically chunked
            so that only a manageable subset of systems is materialized at once.
        T_steps: A list of length `n_lines` of total steps in the temperature dimension of the
            observability grids and grid gradients associated with each synthetically observed
            spectral line. Is a list because different grid dimensions are allowed for each
            separate spectral line (although all lines must be observed on the same PPV cube dimensions).
            See [`make_observability_grids`][iris.chemistry.make_observability_grids].
        interpolation_max_T: A list of length `n_lines` of maximum values in K in the temperature dimension
            of the observability grids and grid gradients associated with each synthetically observed
            spectral line. Is a list because different grid dimensions are allowed for each separate
            spectral line (although all lines must be observed on the same PPV cube dimensions).
            Note that emission is linear (not constant) above these values.
            See [`make_observability_grids`][iris.chemistry.make_observability_grids].
        T_inf: Temperature in K above which no emission or absorption of the spectral line is computed.
            It is assumed the tracer is fully thermally decomposed at this point. It is useful to
            introduce such a temperature ceiling (especially if observing, via a derivative
            [abundance][iris.observation.Abundance], a complex tracer molecule that is not modeled
            in the AREPO chemical network) to prevent errant emissions from high-temperature,
            low-density grid cells. Additionally, because $\text{H}^+$ collisions are not modeled during
            [computation of the level balance][iris.chemistry._make_population_grids].
            If `None`, no temperature ceiling is applied.
            See [`_get_particle_values`][iris.arepo_processing.Snapshot._get_particle_values].
        T_continuum: For diagnostics only. Leave as `None`.
            See [`_compute_single_molecule`][iris.chemistry._compute_single_molecule].
        T_cmb: The brightness temperature in K of the cosmic microwave background.
            See [`SyntheticObserver`][iris.observation.SyntheticObserver].
    """
    def __init__(self) -> None:
        self.lon_pieces: int | None = 8
        self.lat_pieces: int | None = 4
        self.v_subsamples: int | None = None
        self.blur_inputs: bool | None = True
        self.in_blur_kernel_r: int | None = 3
        self.in_blur_kernel_lon: int | None = 5
        self.in_blur_kernel_lat: int | None = 3
        self.in_blur_sigma: float | None = .33
        self.out_blur_fwhm: float | None = None
        self.noise_mean: float | None = None
        self.noise_sigma: float | None = None

        self.k: float | None = 1.380649e-23
        self.h: float | None = 6.626070e-34
        self.c: float | None = 2.997925e8
        self.L: float | None = 6.022141e23

        self.mw_He: float | None = 4.002602
        self.m_He: float | None = self.mw_He / self.L / 1000.
        self.abundance_He: float | None = .1
        self.mw_H2: float | None = 2.01568
        self.m_H2: float | None = self.mw_H2 / self.L / 1000.
        self.ortho_to_para_H2_ratio: float | None = 3.
        self.mw_H: float | None = 1.00784
        self.m_H: float | None = self.mw_H / self.L / 1000.

        self.n_lines: int | None = None
        self.chem_path: typing.Sequence[str] | None = None
        self.transition: typing.Sequence[tuple[int, int]] | None = None
        self.kappa_dust: typing.Sequence[float] | None = None
        self.N_H_TOT_steps: typing.Sequence[int] | None = None
        self.interpolation_max_N_H_TOT: typing.Sequence[float] | None = None
        self.bolic_normalization: typing.Sequence[float] | None = None
        self.abundance_H2_steps: typing.Sequence[int] | None = None
        self.max_level_throughput_resolution: int | None = 2 ** 30
        self.T_steps: typing.Sequence[int] | None = None
        self.interpolation_max_T: typing.Sequence[float] | None = None
        self.T_inf: float | None = 5e4
        self.T_continuum: float | None = 2.73
        self.T_cmb: float | None = 2.73
        return


class CubeHyper(DataClass):
    """
    Contains all hyperparameters pertaining to [cube processing][iris.cube_processing].

    Attributes:
        data_path: A list of length `n_lines` of paths to separate spectral line observations
            (PPV cubes) each stored as a NumPy .np file on the disk. The .np file is assumed
            to contain an array of intensity or temperature values over dimensions
            longitude, latitude, velocity. Each dimension can be of any length, but is assumed,
            if this input type is used, to be specified over a linear projection of galactic
            longitude and latitude between the bounds `v_min, v_max`. Must specify cubes either
            via this option or as raw, unprocessed FITS files via the alternate option `fits_map`.
            See [`make_cube`][iris.cube_processing.make_cube].
        fits_map: A callable (function pointer) that returns a list of length `n_lines` of
            spectral line observations (PPV cubes) as NumPy arrays of intensity or temperature
            values over dimensions longitude, latitude, velocity. Each dimension can be of any
            length, but is assumed to be specified over a linear projection of galactic
            longitude and latitude between the bounds `v_min, v_max`. Must specify cubes either
            via this option or as pre-processed NumPy .np files via the alternate option `data_path`.
            Note that the utility [`load_fits_with_bounds`][iris.cube_processing.load_fits_with_bounds]
            is provided for automatically cropping cubes and reprojecting them over the linear
            longitude-latitude grid. See [`make_cube`][iris.cube_processing.make_cube].
        conversion_raw_to_T_K: A list of length `n_lines` of callables (function pointers)
            that each accept as arguments `raw, hyper` and return a processed `cube`. The arg
            `raw` is assumed to be a raw spectral line observation (PPV cube) loaded as a NumPy
            array via either `data_path` or `fits_map` and converted to a `torch.float32` tensor,
            and the arg `hyper` is a `Hyper` object. Returns `cube` as a `torch.float32` tensor of
            brightness temperatures or Raleigh-Jeans temperatures in K. Is a list because a separate
            conversion is allowed to be applied to each separate line observation, or conversion can
            be skipped for a specific line if that entry is `None`. Use, for example,
            [`corrected_antenna_temperature_to_raleigh_jeans_temperature`][iris.cube_processing.corrected_antenna_temperature_to_raleigh_jeans_temperature].
            If `None`, no conversions are applied.
            See [`make_cube`][iris.cube_processing.make_cube].
        beam_efficiency: The beam efficiency for conversion from antenna temperature to Raleigh-Jeans
            temperature, if applicable. Set to `None` if no conversion is required.
            See [`corrected_antenna_temperature_to_raleigh_jeans_temperature`][iris.cube_processing.corrected_antenna_temperature_to_raleigh_jeans_temperature].
        clean_noise: A list of length `n_lines` of callables (function pointers)
            that each accept as arguments `cube, hyper` and return a processed `clean_cube`. The arg
            `cube` is assumed to be a spectral line observation as a `torch.float32` tensor
            of dimensions `lon_steps, lat_steps, v_steps`, and the arg `hyper` is a `Hyper` object.
            Returns `clean_cube` as a `torch.tensor` of the same dimensions, with noise removed.
            Is a list because a separate cleaning function is allowed to be applied to each
            separate line observation, or cleaning can be skipped for a specific line if that entry
            is `None`. If `clean_noise` is `None`, no cleaning is applied.
            See [`make_cube`][iris.cube_processing.make_cube].
        v_min: The minimal bound in $\text{km}/\text{s}$ of the velocity dimension of all
            spectral line cubes. See [`make_cube`][iris.cube_processing.make_cube].
        v_max: The maximal bound in $\text{km}/\text{s}$ of the velocity dimension of all
            spectral line cubes. See [`make_cube`][iris.cube_processing.make_cube].
        v_steps: The total number of steps in the velocity dimension of all spectral line cubes.
            See [`make_default_cube`][iris.cube_processing.make_default_cube].
        reduction: The latitude reduction to be applied in converting PPV cubes to PV images.
            One of `'mean', 'max'`.
            See [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset],
            [`Reverter`][iris.reversion.Reverter], and
            [`train_reverter`][iris.training.train_reverter].
    """
    def __init__(self) -> None:
        self.data_path: typing.Sequence[str] | None = None
        self.fits_map: typing.Callable[[], typing.Sequence[np.ndarray]] | None = None
        self.conversion_raw_to_T_K: typing.Sequence[
            typing.Callable[[torch.Tensor, Hyper], torch.Tensor] | None] | None = None
        self.beam_efficiency: typing.Sequence[float] | None = None
        self.clean_noise: typing.Sequence[
            typing.Callable[[torch.Tensor, Hyper], torch.Tensor] | None] | None = None
        self.v_min: float | None = None
        self.v_max: float | None = None
        self.v_steps: int | None = 512
        self.reduction: str | None = None
        return
    

class TrainingHyper(DataClass):
    r"""
    Contains all hyperparameters pertaining to [Reverter training][iris.training].

    Attributes:
        validation_data_fraction: The fraction of data to be used during training as validation data.
            See [`Dataset.make_training_and_validation_dataloaders`][iris.arepo_processing.Dataset.make_training_and_validation_dataloaders],
            [`ConcatDataset.make_training_and_validation_dataloaders`][iris.arepo_processing.ConcatDataset.make_training_and_validation_dataloaders], and
            [`train_reverter`][iris.training.train_reverter].
        epochs: The total number of training epochs.
            See [`train_reverter`][iris.training.train_reverter].
        batch_size: The per-GPU batch size during training.
            See [`train_reverter`][iris.training.train_reverter].
        batches_per_update: The number of batches over which gradients are accumulated before
            computing an optimizer step.
            See [`train_reverter`][iris.training.train_reverter].
        physical_loss: The physical loss function to be used during training.
            See [`train_reverter`][iris.training.train_reverter] and
            [`PhysicalLoss`][iris.training.PhysicalLoss].
        density_normalization: The density normalization factor (in $\text{kg}/\text{m}^3) applied
            to the density residual before application of a physical loss function.
            Ensures loss scores are unitless and thus units-invariant.
            See [`PhysicalLoss`][iris.training.PhysicalLoss].
    """
    def __init__(self) -> None:
        self.validation_data_fraction: float | None = .2
        self.epochs: int | None = 32
        self.batch_size: int | None = 8
        self.batches_per_update: int | None = 16
        self.physical_loss: type[training.PhysicalLoss] = training.ScaledDensityLoss
        self.density_normalization: float | None = 1e-19
        return

    @staticmethod
    def optimizer(params: torch.Tensor | typing.Sequence) -> tuple[torch.optim.optimizer.Optimizer,
                                                                   torch.optim.lr_scheduler.LRScheduler]:
        """
        Initializes a training optimizer and scheduler on the model parameters.

        Override this function if using a different optimizer or scheduler.

        Args:
            params: The model parameters.

        Returns:
            A tuple `optimizer, scheduler`.
        """
        optimizer = torch.optim.Adam(params, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 17, 18], gamma=.5)
        return optimizer, scheduler


class SEDIGISM_13C16O(Hyper):
    r"""
    A hyperparameters configuration for observing the CMZ via $^{13}\text{CO}$ 2-1,
    as in the SEDIGISM survey (Schuller et al., 2021).

    The primary configuration used in the IRIS paper. Observes the CMZ within the sky-plane bounds
    $\text{GLON} = \pm 1.5^\circ$ and $\text{GLAT} = \pm 0.375^\circ$ and within the velocity
    bounds $v = \pm 200 \text{km}/\text{s}$. Crops and stitches a corresponding true observational
    cube from the published SEDIGISM $^{13}\text{CO}$ 2-1 cubes (G359, G001). Computes synthetic
    observations in the $^{13}\text{CO}$ 2-1 line via the $^{13}\text{CO}$ .dat file provided
    in the LAMDA database (Schoier et al., 2005), augmented with H collisions from
    Walker et al. (2015) accessed via the BASECOL database (Dubernet et al., 2024). The
    angular resolution for synthetic observation is taken to be 30 arcsec, as in SEDIGISM.
    A dust opacity in this spectral window is taken as the coarse value of
    $1 \times 10^{-3} \text{m}^2/\text{kg}$.

        @ARTICLE{2021MNRAS.500.3064S,
               author = {{Schuller}, F. and {Urquhart}, J.~S. and
                         {Csengeri}, T. and {Colombo}, D. and
                         {Duarte-Cabral}, A. and {Mattern}, M. and
                         {Ginsburg}, A. and {Pettitt}, A.~R. and
                         {Wyrowski}, F. and {Anderson}, L. and
                         {Azagra}, F. and {Barnes}, P. and
                         {Beltran}, M. and {Beuther}, H. and
                         {Billington}, S. and {Bronfman}, L. and
                         {Cesaroni}, R. and {Dobbs}, C. and
                         {Eden}, D. and {Lee}, M.-Y. and
                         {Medina}, S.-N.~X. and {Menten}, K.~M. and
                         {Moore}, T. and {Montenegro-Montes}, F.~M. and
                         {Ragan}, S. and {Rigby}, A. and
                         {Riener}, M. and {Russeil}, D. and
                         {Schisano}, E. and {Sanchez-Monge}, A. and
                         {Traficante}, A. and {Zavagno}, A. and
                         {Agurto}, C. and {Bontemps}, S. and
                         {Finger}, R. and {Giannetti}, A. and
                         {Gonzalez}, E. and {Hernandez}, A.~K. and
                         {Henning}, T. and {Kainulainen}, J. and
                         {Kauffmann}, J. and {Leurini}, S. and
                         {Lopez}, S. and {Mac-Auliffe}, F. and
                         {Mazumdar}, P. and {Molinari}, S. and
                         {Motte}, F. and {Muller}, E. and
                         {Nguyen-Luong}, Q. and {Parra}, R. and
                         {Perez-Beaupuits}, J.-P. and {Schilke}, P. and
                         {Schneider}, N. and {Suri}, S. and
                         {Testi}, L. and {Torstensson}, K. and
                         {Veena}, V.~S. and {Venegas}, P. and
                         {Wang}, K. and {Wienen}, M.},
                title = "{The SEDIGISM survey: First Data Release and overview of the Galactic structure}",
              journal = {\mnras},
             keywords = {surveys, ISM: structure, Galaxy: kinematics and dynamics, radio lines: ISM, Astrophysics - Astrophysics of Galaxies},
                 year = 2021,
                month = jan,
               volume = {500},
               number = {3},
                pages = {3064-3082},
                  doi = {10.1093/mnras/staa2369},
        archivePrefix = {arXiv},
               eprint = {2012.01527},
        primaryClass  = {astro-ph.GA},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.3064S},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

        @ARTICLE{2005A&A...432..369S,
               author = {{Sch{\"o}ier}, F.~L. and {van der Tak}, F.~F.~S.
                       and {van Dishoeck}, E.~F. and {Black}, J.~H.},
                title = "{An atomic and molecular database for analysis of submillimetre line observations}",
              journal = {\aap},
                 year = 2005,
               volume = {432},
                pages = {369-379},
                  doi = {10.1051/0004-6361:20041729},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2005A&A...432..369S}
        }

        @ARTICLE{2015ApJ...811...27W,
               author = {{Walker}, Kyle M. and {Song}, L. and {Yang}, B.~H.
                       and {Groenenboom}, G.~C. and {van der Avoird}, A.
                       and {Balakrishnan}, N. and {Forrey}, R.~C. and {Stancil}, P.~C.},
                title = "{Quantum Calculation of Inelastic CO Collisions with H. II.
                      Pure Rotational Quenching of High Rotational Levels}",
              journal = {\apj},
                 year = 2015,
               volume = {811},
                pages = {27},
                  doi = {10.1088/0004-637X/811/1/27},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2015ApJ...811...27W}
        }

        @ARTICLE{2024A&A...683A..40D,
               author = {{Dubernet}, M.~L. and {Boursier}, C. and
                         {Denis-Alpizar}, O. and {Ba}, Y.~A. and
                         {Moreau}, N. and {Zw{\"o}lf}, C.~M. and
                         {Amor}, M.~A. and {Babikov}, D. and
                         {Balakrishnan}, N. and {Balan{\c{c}}a}, C. and
                         {Ben Khalifa}, M. and {Bergeat}, A. and
                         {Bop}, C.~T. and {Cabrera-Gonz{\'a}lez}, L. and
                         {C{\'a}rdenas}, C. and {Chefai}, A. and
                         {Dagdigian}, P.~J. and {Dayou}, F. and
                         {Demes}, S. and {Desrousseaux}, B. and
                         {Dumouchel}, F. and {Faure}, A. and
                         {Forrey}, R.~C. and {Franz}, J. and
                         {Garc{\'\i}a-V{\'a}zquez}, R.~M. and {Gianturco}, F. and
                         {Godard Palluet}, A. and {Gonz{\'a}lez-S{\'a}nchez}, L. and
                         {Groenenboom}, G.~C. and {Halvick}, P. and
                         {Hammami}, K. and {Khadri}, F. and
                         {Kalugina}, Y. and {Kleiner}, I. and
                         {K{\l}os}, J. and {Lique}, F. and
                         {Loreau}, J. and {Mandal}, B. and
                         {Mant}, B. and {Marinakis}, S. and
                         {Ndaw}, D. and {Pirlot Jankowiak}, P. and
                         {Price}, T. and {Quintas-S{\'a}nchez}, E. and
                         {Ramachandran}, R. and {Sahnoun}, E. and
                         {Santander}, C. and {Stancil}, P.~C. and
                         {Stoecklin}, T. and {Tennyson}, J. and
                         {Tonolo}, F. and {Urz{\'u}a-Leiva}, R. and
                         {Yang}, B. and {Yurtsever}, E. and {{\.Z}{\'o}ltowski}, M.},
                title = "{BASECOL2023 scientific content}",
              journal = {\aap},
             keywords = {standards, astrochemistry, molecular data, molecular processes, astronomical databases: miscellaneous},
                 year = 2024,
                month = mar,
               volume = {683},
                  eid = {A40},
                pages = {A40},
                  doi = {10.1051/0004-6361/202348233},
               adsurl = {https://ui.adsabs.harvard.edu/abs/2024A&A...683A..40D},
              adsnote = {Provided by the SAO/NASA Astrophysics Data System}
        }

    Args:
        r_steps: Sets `self.coordinate_hyper.r_steps`.
    """
    def __init__(self, r_steps: int = 512) -> None:
        super().__init__()
        self.coordinate_hyper.r_steps = r_steps
        self.coordinate_hyper.r_pieces = int(np.ceil(r_steps / 2048))
        self.coordinate_hyper.r_min = 8052.
        self.coordinate_hyper.r_max = 8502.
        self.coordinate_hyper.lon_min = -1.5
        self.coordinate_hyper.lon_max = 1.5
        self.coordinate_hyper.lat_min = -.375
        self.coordinate_hyper.lat_max = .375

        self.observer_hyper.lon_pieces = 8
        self.observer_hyper.lat_pieces = 4
        self.observer_hyper.v_subsamples = 1
        self.observer_hyper.noise_mean = 0.
        self.observer_hyper.noise_sigma = 1.
        self.observer_hyper.out_blur_fwhm = 30.
        self.observer_hyper.n_lines = 1
        self.observer_hyper.chem_path = ['~/IRIS/chem/13C16O.dat']
        self.observer_hyper.transition = [(2, 1)]
        self.observer_hyper.kappa_dust = [1e-3]
        self.observer_hyper.N_H_TOT_steps = [128]
        self.observer_hyper.interpolation_max_N_H_TOT = [1e12]
        self.observer_hyper.bolic_normalization = [1.]
        self.observer_hyper.abundance_H2_steps = [64]
        self.observer_hyper.T_steps = [64]
        self.observer_hyper.interpolation_max_T = [3000.]

        self.cube_hyper.fits_map = self.from_fits
        self.cube_hyper.conversion_raw_to_T_K = None
        self.cube_hyper.clean_noise = None
        self.cube_hyper.v_min = -200.
        self.cube_hyper.v_max = 200.
        self.cube_hyper.reduction = 'mean'
        return

    def from_fits(self) -> list[np.ndarray]:
        r"""
        Crops and stitches a cube from the published SEDIGISM data (Schuller et al., 2021).

            @ARTICLE{2021MNRAS.500.3064S,
                   author = {{Schuller}, F. and {Urquhart}, J.~S. and
                             {Csengeri}, T. and {Colombo}, D. and
                             {Duarte-Cabral}, A. and {Mattern}, M. and
                             {Ginsburg}, A. and {Pettitt}, A.~R. and
                             {Wyrowski}, F. and {Anderson}, L. and
                             {Azagra}, F. and {Barnes}, P. and
                             {Beltran}, M. and {Beuther}, H. and
                             {Billington}, S. and {Bronfman}, L. and
                             {Cesaroni}, R. and {Dobbs}, C. and
                             {Eden}, D. and {Lee}, M.-Y. and
                             {Medina}, S.-N.~X. and {Menten}, K.~M. and
                             {Moore}, T. and {Montenegro-Montes}, F.~M. and
                             {Ragan}, S. and {Rigby}, A. and
                             {Riener}, M. and {Russeil}, D. and
                             {Schisano}, E. and {Sanchez-Monge}, A. and
                             {Traficante}, A. and {Zavagno}, A. and
                             {Agurto}, C. and {Bontemps}, S. and
                             {Finger}, R. and {Giannetti}, A. and
                             {Gonzalez}, E. and {Hernandez}, A.~K. and
                             {Henning}, T. and {Kainulainen}, J. and
                             {Kauffmann}, J. and {Leurini}, S. and
                             {Lopez}, S. and {Mac-Auliffe}, F. and
                             {Mazumdar}, P. and {Molinari}, S. and
                             {Motte}, F. and {Muller}, E. and
                             {Nguyen-Luong}, Q. and {Parra}, R. and
                             {Perez-Beaupuits}, J.-P. and {Schilke}, P. and
                             {Schneider}, N. and {Suri}, S. and
                             {Testi}, L. and {Torstensson}, K. and
                             {Veena}, V.~S. and {Venegas}, P. and
                             {Wang}, K. and {Wienen}, M.},
                    title = "{The SEDIGISM survey: First Data Release and overview of the Galactic structure}",
                  journal = {\mnras},
                 keywords = {surveys, ISM: structure, Galaxy: kinematics and dynamics, radio lines: ISM, Astrophysics - Astrophysics of Galaxies},
                     year = 2021,
                    month = jan,
                   volume = {500},
                   number = {3},
                    pages = {3064-3082},
                      doi = {10.1093/mnras/staa2369},
            archivePrefix = {arXiv},
                   eprint = {2012.01527},
            primaryClass  = {astro-ph.GA},
                   adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.500.3064S},
                  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
            }

        Returns:
            The cropped and stitched cube as a NumPy array in a list of length `n_lines=1`.
        """
        neg = cube_processing.load_fits_with_bounds(path='~/IRIS/data/G359_13CO21_Tmb_DR1.fits',
                                                    lon_steps=round(self.coordinate_hyper.lon_steps / 2),
                                                    lon_min=self.coordinate_hyper.lon_min,
                                                    lon_max=0.,
                                                    lat_steps=self.coordinate_hyper.lat_steps,
                                                    lat_min=self.coordinate_hyper.lat_min,
                                                    lat_max=self.coordinate_hyper.lat_max,
                                                    v_min=self.cube_hyper.v_min,
                                                    v_max=self.cube_hyper.v_max)
        pos = cube_processing.load_fits_with_bounds(path='~/IRIS/data/G001_13CO21_Tmb_DR1.fits',
                                                    lon_steps=round(self.coordinate_hyper.lon_steps / 2),
                                                    lon_min=0.,
                                                    lon_max=self.coordinate_hyper.lon_max,
                                                    lat_steps=self.coordinate_hyper.lat_steps,
                                                    lat_min=self.coordinate_hyper.lat_min,
                                                    lat_max=self.coordinate_hyper.lat_max,
                                                    v_min=self.cube_hyper.v_min,
                                                    v_max=self.cube_hyper.v_max)
        raw = np.concatenate((neg, pos), axis=0)
        return [raw]


class SEDIGISM_13C16O_Foreground(SEDIGISM_13C16O):
    """
    An extension of [`SEDIGISM_13C16O`][iris.hyper.SEDIGISM_13C16O] that observes
    only foreground features.

    To be used as litter during [reverter training][iris.training.train_reverter].
    """
    def __init__(self) -> None:
        super().__init__()
        self.writer_hyper.total_snapshots = 10
        self.writer_hyper.points_per_snapshot = 32

        self.dataset_hyper.CMZ_scale_factor = .5
        self.dataset_hyper.CMZ_scale_range = None
        self.dataset_hyper.CMZ_skew_factor = None
        self.dataset_hyper.CMZ_skew_range = None

        self.coordinate_hyper.r_steps = 2048
        self.coordinate_hyper.r_pieces = 1
        self.coordinate_hyper.r_min = 6000.
        self.coordinate_hyper.r_max = 7500.
        self.coordinate_hyper.r_crop_min_index = 0
        self.coordinate_hyper.r_crop_max_index = 2048

        self.observer_hyper.lon_pieces = 16
        self.observer_hyper.lat_pieces = 8
        return


class SEDIGISM_13C16O_Test(SEDIGISM_13C16O):
    """
    An extension of [`SEDIGISM_13C16O`][iris.hyper.SEDIGISM_13C16O] that generates a test dataset.

    For results visualization.

    Args:
        min_snapshot_index: Sets `self.writer_hyper.min_snapshot_index`.
        max_snapshot_index: Sets `self.writer_hyper.max_snapshot_index`.
    """
    def __init__(self,
                 min_snapshot_index: int | None = None,
                 max_snapshot_index: int | None = None) -> None:
        super().__init__()
        self.writer_hyper.total_snapshots = 8
        self.writer_hyper.points_per_snapshot = 8
        self.writer_hyper.min_snapshot_index = min_snapshot_index
        self.writer_hyper.max_snapshot_index = max_snapshot_index

        self.dataset_hyper.CMZ_scale_factor = None
        self.dataset_hyper.CMZ_scale_range = [0.4, 0.6]
        self.dataset_hyper.CMZ_skew_factor = None
        self.dataset_hyper.CMZ_skew_range = None
        self.dataset_hyper.CMZ_density_factor = None
        self.dataset_hyper.CMZ_density_range = None
        return


class SEDIGISM_13C16O_FullCone_Test(SEDIGISM_13C16O):
    """
    An extension of [`SEDIGISM_13C16O`][iris.hyper.SEDIGISM_13C16O] that generates
    a test dataset for full-cone observations.

    For results visualization.

    Args:
        min_snapshot_index: Sets `self.writer_hyper.min_snapshot_index`.
        max_snapshot_index: Sets `self.writer_hyper.max_snapshot_index`.
    """
    def __init__(self,
                 min_snapshot_index: int | None = None,
                 max_snapshot_index: int | None = None) -> None:
        super().__init__()
        self.writer_hyper.total_snapshots = 4
        self.writer_hyper.points_per_snapshot = 1
        self.writer_hyper.min_snapshot_index = min_snapshot_index
        self.writer_hyper.max_snapshot_index = max_snapshot_index

        self.dataset_hyper.CMZ_scale_factor = None
        self.dataset_hyper.CMZ_scale_range = [0.4, 0.6]
        self.dataset_hyper.CMZ_skew_factor = None
        self.dataset_hyper.CMZ_skew_range = None
        self.dataset_hyper.CMZ_density_factor = None
        self.dataset_hyper.CMZ_density_range = None

        self.coordinate_hyper.r_steps = 16384
        self.coordinate_hyper.r_pieces = 16
        self.coordinate_hyper.r_min = 1302.
        self.coordinate_hyper.r_max = 15702.
        self.coordinate_hyper.r_crop_min_index = 7680
        self.coordinate_hyper.r_crop_max_index = 8192

        self.observer_hyper.lon_pieces = 32
        self.observer_hyper.lat_pieces = 32
        return
