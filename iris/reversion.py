# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Trainable neural networks for imagery reversion.

Reversion is defined as a constrained inverse to observation of a
[physical tensor][iris.arepo_processing.Snapshot.make_physical_tensor]. The initial experiments
detailed in the IRIS paper do not yet reconstruct the entire physical tensor from the entire
observed PPV cube. Rather, a [top-down density image][iris.arepo_processing.columnize_physical_tensor]
is reconstructed from a latitude-reduced observation. In particular, the IRIS paper considers
mean-reductions over the latitude dimension of the cube--yielding longitude-velocity PV images--
although max reductions are also allowed by the code.

Reversion is accomplished by a neural network trained on a [dataset][iris.arepo_processing.Dataset].
This module provides the neural network architecture used in the IRIS paper. The core design is
a convolutional neural network (CNN) with pixelwise attention, structured as an encoder-decoder. The
encoder maps a reduced observation into a latent featural space. The decoder maps the latent featural
object to a top-down density image. The entire neural network has about ~14M trainable parameters.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import typing

import torch

if typing.TYPE_CHECKING:
    from . import hyper as hp


class Reverter(torch.nn.Module):
    r"""
    A trainable neural network for imagery reversion.

    Reverts either a mean- or max-reduced PPV cube in a single spectral line to a
    [top-down density image][iris.arepo_processing.columnize_physical_tensor].
    The core architecture of `Reverter` is an encoder-decoder convolutional neural network (CNN)
    with pixelwise self-attention, which we describe in detail in the IRIS paper
    (subsec: Implementation of Reversion: Architecture).

    For convenience,
    the module expects a full PPV cube and automatically applies the
    `hyper.cube_hyper.reduction` specified in hyperparameters, unless the
    keyword arg `reduce=False` is passed to the module [`forward`][iris.reversion.Reverter.forward]
    method, in which case the expected input is a reduced PV observation. Use `reduce=False` when
    applying to a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset]. The `Reverter`
    also keeps track of its training physical units, which are saved as non-trainable parameters
    in its model state dict. A utility
    [`multi_unit_call`][iris.reversion.Reverter.multi_unit_call] automatically performs the
    unit conversions necessary when applied to an input or output space in different physical
    units than those on which the `Reverter` is trained.

    Attributes:
        temperature: The brightness temperature units of the `Reverter`, as a conversion factor from K.
        intensity: The intensity units of the `Reverter`, as a conversion factor from $\text{kg}/\text{s}^2$.
        v_density: The velocity-density units of the `Reverter`, as a conversion factor from
            $\text{kg} \cdot \text{s}/\text{m}^3$, for use if training on
            [simple observations][iris.observation.SimpleObserver]. See also
            [`SimplyObservedDataset`][iris.arepo_processing.SimplyObservedDataset].
        density: The density units of the `Reverter`, as a conversion factor from $\text{kg}/\text{m}^3$.
        reduction: The latitude reduction performed on an input PPV cube. Either a mean or max reduction.
            Set by `hyper.cube_hyper.reduction`.
        encoder: The encoder CNN.
        decoder: The decoder CNN.

    Args:
        hyper: A hyperparameters object.
        units_hyper: An optional separate hyperparameters object from which to adopt units but not
            other configurations. If `None`, units are taken from `hyper`.

    Raises:
        ValueError: If `hyper.cube_hyper.reduction` is not one of `'mean', 'max'`.
    """

    temperature: torch.nn.Parameter
    intensity: torch.nn.Parameter
    v_density: torch.nn.Parameter
    density: torch.nn.Parameter
    reduction: typing.Callable[[torch.Tensor], torch.Tensor]
    encoder: Encoder
    decoder: Decoder

    def __init__(self, hyper: hp.Hyper, units_hyper: hp.Hyper | None = None) -> None:
        super().__init__()
        if units_hyper is None:
            units_hyper = hyper
        temperature = units_hyper.dataset_hyper._temperature_iris_per_SI
        if temperature is None:
            temperature = 1.0
        length = units_hyper.dataset_hyper._length_iris_per_SI
        if length is None:
            length = 1.0
        mass = units_hyper.dataset_hyper._mass_iris_per_SI
        if mass is None:
            mass = 1.0
        time = units_hyper.dataset_hyper._time_iris_per_SI
        if time is None:
            time = 1.0
        density = mass / length / length / length
        intensity = mass / time / time
        v_density = density * time
        self.temperature = torch.nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32), requires_grad=False)
        self.intensity = torch.nn.Parameter(
            torch.tensor(intensity, dtype=torch.float32), requires_grad=False)
        self.v_density = torch.nn.Parameter(
            torch.tensor(v_density, dtype=torch.float32), requires_grad=False)
        self.density = torch.nn.Parameter(
            torch.tensor(density, dtype=torch.float32), requires_grad=False)

        reduction = hyper.cube_hyper.reduction
        if reduction == 'mean':
            self.reduction = self._reduce_mean
        elif reduction == 'max':
            self.reduction = self._reduce_max
        else:
            raise ValueError("Cube reduction must be one of: 'mean', 'max'.")
        self.encoder = Encoder()
        self.decoder = Decoder()
        return

    def forward(self, inputs: torch.Tensor, reduce: bool = True) -> torch.Tensor:
        """
        The `Reverter` forward pass.
        
        If `reduce`, applies `self.reduction`. Then applies `self.encoder` and `self.decoder`.
            
        Args:
            inputs: The input observations. A batch of either full PPV cubes or latitude-reduced PV images.
            reduce: If `True`, applies `self.reduction`.
            
        Returns:
            A batch of [top-down density images][iris.arepo_processing.columnize_physical_tensor].
        """
        if reduce:
            inputs = self.reduction(inputs)
        latent = self.encoder(inputs)
        outputs = self.decoder(latent)
        return outputs

    def _reduce_mean(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Performs a mean-reduction over the latitude dimension of a PPV cube.

        Args:
            cube: The PPV cube to reduce.

        Returns:
            The latitude-meaned PV image.
        """
        return torch.mean(cube, dim=3)

    def _reduce_max(self, cube: torch.Tensor) -> torch.Tensor:
        """
        Performs a max-reduction over the latitude dimension of a PPV cube.

        Args:
            cube: The PPV cube to reduce.

        Returns:
            The latitude-reduced peak-value image (PV).
        """
        return torch.max(cube, dim=3)[0]

    def multi_unit_call(self,
                        inputs: torch.Tensor,
                        in_units: hp.Hyper,
                        out_units: hp.Hyper,
                        in_space: str = 'T',
                        reduce: bool = True) -> torch.Tensor:
        """
        Wraps the model forward call in input and output unit conversions.

        Used when applying the `Reverter` to an input or output space in different physical
        units than those on which it was trained.

        Args:
            inputs: The input observations. A batch of either full PPV cubes or latitude-reduced PV images.
            in_units: A hyperparameters object specifying the units of the input space.
            out_units: A hyperparameters object specifying the units of the output space.
            in_space: The inputs space. If processing a
                [synthetic observation][iris.observation.SyntheticObserver], options are
                temperature (brightness or Raleigh-Jeans, specify `'T'`) or intensity (specify `'I'`).
                If processing a
                [simple observation][iris.observation.SimpleObserver],
                specify velocity-density (`'vrho'`).
            reduce: If `True`, applies `self.reduction`.

        Returns:
            A batch of [top-down density images][iris.arepo_processing.columnize_physical_tensor].

        Raises:
            ValueError: If `in_space` is not one of `'T', 'I', 'vrho'`.
        """
        if in_space == 'T':
            in_temperature = in_units.dataset_hyper._temperature_iris_per_SI
            inputs = inputs * (self.temperature / in_temperature)
        elif in_space == 'I':
            in_time = in_units.dataset_hyper._time_iris_per_SI
            in_mass = in_units.dataset_hyper._mass_iris_per_SI
            in_intensity = in_mass / in_time / in_time
            inputs = inputs * (self.intensity / in_intensity)
        elif in_space == 'vrho':
            in_length = in_units.dataset_hyper._length_iris_per_SI
            in_time = in_units.dataset_hyper._time_iris_per_SI
            in_mass = in_units.dataset_hyper._mass_iris_per_SI
            in_v_density = in_mass * in_time / in_length / in_length / in_length
            inputs = inputs * (self.v_density / in_v_density)
        else:
            raise ValueError("Arg in_space must be one of 'T', 'I', 'vrho'.")

        out_length = out_units.dataset_hyper._length_iris_per_SI
        out_mass = out_units.dataset_hyper._mass_iris_per_SI
        out_density = out_mass / out_length / out_length / out_length

        outputs = self(inputs, reduce=reduce)
        outputs = outputs * (out_density / self.density)
        return outputs


class Encoder(torch.nn.Module):
    r"""
    The encoder module used by [`Reverter`][iris.reversion.Reverter].

    Maps an input PV image of dimensions `batch, channel=1, lon=512, v=512` into a latent featural
    space of dimensions `batch, channel=2048, lon=1, v=1`, implying a total size reduction factor
    of $512 \cdot 512 / 2048 = 128$. See the IRIS paper for architectural details and discussion
    (subsec: Implementation of Reversion: Architecture).

    Attributes:
        _1_1_convolution: A convolution with `in_channels=1,
                                              out_channels=4,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _1_2_batch_norm: A batch normalization.
        _1_3_leaky_relu: A leaky ReLU.

        _2_1_convolution: A convolution with `in_channels=4,
                                              out_channels=16,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _2_2_batch_norm: A batch normalization.
        _2_3_leaky_relu: A leaky ReLU.

        _3_1_convolution: A convolution with `in_channels=16,
                                              out_channels=32,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _3_2_batch_norm: A batch normalization.
        _3_3_leaky_relu: A leaky ReLU.

        _4_1_convolution: A convolution with `in_channels=32,
                                              out_channels=64,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _4_2_batch_norm: A batch normalization.
        _4_3_leaky_relu: A leaky ReLU.
        _4_4_attention: A pixelwise attention layer.

        _5_1_convolution: A convolution with `in_channels=64,
                                              out_channels=128,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _5_2_batch_norm: A batch normalization.
        _5_3_leaky_relu: A leaky ReLU.
        _5_4_attention: A pixelwise attention layer.

        _6_1_convolution: A convolution with `in_channels=128,
                                              out_channels=256,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _6_2_batch_norm: A batch normalization.
        _6_3_leaky_relu: A leaky ReLU.
        _6_4_attention: A pixelwise attention layer.

        _7_1_convolution: A convolution with `in_channels=256,
                                              out_channels=512,
                                              kernel_size=(4, 4),
                                              stride=(2, 2),
                                              padding=(1, 1),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _7_2_batch_norm: A batch normalization.
        _7_3_leaky_relu: A leaky ReLU.

        _8_1_convolution: A convolution with `in_channels=512,
                                              out_channels=1024,
                                              kernel_size=(2, 2),
                                              stride=(2, 2),
                                              padding=(0, 0),
                                              groups=1,
                                              bias=True`,
                                              dtype=torch.float32.
        _8_2_batch_norm: A batch normalization.
        _8_3_leaky_relu: A leaky ReLU.

        _9_1_convolution: A convolution with `in_channels=1024,
                                              out_channels=2048,
                                              kernel_size=(2, 2),
                                              stride=(1, 1),
                                              padding=(0, 0),
                                              groups=4,
                                              bias=True`,
                                              dtype=torch.float32.
        _9_2_leaky_relu: A batch normalization.
    """

    _1_1_convolution: torch.nn.Conv2d
    _1_2_batch_norm: torch.nn.BatchNorm2d
    _1_3_leaky_relu: torch.nn.LeakyReLU

    _2_1_convolution: torch.nn.Conv2d
    _2_2_batch_norm: torch.nn.BatchNorm2d
    _2_3_leaky_relu: torch.nn.LeakyReLU

    _3_1_convolution: torch.nn.Conv2d
    _3_2_batch_norm: torch.nn.BatchNorm2d
    _3_3_leaky_relu: torch.nn.LeakyReLU

    _4_1_convolution: torch.nn.Conv2d
    _4_2_batch_norm: torch.nn.BatchNorm2d
    _4_3_leaky_relu: torch.nn.LeakyReLU
    _4_4_attention: PixelSelfAttention2d

    _5_1_convolution: torch.nn.Conv2d
    _5_2_batch_norm: torch.nn.BatchNorm2d
    _5_3_leaky_relu: torch.nn.LeakyReLU
    _5_4_attention: PixelSelfAttention2d

    _6_1_convolution: torch.nn.Conv2d
    _6_2_batch_norm: torch.nn.BatchNorm2d
    _6_3_leaky_relu: torch.nn.LeakyReLU
    _6_4_attention: PixelSelfAttention2d

    _7_1_convolution: torch.nn.Conv2d
    _7_2_batch_norm: torch.nn.BatchNorm2d
    _7_3_leaky_relu: torch.nn.LeakyReLU

    _8_1_convolution: torch.nn.Conv2d
    _8_2_batch_norm: torch.nn.BatchNorm2d
    _8_3_leaky_relu: torch.nn.LeakyReLU

    _9_1_convolution: torch.nn.Conv2d
    _9_2_leaky_relu: torch.nn.LeakyReLU

    def __init__(self) -> None:
        super().__init__()
        self._1_1_convolution = torch.nn.Conv2d(in_channels=1,
                                                out_channels=4,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._1_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._1_2_batch_norm = torch.nn.BatchNorm2d(4)
        self._1_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._2_1_convolution = torch.nn.Conv2d(in_channels=4,
                                                out_channels=16,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._2_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._2_2_batch_norm = torch.nn.BatchNorm2d(16)
        self._2_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._3_1_convolution = torch.nn.Conv2d(in_channels=16,
                                                out_channels=32,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._3_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._3_2_batch_norm = torch.nn.BatchNorm2d(32)
        self._3_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._4_1_convolution = torch.nn.Conv2d(in_channels=32,
                                                out_channels=64,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._4_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._4_2_batch_norm = torch.nn.BatchNorm2d(64)
        self._4_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self._4_4_attention = PixelSelfAttention2d(channels=64, num_heads=2)

        self._5_1_convolution = torch.nn.Conv2d(in_channels=64,
                                                out_channels=128,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._5_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._5_2_batch_norm = torch.nn.BatchNorm2d(128)
        self._5_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self._5_4_attention = PixelSelfAttention2d(channels=128, num_heads=4)

        self._6_1_convolution = torch.nn.Conv2d(in_channels=128,
                                                out_channels=256,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._6_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._6_2_batch_norm = torch.nn.BatchNorm2d(256)
        self._6_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        self._6_4_attention = PixelSelfAttention2d(channels=256, num_heads=8)

        self._7_1_convolution = torch.nn.Conv2d(in_channels=256,
                                                out_channels=512,
                                                kernel_size=(4, 4),
                                                stride=(2, 2),
                                                padding=(1, 1),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._7_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._7_2_batch_norm = torch.nn.BatchNorm2d(512)
        self._7_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._8_1_convolution = torch.nn.Conv2d(in_channels=512,
                                                out_channels=1024,
                                                kernel_size=(2, 2),
                                                stride=(2, 2),
                                                padding=(0, 0),
                                                groups=1,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._8_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._8_2_batch_norm = torch.nn.BatchNorm2d(1024)
        self._8_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._9_1_convolution = torch.nn.Conv2d(in_channels=1024,
                                                out_channels=2048,
                                                kernel_size=(2, 2),
                                                stride=(1, 1),
                                                padding=(0, 0),
                                                groups=4,
                                                bias=True,
                                                dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._9_1_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._9_2_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        The encoder forward pass.

        Args:
            inputs: A batch of latitude-reduced PV images.

        Returns:
            A batch of latent featural encodings.
        """
        x = self._1_1_convolution(inputs)
        x = self._1_2_batch_norm(x)
        x = self._1_3_leaky_relu(x)

        x = self._2_1_convolution(x)
        x = self._2_2_batch_norm(x)
        x = self._2_3_leaky_relu(x)

        x = self._3_1_convolution(x)
        x = self._3_2_batch_norm(x)
        x = self._3_3_leaky_relu(x)

        x = self._4_1_convolution(x)
        x = self._4_2_batch_norm(x)
        x = self._4_3_leaky_relu(x)
        x = self._4_4_attention(x)

        x = self._5_1_convolution(x)
        x = self._5_2_batch_norm(x)
        x = self._5_3_leaky_relu(x)
        x = self._5_4_attention(x)

        x = self._6_1_convolution(x)
        x = self._6_2_batch_norm(x)
        x = self._6_3_leaky_relu(x)
        x = self._6_4_attention(x)

        x = self._7_1_convolution(x)
        x = self._7_2_batch_norm(x)
        x = self._7_3_leaky_relu(x)

        x = self._8_1_convolution(x)
        x = self._8_2_batch_norm(x)
        x = self._8_3_leaky_relu(x)

        x = self._9_1_convolution(x)
        outputs = self._9_2_leaky_relu(x)
        return outputs


class Decoder(torch.nn.Module):
    r"""
    The decoder module used by [`Reverter`][iris.reversion.Reverter].

    Maps a latent featural object of dimensions `batch, channel=2048, r=1, lon=1` to a
    [top-down density image][iris.arepo_processing.columnize_physical_tensor] of dimensions
    `batch, channel=1, r=512, lon=512`, implying a total size expansion factor of
    $512 \cdot 512 / 2048 = 128$. See the IRIS paper for architectural details and discussion
    (subsec: Implementation of Reversion: Architecture).

    Attributes:
        _1_1_transpose_convolution: A transpose convolution with `in_channels=2048,
                                                                  out_channels=1024,
                                                                  kernel_size=(2, 2),
                                                                  stride=(1, 1),
                                                                  padding=(0, 0),
                                                                  groups=4,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _1_2_batch_norm: A batch normalization.
        _1_3_leaky_relu: A leaky ReLU.

        _2_1_transpose_convolution: A transpose convolution with `in_channels=1024,
                                                                  out_channels=512,
                                                                  kernel_size=(2, 2),
                                                                  stride=(2, 2),
                                                                  padding=(0, 0),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _2_2_batch_norm: A batch normalization.
        _2_3_leaky_relu: A leaky ReLU.

        _3_1_transpose_convolution: A transpose convolution with `in_channels=512,
                                                                  out_channels=256,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _3_2_batch_norm: A batch normalization.
        _3_3_leaky_relu: A leaky ReLU.

        _4_1_transpose_convolution: A transpose convolution with `in_channels=256,
                                                                  out_channels=128,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _4_2_batch_norm: A batch normalization.
        _4_3_leaky_relu: A leaky ReLU.

        _5_1_transpose_convolution: A transpose convolution with `in_channels=128,
                                                                  out_channels=64,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _5_2_batch_norm: A batch normalization.
        _5_3_leaky_relu: A leaky ReLU.

        _6_1_transpose_convolution: A transpose convolution with `in_channels=64,
                                                                  out_channels=32,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _6_2_batch_norm: A batch normalization.
        _6_3_leaky_relu: A leaky ReLU.

        _7_1_transpose_convolution: A transpose convolution with `in_channels=32,
                                                                  out_channels=16,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _7_2_batch_norm: A batch normalization.
        _7_3_leaky_relu: A leaky ReLU.

        _8_1_transpose_convolution: A transpose convolution with `in_channels=16,
                                                                  out_channels=4,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _8_2_batch_norm: A batch normalization.
        _8_3_leaky_relu: A leaky ReLU.

        _9_1_transpose_convolution: A transpose convolution with `in_channels=4,
                                                                  out_channels=1,
                                                                  kernel_size=(4, 4),
                                                                  stride=(2, 2),
                                                                  padding=(1, 1),
                                                                  groups=1,
                                                                  bias=True,
                                                                  dtype=torch.float32`.
        _9_2_relu: The output hard ReLU used to prevent negative density predictions.
    """

    _1_1_transpose_convolution: torch.nn.ConvTranspose2d
    _1_2_batch_norm: torch.nn.BatchNorm2d
    _1_3_leaky_relu: torch.nn.LeakyReLU

    _2_1_transpose_convolution: torch.nn.ConvTranspose2d
    _2_2_batch_norm: torch.nn.BatchNorm2d
    _2_3_leaky_relu: torch.nn.LeakyReLU

    _3_1_transpose_convolution: torch.nn.ConvTranspose2d
    _3_2_batch_norm: torch.nn.BatchNorm2d
    _3_3_leaky_relu: torch.nn.LeakyReLU

    _4_1_transpose_convolution: torch.nn.ConvTranspose2d
    _4_2_batch_norm: torch.nn.BatchNorm2d
    _4_3_leaky_relu: torch.nn.LeakyReLU

    _5_1_transpose_convolution: torch.nn.ConvTranspose2d
    _5_2_batch_norm: torch.nn.BatchNorm2d
    _5_3_leaky_relu: torch.nn.LeakyReLU

    _6_1_transpose_convolution: torch.nn.ConvTranspose2d
    _6_2_batch_norm: torch.nn.BatchNorm2d
    _6_3_leaky_relu: torch.nn.LeakyReLU

    _7_1_transpose_convolution: torch.nn.ConvTranspose2d
    _7_2_batch_norm: torch.nn.BatchNorm2d
    _7_3_leaky_relu: torch.nn.LeakyReLU

    _8_1_transpose_convolution: torch.nn.ConvTranspose2d
    _8_2_batch_norm: torch.nn.BatchNorm2d
    _8_3_leaky_relu: torch.nn.LeakyReLU

    _9_1_transpose_convolution: torch.nn.ConvTranspose2d
    _9_2_relu: torch.nn.ReLU

    def __init__(self) -> None:
        super().__init__()
        self._1_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=2048,
                                                                   out_channels=1024,
                                                                   kernel_size=(2, 2),
                                                                   stride=(1, 1),
                                                                   padding=(0, 0),
                                                                   groups=4,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._1_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._1_2_batch_norm = torch.nn.BatchNorm2d(1024)
        self._1_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._2_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=1024,
                                                                   out_channels=512,
                                                                   kernel_size=(2, 2),
                                                                   stride=(2, 2),
                                                                   padding=(0, 0),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._2_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._2_2_batch_norm = torch.nn.BatchNorm2d(512)
        self._2_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._3_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=512,
                                                                   out_channels=256,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._3_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._3_2_batch_norm = torch.nn.BatchNorm2d(256)
        self._3_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._4_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=256,
                                                                   out_channels=128,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._4_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._4_2_batch_norm = torch.nn.BatchNorm2d(128)
        self._4_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._5_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=128,
                                                                   out_channels=64,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._5_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._5_2_batch_norm = torch.nn.BatchNorm2d(64)
        self._5_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._6_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=64,
                                                                   out_channels=32,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._6_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._6_2_batch_norm = torch.nn.BatchNorm2d(32)
        self._6_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._7_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=32,
                                                                   out_channels=16,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._7_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._7_2_batch_norm = torch.nn.BatchNorm2d(16)
        self._7_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._8_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=16,
                                                                   out_channels=4,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._8_1_transpose_convolution.weight,
                                       a=0.01,
                                       nonlinearity='leaky_relu')
        self._8_2_batch_norm = torch.nn.BatchNorm2d(4)
        self._8_3_leaky_relu = torch.nn.LeakyReLU(negative_slope=0.01)

        self._9_1_transpose_convolution = torch.nn.ConvTranspose2d(in_channels=4,
                                                                   out_channels=1,
                                                                   kernel_size=(4, 4),
                                                                   stride=(2, 2),
                                                                   padding=(1, 1),
                                                                   groups=1,
                                                                   bias=True,
                                                                   dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(self._9_1_transpose_convolution.weight,
                                       nonlinearity='relu')
        self._9_2_relu = torch.nn.ReLU()
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        The decoder forward pass.

        Args:
            inputs: A batch of latent featural encodings.

        Returns:
            A batch of top-down density images.
        """
        x = self._1_1_transpose_convolution(inputs)
        x = self._1_2_batch_norm(x)
        x = self._1_3_leaky_relu(x)

        x = self._2_1_transpose_convolution(x)
        x = self._2_2_batch_norm(x)
        x = self._2_3_leaky_relu(x)

        x = self._3_1_transpose_convolution(x)
        x = self._3_2_batch_norm(x)
        x = self._3_3_leaky_relu(x)

        x = self._4_1_transpose_convolution(x)
        x = self._4_2_batch_norm(x)
        x = self._4_3_leaky_relu(x)

        x = self._5_1_transpose_convolution(x)
        x = self._5_2_batch_norm(x)
        x = self._5_3_leaky_relu(x)

        x = self._6_1_transpose_convolution(x)
        x = self._6_2_batch_norm(x)
        x = self._6_3_leaky_relu(x)

        x = self._7_1_transpose_convolution(x)
        x = self._7_2_batch_norm(x)
        x = self._7_3_leaky_relu(x)

        x = self._8_1_transpose_convolution(x)
        x = self._8_2_batch_norm(x)
        x = self._8_3_leaky_relu(x)

        x = self._9_1_transpose_convolution(x)
        outputs = self._9_2_relu(x)
        return outputs


class PixelSelfAttention2d(torch.nn.Module):
    """
    Implements a pixelwise self-attention layer.

    Applies a layer norm, followed by a multi-head attention, followed by a layer norm.

    Attributes:
        channels: The number of input/output channels.
        num_heads: The number of attention heads used by the attention block.
        pre_norm: The layer norm applied before attention.
        attention: The attention block.
        post_norm: The layer norm applied after attention.

    Args:
        channels: Sets `self.channels`.
        num_heads: Sets `self.num_heads`.
    """

    channels: int
    num_heads: int
    pre_norm: torch.nn.LayerNorm
    attention: torch.nn.MultiheadAttention
    post_norm: torch.nn.LayerNorm

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.pre_norm = torch.nn.LayerNorm(channels)
        self.attention = torch.nn.MultiheadAttention(embed_dim=channels,
                                                     num_heads=num_heads,
                                                     batch_first=True)
        self.post_norm = torch.nn.LayerNorm(channels)
        return

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the attention layer.

        Args:
            inputs: A batch of multi-channeled images.

        Returns:
            A batch of self-attended multi-channeled images.
        """
        batch, channel, height, width = inputs.shape
        x = inputs.flatten(2).transpose(1, 2)

        y = self.pre_norm(x)
        a, _ = self.attention(y, y, y)
        x = self.post_norm(x + a)

        outputs = x.transpose(1, 2).view(batch, channel, height, width)
        return outputs
