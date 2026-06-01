# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
r"""
Observation of [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor].

Observations are computed in two types:

* [Synthetic observation][iris.observation.SyntheticObserver] and
* [Simple observation][iris.observation.SimpleObserver].

Synthetic observation models a true spectral line observation with a radio telescope
via a full physical model of the radiative transfer process that incorporates:
spontaneous emission, stimulated emission, and absorption of one or multiple spectral lines;
a non-LTE level balance based on a true optically thin assumption; a robust continuum combining
the CMB, thermal dust emission, and dust absorption; post-transfer continuum subtraction;
a configurable antenna resolution; and coarse antenna noise simulation. The output is a tensor of
brightness temperature over dimensions of galactic longitude, latitude, and velocity (a PPV cube).

This module provides the higher-level logic for synthetic observation as well as numerical
solution to the radiative transfer equation. For computational efficiency, it relies on
precomputation of emission and absorption grids and grid gradients over dimensions of
gas density, $\text{H}_2$ abundance, and temperature. Solution of the level systems and
computation of emission and absorption grids is accomplished in [`chemistry`][iris.chemistry].
See [`_compute_single_molecule`][iris.chemistry._compute_single_molecule] in particular
for more details on optically thin level balancing.

In spite of the grid precomputations, by use of a linear interpolation scheme, IRIS synthetic
observation is end-to-end differentiable. This feature is turned off by default, for efficiency,
but can be turned on manually if PyTorch backpropagation through the observer is required.
A hybrid mode of abundance-only differentiability is also provided at greater efficiency
than end-to-end differentiability, which can enable trainable abundance functions if required.
See [SyntheticObserver][iris.observation.SyntheticObserver] for details.

Simple observation is a diagnostic tool that implements a density projection scheme that
offers a coarse analogue to true synthetic observation. The output is a tensor of density
over dimensions of galactic longitude, latitude, and velocity (a PPV cube). The density
value is the mass density of gas, integrated along the line of sight, per differential of
velocity--i.e. the column density of gas per differential velocity or the area-velocity
density of gas mass. It is in units $\text{mass} \cdot \text{time} / \text{length}^2$.

See the IRIS paper (subsec: Comparison Against Density-Tracing) for a deeper theoretical
explanation as to why simple observation is a provides a meaningful characterization of the
theoretical limit of information contained in a synthetic or true observation. Notably,
simple observation is likewise end-to-end differentiable if differentiability mode is
manually enabled. For definitive training and application to true observations in which accuracy
is preferred, however, it is suggested that only synthetic observation be used. See
[Simple observation][iris.observation.SimpleObserver] for details.

In the primary [reversion][iris.reversion.Reverter] resolution of
`r, lon, lat, v = 512, 512, 128, 512`, synthetic observation and simple observation are both
fast, with [SyntheticObserver][iris.observation.SyntheticObserver] requiring about
$\sim 6$ seconds per observation on a 40GB NVIDIA A100 GPU and
[SimpleObserver][iris.observation.SimpleObserver] around an order of magnitude faster
at $\sim 0.6$ seconds. Both operations are memory intensive, however, and must be batched
into manageable bundles of observable rays. This iterative ray batching is automatically
handled by the [`IteratedObserver`][iris.observation.IteratedObserver] classes
[`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver] and
[`IteratedSimpleObserver`][iris.observation.IteratedSimpleObserver], which should be the
primary classes instantiated by the user.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

from contextlib import nullcontext
import typing

import numpy as np
import torch

from . import chemistry
if typing.TYPE_CHECKING:
    import mpi4py
    from . import hyper as hp


class Observer(torch.nn.Module):
    """
    The abstract base class for all observer types.

    Extended by [`SyntheticObserver`][iris.observation.SyntheticObserver],
    [`SimpleObserver`][iris.observation.SimpleObserver], and
    [`IteratedObserver`][iris.observation.IteratedObserver].

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Does not
            modify any behavior in `Observer` itself. Only a flag specifying behavior
            to be implemented by extending classes.
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [reverter][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """

    differentiable_input: bool
    in_blur: VelocityBlur | None

    def __init__(self, hyper: hp.Hyper, *args: any, **kwargs: any) -> None:
        super().__init__()
        self.differentiable_input = False
        if hyper.observer_hyper.blur_inputs:
            self.in_blur = VelocityBlur(hyper)
        else:
            self.in_blur = None
        return


class SyntheticObserver(Observer):
    r"""
    The core synthetic observation class.

    Synthetic observation models a true spectral line observation with a radio telescope
    via a full physical model of the radiative transfer process that incorporates:
    spontaneous emission, stimulated emission, and absorption of one or multiple spectral lines;
    a non-LTE level balance based on a true optically thin assumption; a robust continuum
    combining the CMB, thermal dust emission, and dust absorption; post-transfer continuum
    subtraction; a configurable antenna resolution; and coarse antenna noise simulation.
    The output is a tensor of brightness temperature over dimensions of galactic longitude,
    latitude, and velocity (a PPV cube).

    The line radiative transfer can be solved in one of three modes, specified at runtime
    to [`forward`][iris.observation.SyntheticObserver.forward] via the arg `transfer`:
    optically thick, selectively thin, and optically thin. In optically thick mode,
    line absorption and stimulated emission are fully modeled, as well as spontaneous emission
    of the line, thermal dust emission, and dust absorption. This requires that
    each velocity pixel in the output cube be subsampled down to the resolution of the
    line profile, and then numerically integrated in frequency over each velocity channel
    post-transfer. This is the most computationally intensive mode, because the velocity
    subsampling ultimately requires finer ray-batching to prevent a GPU OOM error.

    In optically thin mode, only spontaneous emission of the line is computed, ignoring
    absorption and stimulated emission. Thermal dust emission is still computed to
    account for nonlinear continuum subtraction in brightness temperature space (see below),
    but dust absorption is also ignored. This enables several computational efficiencies.
    Memory and time are saved by not computing line absorption/stimulated emission.
    The ray solution simplifies to a fixed integral that is computed via a vectorized
    Simpson's Rule as opposed to the iterative BDF2 (see below). Lastly, the radiative transfer
    equation can be analytically integrated in frequency, up to the integral of the Gaussian
    line profile in terms of the standard error function (erf), which eliminates the need
    for post-transfer numerical integration in velocity.

    Selectively thin mode is a hybrid approximation that allows some optically thick behavior
    while still eliminating the need for velocity subsampling/numerical velocity integration.
    Specifically, the line is barred from self-interaction (absorption and stimulated emission)
    but can still absorb or be stimulated by the continuum, and can still itself be absorbed by
    dust. The physical motivation is that the line is assumed to be locally optically thin
    and Doppler-dispersed by large velocity gradients at non-local scale. This version of the
    radiative transfer equation still requires an iteratively stepped ray solution via BDF2,
    but can also still be analytically integrated in frequency, eliminating the need for
    numerical velocity integration.

    The process of computing a synthetic observation is multi-step:

    * If enabled via `hyper.observer_hyper.blur_inputs`,
    a [`VelocityBlur`][iris.observation.VelocityBlur] is applied to the input physical tensor.
    This module applies a configurable Gaussian blur over the velocity channel of the
    physical tensor only. See [`VelocityBlur`][iris.observation.VelocityBlur] for details.
    * An [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor] is called on the
    physical tensor, which determines emission and absorption coefficients for each spectral
    line and for dust.
    See [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor] for details.
    * A [`TransferProcessor`][iris.observation.TransferProcessor] computes a GPU-parallelized
    solution to the radiative transfer equation over all rays within one of the three
    transfer modes described above, in either formal or smooth integration mode, via configurable
    continuum subtraction settings and output units.
    See [`TransferProcessor`][iris.observation.TransferProcessor] for details.
    * A [`BeamBlur`][iris.observation.BeamBlur] Gaussian point-spread convolution is applied
    over the longitude and latitude dimensions of the observed cube to simulate the
    nonzero angular resolution of a radio antenna dish. The beam convolution is configured
    via its full-width-half-maximum (FWHM) in arcsec,
    `hyper.observer_hyper.out_blur_fwhm`. If `out_blur_fwhm` is `None`,
    no beam convolution is applied, yielding an ideal observation at the theoretical limit
    of angular resolution.
    See [`BeamBlur`][iris.observation.BeamBlur] for details.
    * Also note that simulated antenna noise can be added to the output cube via
    [`Noise`][iris.observation.Noise]. As noise, however, is meant to be added stochastically
    during [`Reverter`][iris.reversion.Reverter] [training][iris.training.train_reverter],
    it is treated as an add-on and is not computed within `SyntheticObserver` itself.
    See [`Noise`][iris.observation.Noise] for details on noise modeling and current limitations.

    All stages of this computation are fully differentiable--in particular, by virtue of the
    linearization scheme employed via the observability grid gradients. For efficiency,
    differentiability is turned off by default, but can be manually enabled in one of two modes:
    [end-to-end differentiability][iris.observation.SyntheticObserver.set_requires_grad_all] and
    [abundance-only differentiability][iris.observation.SyntheticObserver.set_requires_grad_abundance].
    In end-to-end differentiability mode, gradients can backpropagate not only to any
    `reauires_grad=True` PyTorch variable in a user-specified [Abundance][iris.observation.Abundance],
    but to any computation prior to
    [`forward`][iris.observation.SyntheticObserver.forward] and/or the input
    physical tensors themselves if they are `requires_grad=True` leaf tensors. In abundance-only
    differentiability mode, gradients can backpropagate to the abundance function and any
    `requires_grad=True` variables in it, but no further. This is a more computationally efficient
    differentiability option if the intent is abundance training. The postponed application of
    the abundance function (as described above) ensures a maximal compute savings. Once enabled,
    differentiability can subsequently be turned back off by calling
    [`set_requires_grad_none`][iris.observation.SyntheticObserver.set_requires_grad_none].

    Note that synthetic observation is memory-intensive, and should be ray-batched to save memory.
    Iterative ray-batching is implemented in full automation by
    [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver], which should be
    the primary class instantiated by the user. Do not implement this class directly unless
    manually implementing a ray-batching scheme, e.g. model-parallel ray-batching across a
    distributed GPU cluster.

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Do not manually
            set this flag. Call
            [`set_requires_grad_all`][iris.observation.SyntheticObserver.set_requires_grad_all]
            instead.
        differentiable_abundance: If `True`, gradients can backpropagate to the
            [Abundance][iris.observation.Abundance] and any `requires_grad=True` variables in it.
            If `True` and `differentiable_input` is `False`, gradients can backpropagate no further
            than the abundance. This is a more computationally efficient differentiability option
            if the intent is abundance training. Do not manually
            set this flag. Call
            [`set_requires_grad_abundance`][iris.observation.SyntheticObserver.set_requires_grad_abundance]
            instead.
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [reverter][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)
        observability_processor: The [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]
            used to compute emission and absorption coefficients.
        transfer_processor: The [`TransferProcessor`][iris.observation.TransferProcessor]
            used for computing ray solutions to the radiative transfer equation.
        out_blur: The [`BeamBlur`][iris.observation.BeamBlur] Gaussian point-spread convolution
            applied over the longitude and latitude dimensions of the observed cube to simulate the
            nonzero angular resolution of a radio antenna dish. The beam convolution is configured
            via its full-width-half-maximum (FWHM) in arcsec,
            `hyper.observer_hyper.out_blur_fwhm`. If `out_blur_fwhm` is `None`,
            `out_blur` is also set to none and no beam convolution is applied, yielding an ideal
            observation at the theoretical limit of angular resolution.

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        abundance: The [`Abundance`][iris.observation.Abundance] passed to
            `self.observability_processor`.
            If `None`, [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]
            currently defaults to the $^{13}\text{CO}$ abundance function employed in the
            IRIS paper, [`Constant_CO_13C16O`][iris.observation.Constant_CO_13C16O].
        units: The units of the input physical tensor and all internal computation of the
            synthetic observation. One of `'iris', 'processing'`. Not the same as the output
            units specified to [`forward`][iris.observation.SyntheticObserver.forward].
        node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support,
            if used during [dataset writing][iris.arepo_processing_write.Writer].
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """

    differentiable_abundance: bool
    observability_processor: ObservabilityProcessor
    transfer_processor: TransferProcessor
    out_blur: BeamBlur | None

    def __init__(self,
                 hyper: hp.Hyper,
                 *args: any,
                 abundance: Abundance | None = None,
                 units: str = 'iris',
                 node_comm: mpi4py.MPI.Intracomm | None = None,
                 **kwargs: any) -> None:
        super().__init__(hyper, 
                         *args, 
                         abundance=abundance, 
                         units=units, 
                         node_comm=node_comm, 
                         **kwargs)

        self.differentiable_abundance = False

        self.observability_processor = ObservabilityProcessor(hyper=hyper,
                                                              abundance=abundance,
                                                              units=units,
                                                              node_comm=node_comm)
        self.transfer_processor = TransferProcessor(observability_processor=self.observability_processor,
                                                    hyper=hyper,
                                                    units=units)
        if hyper.observer_hyper.out_blur_fwhm is None:
            self.out_blur = None
        else:
            self.out_blur = BeamBlur(hyper)
        return

    def forward(self, 
                inputs: torch.Tensor,
                *args: any,
                bypass_blur_in: bool = False,
                bypass_blur_out: bool = False,
                subtraction: str = 'I',
                units: str = 'Trj',
                transfer: str = 'optically thick',
                integration: str = 'smooth',
                **kwargs: any) -> torch.Tensor:
        r"""
        Computes the forward pass of a synthetic observation.

        Since `SyntheticObserver` is a `torch.nn.Module`, this method is automatically called
        when the object itself is called. Example:

        ```python
        observer = SyntheticObserver(hyper)
        observed = observer(physical_tensor)
        ```

        Args:
            inputs: The [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
                to be observed. Has dimensions
                `batch, channel=6, r, lon, lat`. The channel values are
                `v_r, rho, T, abundance_H2, abundance_CO, T_dust`.
            *args: Catch-all for args passed by extending classes or to extended classes.
            bypass_blur_in: If `True`, will skip the application of `self.in_blur`. Used in
                [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver],
                [`PreObservedDataset.add_tensor`][iris.arepo_processing.PreObservedDataset.add_tensor],
                and [`train_reverter`][iris.training.train_reverter] (when training with a
                [`StandardDataset`][iris.arepo_processing.StandardDataset]) where `in_blur`
                is manually applied by the external caller.
            bypass_blur_out: If `True`, will skip the application of `self.out_blur`. Used in
                [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver]
                where `out_blur` is manually applied by the external caller.
            subtraction: The continuum subtraction mode passed to `self.transfer_processor`.
                One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube. Note that `subtraction` is independent
                of `units`.
            units: The output units passed to `self.transfer_processor`.
                One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`SyntheticObserver.__init__`][iris.observation.SyntheticObserver]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`SyntheticObserver.__init__`][iris.observation.SyntheticObserver]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.
            transfer: The transfer type passed to
                [`observability_processor.forward`][iris.observation.ObservabilityProcessor.forward] and
                [`transfer_processor.forward`][iris.observation.TransferProcessor.forward].
                One of `'optically thick', 'selectively thin', 'optically thin'`.
            integration: The integration type passed to
                [`transfer_processor.forward`][iris.observation.TransferProcessor.forward].
                One of `'formal', 'smooth'`.
            **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = not self.differentiable_input
        with nullcontext() if self.differentiable_input else torch.no_grad():
            if inplace and not isinstance(self, IteratedObserver):
                inputs = inputs.clone()
            if not bypass_blur_in and self.in_blur is not None:
                # Apply velocity blurring for smoother observation.
                inputs = self.in_blur(inputs, inplace=inplace)
        # Compute emission and absorption coefficients.
        observables = self.observability_processor(inputs,
                                                   inplace=inplace,
                                                   transfer=transfer)
        # Compute ray solutions to the radiative transfer equation.
        observed = self.transfer_processor(observables,
                                           inplace=inplace,
                                           subtraction=subtraction,
                                           units=units,
                                           transfer=transfer,
                                           integration=integration)
        with nullcontext() if self.differentiable_abundance else torch.no_grad():
            if not bypass_blur_out and self.out_blur is not None:
                # Apply beam blurring to simulate angular resolution of the telescope.
                observed = self.out_blur(observed)
        return observed

    def set_requires_grad_all(self) -> None:
        """
        Enables end-to-end differentiability.

        Sets `self.differentiable_input = True` and `self.differentiable_abundance = True`, and calls
        [observability_processor.set_requires_grad_all][iris.observation.ObservabilityProcessor.set_requires_grad_all]
        and
        [transfer_processor.set_requires_grad_all][iris.observation.TransferProcessor.set_requires_grad_all].
        """
        self.differentiable_input = True
        self.differentiable_abundance = True
        self.observability_processor.set_requires_grad_all()
        self.transfer_processor.set_requires_grad_all()
        return

    def set_requires_grad_abundance(self) -> None:
        """
        Enables abundance-only differentiability.

        Sets `self.differentiable_input = False` and `differentiable_abundance = True`, and calls
        [observability_processor.set_requires_grad_abundance][iris.observation.ObservabilityProcessor.set_requires_grad_abundance]
        and
        [transfer_processor.set_requires_grad_all][iris.observation.TransferProcessor.set_requires_grad_all].
        """
        self.differentiable_input = False
        self.differentiable_abundance = True
        self.observability_processor.set_requires_grad_abundance()
        self.transfer_processor.set_requires_grad_all()
        return

    def set_requires_grad_none(self) -> None:
        """
        Disables all differentiability.

        Sets `self.differentiable_input = False` and `differentiable_abundance = False`, and calls
        [observability_processor.set_requires_grad_none][iris.observation.ObservabilityProcessor.set_requires_grad_none]
        and
        [transfer_processor.set_requires_grad_none][iris.observation.TransferProcessor.set_requires_grad_none].
        """
        self.differentiable_input = False
        self.differentiable_abundance = False
        self.observability_processor.set_requires_grad_none()
        self.transfer_processor.set_requires_grad_none()
        return

    def to_Jy_per_Sr(self, I: torch.Tensor) -> torch.Tensor:
        r"""
        Converts a tensor in the intensity units specified to
        [`SyntheticObserver.__init__`][iris.observation.SyntheticObserver]
        (one of `'iris', 'processing'`) into $\text{Jy}/\text{sr}$.

        Args:
            I: The tensor to convert.

        Returns:
            The input tensor converted into $\text{Jy}/\text{sr}$.
        """
        return self.transfer_processor.to_Jy_per_Sr(I)

    def to_K(self, T: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor in the temperature units specified to
        [`SyntheticObserver.__init__`][iris.observation.SyntheticObserver]
        (one of `'iris', 'processing'`) into K.

        Args:
            T: The temperature tensor to convert.

        Returns:
            The input temperature tensor converted into K.
        """
        return self.transfer_processor.to_K(T)


class SimpleObserver(Observer):
    r"""
    The core simple observation class.

    Simple observation is a diagnostic tool that implements a density projection scheme that
    offers a coarse analogue to true
    [synthetic observation][iris.observation.SyntheticObserver]. The output is a tensor of density
    over dimensions of galactic longitude, latitude, and velocity (a PPV cube). The density
    value is the mass density of gas, integrated along the line of sight, per differential of
    velocity--i.e. the column density of gas per differential velocity or the area-velocity
    density of gas mass. It is in units $\text{mass} \cdot \text{time} / \text{length}^3$.
    See the IRIS paper (subsec: Comparison Against Density-Tracing) for a deeper theoretical
    explanation as to why simple observation is a provides a meaningful characterization of the
    theoretical limit of information contained in a synthetic or true observation.
    For definitive training and application to true observations in which accuracy
    is preferred, however, it is suggested that only synthetic observation be used.

    For these reasons, simple observation may provide a probe into the ideal performance
    limit of [reversion][iris.reversion.Reverter] in the absence of the many confounding
    factors of synthetic observation that may serve to obscure or dilute the information
    content of an observation. For definitive training
    and application to true observations in which accuracy is preferred, however, it is
    suggested that only synthetic observation be used.

    By virtue of the velocity-wise soft-binning scheme implemented in
    [`forward`][iris.observation.SimpleObserver.forward], simple observation is end-to-end
    differentiable. For efficiency, differentiability is turned off by default, but can be
    manually enabled by calling
    [`set_requires_grad_all`][iris.observation.SimpleObserver.set_requires_grad_all].
    In end-to-end differentiability mode, gradients can backpropagate to any computation
    prior to [`SyntheticObserver.forward`][iris.observation.SimpleObserver.forward]
    and/or the input physical tensors themselves if they are `requires_grad=True` leaf tensors.
    Once enabled, differentiability can subsequently be turned back off by calling
    [`set_requires_grad_none`][iris.observation.SimpleObserver.set_requires_grad_none].

    Note that simple observation is memory-intensive, and should be ray-batched to save memory.
    Iterative ray-batching is implemented in full automation by
    [`IteratedSimpleObserver`][iris.observation.IteratedSimpleObserver], which should be
    the primary class instantiated by the user. Do not implement this class directly unless
    manually implementing a ray-batching scheme, e.g. model-parallel ray-batching across a
    distributed GPU cluster.

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Do not manually
            set this flag. Call
            [`set_requires_grad_all`][iris.observation.SimpleObserver.set_requires_grad_all]
            instead.
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [reverter][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)
        v_density_per_SI: The conversion factor, from SI units to `units`, for the output
            units of velocity-density ($\text{mass} / [\text{area} \cdot \text{velocity}]$).
            A `torch.float32` scalar.
        v_min: The minimal velocity bound of the output PPV cube. A `torch.float32` scalar.
        v_max: The maximal velocity bound of the output PPV cube. A `torch.float32` scalar.
        v_steps: The number of steps in velocity dimension of the output PPV cube. A `torch.int32` scalar.
        dv: The step size of the velocity dimension of the output PPV cube. A `torch.float32` scalar.
        dr_dv: The ratio of `r` step size to `v` step size. A `torch.float32` scalar.

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        units: The units of the input physical tensor and all internal computation of the
            synthetic observation. One of `'iris', 'processing'`. Not the same as the output
            units specified to [`forward`][iris.observation.SimpleObserver.forward].
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """

    v_density_per_SI: torch.nn.Parameter
    v_min: torch.nn.Parameter
    v_max: torch.nn.Parameter
    v_steps: torch.nn.Parameter
    dv: torch.nn.Parameter
    dr_dv: torch.nn.Parameter

    def __init__(self, hyper: hp.Hyper, *args: any, units: str = 'iris', **kwargs: any) -> None:
        super().__init__(hyper, *args, units=units, **kwargs)
        if units == 'iris':
            time = hyper.dataset_hyper._time_iris_per_SI
            mass = hyper.dataset_hyper._mass_iris_per_SI
            length = hyper.dataset_hyper._length_iris_per_SI
            area = length * length
            length_per_parsec = hyper.dataset_hyper._length_iris_per_parsec
            velocity = length / time * 1000
        elif units == 'processing':
            mass = 1000 / hyper.writer_hyper.mass_g_per_processing
            length = 100 / hyper.writer_hyper.length_cm_per_processing
            area = length * length
            velocity = 100000 / hyper.writer_hyper.velocity_cm_per_s_per_processing
            length_per_parsec = 1 / hyper.writer_hyper._length_parsec_per_processing
        else:
            raise ValueError("Invalid units provided to SimpleObserver. Must be 'iris' or 'processing'.")

        self.v_density_per_SI = torch.nn.Parameter(
            torch.tensor(mass / area / velocity, dtype=torch.float32), requires_grad=False)
        v_min = hyper.cube_hyper.v_min * velocity
        v_max = hyper.cube_hyper.v_max * velocity
        v_steps = hyper.cube_hyper.v_steps
        dv = (v_max - v_min) / (v_steps - 1)
        self.v_min = torch.nn.Parameter(torch.tensor(v_min, dtype=torch.float32), requires_grad=False)
        self.v_max = torch.nn.Parameter(torch.tensor(v_max, dtype=torch.float32), requires_grad=False)
        self.v_steps = torch.nn.Parameter(torch.tensor(v_steps, dtype=torch.int32), requires_grad=False)
        self.dv = torch.nn.Parameter(torch.tensor(dv, dtype=torch.float32), requires_grad=False)
        r_min = hyper.coordinate_hyper.r_min
        r_max = hyper.coordinate_hyper.r_max
        r_steps = hyper.coordinate_hyper.r_steps
        dr = (r_max - r_min) / (r_steps - 1) * length_per_parsec
        self.dr_dv = torch.nn.Parameter(torch.tensor(dr / dv, dtype=torch.float32), requires_grad=False)
        return

    def forward(self,
                inputs: torch.Tensor,
                *args: any,
                bypass_blur_in: bool = False,
                units: str = 'vrho',
                **kwargs: any) -> torch.Tensor:
        r"""
        Computes the forward pass of a simple observation.

        Since `SimpleObserver` is a `torch.nn.Module`, this method is automatically called
        when the object itself is called. Example:

        ```python
        observer = SimpleObserver(hyper)
        observed = observer(physical_tensor)
        ```

        Args:
            inputs: The [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
                to be observed. Has dimensions
                `batch, channel=6, r, lon, lat`. The channel values are
                `v_r, rho, T, abundance_H2, abundance_CO, T_dust`.
            *args: Catch-all for args passed by extending classes or to extended classes.
            bypass_blur_in: If `True`, will skip the application of `self.in_blur`. Used in
                [`IteratedSimpleObserver`][iris.observation.IteratedSimpleObserver],
                [`PreObservedDataset.add_tensor`][iris.arepo_processing.PreObservedDataset.add_tensor],
                and [`train_reverter`][iris.training.train_reverter] (when training with a
                [`StandardDataset`][iris.arepo_processing.StandardDataset]) where `in_blur`
                is manually applied by the external caller.
            units: The output units. One of `'vrho', 'vrho SI'. If `'vrho'`, the output is returned
                as a PPV cube of velocity-density ($\text{mass} / (\text{area} \cdot \text{velocity})$)
                in whatever units are provided to
                [`SimpleObserver.__init__`][iris.observation.SimpleObserver]--one of `'iris', 'processing'`.
                If `'vrho SI'`, the output is returned as a PPV cube of velocity-density in SI units.
            **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

        Returns:
            The observed PPV cube. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Raises:
            ValueError: If `units` is not one of `'vrho', 'vrho SI'.
        """
        inplace = not self.differentiable_input
        with nullcontext() if self.differentiable_input else torch.no_grad():
            if inplace and not isinstance(self, IteratedObserver):
                inputs = inputs.clone()
            if not bypass_blur_in and self.in_blur is not None:
                # Apply velocity blurring for smoother observation.
                inputs = self.in_blur(inputs, inplace=inplace)

            # Convert from volume density to velocity density.
            # The entire observation computation is linear with respect to
            # this conversion coefficient, so the conversion can be performed up-front.
            # Equivalent to integrating over r and dividing by the v channel size
            # to attain a per-channel average velocity-density.
            if inplace:
                rho = inputs[:, 1, :, :, :].mul_(self.dr_dv)
                v_r = inputs[:, 0, :, :, :]
            else:
                rho = inputs[:, 1, :, :, :] * self.dr_dv
                v_r = inputs[:, 0, :, :, :].clone()

            # Soft-bin physical tensor cells by velocity.
            # Soft-binning as opposed to hard-binning is used for differentiability.
            v_coords = v_r.sub_(self.v_min).div_(self.dv)
            v_indices = v_coords.floor().long()
            v_indices = v_indices.clamp_(min=0, max=self.v_steps - 2)
            weight = v_coords.sub_(v_indices)
            clamped = weight.clamp_(min=0, max=1)
            lower_weight = (1 - clamped) * (weight > -.5)
            upper_weight = clamped * (weight < 1.5)

            # Map physical tensor into an expanded r, lon, lat, v space
            # according to the soft bins.
            ii, jj, kk, ll = torch.meshgrid(torch.arange(v_indices.shape[0]),
                                            torch.arange(v_indices.shape[1]),
                                            torch.arange(v_indices.shape[2]),
                                            torch.arange(v_indices.shape[3]),
                                            indexing='ij')
            space = torch.zeros(list(rho.shape) + [self.v_steps], dtype=torch.float32, device=rho.device)
            space[ii, jj, kk, ll, v_indices] += rho * lower_weight
            space[ii, jj, kk, ll, v_indices + 1] += rho * upper_weight
            space = space.unsqueeze(dim=1)
            # Integrate the expanded space over r.
            # Multiplication by dr and division by dv has already been performed.
            # sight_integrated yields an average velocity-density per velocity channel.
            sight_integrated = torch.sum(space, dim=2)

            if units == 'vrho':
                pass
            elif units == 'vrho SI':
                sight_integrated = sight_integrated / self.v_density_per_SI
            else:
                raise ValueError("Invalid units provided to SimpleObserver.forward. "
                                 "Must be one of 'vrho', 'vrho SI'.")
        return sight_integrated

    def set_requires_grad_all(self) -> None:
        """
        Enables end-to-end differentiability.

        Sets `self.differentiable_input = True`.
        """
        self.differentiable_input = True
        return

    def set_requires_grad_none(self) -> None:
        """
        Disables all differentiability.

        Sets `self.differentiable_input = False`.
        """
        self.differentiable_input = False
        return


class IteratedObserver(Observer):
    """
    The base class for all iterated observer types.

    Performs iterative ray-batching of an [`Observer`][iris.observation.Observer].
    Is a parent class that should not be directly instantiated by the user. Extended by
    the user-facing [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver]
    and [`IteratedSimpleObserver`][iris.observation.IteratedSimpleObserver] classes.

    Since both [synthetic observation][iris.observation.SyntheticObserver] and
    [simple observation][iris.observation.SimpleObserver] are very memory intensive,
    they cannot be performed as a single forward pass over a
    [physical tensor][iris.arepo_processing.Snapshot.make_physical_tensor] of
    practical dimensions on a single GPU. Therefore, these observations must
    be ray-batched in order to avoid a CUDA OOM error. Ray-batching refers to chunking
    the observation over small patches of latitude and longitude.

    This class implements a basic iterative ray-batching scheme, whereby the sky plane
    is divided into a grid of `self.lon_pieces` longitude sections and `self.lat_pieces`
    latitude sections that are iteratively fed to the `forward` method of the base observer class, e.g.
    [`SyntheticObserver.forward`][iris.observation.SyntheticObserver.forward]. The values
    of `lon_pieces` and `lat_pieces` need not divide `hyper.coordinate_hyper.lon_steps` and
    `hyper.coordinate_hyper.lat_steps`, respectively. The last row and column
    of ray batches will be truncated if necessary to fit the sky plane. If the `IteratedObserver`
    is a `SyntheticObserver` and [`forward`][iris.observation.IteratedObserver.forward]
    receives the keyword arg `transfer='optically thick'`, then `lon_pieces` will be multiplied by
    `2 * self.v_subsamples` to prevent an OOM error
    from the velocity subsampling (no multiplication is applied if `v_subsamples == 0`).

    This ray-batching scheme can be performed in one of two modes: Full GPU mode, by setting
    `self.cpu_batch = False`, and CPU-batching mode, by setting `self.cpu_batch = True`. In full GPU
    mode, it is expected that both the input physical tensor and the `IteratedObserver` are
    moved to the GPU prior to calling the forward pass. In CPU-batching mode, it is expected
    that the `IteratedObserver` is moved to the GPU and then passed a CPU physical tensor.
    The physical tensor is moved to the GPU a single ray batch at a time. Full GPU mode is the
    standard configuration, and is slightly preferable in standard cases since `self.in_blur`
    is applied over the entire physical tensor before passing to the base observer. For very
    large physical tensors such as full-cone observations, however, CPU batching prevents a
    CUDA OOM error from the physical tensor itself exceeding GPU memory capacity. For CPU batching,
    `self.in_blur` is applied separately by the base observer over each ray batch.

    Note that this simple iterative ray-batching scheme is not suitable if observer differentiability
    is required. For instance, if an `IteratedSyntheticObserver` is configured in end-to-end
    differentiability mode by calling
    [`IteratedSyntheticObserver.set_requires_grad_all`][iris.observation.SyntheticObserver.set_requires_grad_all],
    then GPU memory will not be freed after each iterative step since PyTorch must retain
    a complete computational graph for subsequent gradient backpropagation. In such cases,
    an alternative ray-batching scheme should be used instead of `IteratedObserver`,
    e.g. one of the following:

    * Ray-batched forwards pass; subsequent computations that are independent of other
    ray batches; backwards pass on the ray-batch only, destroying the graph;
    iterate over all ray batches.
    * Model-parallel ray-batched forward pass over a distributed GPU cluster;
    full-output computations; model-parallel backward pass.
    * Ray-batched forwards pass; GPU-to-CPU memory swapping scheme of the computational graph;
    iterate over all ray batches; full-output computations; backward pass via custom
    swap-smart backpropagation routine.

    Since differentiable observers, while implemented, are not currently utilized by the IRIS
    project, these alternative ray-batching schemes are left to user implementations
    or future releases.

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Is primarily
            a flag specifying behavior to be implemented by extending classes, and should
            not be manually set. Call `set_requires_grad_all` instead. (See
            [`SyntheticObserver.set_requires_grad_all`][iris.observation.SyntheticObserver.set_requires_grad_all] and
            [`SimpleObserver.set_requires_grad_all`][iris.observation.SimpleObserver.set_requires_grad_all].)
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [reverter][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)
        lon_pieces: The number of longitude sections into which the sky plane is ray-batched.
            If the `IteratedObserver` is a `SyntheticObserver` and
            [`forward`][iris.observation.IteratedObserver.forward] receives the keyword
            arg `transfer='optically thick'`, then `lon_pieces` will be multiplied by
            `2 * self.v_subsamples` prior to ray-batching
            (no multiplication is applied if `v_subsamples == 0`).
            Set by `hyper.observer_hyper.lat_pieces`.
        lat_pieces: The number of latitude sections into which the sky plane is ray-batched.
            Set by `hyper.observer_hyper.lon_pieces`.
        lon_steps: The number of total steps in the longitude dimension.
            Set by `hyper.coordinate_hyper.lon_steps`.
        lat_steps: The number of total steps in the latitude dimension.
            Set by `hyper.coordinate_hyper.lat_steps`.
        v_subsamples: The number of velocity samples per velocity channel.
            If the `IteratedObserver` is a `SyntheticObserver` and
            [`forward`][iris.observation.IteratedObserver.forward] receives the keyword
            arg `transfer='optically thick'`, then
            `self.lon_pieces` will be multiplied by
            `2 * v_subsamples` prior to ray-batching (no multiplication is applied if `v_subsamples == 0`).
            Set by `hyper.observer_hyper.v_subsamples`.
        cpu_batch: If `True`, applies CPU batching. Otherwise, operates in full GPU mode. In full GPU
            mode, it is expected that both the input physical tensor and the `IteratedObserver` are
            moved to the GPU prior to calling the forward pass. In CPU-batching mode, it is expected
            that the `IteratedObserver` is moved to the GPU and then passed a CPU physical tensor.
            The physical tensor is moved to the GPU a single ray batch at a time. Full GPU mode is the
            standard configuration, and is slightly preferable in standard cases since
            `self.in_blur` is applied over the entire
            physical tensor before passing to the base observer. For very large physical tensors
            such as full-cone observations, however, CPU batching prevents a CUDA OOM error from
            the physical tensor itself exceeding GPU memory capacity. For CPU batching, `in_blur`
            is applied separately by the base observer over each ray batch.

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        cpu_batch: Sets `self.cpu_batch`.
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """

    lon_pieces: int
    lat_pieces: int
    lon_steps: int
    lat_steps: int
    v_subsamples: int
    cpu_batch: bool

    def __init__(self, hyper: hp.Hyper, *args: any, cpu_batch: bool = False, **kwargs: any) -> None:
        super().__init__(hyper,
                         *args, 
                         cpu_batch=cpu_batch,
                         **kwargs)
        self.lon_pieces = hyper.observer_hyper.lon_pieces
        self.lat_pieces = hyper.observer_hyper.lat_pieces
        self.lon_steps = hyper.coordinate_hyper.lon_steps
        self.lat_steps = hyper.coordinate_hyper.lat_steps
        self.v_subsamples = hyper.observer_hyper.v_subsamples
        self.cpu_batch = cpu_batch
        return

    def forward(self,
                inputs: torch.Tensor,
                *args: any,
                bypass_blur_in: bool = False,
                gpu: int | None = None,
                transfer: str = 'optically thick',
                **kwargs: any) -> torch.Tensor:
        r"""
        Iteratively ray-batches the `forward` method of the base observer.

        Args:
            inputs: The [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
                to be observed. Has dimensions
                `batch, channel=6, r, lon, lat`. The channel values are
                `v_r, rho, T, abundance_H2, abundance_CO, T_dust`.
            *args: Catch-all for args passed by extending classes or to extended classes.
            bypass_blur_in: If `True`, will skip the application of
                `self.in_blur`. Used in
                [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver],
                [`PreObservedDataset.add_tensor`][iris.arepo_processing.PreObservedDataset.add_tensor],
                and [`train_reverter`][iris.training.train_reverter] (when training with a
                [`StandardDataset`][iris.arepo_processing.StandardDataset]) where `in_blur`
                is manually applied by the external caller.
            gpu: A GPU access key from a GPU manager, if applicable
                (i.e. if `self.cpu_batch` and alled within a
                [`Writer`][iris.arepo_processing_write.Writer] process).
            transfer: The transfer type. If the `IteratedObserver` is an
                [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver].
                and `transfer == 'optically thick'`,
                `self.lon_pieces` is multiplied by
                `2 * self.v_subsamples` prior to ray-batching
                (no multiplication is applied if `v_subsamples == 0`).
            **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
            
        Returns:
            The observed PPV cube. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = not self.differentiable_input
        cpu_batch = self.cpu_batch and torch.cuda.is_available()
        with nullcontext() if self.differentiable_input else torch.no_grad():
            if not cpu_batch:
                if inplace:
                    inputs = inputs.clone()
                if not bypass_blur_in and self.in_blur is not None:
                    inputs = self.in_blur(inputs, inplace=inplace)
                    bypass_blur_in = True

        if isinstance(self, SyntheticObserver) and transfer == 'optically thick' and self.v_subsamples > 0:
            lon_pieces = self.lon_pieces * self.v_subsamples * 2
        else:
            lon_pieces = self.lon_pieces
        lat_pieces = self.lat_pieces
        lon_groups = []
        lon_hi = 0
        for lon in range(1, lon_pieces + 1):
            group = []
            lon_lo = lon_hi
            lon_hi = min(int(lon * self.lon_steps / lon_pieces), self.lon_steps)
            lat_hi = 0
            for lat in range(1, lat_pieces + 1):
                lat_lo = lat_hi
                lat_hi = min(int(lat * self.lat_steps / lat_pieces), self.lat_steps)
                ray_batch = inputs[:, :, :, lon_lo:lon_hi, lat_lo:lat_hi]
                if cpu_batch:
                    ray_batch = ray_batch.cuda(gpu)
                observed_bundle = super().forward(ray_batch,
                                                  bypass_blur_in=bypass_blur_in,
                                                  transfer=transfer,
                                                  **kwargs)
                group.append(observed_bundle)
                del ray_batch
            lon_groups.append(torch.cat(group, dim=3))
        outputs = torch.cat(lon_groups, dim=2)
        return outputs


class IteratedSyntheticObserver(IteratedObserver, SyntheticObserver):
    """
    Iteratively ray-batches a synthetic observation by extending both
    [`IteratedObserver`][iris.observation.IteratedObserver] and
    [`SyntheticObserver`][iris.observation.SyntheticObserver].

    A user-facing class. Is the primary [`Observer`][iris.observation.Observer] type that
    should be implemented in most use-cases. See
    [`SyntheticObserver`][iris.observation.SyntheticObserver] for details on synthetic
    observation and
    [`IteratedObserver`][iris.observation.IteratedObserver] for details on the iterative
    ray-batching scheme.

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Do not manually
            set this flag. Call
            [`set_requires_grad_all`][iris.observation.SyntheticObserver.set_requires_grad_all]
            instead.
        differentiable_abundance: If `True`, gradients can backpropagate to the
            [Abundance][iris.observation.Abundance] and any `requires_grad=True` variables in it.
            If `True` and `differentiable_input` is `False`, gradients can backpropagate no further
            than the abundance. This is a more computationally efficient differentiability option
            if the intent is abundance training. Do not manually
            set this flag. Call
            [`set_requires_grad_abundance`][iris.observation.SyntheticObserver.set_requires_grad_abundance]
            instead.
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [reverter][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)
        observability_processor: The [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]
            used to compute emission and absorption coefficients.
        transfer_processor: The [`TransferProcessor`][iris.observation.TransferProcessor]
            used for computing ray solutions to the radiative transfer equation.
        out_blur: The [`BeamBlur`][iris.observation.BeamBlur] Gaussian point-spread convolution
            applied over the longitude and latitude dimensions of the observed cube to simulate the
            nonzero angular resolution of a radio antenna dish. The beam convolution is configured
            via its full-width-half-maximum (FWHM) in arcsec,
            `hyper.observer_hyper.out_blur_fwhm`. If `out_blur_fwhm` is `None`,
            `out_blur` is also set to none and no beam convolution is applied, yielding an ideal
            observation at the theoretical limit of angular resolution.
        lon_edges: The longitude edge indices of the ray batches.
        lat_edges: The latitude edge indices of the ray batches.
        cpu_batch: If `True`, applies CPU batching. Otherwise, operates in full GPU mode. In full GPU
            mode, it is expected that both the input physical tensor and the `IteratedSyntheticObserver` are
            moved to the GPU prior to calling the forward pass. In CPU-batching mode, it is expected
            that the `IteratedSyntheticObserver` is moved to the GPU and then passed a CPU physical tensor.
            The physical tensor is moved to the GPU a single ray batch at a time. Full GPU mode is the
            standard configuration, and is slightly preferable in standard cases since
            `self.in_blur` is applied over the entire
            physical tensor before passing to the base observer. For very large physical tensors
            such as full-cone observations, however, CPU batching prevents a CUDA OOM error from
            the physical tensor itself exceeding GPU memory capacity. For CPU batching, `in_blur`
            is applied separately by the base observer over each ray batch.

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        cpu_batch: Sets `self.cpu_batch`.
        abundance: The [`Abundance`][iris.observation.Abundance] passed to `self.observability_processor`.
            If `None`, [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]
            currently defaults to the $^{13}\text{CO}$ abundance function employed in the
            IRIS paper, [`Constant_CO_13C16O`][iris.observation.Constant_CO_13C16O].
        units: The units of the input physical tensor and all internal computation of the
            synthetic observation. One of `'iris', 'processing'`. Not the same as the output
            units specified to [`forward`][iris.observation.SyntheticObserver.forward].
        node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support,
            if used during [dataset writing][iris.arepo_processing_write.Writer].
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.
    """
    def forward(self,
                inputs: torch.Tensor,
                *args: any,
                bypass_blur_out: bool = False,
                **kwargs: any) -> torch.Tensor:
        r"""
        Wraps the iterative ray-batching implemented in
        [`IteratedObserver.forward`][iris.observation.IteratedObserver.forward]
        in order to apply `self.out_blur` as a single pass over the entire PPV cube.

        If this method were not implemented, the default behavior would be that the
        [`BeamBlur`][iris.observation.BeamBlur] of the synthetic observation process would be
        applied separately over the PPV cubes of every ray batch. This could introduce
        edge artifacts into the combined PPV cube, tracing the outline of the ray-batching
        grid. This method prevents the emergence of these artifacts by performing the
        beam convolution over the entire PPV cube at once.

        Args:
            inputs: The [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
                to be observed. Has dimensions
                `batch, channel=6, r, lon, lat`. The channel values are
                `v_r, rho, T, abundance_H2, abundance_CO, T_dust`.
            *args: Catch-all for args passed by extending classes or to extended classes.
            bypass_blur_out: If `True`, will skip the application of `self.out_blur`.
            **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

        Returns:
            The observed PPV cube. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        observed = super().forward(inputs, *args, bypass_blur_out=True, **kwargs)
        with nullcontext() if self.differentiable_abundance else torch.no_grad():
            if not bypass_blur_out and self.out_blur is not None:
                observed = self.out_blur(observed)
        return observed


class IteratedSimpleObserver(IteratedObserver, SimpleObserver):
    r"""
    Iteratively ray-batches a synthetic observation by extending both
    [`IteratedObserver`][iris.observation.IteratedObserver] and
    [`SimpleObserver`][iris.observation.SimpleObserver].

    A user-facing class. Is the primary [`Observer`][iris.observation.Observer] type that
    should be implemented for simple observation in most cases. See
    [`SimpleObserver`][iris.observation.SimpleObserver] for details on synthetic
    observation and
    [`IteratedObserver`][iris.observation.IteratedObserver] for details on the iterative
    ray-batching scheme.

    Attributes:
        differentiable_input: Whether end-to-end differentiability of the observer is enabled.
            If `True`, gradients can backpropagate all the way through the observer to
            the inputs. By default, is `False` for computational efficiency. Do not manually
            set this flag. Call
            [`set_requires_grad_all`][iris.observation.SimpleObserver.set_requires_grad_all]
            instead.
        in_blur: A preblur applied to the velocity channel of the input only
            ([`VelocityBlur`][iris.observation.VelocityBlur]). Enabled via the flag
            `hyper.observer_hyper.blur_inputs`, `None` if not enabled.
            The nearest-neighbor [interpolation][iris.arepo_processing.Snapshot._interpolate]
            scheme employed during AREPO snapshot processing yields regions of constant velocity, which
            appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
            By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
            observations for more reliable [`reverter`][iris.reversion.Reverter] training. Size of
            the Gaussian blurring kernel is configured via
            `hyper.observer_hyper.in_blur_kernel_r`
            (`r` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lon`
            (`lon` size of the Gaussian kernel in pixels),
            `hyper.observer_hyper.in_blur_kernel_lat`
            (`lat` size of the Gaussian kernel in pixels), and
            `hyper.observer_hyper.in_blur_sigma`
            (spatial standard deviation of the Gaussian kernel in pixels).
            (See [`VelocityBlur`][iris.observation.VelocityBlur] for details.)
        v_density_per_SI: The conversion factor, from SI units to `units`, for the output
            units of velocity-density ($\text{mass} / (\text{area} \cdot \text{velocity})$).
        v_min: The minimal velocity bound of the output PPV cube.
        v_max: The maximal velocity bound of the output PPV cube.
        v_steps: The number of steps in velocity dimension of the output PPV cube.
        dv: The step size of the velocity dimension of the output PPV cube.
        dr_dv: The ratio of `r` step size to `v` step size.
        lon_edges: The longitude edge indices of the ray batches.
        lat_edges: The latitude edge indices of the ray batches.
        cpu_batch: If `True`, applies CPU batching. Otherwise, operates in full GPU mode. In full GPU
            mode, it is expected that both the input physical tensor and the `IteratedSimpleObserver` are
            moved to the GPU prior to calling the forward pass. In CPU-batching mode, it is expected
            that the `IteratedSimpleObserver` is moved to the GPU and then passed a CPU physical tensor.
            The physical tensor is moved to the GPU a single ray batch at a time. Full GPU mode is the
            standard configuration, and is slightly preferable in standard cases since
            `self.in_blur` is applied over the entire
            physical tensor before passing to the base observer. For very large physical tensors
            such as full-cone observations, however, CPU batching prevents a CUDA OOM error from
            the physical tensor itself exceeding GPU memory capacity. For CPU batching, `in_blur`
            is applied separately by the base observer over each ray batch.

    Args:
        hyper: A hyperparameters object.
        *args: Catch-all for args passed by extending classes or to extended classes.
        cpu_batch: Sets `self.cpu_batch`.
        units: The units of the input physical tensor and all internal computation of the
            synthetic observation. One of `'iris', 'processing'`. Not the same as the output
            units specified to [`forward`][iris.observation.SimpleObserver.forward].
        **kwargs: Catch-all for keyword args passed by extending classes or to extended classes.

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """
    pass


class ObservabilityProcessor(torch.nn.Module):
    r"""
    Computes emission and absorption coefficients of a physical tensor.

    Determines emission and absorption coefficients for each spectral line and for dust.
    The dust absorption coefficient is determined from the density of a physical tensor
    according to a single, constant dust opacity per spectral line, specified in
    `self.hyper.observer_hyper.kappa_dust`. The dust absorption tensor is
    thus treated as constant over the small frequency window of the line observation,
    but not over all separate lines, which may vary substantially in frequency, and so
    has shape `batch, n_lines, r_steps, lon_steps, lat_steps, 1`. Dust emission is computed
    based on dust absorption according to a blackbody source function that is allowed
    to vary across the frequency dimension, and so has shape
    `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps`. The emission and absorption
    tensors of the spectral line also vary over frequency space within each line observation,
    and so both have shapes `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps`.
    These tensors occupy a large memory footprint on the GPU, which is the primary compute
    bottleneck for synthetic observation.

    In order to compute these observability coefficients efficiently
    at runtime, `ObservabilityProcessor` relies on grids and grid gradients of
    line emission and absorption coefficients computed over dimensions of gas density,
    $\text{H}_2$ abundance, and temperature in order to perform a fast linear interpolation.
    These grids are computed based on a non-LTE, optically thin assumption by manual
    solution of the level systems. These grids and grid gradients are computed a single time only,
    upon instantiation of the `SyntheticObserver` via
    [`chemistry.make_observability_grids`][iris.chemistry.make_observability_grids].
    The linear solution of the levels system is GPU-parallelized via PyTorch, and, depending
    upon the grid dimensions configured in `self.hyper.observer_hyper`, the instantiation
    process takes about 2-3 seconds on an NVIDIA A100 GPU. See
    [`chemistry.make_observability_grids`][iris.chemistry.make_observability_grids] for details
    on grid configuration, and note that observability coefficients are linear (not constant)
    between grid steps and outside grid bounds via the grid gradients interpolation.

    The grids are separated from the line profile, which is computed and applied at runtime,
    eliminating the necessity of adding a costly frequency dimension to the grids. One of
    three transfer modes can be specified to
    [`forward`][iris.observation.ObservabilityProcessor.forward]: optically thick mode,
    selectively thin mode, and optically thin mode. See
    [`TransferProcessor`][iris.observation.TransferProcessor] for details on these separate
    modes. In optically thick mode, a Gaussian line profile is computed based on pure
    Doppler (thermal) broadening. Natural broadening (via the Lorentz profile or combined
    Voigt profile) is ignored, as it is typically only dominant at the profile wings. In
    optically thin or selectively thin mode, the Gaussian profile is integrated via an
    analytic expression in terms of the standard error function (erf).

    The grids are also computed per tracer abundance, where tracer abundance is
    expressed as a fraction of total H atom number density, as both the observability
    coefficients and the level systems themselves are linear with respect to this variable.
    As the final step of observability determination, a configurable
    [abundance function][iris.observation.Abundance] is applied via multiplication to each line.
    Postponing abundance application to the final step allows more efficient gradient
    backpropagation when configured in abundance-only differentiability mode.

    Observability determination is fully differentiable--in particular, by virtue of the
    linearization scheme employed via the grid gradients. For efficiency,
    differentiability is turned off by default, but can be manually enabled in one of two modes:
    [end-to-end differentiability][iris.observation.ObservabilityProcessor.set_requires_grad_all] and
    [abundance-only differentiability][iris.observation.ObservabilityProcessor.set_requires_grad_abundance].
    In end-to-end differentiability mode, gradients can backpropagate to both the user-specified
    [Abundance][iris.observation.Abundance] and to the inputs of
    [`forward`][iris.observation.ObservabilityProcessor.forward]. In abundance-only
    differentiability mode, gradients can backpropagate to the abundance function, but no further.
    The postponed application of the abundance function ensures a maximal compute savings. Once enabled,
    differentiability can subsequently be turned back off by calling
    [`set_requires_grad_none`][iris.observation.SyntheticObserver.set_requires_grad_none]. See
    [`SyntheticObserver`][iris.observation.SyntheticObserver] for more details on differentiability.
        
    Attributes:
        hyper: A hyperparameters object.
        abundance: An abundance function.
        differentiable_input: Whether end-to-end differentiability is enabled.
            If `True`, gradients can backpropagate all the way through the `ObservabilityProcessor` to
            the inputs. By default, is `False` for computational efficiency. Do not manually
            set this flag. Call
            [`set_requires_grad_all`][iris.observation.ObservabilityProcessor.set_requires_grad_all]
            instead.
        differentiable_abundance: If `True`, gradients can backpropagate to the
            [Abundance][iris.observation.Abundance] and any `requires_grad=True` variables in it.
            If `True` and `differentiable_input` is `False`, gradients can backpropagate no further
            than the abundance. This is a more computationally efficient differentiability option
            if the intent is abundance training. Do not manually
            set this flag. Call
            [`set_requires_grad_abundance`][iris.observation.ObservabilityProcessor.set_requires_grad_abundance]
            instead.
        h: The Planck constant. A `torch.float32` scalar.
        k: The Boltzmann constant. A `torch.float32` scalar.
        c: The vacuum speed of light. A `torch.float32` scalar.
        n_lines: The number of spectral lines to be observed.
        rho: The density grid over which the observability grids are computed. Allows separate
            grid dimensions to be configured for each separate spectral line. See
            [`make_observability_grids`][iris.chemistry.make_observability_grids]
            for details on grid configuration.
            A list of `torch.float32` tensors of size `(N_H_TOT_steps[line],)` where
            `self.hyper.observer_hyper.N_H_TOT_steps` is set per spectral line in hyperparameters.
        dN_bolic: The linear step width of the `N_H_TOT` grid in arc-hyperbolic-sine space.
            A list of length `n_lines` of `torch.float32` scalars.
        abundance_H2: The $\text{H}_2$ abundance grid over which the observability grids are computed. 
            Allows separate grid dimensions to be configured for each separate spectral line. See
            [`make_observability_grids`][iris.chemistry.make_observability_grids]
            for details on grid configuration.
            A list of `torch.float32` tensors of size `(abundance_H2_steps[line],)` where
            `self.hyper.observer_hyper.abundance_H2_steps` is set per spectral line in hyperparameters.
        d_ab: The linear step width of the `abundance_H2` grid.
            A list of length `n_lines` of `torch.float32` scalars.
        T: The temperature abundance grid over which the observability grids are computed. 
            Allows separate grid dimensions to be configured for each separate spectral line. See
            [`make_observability_grids`][iris.chemistry.make_observability_grids]
            for details on grid configuration.
            A list of `torch.float32` tensors of size `(T_steps[line],)` where
            `self.hyper.observer_hyper.T_steps` is set per spectral line in hyperparameters.
        dT: The linear step width of the `T` grid.
            A list of length `n_lines` of `torch.float32` scalars.
        nu_ul: The transition frequency of each spectral line.
            A `torch.float32` tensor of dimensions `(n_lines,)`.
        tracer_mass: The mass of each tracer molecule.
            A `torch.float32` tensor of dimensions `(n_lines,)`.
        nu_channel_width: The frequency width associated with the step size of the coarse velocity grid,
            over which the eventual PPV cube is computed by
            [`TransferProcessor`][iris.observation.TransferProcessor].
            A `torch.float32` scalar.
        nu: The coarse velocity grid, over which the eventual PPV cube is computed by
            [`TransferProcessor`][iris.observation.TransferProcessor]. A `torch.float32` tensor of
            dimensions `batch=1, n_lines, r_steps, lon_steps, lat_steps, v_steps`.
        nu_edges: The edges of the coarse velocity grid. A `torch.float32` tensor of
            dimensions `batch=1, n_lines, r_steps, lon_steps, lat_steps, v_steps + 1`.
        nu_fine: The fine frequency grid over which line emission and absorption coefficients
            are computed in optically thick transfer mode. A `torch.float32` tensor of
            dimensions `batch=1, n_lines, r_steps, lon_steps, lat_steps, fine_steps`,
            where `fine_steps = v_steps` when `self.hyper.observer_hyper.v_subsamples`
            is `0` and `fine_steps = 2 * v_steps * v_subsamples + 1` otherwise.
        number_ism_molecular_mass: The average mass of the ISM per ``iris_number_unit`` H atoms.
            Varies depending upon the `self.hyper.observer_hyper.abundance_He` set in hyperparameters.
        bolic_normalization: A normalization factor used for mapping to the hyperbolic density-grid space.
            A `torch.float32` tensor of dimensions `n_lines`.
        kappa_dust: The dust opacity. Dust is coarsely assumed to be of a single species, uniformly
            distributed throughout the ISM, varying in opacity across the wide frequency gaps
            between separate spectral lines, but constant in opacity across the small frequency
            span of each line cube. Set by `self.hyper.observer_hyper.kappa_dust`
            as a single opacity per line. Dust is assumed to occupy a standard mass fraction
            of the ISM. That is, supposing, as an example, a standard ratio of 1:100, then the ratio of
            gas mass to dust mass in the ISM is assumed to be 100. Each opacity should be specified
            per unit of gas mass incorporating this standard fraction, as opposed to per unit dust mass.
            A `torch.float32` tensor of dimensions `(n_lines,)`.
        j: The line emission grids, expressed as a frequency-independent emission coefficient,
            i.e. before application of the line profile, per tracer abundance, where tracer abundance
            is expressed as a fraction of total H atom number density. A list of length `n_lines`
            of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        dj_drho: The partial derivative of each emission grid `j` with respect to the density dimension.
            A list of length `n_lines` of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        dj_d_ab: The partial derivative of each emission grid `j` with respect to the $\text{H}_2$
            abundance dimension. A list of length `n_lines` of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        dj_dT: The partial derivative of each emission grid `j` with respect to the temperature dimension.
            A list of length `n_lines` of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        alpha: The line absorption grids, expressed as a frequency-independent absorption coefficient,
            i.e. before application of the line profile, per tracer abundance, where tracer abundance
            is expressed as a fraction of total H atom number density. A list of length `n_lines`
            of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        d_alpha_drho: The partial derivative of each absorption grid `alpha` with respect to the density
            dimension. A list of length `n_lines` of `torch.float32` tensors, each of dimensions
            `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        d_alpha_d_ab: The partial derivative of each absorption grid `alpha` with respect to the
            $\text{H}_2$ abundance dimension. A list of length `n_lines` of `torch.float32` tensors,
            each of dimensions `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        d_alpha_dT: The partial derivative of each absorption grid `alpha` with respect to the
            temperature dimension. A list of length `n_lines` of `torch.float32` tensors,
            each of dimensions `N_H_TOT_steps[line], abundance_H2_steps[line], T_steps[line]', where
            `self.hyper.observer_hyper.N_H_TOT_steps`,
            `self.hyper.observer_hyper.abundance_H2_steps`, and
            `self.hyper.observer_hyper.T_steps` are set per spectral line in hyperparameters.
        
    Args:
        hyper: A hyperparameters object. Sets `self.hyper`.
        abundance: A user-specified abundance function. Sets `self.abundance`.
            See [`Abundance`][iris.observation.Abundance] for details. If `None`, defaults to the
            [`Constant_CO_13C16O`][iris.observation.Constant_CO_13C16O] abundance used in the IRIS paper.
        units: The input and output units. One of `'iris', 'processing'`.
        node_comm: An MPI node intracomm used to communicate with the GPU manager for GPU support,
            if used during [dataset writing][iris.arepo_processing_write.Writer].
        
    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """
    
    hyper: hp.Hyper
    abundance: Abundance
    differentiable_input: bool
    differentiable_input: bool
    h: torch.nn.Parameter
    k: torch.nn.Parameter
    c: torch.nn.Parameter
    n_lines: int
    rho: torch.nn.ParameterList
    dN_bolic: torch.nn.ParameterList
    abundance_H2: torch.nn.ParameterList
    d_ab: torch.nn.ParameterList
    T: torch.nn.ParameterList
    dT: torch.nn.ParameterList
    nu_ul: torch.nn.Parameter
    tracer_mass: torch.nn.Parameter
    nu_channel_width: torch.nn.Parameter
    nu: torch.nn.Parameter
    nu_edges: torch.nn.Parameter
    nu_fine: torch.nn.Parameter
    number_ism_molecular_mass: torch.nn.Parameter
    bolic_normalization: torch.nn.Parameter
    kappa_dust: torch.nn.Parameter
    j: torch.nn.ParameterList
    dj_drho: torch.nn.ParameterList
    dj_d_ab: torch.nn.ParameterList
    dj_dT: torch.nn.ParameterList
    alpha: torch.nn.ParameterList
    d_alpha_drho: torch.nn.ParameterList
    d_alpha_d_ab: torch.nn.ParameterList
    d_alpha_dT: torch.nn.ParameterList
    
    def __init__(self,
                 hyper: hp.Hyper,
                 abundance: Abundance | None = None,
                 units: str = 'iris',
                 node_comm: mpi4py.MPI.Intracomm | None = None) -> None:
        super().__init__()
        self.hyper = hyper

        if abundance is None:
            self.abundance = Constant_CO_13C16O(hyper, units=units)
        else:
            abundance.set_units(hyper, units)
            self.abundance = abundance

        self.differentiable_input = False
        self.differentiable_abundance = False

        if units == 'iris':
            number_unit = hyper.dataset_hyper.iris_number_unit
            mass = hyper.dataset_hyper._mass_iris_per_SI
            time = hyper.dataset_hyper._time_iris_per_SI
            length = hyper.dataset_hyper._length_iris_per_SI
            number_density = 1 / number_unit / length / length / length
            temperature = hyper.dataset_hyper._temperature_iris_per_SI
            velocity = length / time
            acceleration = velocity / time
            force = mass * acceleration
            energy = force * length
        elif units == 'processing':
            number_unit = hyper.dataset_hyper.iris_number_unit
            length = 100 / hyper.writer_hyper.length_cm_per_processing
            number_density = 1 / number_unit / length / length / length
            velocity = 100 / hyper.writer_hyper.velocity_cm_per_s_per_processing
            time = length / velocity
            mass = 1000 / hyper.writer_hyper.mass_g_per_processing
            temperature = 1 / hyper.writer_hyper.temperature_K_per_processing
            acceleration = velocity / time
            force = mass * acceleration
            energy = force * length
        else:
            raise ValueError("Invalid units provided to ObservabilityProcessor. Must be 'iris' or 'processing'.")

        self.h = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.h * energy * time,
                         dtype=torch.float32), requires_grad=False)
        self.k = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.k * energy / temperature,
                         dtype=torch.float32), requires_grad=False)
        self.c = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.c * velocity,
                         dtype=torch.float32), requires_grad=False)

        self.n_lines = hyper.observer_hyper.n_lines
        observability_grids = chemistry.make_observability_grids(hyper=hyper,
                                                                 units=units,
                                                                 node_comm=node_comm)

        self.rho = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['rho'], requires_grad=False) for i in range(self.n_lines)])
        self.dN_bolic = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['dN_bolic'], requires_grad=False) for i in range(self.n_lines)])
        self.abundance_H2 = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['abundance_H2'], requires_grad=False) for i in range(self.n_lines)])
        self.d_ab = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['d_ab'], requires_grad=False) for i in range(self.n_lines)])
        self.T = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['T'], requires_grad=False) for i in range(self.n_lines)])
        self.dT = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['dT'], requires_grad=False) for i in range(self.n_lines)])
        self.nu_ul = torch.nn.Parameter(
            torch.tensor([observability_grids[i]['nu_ul'] for i in range(self.n_lines)], dtype=torch.float32),
            requires_grad=False)
        self.tracer_mass = torch.nn.Parameter(
            torch.tensor([observability_grids[i]['molecular_weight'] / hyper.observer_hyper.L / 1000 * mass
                          for i in range(self.n_lines)], dtype=torch.float32),
            requires_grad=False)

        v_min = hyper.cube_hyper.v_min * 1000 * velocity
        v_max = hyper.cube_hyper.v_max * 1000 * velocity
        v_steps = hyper.cube_hyper.v_steps
        v = torch.linspace(v_min, v_max, v_steps, dtype=torch.float32)
        dv = (v_max - v_min) / (v_steps - 1)
        v_edges = torch.linspace(v_min - dv / 2, v_max + dv / 2, v_steps + 1)
        v_subsamples = hyper.observer_hyper.v_subsamples
        if v_subsamples == 0:
            fine_steps = v_steps
            v_fine = v
        else:
            fine_steps = 2 * v_steps * v_subsamples + 1
            v_fine = torch.linspace(v_min - dv / 2, v_max + dv / 2, fine_steps, dtype=torch.float32)

        nu_channel_width = dv / self.c * self.nu_ul
        self.nu_channel_width = torch.nn.Parameter(nu_channel_width, requires_grad=False)
        nu = self._doppler(self.nu_ul.unsqueeze(dim=1), v.unsqueeze(dim=0)).view(1, self.n_lines, 1, 1, 1, v_steps)
        self.nu = torch.nn.Parameter(nu, requires_grad=False)
        nu_edges = self._doppler(self.nu_ul.unsqueeze(dim=1), v_edges.unsqueeze(dim=0)).view(1, self.n_lines, 1, 1, 1, v_steps + 1)
        self.nu_edges = torch.nn.Parameter(nu_edges, requires_grad=False)
        nu_fine = self._doppler(self.nu_ul.unsqueeze(dim=1), v_fine.unsqueeze(dim=0)).view(1, self.n_lines, 1, 1, 1, fine_steps)
        self.nu_fine = torch.nn.Parameter(nu_fine, requires_grad=False)

        number_ism_molecular_mass = (hyper.observer_hyper.m_H
                              + hyper.observer_hyper.abundance_He
                              * hyper.observer_hyper.m_He)
        self.number_ism_molecular_mass = torch.nn.Parameter(
            torch.tensor(number_ism_molecular_mass * (mass * number_unit), dtype=torch.float32), requires_grad=False)

        bolic_normalization = torch.tensor(hyper.observer_hyper.bolic_normalization, dtype=torch.float32)
        self.bolic_normalization = torch.nn.Parameter(bolic_normalization * number_density, requires_grad=False)

        kappa_dust = torch.tensor(hyper.observer_hyper.kappa_dust,
                                  dtype=torch.float32) * length * length / mass
        self.kappa_dust = torch.nn.Parameter(
            kappa_dust.view(1, self.n_lines, 1, 1, 1, 1), requires_grad=False)
        self.j = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['emission_factor'], requires_grad=False) for i in
            range(self.n_lines)])
        self.dj_drho = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['dj_drho'], requires_grad=False) for i in
            range(self.n_lines)])
        self.dj_d_ab = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['dj_d_ab'], requires_grad=False) for i in
            range(self.n_lines)])
        self.dj_dT = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['dj_dT'], requires_grad=False) for i in range(self.n_lines)])
        self.alpha = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['absorption_factor'], requires_grad=False) for i in
            range(self.n_lines)])
        self.d_alpha_drho = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['d_alpha_drho'], requires_grad=False) for i in
            range(self.n_lines)])
        self.d_alpha_d_ab = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['d_alpha_d_ab'], requires_grad=False) for i in
            range(self.n_lines)])
        self.d_alpha_dT = torch.nn.ParameterList([
            torch.nn.Parameter(observability_grids[i]['d_alpha_dT'], requires_grad=False) for i in
            range(self.n_lines)])
        return
    
    def forward(self,
                inputs: torch.Tensor,
                inplace: bool = True,
                transfer: str = 'optically thick') -> torch.Tensor:
        """
        Computes emission and absorption coefficients of a physical tensor.

        Args:
            inputs: The [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor]
                for which observability coefficients will be determined. Has dimensions
                `batch, n_lines, r_steps, lon_steps, lat_steps`.
            inplace: If `True`, makes use of in-place optimizations, unless the differentiability mode
                requires these optimizations to be turned off.
            transfer: The transfer type. One of `'optically thick', 'selectively thin', 'optically thin'`.

        Returns:
            A tuple `j, alpha, j_dust, alpha_dust`. The line emission and absorption,
            `j, alpha`, have dimensions `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps`,
            where `v_steps` corresponds to the fine velocity grid in optically thick mode and the
            coarse velocity grid in optically thin and selectively thin mode. The dust emission
            `j_dust` has dimensions `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps`,
            where `v_steps` is always the coarse value. The dust absorption coefficient
            `alpha_dust` is taken to be constant in velocity over each separate line, and
            so has dimensions `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps=1`.
            In optically thin mode, `alpha` will be `None` and will not be computed. In order to
            compute `j_dust`, `alpha_dust` must still be computed, and so will still be returned.

        Raises:
            ValueError: If `transfer` is not one of
                `'optically thick', 'selectively thin', 'optically thin'`.
        """
        with nullcontext() if self.differentiable_input else torch.no_grad():
            v_r = inputs[:, 0, :, :, :].unsqueeze(dim=-1)
            rho = inputs[:, 1, :, :, :].unsqueeze(dim=-1)
            if inplace and not self.differentiable_abundance:
                T = inputs[:, 2, :, :, :].unsqueeze(dim=-1)
            else:
                T = inputs[:, 2, :, :, :].clone().unsqueeze(dim=-1)
            abundance_H2 = inputs[:, 3, :, :, :].unsqueeze(dim=-1)
            abundance_CO = inputs[:, 4, :, :, :].unsqueeze(dim=-1)
            T_dust = inputs[:, 5, :, :, :].unsqueeze(dim=-1)

            if transfer != 'optically thick' and transfer != 'selectively thin' and transfer != 'optically thin':
                raise ValueError("Invalid transfer provided to ObservabilityProcessor.forward. "
                                 "Must be one of 'optically thick', 'selectively thin', 'optically thin'.")

            N_H_TOT = rho / self.number_ism_molecular_mass
            T_not_finite = ~T.isfinite()
            j_per_abundance = []
            alpha_per_abundance = []
            for line in range(self.n_lines):
                j_per_abundance_per_line, alpha_per_abundance_per_line = self._emission_absorption_per_abundance(
                    v_r,
                    rho,
                    N_H_TOT,
                    abundance_H2,
                    T,
                    line,
                    transfer)
                j_per_abundance_per_line.masked_fill_(T_not_finite, 0)
                j_per_abundance.append(j_per_abundance_per_line)
                if transfer != 'optically thin':
                    alpha_per_abundance_per_line.masked_fill_(T_not_finite, 0)
                    alpha_per_abundance.append(alpha_per_abundance_per_line)

            j_per_abundance = torch.stack(j_per_abundance, dim=1)
            if transfer != 'optically thin':
                alpha_per_abundance = torch.stack(alpha_per_abundance, dim=1)
            alpha_dust = self.kappa_dust * rho.unsqueeze(dim=1)
            j_dust = self._dust_emission(alpha_dust,
                                         self._inverse_doppler(self.nu, v_r),
                                         T_dust.unsqueeze(dim=1),
                                         inplace=True)

            N_H_TOT.masked_fill_(T_not_finite, 0)
            T.masked_fill_(T_not_finite, 0)

        with nullcontext() if self.differentiable_abundance else torch.no_grad():
            tracer_abundances = self.abundance(N_H_TOT, T, abundance_H2, abundance_CO)
            if not self.differentiable_abundance:
                j = j_per_abundance.mul_(tracer_abundances)
                if transfer == 'optically thin':
                    alpha = None
                else:
                    alpha = alpha_per_abundance.mul_(tracer_abundances)
            else:
                j = j_per_abundance * tracer_abundances
                if transfer == 'optically thin':
                    alpha = None
                else:
                    alpha = alpha_per_abundance * tracer_abundances
        return j, alpha, j_dust, alpha_dust

    def _emission_absorption_per_abundance(self,
                                           v_r: torch.Tensor,
                                           rho: torch.Tensor,
                                           N_H_TOT: torch.Tensor,
                                           abundance_H2: torch.Tensor,
                                           T: torch.Tensor,
                                           line: int,
                                           transfer: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Computes the line emission and absorption coefficients, per tracer abundance, where
        tracer abundance is expressed as a fraction of total H atom number density.

        Performs a differentiable linear interpolation of the observability grids over
        each physical tensor cell. In optically thick mode, applies the Gaussian line profile
        derived from Doppler (thermal) broadening. In optically thin or selectively thin
        mode, applies the integrated line profile.

        Args:
            v_r: The velocity channel of the physical tensor.
            rho: The density channel of the physical tensor.
            N_H_TOT: The total H atom number density of the physical tensor, computed from `rho`.
            abundance_H2: The $\text{H}_2$ abundance channel of the physical tensor.
            T: The temperature channel of the physical tensor.
            line: The index of the line to be computed, in `range(n_lines)`.
            transfer: The transfer type. One of `'optically thick', 'selectively thin', 'optically thin'`.

        Returns:
            A tuple `j_per_abundance, alpha_per_abundance` of the per-abundance emission and
            absorption coefficients. In optically thin mode, `alpha_per_abundance` will be `None`.
        """
        if transfer == 'optically thick':
            nu_fine = self._inverse_doppler(self.nu_fine[:, line], v_r)
            profile = self._line_profile(nu_fine, T, line)
        else:
            nu_edges = self._inverse_doppler(self.nu_edges[:, line], v_r)
            profile = self._integrated_line_profile(nu_edges, T, line)

        rho_indices = (torch.asinh(N_H_TOT / self.bolic_normalization[line]) / self.dN_bolic[line] + .5).floor_().long()
        rho_indices = rho_indices.clamp_(min=0, max=self.rho[line].shape[0] - 1)
        ab_indices = (abundance_H2 / self.d_ab[line] + .5).floor_().long()
        ab_indices = ab_indices.clamp_(min=0, max=self.abundance_H2[line].shape[0] - 1)
        T_indices = (T / self.dT[line] + .5).floor_().long()
        T_indices = T_indices.clamp_(min=0, max=self.T[line].shape[0] - 1)

        dj_rho = self.dj_drho[line][rho_indices, ab_indices, T_indices] * (rho - self.rho[line][rho_indices])
        dj_ab = self.dj_d_ab[line][rho_indices, ab_indices, T_indices] * (abundance_H2 - self.abundance_H2[line][ab_indices])
        dj_T = self.dj_dT[line][rho_indices, ab_indices, T_indices] * (T - self.T[line][T_indices])
        dj = dj_rho + dj_ab + dj_T
        j_factor = self.j[line][rho_indices, ab_indices, T_indices] + dj
        j_per_abundance = j_factor * profile

        if transfer == 'optically thin':
            alpha_per_abundance = None
        else:
            d_alpha_rho = self.d_alpha_drho[line][rho_indices, ab_indices, T_indices] * (rho - self.rho[line][rho_indices])
            d_alpha_ab = self.d_alpha_d_ab[line][rho_indices, ab_indices, T_indices] * (abundance_H2 - self.abundance_H2[line][ab_indices])
            d_alpha_T = self.d_alpha_dT[line][rho_indices, ab_indices, T_indices] * (T - self.T[line][T_indices])
            d_alpha = d_alpha_rho + d_alpha_ab + d_alpha_T
            alpha_factor = self.alpha[line][rho_indices, ab_indices, T_indices] + d_alpha
            alpha_per_abundance = alpha_factor * profile

        return j_per_abundance, alpha_per_abundance

    def _line_profile(self, nu: torch.Tensor, T: torch.Tensor, line: int) -> torch.Tensor:
        """
        Computes a Gaussian line profile based on Doppler (thermal) broadening.

        Natural broadening (via the Lorentz profile or combined Voigt profile) is ignored,
        as it is typically only dominant at the profile wings.

        Args:
            nu: The frequencies over which to compute the line profile.
            T: The temperatures over which to compute the line profile.
            line: The index of the line to be computed, in `range(n_lines)`.

        Returns:
            The line profile tensor.
        """
        nu_ul = self.nu_ul[line]
        c = self.c
        k = self.k
        tracer_mass = self.tracer_mass[line]
        delta_nu_doppler = nu_ul / c * torch.sqrt(2 * k * T / tracer_mass)
        pi = torch.tensor(torch.pi, dtype=torch.float32, device=nu.device)
        profile = (torch.exp(-torch.square(nu - nu_ul) / torch.square(delta_nu_doppler)) /
                   delta_nu_doppler / torch.sqrt(pi))
        return profile

    def _integrated_line_profile(self, nu_edges: torch.Tensor, T: torch.Tensor, line: int) -> torch.Tensor:
        """
        Computes a Gaussian line profile based on Doppler (thermal) broadening,
        integral-averaged over specified frequency channel edges.

        This is the exact frequency integral of
        [`_line_profile`][iris.observation.ObservabilityProcessor._line_profile],
        normalized by the channel width of the coarse frequency grid. Note that the channel width
        is not computed dynamically from the edges, which are all assumed to be spaced equally
        according to the coarse frequency channel width.

        Args:
            nu_edges: The edges of the frequency channels over which to compute the integrated profile.
            T: The temperatures over which to compute the integrated profile.
            line: The index of the line to be computed, in `range(n_lines)`.

        Returns:
            The integral-averaged line profile tensor.
        """
        nu_ul = self.nu_ul[line]
        c = self.c
        k = self.k
        tracer_mass = self.tracer_mass[line]
        delta_nu_doppler = nu_ul / c * torch.sqrt(2 * k * T / tracer_mass)
        integrated_profile = torch.erf(
            (nu_edges - nu_ul) / delta_nu_doppler) / (2 * self.nu_channel_width)
        return integrated_profile[:, :, :, :, 1:] - integrated_profile[:, :, :, :, :-1]

    def _dust_emission(self,
                       alpha: torch.Tensor,
                       nu: torch.Tensor,
                       T: torch.Tensor,
                       inplace: bool = True) -> torch.Tensor:
        """
        Computes dust emission coefficients, given dust absorption coefficients,
        assuming a blackbody source function.

        Computes the blackbody source function via Planck's Law.

        Args:
            alpha: The dust absorption coefficients.
            nu: The frequencies over which to compute the dust emission coefficients.
            T: The temperatures over which to compute the dust emission coefficients.
            inplace: If `True`, will use in-place optimizations.

        Returns:
            The dust emission coefficients.
        """
        S = 2 * self.h * nu * nu * nu / self.c / self.c / torch.expm1(self.h * nu / self.k / T)
        S.masked_fill_(~torch.isfinite(S), 0)
        if inplace:
            j = S.mul_(alpha)
        else:
            j = S * alpha
        return j

    def _doppler(self, nu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the non-relativistic Doppler shift.

        In other words, computes the radiative frequency observed from an emitter moving with
        velocity `v` with respect to the observer and radiating at a frequency `nu`
        in the moving reference frame.

        Args:
            nu: The frequencies to be shifted.
            v: The velocities of the moving emitters relative to the observer frame.

        Returns:
            The Doppler-shifted frequencies.
        """
        c = self.c
        return (1 + v / c) * nu

    def _inverse_doppler(self, nu: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse of the non-relativistic Doppler shift.

        In other words, computes, in the emitter reference frame, the radiative frequency
        of an emission observed at a frequency `nu` from an emitter moving with
        velocity `v` with respect to the observer. In the relativistic longitudinal
        Doppler shift, the inverse shift is equivalent to a shift by the negative velocity.
        Because the non-relativistic approximation is used, the inverse and the negative
        doppler are slightly different, so the inverse is preferred for internal consistency.

        Args:
            nu: The frequencies to be shifted.
            v: The velocities of the moving emitters relative to the observer frame.

        Returns:
            The Doppler-shifted frequencies.
        """
        c = self.c
        return nu / (1 + v / c)

    def set_requires_grad_all(self) -> None:
        """
        Enables end-to-end differentiability.

        Sets `self.differentiable_input = True`
            and `self.differentiable_abundance = True`.
        """
        self.differentiable_input = True
        self.differentiable_abundance = True
        return

    def set_requires_grad_abundance(self) -> None:
        """
        Enables abundance-only differentiability.

        Sets `self.differentiable_input = False`
            and `self.differentiable_abundance = True`.
        """
        self.differentiable_input = False
        self.differentiable_abundance = True
        return

    def set_requires_grad_none(self) -> None:
        """
        Disables all differentiability.

        Sets `self.differentiable_input = False`
            and `self.differentiable_abundance = False`.
        """
        self.differentiable_input = False
        self.differentiable_abundance = False
        return


class TransferProcessor(torch.nn.Module):
    r"""
    A class for solving the radiative transfer equation.

    Computes a parallelized solution to the radiative transfer equation over all rays
    within one of the three transfer modes and one of two integration schemes. The integration
    schemes, specified to [`forward`][iris.observation.TransferProcessor.forward] at runtime
    via the arg `integration`, are: formal and smooth. In formal integration, the formal,
    exponential-form radiative transfer solution is applied assuming emission and absorption
    coefficients are cell-wise constant. In smooth integration, emission and absorption coefficients
    are treated as radial pixel-center values, and a smoothing is applied between values. The
    default behavior is smooth integration, which is 1.5 to 2 times as fast due to the elimination
    of transcendental operations (exp). The transfer modes, specified at runtime to
    [`forward`][iris.observation.TransferProcessor.forward] via the arg `transfer`, are:
    optically thick, selectively thin, and optically thin.

    In optically thick mode, line absorption and stimulated emission are fully modeled, as well
    as spontaneous emission of the line, thermal dust emission, and dust absorption. This requires
    that each velocity pixel in the output cube be subsampled down to the resolution of the
    line profile, and then numerically integrated in frequency over each velocity channel
    post-transfer. This is the most computationally intensive mode, because the velocity
    subsampling ultimately requires finer ray-batching to prevent a GPU OOM error. Additionally,
    an iteratively stepped solution is required (see below).

    In optically thin mode, only spontaneous emission of the line is computed, ignoring
    absorption and stimulated emission. Thermal dust emission is still computed to
    account for nonlinear continuum subtraction in brightness temperature space
    (not to be conflated with Raleigh-Jeans temperature space, see below),
    but dust absorption is also ignored. This enables several computational efficiencies.
    Memory and time are saved by not computing line absorption/stimulated emission.
    The ray solution simplifies to a fixed integral that is computed via a vectorized
    piecewise-constant integration in formal mode or Simpson's Rule in smooth mode, as opposed to
    the iteratively stepped solutions applied in optically thick and selectively thin mode.
    Lastly, the radiative transfer equation can be analytically integrated in frequency, up to
    the integral of the Gaussian line profile in terms of the standard error function (erf),
    which eliminates the need for post-transfer numerical integration in velocity.

    Selectively thin mode is a hybrid approximation that allows some optically thick behavior
    while still eliminating the need for velocity subsampling/numerical velocity integration.
    Specifically, the line is barred from self-interaction (absorption and stimulated emission)
    but can still absorb or be stimulated by the continuum, and can still itself be absorbed by
    dust. The physical motivation is that the line is assumed to be locally optically thin
    and Doppler-dispersed by large velocity gradients at non-local scale. This version of the
    radiative transfer equation requires an iteratively stepped ray solution (see below),
    but can also be analytically integrated in frequency, eliminating the need for numerical
    velocity integration.

    In optically thick or selectively thin mode, one of two iteratively stepped solution methods
    is applied. In formal integration, the formal, exponential-form solution is applied to each
    radial step assuming stepwise constancy of emission and absorption coefficients. In smooth
    integration, in which emission and absorption coefficients are treated as radial-step-center
    values rather than stepwise constant, the transfer equation is treated as a purely numerical
    ODE with no formal solution. In this case, care must be taken because the transfer ODE is a 
    stiff ODE, which is susceptible to spontaneous divergence, in particular when explicit
    solutions such as RK4 are applied. For stability, `TransferProcessor` implements the
    A-stable BDF2 method. This is an implicit method, which requires solution of an equation for
    each `r` step. Since, however, the radiative transfer equation is a linear ODE,
    this step equation has an algebraic solution that is hard-coded into `TransferProcessor`,
    and which yields time complexity equivalent to the application of an explicit method.
    In optically thick mode, the subsamples of each velocity channel are then integral-averaged
    via Simpson's Rule.

    Once independent solutions are attained to the continuum transfer and
    line + continuum transfer, two steps remain in transfer solving: optional conversion
    to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
    The order of these remaining steps is configurable. Conversion from intensity space 
    to brightness temperature space is computed via Planck's Law. Conversion from
    intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
    Approximation of Planck's Law. The specific temperature type should be chosen to match 
    the true observation processing pipeline. In continuum subtraction, the continuum cube is
    subtracted from the line + continuum cube to yield a true line observation. Post-transfer
    continuum subtraction is necessary due to multiple nonlinearities, which we describe in
    detail in the IRIS paper (subsec: Continuum Subtraction).

    [`TransferProcessor.forward`][iris.observation.TransferProcessor.forward] provides
    configurability of all permutations of the following options: whether the continuum
    is subtracted in intensity or brightness-temperature space, whether the continuum cube
    is the output, and whether the output cube is returned in intensity, brightness-temperature,
    or Raleigh-Jeans temperature units.
    The preferred mode for the user depends entirely on the true observation data processing 
    pipeline being modeled, but 
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
    and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
    Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
    specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
    is linear in intensity. In all cases, the output of `TransferProcessor`
    is a PPV cube with dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.

    All stages of transfer solving are fully differentiable. For efficiency,
    differentiability is turned off by default, but can be manually enabled via
    [set_requires_grad_all][iris.observation.TransferProcessor.set_requires_grad_all]. See
    [`SyntheticObserver`][iris.observation.SyntheticObserver] for more details on
    differentiability modes.

    Attributes:
        differentiable: If `True`, gradients can propagate through the `TransferProcessor`. See
            [`SyntheticObserver`][iris.observation.SyntheticObserver] for more details on
            differentiability modes.
        h: The Planck constant. A `torch.float32` scalar.
        k: The Boltzmann constant. A `torch.float32` scalar.
        c: The vacuum speed of light. A `torch.float32` scalar.
        intensity_Jy_per_Sr: Conversion factor from $\text{Jy}/\text{sr}$ to the intensity units
            dictated by the arg `units`. A `torch.float32` scalar.
        temperature: Conversion factor from K to the temperature units dictated by the arg `units`.
            A `torch.float32` scalar.
        ds: The `r` step size, labeled `ds` per convention. A `torch.float32` scalar.
        nu: The frequency channels (not subsampled) associated with the output velocity channels.
            A `torch.float32` tensor of dimensions `batch=1, n_lines, lon=1, lat=1, v_steps`.
        v_subsamples: The number of velocity samples per velocity channel. Only used in
            [`forward`][iris.observation.TransferProcessor.forward] if called with
            `transfer='optically thick'`. Note that each velocity sample has a left and right
            edge neighbor, per the Simpson's Rule integration used to compute the channel
            intensity mean. Edges coincide, but the sample point does not serve as the edge of
            a separate sample. There are `2 * v_steps * v_subsamples + 1` total velocity points
            in the fine velocity grid.
        background_intensity: The intensity of the CMB over all observed frequencies/velocities.
            A `torch.float32` tensor of dimensions `batch=1, n_lines, lon=1, lat=1, v_steps`.

    Args:
        observability_processor: The `ObservabilityProcessor` to which the `TransferProcessor` is coupled.
        hyper: A hyperparameters object.
        units: The units for input and computation. One of `'iris', 'processing'`. Not the same
            as the output units specified to
            [`forward`][iris.observation.TransferProcessor.forward].

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """

    differentiable: bool
    h: torch.nn.Parameter
    k: torch.nn.Parameter
    c: torch.nn.Parameter
    intensity_Jy_per_Sr: torch.nn.Parameter
    temperature: torch.nn.Parameter
    ds: torch.nn.Parameter
    nu: torch.nn.Parameter
    v_subsamples: int
    background_intensity: torch.nn.Parameter

    def __init__(self,
                 observability_processor: ObservabilityProcessor,
                 hyper: hp.Hyper,
                 units: str = 'iris') -> None:
        super().__init__()
        if units == 'iris':
            mass = hyper.dataset_hyper._mass_iris_per_SI
            time = hyper.dataset_hyper._time_iris_per_SI
            length = hyper.dataset_hyper._length_iris_per_SI
            length_per_parsec = hyper.dataset_hyper._length_iris_per_parsec
            temperature = hyper.dataset_hyper._temperature_iris_per_SI
            velocity = length / time
            acceleration = velocity / time
            force = mass * acceleration
            energy = force * length
        elif units == 'processing':
            length = 100 / hyper.writer_hyper.length_cm_per_processing
            length_per_parsec = 1 / hyper.writer_hyper._length_parsec_per_processing
            velocity = 100 / hyper.writer_hyper.velocity_cm_per_s_per_processing
            time = length / velocity
            mass = 1000 / hyper.writer_hyper.mass_g_per_processing
            temperature = 1 / hyper.writer_hyper.temperature_K_per_processing
            acceleration = velocity / time
            force = mass * acceleration
            energy = force * length
        else:
            raise ValueError("Invalid units provided to TransferProcessor. Must be 'iris' or 'processing'.")

        self.differentiable = False

        self.h = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.h * energy * time,
                         dtype=torch.float32), requires_grad=False)
        self.k = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.k * energy / temperature,
                         dtype=torch.float32), requires_grad=False)
        self.c = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.c * velocity,
                         dtype=torch.float32), requires_grad=False)

        self.intensity_Jy_per_Sr = torch.nn.Parameter(
            torch.tensor(mass / time / time * 1e-26,
                         dtype=torch.float32), requires_grad=False)
        self.temperature = torch.nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

        r_min = hyper.coordinate_hyper.r_min
        r_max = hyper.coordinate_hyper.r_max
        r_steps = hyper.coordinate_hyper.r_steps
        self.ds = torch.nn.Parameter(
            torch.tensor((r_max - r_min) / (r_steps - 1) * length_per_parsec,
                         dtype=torch.float32), requires_grad=False)

        self.nu = torch.nn.Parameter(observability_processor.nu.squeeze(dim=2), requires_grad=False)
        self.v_subsamples = hyper.observer_hyper.v_subsamples
        T_cmb = hyper.observer_hyper.T_cmb * self.temperature
        self.background_intensity = torch.nn.Parameter(
            self._brightness_temperature_to_intensity(self.nu, T_cmb), requires_grad=False)
        return
    
    def forward(self,
                inputs: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor],
                inplace: bool = True,
                subtraction: str = 'I',
                units: str = 'Trj',
                transfer: str = 'optically thick',
                integration: str = 'smooth') -> torch.Tensor:
        """
        A wrapper method for a radiative transfer solution.

        Based on the args `transfer` and `integration`, calls one of:

        * [`_optically_thick_transfer`][iris.observation.TransferProcessor._optically_thick_transfer],
        * [`_optically_thick_transfer_bdf2`][iris.observation.TransferProcessor._optically_thick_transfer_bdf2],
        * [`_selectively_thin_transfer`][iris.observation.TransferProcessor._selectively_thin_transfer],
        * [`_selectively_thin_transfer_bdf2`][iris.observation.TransferProcessor._selectively_thin_transfer_bdf2],
        * [`_optically_thin_transfer`][iris.observation.TransferProcessor._optically_thin_transfer], or
        * [`_optically_thin_transfer_simpson`][iris.observation.TransferProcessor._optically_thin_transfer_simpson].

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.
            transfer: The transfer mode. One of `'optically thick', 'selectively thin', 'optically thin'`.
            integration: The integration scheme. One of `'formal', 'smooth'`.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Raises:
            ValueError: If `transfer` is not one of `'optically thick', 'selectively thin', 'optically thin'`.
            ValueError: If `integration` is not one of `'formal', 'smooth'`.
        """
        if transfer == 'optically thick':
            if integration == 'formal':
                outputs = self._optically_thick_transfer_formal(inputs=inputs,
                                                                inplace=inplace,
                                                                subtraction=subtraction,
                                                                units=units)
            elif integration == 'smooth':
                outputs = self._optically_thick_transfer_bdf2(inputs=inputs,
                                                              inplace=inplace,
                                                              subtraction=subtraction,
                                                              units=units)
            else:
                raise ValueError("Invalid integration provided to TransferProcessor.forward. "
                                 "Must be one of 'formal', 'smooth'.")
        elif transfer == 'selectively thin':
            if integration == 'formal':
                outputs = self._selectively_thin_transfer_formal(inputs=inputs,
                                                                 inplace=inplace,
                                                                 subtraction=subtraction,
                                                                 units=units)
            elif integration == 'smooth':
                outputs = self._selectively_thin_transfer_bdf2(inputs=inputs,
                                                               inplace=inplace,
                                                               subtraction=subtraction,
                                                               units=units)
            else:
                raise ValueError("Invalid integration provided to TransferProcessor.forward. "
                                 "Must be one of 'formal', 'smooth'.")
        elif transfer == 'optically thin':
            if integration == 'formal':
                outputs = self._optically_thin_transfer_formal(inputs=inputs,
                                                               inplace=inplace,
                                                               subtraction=subtraction,
                                                               units=units)
            elif integration == 'smooth':
                outputs = self._optically_thin_transfer_simpson(inputs=inputs,
                                                                inplace=inplace,
                                                                subtraction=subtraction,
                                                                units=units)
            else:
                raise ValueError("Invalid integration provided to TransferProcessor.forward. "
                                 "Must be one of 'formal', 'smooth'.")
        else:
            raise ValueError("Invalid transfer provided to TransferProcessor.forward. "
                             "Must be one of 'optically thick', 'selectively thin', 'optically thin'.")
        return outputs
    
    def _optically_thick_transfer_formal(self,
                                         inputs: tuple[torch.Tensor,
                                                       torch.Tensor | None,
                                                       torch.Tensor,
                                                       torch.Tensor],
                                         inplace: bool = True,
                                         subtraction: str = 'I',
                                         units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes an optically thick radiative transfer solution via formal integration.

        In optically thick mode, line absorption and stimulated emission are fully modeled,
        as well as spontaneous emission of the line, thermal dust emission, and dust absorption.
        This requires that each velocity pixel in the output cube be subsampled down to the
        resolution of the line profile, and then numerically integrated in frequency over each
        velocity channel post-transfer. This is the most computationally intensive transfer mode,
        because the velocity subsampling ultimately requires finer ray-batching to prevent a
        GPU OOM error.

        The solution method is broken down into a number of distinct steps. First, independent
        solutions are computed for the continuum and line + continuum transfer equations. The
        continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, however, the line + continuum transfer
        may vary significantly over the coarse velocity channel width. Therefore, it is solved
        on a fine velocity grid with an upsampling ratio determined by `self.v_subsamples`.
        (If `v_subsamples` is 0, the fine grid reverts to the coarse grid.)

        These two separate radiative transfer equations take the form
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} - \alpha_\text{dust}I_\text{continuum} $$
        and
        $$ \frac{dI_\text{total}}{ds} = (j_\text{line} + j_\text{dust}) -
        (\alpha_\text{line} + \alpha_\text{dust})I_\text{total} \text{ .} $$
        These transfer equations are each solved via the formal, exponential-form solution assuming
        emission and absorption coefficients are radial-stepwise-constant. For low-optical-depth
        cells where the exponential-form solution is numerically unstable, the linear emission-only
        solution is applied. See the IRIS paper for more theoretical details.

        The solved line + continuum cube must then downsampled onto the coarse velocity grid by averaging
        the fine values over each coarse channel. Rather than computing a naive mean of discrete samples,
        an integral mean over each velocity channel is computed using Simpson's Rule. By construction
        of the fine velocity grid, the coarse channel width cancels from the computation, yielding
        a discrete Simpson mean that achieves substantially greater accuracy than a naive mean
        assuming an underlying spectrum that is smooth and locally quadratic on the scale of the fine
        grid. This Simpson mean is computed in an efficient, vectorized form.

        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer
        continuum subtraction is necessary due to multiple nonlinearities, which we describe in
        detail in the IRIS paper (subsec: Continuum Subtraction).

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = inplace and not self.differentiable
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h = self.ds
            batch, n_lines, r_steps, lon_steps, lat_steps, v_steps = j_dust.shape
            _, _, _, _, _, v_steps_fine = j.shape

            # Expand the coarse dust emission and background intensities onto the fine velocity grid.
            # The dust absorption need not be expanded, since it is constant over the entire
            # velocity extent of the cube, with a broadcasting v dimension of 1.
            v_subsamples = self.v_subsamples
            if v_subsamples > 0:
                j_dust_fine = j_dust.unsqueeze(-1).expand(
                    batch, n_lines, r_steps, lon_steps, lat_steps, v_steps, 2 * v_subsamples).reshape(
                    batch, n_lines, r_steps, lon_steps, lat_steps, v_steps_fine - 1)
                j_dust_fine = torch.cat((j_dust_fine, j_dust_fine[..., -1:]), dim=-1)
                background_intensity_fine = self.background_intensity.unsqueeze(-1).expand(
                    1, n_lines, 1, 1, v_steps, 2 * v_subsamples).reshape(
                    1, n_lines, 1, 1, v_steps_fine - 1)
                background_intensity_fine = torch.cat(
                    (background_intensity_fine, background_intensity_fine[..., -1:]), dim=-1)
            else:
                j_dust_fine = j_dust
                background_intensity_fine = self.background_intensity
            # Precompute partial terms in the formal step solution for efficiency.
            thin_epsilon = 1e-6
            if inplace:
                j += j_dust_fine
                h_j = h * j
                h_j_dust = h * j_dust
                alpha += alpha_dust
                S = j.div_(alpha)
                S_dust = j_dust.div_(alpha_dust)
                nd_tau = alpha.mul_(-h)
                nd_tau_dust = alpha_dust.mul_(-h)
                thin = torch.abs(nd_tau) < thin_epsilon
                thin_dust = torch.abs(nd_tau_dust) < thin_epsilon
                exp_d_tau = nd_tau.exp_()
                exp_d_tau_dust = nd_tau_dust.exp_()
            else:
                j = j + j_dust_fine
                h_j = h * j
                h_j_dust = h * j_dust
                alpha = alpha + alpha_dust
                S = j / alpha
                S_dust = j_dust / alpha_dust
                nd_tau = -h * alpha
                nd_tau_dust = -h * alpha_dust
                thin = torch.abs(nd_tau) < thin_epsilon
                thin_dust = torch.abs(nd_tau_dust) < thin_epsilon
                exp_d_tau = torch.exp(nd_tau)
                exp_d_tau_dust = torch.exp(nd_tau_dust)

            # Iterate over all radial steps.
            I_v = background_intensity_fine
            I_continuum = self.background_intensity
            for s in range(r_steps):
                I_v_exp = torch.nn.functional.relu(
                    (I_v - S[:, :, s, :, :, :]) * exp_d_tau[:, :, s, :, :, :] + S[:, :, s, :, :, :])
                I_v_lin = I_v + h_j[:, :, s, :, :, :]
                exp_bad = ~torch.isfinite(I_v_exp) | thin[:, :, s, :, :, :]
                I_v = torch.where(exp_bad, I_v_lin, I_v_exp)

                I_continuum_exp = torch.nn.functional.relu(
                    (I_continuum - S_dust[:, :, s, :, :, :]) * exp_d_tau_dust[:, :, s, :, :, :] +
                    S_dust[:, :, s, :, :, :])
                I_continuum_lin = I_continuum + h_j_dust[:, :, s, :, :, :]
                exp_bad = ~torch.isfinite(I_continuum_exp) | thin_dust[:, :, s, :, :, :]
                I_continuum = torch.where(exp_bad, I_continuum_lin, I_continuum_exp)

            # Downsample the line + continuum cube onto the coarse velocity grid.
            # Computes an integral mean via Simpson's Rule.
            # The coarse velocity channel width cancels. (Nifty!)
            if v_subsamples > 0:
                I_d_nu_normed = I_v[:, :, :, :, 1:-1:2].clone()
                I_d_nu_normed *= 4
                I_d_nu_normed += I_v[:, :, :, :, :-2:2]
                I_d_nu_normed += I_v[:, :, :, :, 2::2]
                I_d_nu_normed /= 6 * v_subsamples
                I = I_d_nu_normed.view(*I_continuum.shape, v_subsamples).sum(dim=-1)
            else:
                I = I_v

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs

    def _optically_thick_transfer_bdf2(self,
                                       inputs: tuple[torch.Tensor,
                                                     torch.Tensor | None,
                                                     torch.Tensor,
                                                     torch.Tensor],
                                       inplace: bool = True,
                                       subtraction: str = 'I',
                                       units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes an optically thick radiative transfer solution via BDF2.

        In optically thick mode, line absorption and stimulated emission are fully modeled,
        as well as spontaneous emission of the line, thermal dust emission, and dust absorption.
        This requires that each velocity pixel in the output cube be subsampled down to the
        resolution of the line profile, and then numerically integrated in frequency over each
        velocity channel post-transfer. This is the most computationally intensive transfer mode,
        because the velocity subsampling ultimately requires finer ray-batching to prevent a
        GPU OOM error.

        The solution method is broken down into a number of distinct steps. First, independent
        solutions are computed for the continuum and line + continuum transfer equations. The
        continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, however, the line + continuum transfer
        may vary significantly over the coarse velocity channel width. Therefore, it is solved
        on a fine velocity grid with an upsampling ratio determined by `self.v_subsamples`.
        (If `v_subsamples` is 0, the fine grid reverts to the coarse grid.)

        These two separate radiative transfer equations take the form
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} - \alpha_\text{dust}I_\text{continuum} $$
        and
        $$ \frac{dI_\text{total}}{ds} = (j_\text{line} + j_\text{dust}) -
        (\alpha_\text{line} + \alpha_\text{dust})I_\text{total} \text{ .} $$
        Both are stiff ODEs that are susceptible to spontaneous divergence when solved via an explicit
        method such as RK4. Instead, the first step is computed via the Trapezoidal Rule, and the
        remaining steps are computed by Second-Order Backwards Differentiation Formula (BDF2), which are
        both A-stable. These are both implicit methods, which define the step in terms of a step equation,
        and which in general may be nonlinear and require an iterative solution. Since the transfer 
        equations are linear, however, the step equations have explicit algebraic solutions that are 
        hard-coded into this function, such that the resulting step computation is equivalent in 
        efficiency to that dictated by an explicit method.
        
        The solved line + continuum cube must then downsampled onto the coarse velocity grid by averaging
        the fine values over each coarse channel. Rather than computing a naive mean of discrete samples,
        an integral mean over each velocity channel is computed using Simpson's Rule. By construction
        of the fine velocity grid, the coarse channel width cancels from the computation, yielding
        a discrete Simpson mean that achieves substantially greater accuracy than a naive mean
        assuming an underlying spectrum that is smooth and locally quadratic on the scale of the fine
        grid. This Simpson mean is computed in an efficient, vectorized form.

        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer
        continuum subtraction is necessary due to multiple nonlinearities, which we describe in
        detail in the IRIS paper (subsec: Continuum Subtraction).

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = inplace and not self.differentiable
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h = self.ds
            batch, n_lines, r_steps, lon_steps, lat_steps, v_steps = j_dust.shape
            _, _, _, _, _, v_steps_fine = j.shape

            # Expand the coarse dust emission and background intensities onto the fine velocity grid.
            # The dust absorption need not be expanded, since it is constant over the entire
            # velocity extent of the cube, with a broadcasting v dimension of 1.
            v_subsamples = self.v_subsamples
            if v_subsamples > 0:
                j_dust_fine = j_dust.unsqueeze(-1).expand(
                    batch, n_lines, r_steps, lon_steps, lat_steps, v_steps, 2 * v_subsamples).reshape(
                    batch, n_lines, r_steps, lon_steps, lat_steps, v_steps_fine - 1)
                j_dust_fine = torch.cat((j_dust_fine, j_dust_fine[..., -1:]), dim=-1)
                background_intensity_fine = self.background_intensity.unsqueeze(-1).expand(
                    1, n_lines, 1, 1, v_steps, 2 * v_subsamples).reshape(
                    1, n_lines, 1, 1, v_steps_fine - 1)
                background_intensity_fine = torch.cat(
                    (background_intensity_fine, background_intensity_fine[..., -1:]), dim=-1)
            else:
                j_dust_fine = j_dust
                background_intensity_fine = self.background_intensity
            # Precompute partial terms in the step equations for efficiency.
            if inplace:
                j += j_dust_fine
                alpha += alpha_dust
                h_j_continuum = j_dust.mul_(2 * h)
                h_alpha_continuum = alpha_dust.mul_(2 * h)
                h_j = j.mul_(2 * h)
                h_alpha = alpha.mul_(2 * h)
            else:
                h_j_continuum = 2 * h * j_dust
                h_alpha_continuum = 2 * h * alpha_dust
                h_j = 2 * h * (j + j_dust_fine)
                h_alpha = 2 * h * (alpha + alpha_dust)

            # Compute the first continuum step via the Trapezoidal Rule.
            I_continuum_last = self.background_intensity
            I_continuum = torch.nn.functional.relu(
                (h_j_continuum[:, :, 0, :, :, :] +
                 h_j_continuum[:, :, 1, :, :, :] +
                 (4 - h_alpha_continuum[:, :, 0, :, :, :]) * I_continuum_last) /
                (4 + h_alpha_continuum[:, :, 1, :, :, :]))

            # Compute the first line + continuum step via the Trapezoidal Rule.
            I_v_last = background_intensity_fine
            I_v = torch.nn.functional.relu(
                (h_j[:, :, 0, :, :, :] +
                 h_j[:, :, 1, :, :, :] +
                 (4 - h_alpha[:, :, 0, :, :, :]) * I_v_last) /
                (4 + h_alpha[:, :, 1, :, :, :]))

            # Iterate over remaining steps.
            for s in range(r_steps - 2):
                # Compute the continuum step via BDF2.
                I_continuum_next = torch.nn.functional.relu(
                    (h_j_continuum[:, :, s + 2, :, :, :] + 4 * I_continuum - I_continuum_last) /
                    (3 + h_alpha_continuum[:, :, s + 2, :, :, :]))

                # Compute the line + continuum step via BDF2.
                I_v_next = torch.nn.functional.relu(
                    (h_j[:, :, s + 2, :, :, :] + 4 * I_v - I_v_last) /
                    (3 + h_alpha[:, :, s + 2, :, :, :]))

                I_continuum_last, I_continuum = I_continuum, I_continuum_next
                I_v_last, I_v = I_v, I_v_next

            # Downsample the line + continuum cube onto the coarse velocity grid.
            # Computes an integral mean via Simpson's Rule.
            # The coarse velocity channel width cancels. (Nifty!)
            if v_subsamples > 0:
                I_d_nu_normed = I_v[:, :, :, :, 1:-1:2].clone()
                I_d_nu_normed *= 4
                I_d_nu_normed += I_v[:, :, :, :, :-2:2]
                I_d_nu_normed += I_v[:, :, :, :, 2::2]
                I_d_nu_normed /= 6 * v_subsamples
                I = I_d_nu_normed.view(*I_continuum.shape, v_subsamples).sum(dim=-1)
            else:
                I = I_v

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs

    def _selectively_thin_transfer_formal(self,
                                          inputs: tuple[torch.Tensor,
                                                        torch.Tensor | None,
                                                        torch.Tensor,
                                                        torch.Tensor],
                                          inplace: bool = True,
                                          subtraction: str = 'I',
                                          units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes a selectively thin radiative transfer solution via formal integration.

        Selectively thin mode is a hybrid approximation that allows some optically thick behavior
        while still eliminating the need for velocity subsampling/numerical velocity integration.
        Specifically, the line is barred from self-interaction (absorption and stimulated emission)
        but can still absorb or be stimulated by the continuum, and can still itself be absorbed by
        dust. The physical motivation is that the line is assumed to be locally optically thin
        and Doppler-dispersed by large velocity gradients at non-local scale. This version of the
        radiative transfer equation still requires an iteratively stepped ray solution,
        but can also be analytically integrated in frequency/velocity, up to the integral of the
        Gaussian line profile in terms of the standard error function (erf), which eliminates the need
        for post-transfer numerical integration in velocity. Note that optically thick transfer
        is still faster, on average, than selectively thin transfer when `self.v_subsamples` is `0`,
        but becomes slower than selectively thin transfer for `v_subsamples > 0` with increasing
        performance loss when implemented within an
        [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver].

        Independent solutions are computed for the continuum and line + continuum transfer equations.
        The continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, the line + continuum transfer
        may vary significantly over the coarse velocity channel width, but the line emission
        and absorption coefficients are analytically integrated in frequency in this mode by the
        [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]. This analytic integration
        is fine under the restricted assumption of non-self-interaction of the line, in which case
        the transfer equations become
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} - \alpha_\text{dust}I_\text{continuum} $$
        and
        $$ \frac{d\Bar{I}_\text{total}}{ds} = (\Bar{j}_\text{line} + j_\text{dust}) -
        \Bar{\alpha}_\text{line}I_\text{continuum} - \alpha_\text{dust}\Bar{I}_\text{total} \text{ .} $$
        This is a system of two ODEs that are solved in tandem via the formal, exponential-form 
        solution assuming emission and absorption coefficients are radial-stepwise-constant. 
        For low-optical-depth cells where the exponential-form solution is numerically unstable, 
        the linear emission-only solution is applied. See the IRIS paper for more theoretical details.

        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer
        continuum subtraction is necessary due to multiple nonlinearities, which we describe in
        detail in the IRIS paper (subsec: Continuum Subtraction).

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = inplace and not self.differentiable
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h = self.ds
            batch, n_lines, r_steps, lon_steps, lat_steps, v_steps = j.shape

            # Precompute partial terms in the formal step solution for efficiency.
            thin_epsilon = 1e-6
            if inplace:
                j += j_dust
                h_j = h * j
                h_j_dust = h * j_dust
                alpha += alpha_dust
                alpha_ratio = alpha.div_(alpha_dust)
                S = j.div_(alpha_dust)
                S_dust = j_dust.div_(alpha_dust)
                nd_tau_dust = alpha_dust.mul_(-h)
                thin_dust = torch.abs(nd_tau_dust) < thin_epsilon
                exp_d_tau_dust = nd_tau_dust.exp_()
            else:
                j = j + j_dust
                h_j = h * j
                h_j_dust = h * j_dust
                alpha = alpha + alpha_dust
                alpha_ratio = alpha / alpha_dust
                S = j / alpha_dust
                S_dust = j_dust / alpha_dust
                nd_tau_dust = -h * alpha_dust
                thin_dust = torch.abs(nd_tau_dust) < thin_epsilon
                exp_d_tau_dust = torch.exp(nd_tau_dust)

            # Iterate over all radial steps.
            I = self.background_intensity
            I_continuum = self.background_intensity
            for s in range(r_steps):
                I_continuum_exp = torch.nn.functional.relu(
                    (I_continuum - S_dust[:, :, s, :, :, :]) * exp_d_tau_dust[:, :, s, :, :, :] +
                    S_dust[:, :, s, :, :, :])
                I_continuum_lin = I_continuum + h_j_dust[:, :, s, :, :, :]
                exp_bad = ~torch.isfinite(I_continuum_exp) | thin_dust[:, :, s, :, :, :]
                I_continuum = torch.where(exp_bad, I_continuum_lin, I_continuum_exp)

                S[:, :, s, :, :, :] -= alpha_ratio[:, :, s, :, :, :] * I_continuum
                I_exp = torch.nn.functional.relu(
                    (I - S[:, :, s, :, :, :]) * exp_d_tau_dust[:, :, s, :, :, :] +
                    S[:, :, s, :, :, :])
                I_lin = I + h_j[:, :, s, :, :, :]
                exp_bad = ~torch.isfinite(I_exp) | thin_dust[:, :, s, :, :, :]
                I = torch.where(exp_bad, I_lin, I_exp)

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs

    def _selectively_thin_transfer_bdf2(self,
                                        inputs: tuple[torch.Tensor,
                                                      torch.Tensor | None,
                                                      torch.Tensor,
                                                      torch.Tensor],
                                        inplace: bool = True,
                                        subtraction: str = 'I',
                                        units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes a selectively thin radiative transfer solution via BDF2.

        Selectively thin mode is a hybrid approximation that allows some optically thick behavior
        while still eliminating the need for velocity subsampling/numerical velocity integration.
        Specifically, the line is barred from self-interaction (absorption and stimulated emission)
        but can still absorb or be stimulated by the continuum, and can still itself be absorbed by
        dust. The physical motivation is that the line is assumed to be locally optically thin
        and Doppler-dispersed by large velocity gradients at non-local scale. This version of the
        radiative transfer equation still requires an iteratively stepped ray solution,
        but can also be analytically integrated in frequency/velocity, up to the integral of the
        Gaussian line profile in terms of the standard error function (erf), which eliminates the need
        for post-transfer numerical integration in velocity. Note that optically thick transfer
        is still faster, on average, than selectively thin transfer when `self.v_subsamples` is `0`,
        but becomes slower than selectively thin transfer for `v_subsamples > 0` with increasing
        performance loss when implemented within an
        [`IteratedSyntheticObserver`][iris.observation.IteratedSyntheticObserver].

        Independent solutions are computed for the continuum and line + continuum transfer equations.
        The continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, the line + continuum transfer
        may vary significantly over the coarse velocity channel width, but the line emission
        and absorption coefficients are analytically integrated in frequency in this mode by the
        [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]. This analytic integration
        is fine under the restricted assumption of non-self-interaction of the line, in which case
        the transfer equations become
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} - \alpha_\text{dust}I_\text{continuum} $$
        and
        $$ \frac{d\Bar{I}_\text{total}}{ds} = (\Bar{j}_\text{line} + j_\text{dust}) -
        \Bar{\alpha}_\text{line}I_\text{continuum} - \alpha_\text{dust}\Bar{I}_\text{total} \text{ .} $$

        This is a system of two ODEs that are solved in tandem. Both are stiff equations that
        are each susceptible to spontaneous divergence when solved via an explicit method such as RK4.
        Instead, the first step is computed via the Trapezoidal Rule, and the
        remaining steps are computed by Second-Order Backwards Differentiation Formula (BDF2), which are
        both A-stable. These are both implicit methods, which define the step in terms of a step equation,
        and which in general may be nonlinear and require an iterative solution. Since the transfer
        equations are linear, however, the step equations have explicit algebraic solutions that are
        hard-coded into this function, such that the resulting step computation is equivalent in
        efficiency to that dictated by an explicit method.

        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer
        continuum subtraction is necessary due to multiple nonlinearities, which we describe in
        detail in the IRIS paper (subsec: Continuum Subtraction).

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        inplace = inplace and not self.differentiable
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h = self.ds
            batch, n_lines, r_steps, lon_steps, lat_steps, v_steps = j.shape

            # Precompute partial terms in the step equations for efficiency and
            # compute the first continuum step via the Trapezoidal Rule.
            h_j_continuum = 2 * h * j_dust
            h_alpha_continuum = 2 * h * alpha_dust
            I_continuum_last = self.background_intensity
            I_continuum = torch.nn.functional.relu(
                (h_j_continuum[:, :, 0, :, :, :] +
                 h_j_continuum[:, :, 1, :, :, :] +
                 (4 - h_alpha_continuum[:, :, 0, :, :, :]) * I_continuum_last) /
                (4 + h_alpha_continuum[:, :, 1, :, :, :]))

            # Precompute partial terms in the step equations for efficiency and
            # compute the first line + continuum step via the Trapezoidal Rule.
            if inplace:
                f = j.add_(j_dust)
            else:
                f = j + j_dust
            f[:, :, 0, :, :, :] -= alpha[:, :, 0, :, :, :] * I_continuum_last
            f[:, :, 1, :, :, :] -= alpha[:, :, 1, :, :, :] * I_continuum
            I_last = self.background_intensity
            I = torch.nn.functional.relu(
                ((4 - h_alpha_continuum[:, :, 0, :, :, :]) * I_last +
                 2 * h * (f[:, :, 0, :, :, :] + f[:, :, 1, :, :, :])) /
                (4 + h_alpha_continuum[:, :, 1, :, :, :]))

            # Iterate over remaining steps.
            for s in range(r_steps - 2):
                # Compute the continuum step via BDF2.
                I_continuum_next = torch.nn.functional.relu(
                    (h_j_continuum[:, :, s + 2, :, :, :] + 4 * I_continuum - I_continuum_last) /
                    (3 + h_alpha_continuum[:, :, s + 2, :, :, :]))

                # Compute the line + continuum step via BDF2.
                f[:, :, s + 2, :, :, :] -= alpha[:, :, s + 2, :, :, :] * I_continuum_next
                I_next = torch.nn.functional.relu(
                    (2 * h * f[:, :, s + 2, :, :, :] + 4 * I - I_last) /
                    (3 + h_alpha_continuum[:, :, s + 2, :, :, :]))

                I_continuum_last, I_continuum = I_continuum, I_continuum_next
                I_last, I = I, I_next

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs
    
    def _optically_thin_transfer_formal(self,
                                        inputs: tuple[torch.Tensor,
                                                      torch.Tensor | None,
                                                      torch.Tensor,
                                                      torch.Tensor],
                                        inplace: bool = True,
                                        subtraction: str = 'I',
                                        units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes an optically thin radiative transfer solution with radial-stepwise-constant integration.

        In optically thin mode, only spontaneous emission of the line is computed, ignoring
        absorption and stimulated emission. Thermal dust emission is still computed to
        account for nonlinear continuum subtraction in brightness temperature space (see below),
        but dust absorption is also ignored. This enables several computational efficiencies.
        Memory and time are saved by not computing line absorption/stimulated emission.
        The ray solution simplifies to a fixed integral that is computed via a vectorized
        Simpson's Rule as opposed to an iterative method. Lastly, the radiative transfer
        equation can be analytically integrated in frequency, up to the integral of the Gaussian
        line profile in terms of the standard error function (erf), which eliminates the need
        for post-transfer numerical integration in velocity.

        Independent solutions are computed for the continuum and line + continuum transfer equations.
        The continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, the line + continuum transfer
        may vary significantly over the coarse velocity channel width, but the line emission
        coefficient is analytically integrated in frequency in this mode by the
        [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]. The optically thin
        transfer equations then become
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} $$
        and
        $$ \frac{d\Bar{I}_\text{total}}{ds} = \Bar{j}_\text{line} + j_\text{dust} \text{ .} $$
        Both equations are directly integrated as cell-wise constant.

        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer 
        continuum subtraction is still necessary, even without absorption or stimulated emission, 
        if subtraction is computed in brightness temperature space since Planck's Law 
        (unlike the Raleigh-Jeans Approximation to Planck's Law) is nonlinear in intensity.

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: If `True` and not `self.differentiable`,
                will leverage in-place optimizations to conserve GPU memory.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h = self.ds

            # Integrate the line transfer as stepwise constant.
            if inplace:
                dI_line = j.mul_(h)
            else:
                dI_line = j * h
            I_line = torch.sum(dI_line, dim=2)

            # Integrate the continuum transfer as stepwise constant.
            if inplace:
                dI_continuum = j_dust.mul_(h)
            else:
                dI_continuum = j_dust * h
            I_continuum = torch.sum(dI_continuum, dim=2).add_(self.background_intensity)

            # Generate the line + continuum cube in intensity space.
            I = I_line.add_(I_continuum)

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs

    def _optically_thin_transfer_simpson(self,
                                         inputs: tuple[torch.Tensor,
                                                       torch.Tensor | None,
                                                       torch.Tensor,
                                                       torch.Tensor],
                                         inplace: bool = True,
                                         subtraction: str = 'I',
                                         units: str = 'Trj') -> torch.Tensor:
        r"""
        Computes an optically thin radiative transfer solution with Simpson smooth integration.

        In optically thin mode, only spontaneous emission of the line is computed, ignoring
        absorption and stimulated emission. Thermal dust emission is still computed to
        account for nonlinear continuum subtraction in brightness temperature space (see below),
        but dust absorption is also ignored. This enables several computational efficiencies.
        Memory and time are saved by not computing line absorption/stimulated emission.
        The ray solution simplifies to a fixed integral that is computed via a vectorized
        Simpson's Rule as opposed to an iterative method. Lastly, the radiative transfer
        equation can be analytically integrated in frequency, up to the integral of the Gaussian
        line profile in terms of the standard error function (erf), which eliminates the need
        for post-transfer numerical integration in velocity.

        Independent solutions are computed for the continuum and line + continuum transfer equations.
        The continuum is solved over the coarse velocity grid. Because the continuum can be treated
        as constant over a velocity channel width, the continuum solution at the channel centers
        can also be taken to be the channel-averaged solution. Depending on the scale of the coarse
        velocity grid with respect to the average line width, the line + continuum transfer
        may vary significantly over the coarse velocity channel width, but the line emission
        coefficient is analytically integrated in frequency in this mode by the
        [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor]. The optically thin
        transfer equations then become
        $$ \frac{dI_\text{continuum}}{ds} = j_\text{dust} $$
        and
        $$ \frac{d\Bar{I}_\text{total}}{ds} = \Bar{j}_\text{line} + j_\text{dust} \text{ .} $$
        Both equations are directly integrated via a fast, vectorized Simpson's Rule.
        
        Following independent solution of the continuum and line + continuum cubes over the coarse
        velocity grid, two steps remain in the transfer computation: optional conversion
        to brightness temperature or Raleigh-Jeans temperature, and continuum subtraction. 
        The order of these remaining steps is configurable. Conversion from intensity space 
        to brightness temperature space is computed via Planck's Law. Conversion from
        intensity space to Raleigh-Jeans temperature space is computed via the Raleigh-Jeans
        Approximation of Planck's Law. The specific temperature type should be chosen to match 
        the true observation processing pipeline. In continuum subtraction, the continuum cube is
        subtracted from the line + continuum cube to yield a true line observation. Post-transfer 
        continuum subtraction is still necessary, even without absorption or stimulated emission, 
        if subtraction is computed in brightness temperature space since Planck's Law 
        (unlike the Raleigh-Jeans Approximation to Planck's Law) is nonlinear in intensity.

        Configurability of all permutations of the following options is provided by the args
        `subtraction` and `units`: whether the continuum
        is subtracted in intensity or brightness-temperature space, whether the continuum cube
        is the output, and whether the output cube is returned in intensity, brightness-temperature,
        or Raleigh-Jeans temperature units.
        The preferred mode for the user depends entirely on the true observation data processing 
        pipeline being modeled, but 
        [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset]
        and [`Reverter`][iris.reversion.Reverter] currently support only temperature units. 
        Note that if subtraction in Raleigh-Jeans temperature space is desired, the user should
        specify subtraction in intensity space, which is equivalent since Raleigh-Jeans temperature
        is linear in intensity. In all cases, the output is a PPV cube with dimensions 
        `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Args:
            inputs: The observability variables (emission and absorption coefficients).
                A tuple of `j, alpha, j_dust, alpha_dust`.
            inplace: In optically thin Simpson mode, there are no non-differentiable in-place
                optimizations used. (In-place optimizations are used, but they are fully
                differentiable.) This arg is retained only for consistency with
                [`_optically_thick_transfer_formal`][iris.observation.TransferProcessor._optically_thick_transfer_formal],
                [`_optically_thick_transfer_bdf2`][iris.observation.TransferProcessor._optically_thick_transfer_bdf2],
                [`_selectively_thin_transfer_formal`][iris.observation.TransferProcessor._selectively_thin_transfer_formal]
                [`_selectively_thin_transfer_bdf2`][iris.observation.TransferProcessor._selectively_thin_transfer_bdf2].
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is subtracted from
                the full cube in brightness temperature space. If `'I'`, the continuum cube is
                subtracted from the full cube in intensity space. The subtracted cubes will differ
                since Planck's Law, according to which intensities are converted to brightness
                temperatures, is nonlinear outside the Rayleigh-Jeans regime. If `'continuum'`,
                this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'I', 'I Jy per Sr'`.
                If `'Tb'`, the output is returned as a PPV cube of brightness temperature in whatever
                units are provided to
                [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'Tb K'`, the output is returned as a PPV cube of brightness temperature in K.
                If `'I'`, the output is returned as a PPV cube of intensity in whatever units are provided
                to [`TransferProcessor.__init__`][iris.observation.TransferProcessor]--one of
                `'iris', 'processing'`.
                If `'I Jy per Sr'`, the output is returned as a PPV cube of intensity in $\text{Jy}/\text{sr}$.

        Returns:
            The observed PPV cube or cubes. Has dimensions `batch, n_lines, lon_steps, lat_steps, v_steps`.
        """
        with nullcontext() if self.differentiable else torch.no_grad():
            j, alpha, j_dust, alpha_dust = inputs
            h_3 = self.ds / 3

            # Integrate the line transfer via Simpson's Rule.
            dI_line = j[:, :, 1:-1:2, :, :, :].clone()
            dI_line *= 4
            dI_line += j[:, :, :-2:2, :, :, :]
            dI_line += j[:, :, 2::2, :, :, :]
            dI_line *= h_3
            I_line = torch.sum(dI_line, dim=2)

            # Integrate the continuum transfer via Simpson's Rule.
            dI_continuum = j_dust[:, :, 1:-1:2, :, :, :].clone()
            dI_continuum *= 4
            dI_continuum += j_dust[:, :, :-2:2, :, :, :]
            dI_continuum += j_dust[:, :, 2::2, :, :, :]
            dI_continuum *= h_3
            I_continuum = torch.sum(dI_continuum, dim=2).add_(self.background_intensity)

            # Generate the line + continuum cube in intensity space.
            I = I_line.add_(I_continuum)

            outputs = self._subtract_and_convert(I=I,
                                                 I_continuum=I_continuum,
                                                 subtraction=subtraction,
                                                 units=units)
        return outputs

    def _subtract_and_convert(self,
                              I: torch.Tensor,
                              I_continuum: torch.Tensor,
                              subtraction: str = 'I',
                              units: str = 'Trj') -> torch.Tensor:
        r"""
        Applies the post-transfer subtraction mode and converts output units.

        Args:
            I: The line + continuum intensity cube. Has dimensions
                `batch, n_lines, lon_steps, lat_steps, v_steps`.
            I_continuum: The continuum intensity cube. Has dimensions
                `batch, n_lines, lon_steps, lat_steps, v_steps`.
            subtraction: One of `'Tb', 'I', 'continuum'`. If `'Tb'`, the continuum cube is
                subtracted from the full cube in brightness temperature space. If `'I'`, the
                continuum cube is subtracted from the full cube in intensity space. If
                `'continuum'`, this option outputs the continuum cube.
            units: The output units. One of `'Tb', 'Tb K', 'Trj', 'Trj K', 'I', 'I Jy per Sr'`.

        Returns:
            The subtracted or continuum-only PPV cube in the requested units. Has dimensions
            `batch, n_lines, lon_steps, lat_steps, v_steps`.

        Raises:
            ValueError: If `subtraction` is not one of `'Tb', 'I', 'continuum'`.
            ValueError: If `units` is not one of `'Tb', 'Tb K', 'Trj', 'Trj K', 'I', 'I Jy per Sr'`.
        """
        if subtraction == 'I':
            I -= I_continuum
            if units == 'I':
                outputs = I
            elif units == 'I Jy per Sr':
                outputs = self.to_Jy_per_Sr(I, inplace=True)
            elif units == 'Trj':
                outputs = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
            elif units == 'Trj K':
                Trj = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
                outputs = self.to_K(Trj, inplace=True)
            elif units == 'Tb':
                outputs = self._intensity_to_brightness_temperature(self.nu, I)
            elif units == 'Tb K':
                Tb = self._intensity_to_brightness_temperature(self.nu, I)
                outputs = self.to_K(Tb, inplace=True)
            else:
                raise ValueError("Invalid units provided to TransferProcessor._subtract_and_convert. "
                                 "Must be one of 'Tb', 'Tb K', 'Trj', 'Trj K', 'I', 'I Jy per Sr'.")
        elif subtraction == 'Tb':
            Tb = self._intensity_to_brightness_temperature(self.nu, I)
            Tb_continuum = self._intensity_to_brightness_temperature(self.nu, I_continuum)
            Tb -= Tb_continuum
            if units == 'Tb':
                outputs = Tb
            elif units == 'Tb K':
                outputs = self.to_K(Tb, inplace=True)
            elif units == 'Trj':
                I = self._brightness_temperature_to_intensity(self.nu, Tb)
                outputs = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
            elif units == 'Trj K':
                I = self._brightness_temperature_to_intensity(self.nu, Tb)
                Trj = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
                outputs = self.to_K(Trj, inplace=True)
            elif units == 'I':
                outputs = self._brightness_temperature_to_intensity(self.nu, Tb)
            elif units == 'I Jy per Sr':
                I = self._brightness_temperature_to_intensity(self.nu, Tb)
                outputs = self.to_Jy_per_Sr(I, inplace=True)
            else:
                raise ValueError("Invalid units provided to TransferProcessor._subtract_and_convert. "
                                 "Must be one of 'Tb', 'Tb K', 'Trj', 'Trj K', 'I', 'I Jy per Sr'.")
        elif subtraction == 'continuum':
            I = I_continuum
            if units == 'I':
                outputs = I
            elif units == 'I Jy per Sr':
                outputs = self.to_Jy_per_Sr(I, inplace=True)
            elif units == 'Trj':
                outputs = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
            elif units == 'Trj K':
                Trj = self._intensity_to_raleigh_jeans_temperature(self.nu, I)
                outputs = self.to_K(Trj, inplace=True)
            elif units == 'Tb':
                outputs = self._intensity_to_brightness_temperature(self.nu, I)
            elif units == 'Tb K':
                Tb = self._intensity_to_brightness_temperature(self.nu, I)
                outputs = self.to_K(Tb, inplace=True)
            else:
                raise ValueError("Invalid units provided to TransferProcessor._subtract_and_convert. "
                                 "Must be one of 'Tb', 'Tb K', 'Trj', 'Trj K', 'I', 'I Jy per Sr'.")
        else:
            raise ValueError("Invalid subtraction mode provided to TransferProcessor._subtract_and_convert. "
                             "Must be one of 'Tb', 'I', 'continuum'.")
        return outputs

    def _brightness_temperature_to_intensity(self, nu: torch.Tensor, Tb: torch.Tensor) -> torch.Tensor:
        """
        Converts brightness temperature to intensity via Planck's Law.

        Brightness temperature may be alternately defined via either Planck's Law
        or the Raleigh-Jeans Approximation, which is a linearization of Planck's Law.
        This function uses the Planck definition. For the Raleigh-Jeans equivalent, use
        [`_raleigh_jeans_temperature_to_intensity`][iris.observation.TransferProcessor._raleigh_jeans_temperature_to_intensity].
        While the results can differ substantially at low temperatures, the Raleigh-Jeans
        temperature is sometimes preferred as the final output units of true observations.
        Therefore, the definition used should be chosen to match that of the true telescope data.
        
        Args:
            nu: The frequency channels of the brightness temperature tensor.
            Tb: The brightness temperature to convert to intensity.
        
        Returns:
            The converted intensity tensor.
        """
        I = 2 * self.h * nu * nu * nu / self.c / self.c / torch.expm1(self.h * nu / self.k / Tb)
        I = torch.where(torch.isfinite(I), I, 0)
        return I

    def _raleigh_jeans_temperature_to_intensity(self, nu: torch.Tensor, Trj: torch.Tensor) -> torch.Tensor:
        """
        Converts brightness temperature to intensity via the Raleigh-Jeans Approximation.

        Brightness temperature may be alternately defined via either Planck's Law
        or the Raleigh-Jeans Approximation, which is a linearization of Planck's Law.
        This function uses the Raleigh-Jeans definition. For the Planck equivalent, use
        [`_brightness_temperature_to_intensity`][iris.observation.TransferProcessor._brightness_temperature_to_intensity].
        While the results can differ substantially at low temperatures, the Raleigh-Jeans
        temperature is sometimes preferred as the final output units of true observations.
        Therefore, the definition used should be chosen to match that of the true telescope data.

        Args:
            nu: The frequency channels of the brightness temperature tensor.
            Tb: The brightness temperature to convert to intensity.

        Returns:
            The converted intensity tensor.
        """
        I = 2 * self.k * nu * nu * Trj / self.c / self.c
        return I

    def _intensity_to_brightness_temperature(self, nu: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Converts intensity to brightness temperature via Planck's Law.

        Brightness temperature may be alternately defined via either Planck's Law
        or the Raleigh-Jeans Approximation, which is a linearization of Planck's Law.
        This function uses the Planck definition. For the Raleigh-Jeans equivalent, use
        [`_intensity_to_raleigh_jeans_temperature`][iris.observation.TransferProcessor._intensity_to_raleigh_jeans_temperature].
        While the results can differ substantially at low temperatures, the Raleigh-Jeans
        temperature is sometimes preferred as the final output units of true observations.
        Therefore, the definition used should be chosen to match that of the true telescope data.

        Args:
            nu: The frequency channels of the intensity tensor.
            Tb: The intensity to convert to brightness temperature.

        Returns:
            The converted brightness temperature tensor.
        """
        Tb = self.h * nu / self.k / torch.log1p(2 * self.h * nu * nu * nu / I / self.c / self.c)
        Tb = torch.where(torch.isfinite(Tb), Tb, 0)
        return Tb

    def _intensity_to_raleigh_jeans_temperature(self, nu: torch.Tensor, I: torch.Tensor) -> torch.Tensor:
        """
        Converts intensity to brightness temperature via the Raleigh-Jeans Approximation.

        Brightness temperature may be alternately defined via either Planck's Law
        or the Raleigh-Jeans Approximation, which is a linearization of Planck's Law.
        This function uses the Raleigh-Jeans definition. For the Planck equivalent, use
        [`_intensity_to_brightness_temperature`][iris.observation.TransferProcessor._intensity_to_brightness_temperature].
        While the results can differ substantially at low temperatures, the Raleigh-Jeans
        temperature is sometimes preferred as the final output units of true observations.
        Therefore, the definition used should be chosen to match that of the true telescope data.

        Args:
            nu: The frequency channels of the intensity tensor.
            Tb: The intensity to convert to brightness temperature.

        Returns:
            The converted brightness temperature tensor.
        """
        Trj = self.c * self.c * I / (2 * self.k * nu * nu)
        return Trj

    def to_Jy_per_Sr(self, I: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        r"""
        Converts an intensity tensor in the internal units specified to
        [`TransferProcessor.__init__`][iris.observation.TransferProcessor]
        into $\text{Jy}/\text{sr}$.
            
        Args:
            I: The intensity tensor to convert.
            inplace: If `True`, performs the conversion in place.
            
        Returns:
            The converted intensity tensor in $\text{Jy}/\text{sr}$.
        """
        if inplace:
            I /= self.intensity_Jy_per_Sr
            return I
        else:
            return I / self.intensity_Jy_per_Sr

    def to_K(self, T: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """
        Converts a temperature tensor in the internal units specified to
        [`TransferProcessor.__init__`][iris.observation.TransferProcessor] into K.

        Args:
            T: The temperature tensor to convert.
            inplace: If `True`, performs the conversion in place.

        Returns:
            The converted temperature tensor in K
        """
        if inplace:
            T /= self.temperature
            return T
        else:
            return T / self.temperature
        
    def set_requires_grad_all(self) -> None:
        """
        Enables differentiability.

        Sets `self.differentiable = True`.
        """
        self.differentiable = True
        return

    def set_requires_grad_none(self) -> None:
        """
        Disables differentiability.

        Sets `self.differentiable = False`.
        """
        self.differentiable = False
        return


class VelocityBlur(torch.nn.Module):
    """
    Applies a Gaussian convolution over the velocity channel of a
    [physical tensor][iris.arepo_processing.make_physical_tensor].
        
    Intended use is as an [`Observer.in_blur`][iris.observation.Observer] applied to
    a physical tensor prior to observation. The nearest-neighbor
    [interpolation][iris.arepo_processing.Snapshot._interpolate]
    scheme employed during AREPO snapshot processing yields regions of constant velocity, which
    appear in an observation as flat streaks with jump discontinuities at the cell boundaries.
    By smoothing the velocity transitions at cell boundaries, `VelocityBlur` yields smoother
    observations for more reliable [`reverter`][iris.reversion.Reverter] training.

    Velocity blurring is achieved by convolution with a Gaussian kernel. The kernel sum
    is normalized to a value of 1 such that the velocity value of any pixel in the output
    physical tensor becomes the Gaussian-weighted average of its neighborhood. The edges of
    the convolution are reflection-padded such that the output dimensions equal the input
    dimensions. The kernel size is configured via
    `hyper.observer_hyper.in_blur_kernel_r`
    (`r` size of the Gaussian kernel in pixels),
    `hyper.observer_hyper.in_blur_kernel_lon`
    (`lon` size of the Gaussian kernel in pixels),
    `hyper.observer_hyper.in_blur_kernel_lat`
    (`lat` size of the Gaussian kernel in pixels), and
    `hyper.observer_hyper.in_blur_sigma`
    (spatial standard deviation of the Gaussian kernel in pixels).
        
    Attributes:
        kernel_r: The `r` size of the Gaussian kernel in pixels. Set by
            `hyper.observer_hyper.in_blur_kernel_r`.
        kernel_lon: The `lon` size of the Gaussian kernel in pixels. Set by
            `hyper.observer_hyper.in_blur_kernel_lon`.
        kernel_lat: The `lat` size of the Gaussian kernel in pixels. Set by
            `hyper.observer_hyper.in_blur_kernel_lat`.
        sigma: The spatial standard deviation of the Gaussian kernel in pixels. Set by
            `hyper.observer_hyper.in_blur_sigma`.
        velocity_convolution: The 3D convolution applied to the velocity channel of the
            input physical tensor. The convolution is computed over the dimensions `r, lon, lat`.
        
    Args:
        hyper: A hyperparameters object.
    """

    kernel_r: int
    kernel_lon: int
    kernel_lat: int
    sigma: float
    velocity_convolution: torch.nn.Conv3d
    
    def __init__(self, hyper: hp.Hyper) -> None:
        super().__init__()
        self.kernel_r = hyper.observer_hyper.in_blur_kernel_r
        self.kernel_lon = hyper.observer_hyper.in_blur_kernel_lon
        self.kernel_lat = hyper.observer_hyper.in_blur_kernel_lat
        self.sigma = hyper.observer_hyper.in_blur_sigma

        self.velocity_convolution = torch.nn.Conv3d(in_channels=1,
                                                    out_channels=1,
                                                    kernel_size=(self.kernel_r, self.kernel_lon, self.kernel_lat),
                                                    stride=(1, 1, 1),
                                                    padding='same',
                                                    padding_mode='reflect',
                                                    groups=1,
                                                    bias=False)
        self._gaussian_init(self.velocity_convolution.weight)
        return

    def forward(self, physical_tensor: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        """
        The forward pass of `VelocityBlur`. Convolves the input physical tensor
        with the Gaussian kernel (`self.velocity_convolution`).

        Args:
            physical_tensor: The input [physical tensor][iris.arepo_processing.make_physical_tensor]
                to be blurred.
            inplace: If `True`, performs the blur in-place over the input physical tensor.

        Returns:
            The blurred physical tensor.
        """
        if inplace:
            v_r = physical_tensor[:, 0:1, :, :, :]
            v_r = self.velocity_convolution(v_r)
            physical_tensor[:, 0:1, :, :, :] = v_r
        else:
            v_r, rho, T, abundance_H2, abundance_CO, T_dust = torch.split(physical_tensor, 1, dim=1)
            v_r = self.velocity_convolution(v_r)
            physical_tensor = torch.cat((v_r, rho, T, abundance_H2, abundance_CO, T_dust), dim=1)
        return physical_tensor

    def _gaussian_init(self, weight: torch.Tensor) -> None:
        """
        Initializes the weights of the blur kernel (`self.velocity_convolution`)
        according to a Gaussian density over `r, lon, lat`.

        The dimensions `r, lon, lat` are treated symmetrically, i.e. the projection of the kernel
        onto any one dimension is a single-variable Gaussian of identical standard deviation,
        set as `self.sigma`. The sum of all kernel values
        is normalized to 1 such that the velocity value of any pixel in the output
        [physical tensor][iris.arepo_processing.make_physical_tensor] becomes the
        Gaussian-weighted average of its neighborhood.

        Args:
            weight: The kernel weights to be initialized.
        """
        r = torch.arange(self.kernel_r, dtype=torch.float32) - self.kernel_r // 2
        lon = torch.arange(self.kernel_lon, dtype=torch.float32) - self.kernel_lon // 2
        lat = torch.arange(self.kernel_lat, dtype=torch.float32) - self.kernel_lat // 2
        gaussian_r = torch.exp(-.5 * torch.square(r / torch.tensor(self.kernel_r * self.sigma)))
        gaussian_lon = torch.exp(-.5 * torch.square(lon / torch.tensor(self.kernel_lon * self.sigma)))
        gaussian_lat = torch.exp(-.5 * torch.square(lat / torch.tensor(self.kernel_lat * self.sigma)))
        kernel = torch.einsum('i,j,k->ijk', gaussian_r, gaussian_lon, gaussian_lat)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        weight.data = kernel
        weight.requires_grad = False
        return
    
    
class BeamBlur(torch.nn.Module):
    """
    A Gaussian point-spread convolution applied over the longitude and latitude dimensions
    of an observed PPV cube to simulate the nonzero angular resolution of a radio antenna dish.

    The beam blur is achieved by convolution with a Gaussian kernel. The kernel sum
    is normalized to a value of 1 such that the intensity/brightness-temperature value of
    any pixel in the output physical tensor becomes the Gaussian-weighted average of its
    neighborhood. The edges of the convolution are reflection-padded such that the output
    dimensions equal the input dimensions. The longitude/latitude size of the beam convolution
    is configured via its full-width-half-maximum (FWHM) in arcsec, set by
    `hyper.observer_hyper.out_blur_fwhm`. If `out_blur_fwhm` is `None`,
    this beam convolution module is not applied to the observed cube, yielding an ideal
    observation at the theoretical limit of angular resolution.

    Attributes:
        lon_sigma: The standard deviation corresponding to
            `hyper.observer_hyper.out_blur_fwhm`, converted into units of
            longitude pixels. (May differ from `lat_sigma` since longitude/latitude pixels may
            cover different angular widths.)
        kernel_lon: The longitude width of the convolutional kernel in units of longitude pixels.
            Set to `min(int(10 * lon_sigma), lon_steps)`.
        lat_sigma: The standard deviation corresponding to
            `hyper.observer_hyper.out_blur_fwhm`, converted into units of
            latitude pixels. (May differ from `lon_sigma` since longitude/latitude pixels may
            cover different angular widths.)
        kernel_lat: The latitude width of the convolutional kernel in units of latitude pixels.
            Set to `min(int(10 * lat_sigma), lat_steps)`.
        beam_convolution: The 2D convolution applied over the `lon, lat` dimensions of a PPV cube.

    Args:
        hyper: A hyperparameters object.
    """

    lon_sigma: float
    kernel_lon: int
    lat_sigma: float
    kernel_lat: int
    beam_convolution: torch.nn.Conv2d

    def __init__(self, hyper: hp.Hyper) -> None:
        super().__init__()
        fwhm = hyper.observer_hyper.out_blur_fwhm / 3600        # in deg
        sigma_conversion = 2 * torch.sqrt(2 * torch.log(torch.tensor(2, dtype=torch.float32)))

        lon_steps = hyper.coordinate_hyper.lon_steps
        lon_min = hyper.coordinate_hyper.lon_min
        lon_max = hyper.coordinate_hyper.lon_max
        self.lon_sigma = fwhm / (lon_max - lon_min) * lon_steps / sigma_conversion
        self.kernel_lon = min(int(10 * self.lon_sigma), lon_steps)
        if self.kernel_lon % 2 == 0:
            if self.kernel_lon == lon_steps:
                self.kernel_lon -= 1
            else:
                self.kernel_lon += 1

        lat_steps = hyper.coordinate_hyper.lat_steps
        lat_min = hyper.coordinate_hyper.lat_min
        lat_max = hyper.coordinate_hyper.lat_max
        self.lat_sigma = fwhm / (lat_max - lat_min) * lat_steps / sigma_conversion
        self.kernel_lat = min(int(10 * self.lat_sigma), lat_steps)
        if self.kernel_lat % 2 == 0:
            if self.kernel_lat == lat_steps:
                self.kernel_lat -= 1
            else:
                self.kernel_lat += 1

        n_lines = hyper.observer_hyper.n_lines

        # Setting groups=n_lines ensures different line channels are not convolved together.
        self.beam_convolution = torch.nn.Conv2d(in_channels=n_lines,
                                                out_channels=n_lines,
                                                kernel_size=(self.kernel_lon, self.kernel_lat),
                                                stride=(1, 1),
                                                padding='same',
                                                padding_mode='reflect',
                                                groups=n_lines,
                                                bias=False)
        self._gaussian_init(self.beam_convolution.weight)
        return

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of `BeamBlur`. Convolves the input PPV cube
        with the Gaussian kernel (`self.beam_convolution`).

        Args:
            observed: The input PPV cube to be blurred.

        Returns:
            The blurred PPV cube.
        """
        batch, channel, lon, lat, v = observed.shape
        observed = observed.permute(4, 0, 1, 2, 3)
        observed = observed.reshape(-1, channel, lon, lat)
        observed = self.beam_convolution(observed)
        observed = observed.reshape(v, batch, channel, lon, lat)
        observed = observed.permute(1, 2, 3, 4, 0).contiguous()
        return observed

    def _gaussian_init(self, weight: torch.Tensor) -> None:
        """
        Initializes the weights of the blur kernel (`self.beam_convolution`)
        according to a Gaussian density (point-spread function) over `lon, lat`.

        The projection of the kernel onto either the longitude or latitude dimension is a
        single-variable Gaussian density of standard deviation in pixel units of
        `self.lon_sigma` or `self.lat_sigma`, respectively. Both are
        scaled according to the longitude/latitude pixel sizes such that the
        angular full-width-half-maximum of each projected Gaussian equals
        `hyper.observer_hyper.out_blur_fwhm`. The sum of all kernel values
        is normalized to 1 such that the intensity/brightness-temperature value of any pixel
        in the output PPV cube becomes the Gaussian-weighted average of its neighborhood.

        Args:
            weight: The kernel weights to be initialized.
        """
        lon = torch.arange(self.kernel_lon, dtype=torch.float32) - self.kernel_lon // 2
        lat = torch.arange(self.kernel_lat, dtype=torch.float32) - self.kernel_lat // 2
        gaussian_lon = torch.exp(-.5 * torch.square(lon / self.lon_sigma))
        gaussian_lat = torch.exp(-.5 * torch.square(lat / self.lat_sigma))
        kernel = torch.einsum('i,j->ij', gaussian_lon, gaussian_lat)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0)
        weight.data = kernel
        weight.requires_grad = False
        return


class Noise(torch.nn.Module):
    """
    Adds noise to a PPV cube.

    Real radio observations are not ideal images. They incorporate stochastic noise from a variety
    of sources such as electrical noise in the receiver and atmospheric interference. Accurately
    modeling all such effects to mimic the exact signature of a particular radio telescope
    poses a strong challenge. For the IRIS project, however, as pertains to
    [training][iris.training.train_reverter] a [`Reverter`][iris.reversion.Reverter] to
    intelligently ignore noise artifacts while reverting a PPV cube, achieving a perfectly
    accurate noise signature is likely not as relevant as exposing the `Reverter` to a wide
    variety of noise signatures. This module side-steps the issue of faithful noise modeling
    and employs a simple but varied addition of pixelwise Gaussian noise.

    The noise produced by this class is configurable by three variables:

    * `self.mean`, set by `hyper.observer_hyper.noise_mean`--the mean of the
    Gaussian noise distribution in K;
    * `self.sigma`, set by `self.noise_sigma`--the standard deviation of the
    Gaussian noise distribution in K; and
    * `self.fade`, set by the arg `fade`--if `True`, the mean
    and standard deviation of the Gaussian noise distribution are themselves scaled
    by an additional random variable sampled according to the uniform distribution
    over the unit interval.

    Noise is computed as a random tensor over the entire PPV cube, where each pixel value
    is independent. This is a very rudimentary noise-signature, as true observational noise
    may appear on a variety of angular scales, some exceeding the pixel size. Also note
    that this class only supports cubes in units of brightness/Raleigh-Jeans temperature
    as opposed to intensity. More sophisticated noise implementations are left to user
    implementations or future releases.

    The `Noise` class is implemented to apply to both full PPV cubes output by
    [`SyntheticObserver`][iris.observation.SyntheticObserver] and longitude-velocity
    reductions produced by
    [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset].
    If noise is generated over the full cube and then reduced over the latitude dimension,
    its standard deviation will also be reduced. `Noise` accounts for this discrepancy by
    implementing two mode options in its
    [`forward`][iris.observation.Noise.forward] method--`mode='cube'` and `mode='lv'`.
    In `mode='lv'`, in which noise is added to a cube that is already latitude-reduced,
    a [mean reduction][iris.arepo_processing.PreObservedDataset._reduce_mean] is assumed,
    and the exact correction is computed by scaling `self.sigma` by a factor of `1/sqrt(lon_steps)`.

    Attributes:
        temperature: The brightness-temperature conversion from K to the units specified
            in the arg `units` (one of `'iris', 'processing'`). A `torch.float32` scalar.
        fade: If `True`, the mean and standard deviation of the Gaussian noise distribution
            are themselves scaled by an additional random variable sampled according to the
            uniform distribution over the unit interval.
        mean: The mean of the Gaussian noise distribution in K. A `torch.float32` scalar.
        sigma: The standard deviation of the Gaussian noise distribution in K. A `torch.float32` scalar.
        sigma_lv: The reduction-adjusted standard deviation of the Gaussian noise distribution in K.
            A [mean reduction][iris.arepo_processing.PreObservedDataset._reduce_mean] is assumed,
            and the exact correction is computed by scaling `self.sigma`
            by a factor of `1/sqrt(lon_steps)`. A `torch.float32` scalar.

    Args:
        hyper: A hyperparameters object.
        units_hyper: An optional second hyperparameters object from which to take units.
            The attributes `self.mean`, `self.sigma`, and `self.sigma_lv`
            are all still set by `hyper`.
        units: The units type. One of `'iris', 'processing'`. Not the same as the input/output
            `units` specified as an argument in [`forward`][iris.observation.Noise.forward].
        fade: Sets `self.fade`.

    Raises:
        ValueError: If `units` is not one of `'iris', 'processing'`.
    """

    temperature: torch.nn.Parameter
    fade: bool
    mean: torch.nn.Parameter
    sigma: torch.nn.Parameter
    sigma_lv: torch.nn.Parameter

    def __init__(self,
                 hyper: hp.Hyper,
                 units_hyper: hp.Hyper | None = None,
                 units: str = 'iris',
                 fade: bool = False) -> None:
        super().__init__()
        if units_hyper is None:
            units_hyper = hyper
        if units == 'iris':
            temperature = units_hyper.dataset_hyper._temperature_iris_per_SI
        elif units == 'processing':
            temperature = 1 / hyper.writer_hyper.temperature_K_per_processing
        else:
            raise ValueError("Invalid units provided to Noise. Must be 'iris' or 'processing'.")
        self.temperature = torch.nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

        self.fade = fade

        self.mean = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.noise_mean, dtype=torch.float32),
            requires_grad=False)
        self.sigma = torch.nn.Parameter(
            torch.tensor(hyper.observer_hyper.noise_sigma, dtype=torch.float32),
            requires_grad=False)
        self.sigma_lv = torch.nn.Parameter(
            self.sigma / np.sqrt(hyper.coordinate_hyper.lon_steps), requires_grad=False)
        return

    def forward(self,
                inputs: torch.Tensor,
                inplace: bool = False,
                units: str = 'Trj',
                mode: str = 'cube') -> torch.Tensor:
        """
        Applies Gaussian noise to an input observation.

        Args:
            inputs: The PPV cube or latitude-reduced cube to which noise is added.
                Assumed to be in the units of brightness temperature dictated by `units`.
            inplace: If `True`, noise is added in-place over the input.
            units: The input/output units. One of `'Tb', 'Tb K'`.
                If `'Tb'` or `'Trj'`, units are brightness temperature or Raleigh-Jeans temperature 
                in the system dictated by the `units` arg passed to the `Noise` constructor 
                (one of `'iris', 'processing'`).
                If `'Tb K'` or `'Trj K'`, units are brightness temperature or Raleigh-Jeans temperature 
                in K.
            mode: The input mode. One of `'cube', 'lv'`.
                If `'cube'`, expects a full PPV cube as an input and applies `self.sigma`.
                If `'lv'`, expects a latitude-reduced cube as input and applies `self.sigma_lv`.

        Returns:
            The noise-added observation.

        Raises:
            ValueError: If `mode` is not one of `'cube', 'lv'`.
            ValueError: If `units` is not one of `'Tb', 'Tb K', 'Trj', 'Trj K'`.
        """
        if mode == 'cube':
            sigma = self.sigma
        elif mode == 'lv':
            sigma = self.sigma_lv
        else:
            raise ValueError("Invalid mode provided to Noise.forward. "
                             "Must be one of 'cube', 'lv'.")
        if units == 'Trj' or units == 'Tb':
            mean = self.mean * self.temperature
            sigma = sigma * self.temperature
        elif units == 'Trj K' or units == 'Tb K':
            mean = self.mean
        else:
            raise ValueError("Invalid units provided to Noise.forward. "
                             "Must be one of 'Tb', 'Tb K', 'Trj', 'Trj K'.")
        noise = mean + sigma * torch.randn_like(inputs, dtype=torch.float32, device=inputs.device)
        if self.fade:
            noise *= torch.rand(1, dtype=torch.float32, device=inputs.device).squeeze()
        if inplace:
            outputs = inputs.add_(noise)
        else:
            outputs = inputs + noise
        return outputs


class Abundance(torch.nn.Module):
    r"""
    The base class for tracer abundance functions.

    Emission and absorption of a spectral line depend upon the abundance of the tracer molecule.
    For [`ObservabilityProcessor`][iris.observation.ObservabilityProcessor], tracer abundance
    must be expressed as a fraction of total H atom number density. IRIS is designed to
    process AREPO simulations that track the abundances of $\text{H}_2$, $\text{H}^+$, and CO.
    All tracer abundances must be derived from these base quantities. The purpose of `Abundance`
    is to implement a [`forward`][iris.observation.Abundance.forward] function that receives
    a pixelwise total-H-atom number density, gas temperature, $\text{H}_2$ abundance per
    total-H-atom number density, and CO abundance per total-H-atom number density, and returns a
    pixelwise tracer abundance per total-H-atom number density. If the observation is in multiple
    spectral lines, `forward` should return multiple abundances stacked along the channel dimension.
    Note that the output is a fractional abundance, not a raw number density. This is purely a
    decision of design convention rather than computational expedience.

    An `Abundance` is, in theory, trainable. A
    [`SyntheticObserver`][iris.observation.SyntheticObserver] in either
    [end-to-end differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_all]
    or the more efficient
    [abundance-only differentiability mode][iris.observation.SyntheticObserver.set_requires_grad_abundance]
    can provide gradient backpropagation to a user-specified `Abundance` with trainable parameters.
    Depending upon the user-specified training setup, the `Abundance` can then be optimized such
    that some set of synthetic observations matches some real observational prior. The challenge
    is only in specifying a dataset, training setup, and prior-based loss function that sufficiently
    constrains an `Abundance` that is intended to be applied pointwise based upon total-H-atom
    number density, gas temperature, $\text{H}_2$ abundance, and CO abundance, i.e. with no
    spatial dependence. Such implementations are left to future applications and releases.

    Attributes:
        number_density: The conversion factor of number density from SI units to the units dictated
            by the arg `units` (one of `'iris', 'processing'`). A `torch.float32` scalar.
        temperature: The conversion factor of temperature from SI units to the units dictated
            by the arg `units` (one of `'iris', 'processing'`). A `torch.float32` scalar.

    Args:
        hyper: A hyperparameters object.
        units: The units of the abundance function. One of `'iris', 'processing'`.
    """

    number_density: torch.nn.Parameter
    temperature: torch.nn.Parameter

    def __init__(self, hyper: hp.Hyper, units: str | None = 'iris') -> None:
        super().__init__()
        self.number_density = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
        self.temperature = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
        if units:
            self.set_units(hyper, units)
        return

    def forward(self,
                N_H_TOT: torch.Tensor,
                T: torch.Tensor,
                abundance_H2: torch.Tensor,
                abundance_CO: torch.Tensor) -> torch.Tensor:
        r"""
        Computes tracer abundance.

        An abstract method signature to be overridden by the extending class. If implementing
        a single-tracer abundance, do not forget to expand the channel/lines dimension to a size of 1.

        Args:
            N_H_TOT: Total-H-atom number density. Includes H, $\text{H}_2$, and $\text{H}^+$, although
                $\text{H}^+$ is ignored in terms of computing the level balance
                (see [`_compute_single_molecule`][iris.chemistry._compute_single_molecule]).
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            T: Gas temperature. Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            abundance_H2: $\text{H}_2$ abundance per total-H-atom number density.
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            abundance_CO: CO abundance per total-H-atom number density.
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.

        Returns:
            A tracer abundance or tracer abundances per total-H-atom number density.
                Expressed as a fractional abundance, not a raw number density.
                Has dimensions `batch, n_lines, r_steps, lon_steps, lat_steps, v_steps`.
        """
        pass

    def set_units(self, hyper: hp.Hyper, units: str) -> None:
        """
        Sets the units of the abundance function.

        Args:
            hyper: A hyperparameters object.
            units: The units of the abundance function. One of `'iris', 'processing'`.

        Raises:
            ValueError: If `units` is not one of `'iris', 'processing'`.
            ValueError: If units are not found in `hyper`.
        """
        if units == 'iris':
            length = hyper.dataset_hyper._length_iris_per_SI
            temperature = hyper.dataset_hyper._temperature_iris_per_SI
            if length is None or temperature is None:
                raise ValueError("IRIS units not available in hyper.")
            number_unit = hyper.dataset_hyper.iris_number_unit
            number_volume = number_unit * length * length * length
            number_density = 1 / number_volume
        elif units == 'processing':
            length_cm_per_processing = hyper.writer_hyper.length_cm_per_processing
            if length_cm_per_processing is None:
                raise ValueError("Processing units not available in hyper.")
            temperature = 1 / hyper.writer_hyper.temperature_K_per_processing
            length = 100 / length_cm_per_processing
            number_unit = hyper.dataset_hyper.iris_number_unit
            number_volume = number_unit * length * length * length
            number_density = 1 / number_volume
        else:
            raise ValueError("Invalid units provided to Abundance. Must be 'iris' or 'processing'.")

        self.number_density.data = torch.tensor(number_density, dtype=torch.float32)
        self.temperature.data = torch.tensor(temperature, dtype=torch.float32)
        return


class Constant_CO_13C16O(Abundance):
    r"""
    Implements an abundance of the CO isotopologue $^{13}\text{CO}$ as a constant factor of
    ``peak`` times total CO abundance.
    $1.5 \times 10^{-2}$ times total CO abundance.

    Attributes:
        number_density: The conversion factor of number density from SI units to the units dictated
            by the arg `units` (one of `'iris', 'processing'`). A `torch.float32` scalar.
        temperature: The conversion factor of temperature from SI units to the units dictated
            by the arg `units` (one of `'iris', 'processing'`). A `torch.float32` scalar.
        peak: The constant abundance of $^{13}\text{CO}$ as a fraction of total CO abundance.

    Args:
        hyper: A hyperparameters object.
        peak: Sets ``self.peak``.
        units: The units of the abundance function. One of `'iris', 'processing'`.
        trainable: Tells PyTorch whether to track a gradient of `self.peak`.
            Set `True` if training `peak` against some known observational prior.
            (See [`Abundance`][iris.observation.Abundance] for details on trainable abundances.)
    """

    peak: torch.nn.Parameter

    def __init__(self, hyper: hp.Hyper, peak: float = 4e-2, units: str = 'iris', trainable: bool = False) -> None:
        super().__init__(hyper, units=units)
        self.peak = torch.nn.Parameter(torch.tensor(peak, dtype=torch.float32), requires_grad=trainable)
        return

    def forward(self,
                N_H_TOT: torch.Tensor,
                T: torch.Tensor,
                abundance_H2: torch.Tensor,
                abundance_CO: torch.Tensor) -> torch.Tensor:
        r"""
        Computes $^{13}\text{CO}$ abundance as a constant function `abundance_13CO = peak * abundance_CO`.

        Args:
            N_H_TOT: Total-H-atom number density. Includes H, $\text{H}_2$, and $\text{H}^+$, although
                $\text{H}^+$ is ignored in terms of computing the level balance
                (see [`_compute_single_molecule`][iris.chemistry._compute_single_molecule]).
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            T: Gas temperature. Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            abundance_H2: $\text{H}_2$ abundance per total-H-atom number density.
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.
            abundance_CO: CO abundance per total-H-atom number density.
                Has dimensions `batch, r_steps, lon_steps, lat_steps, v_steps`.

        Returns:
            $^{13}\text{CO}$ abundance per total-H-atom number density.
                Expressed as a fractional abundance, not a raw number density.
                Has dimensions `batch, n_lines=1, r_steps, lon_steps, lat_steps, v_steps`.
        """
        abundance = abundance_CO * torch.abs(self.peak)
        return abundance.unsqueeze(dim=1)
