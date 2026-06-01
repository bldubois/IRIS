# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
Implements the training and testing setups for a [`Reverter`][iris.reversion.Reverter].

The primary function of this module is [`train_reverter`][iris.training.train_reverter], which
implements a robust training setup. In addition to the training setup, this module provides a
[test function][iris.training.test_reverter] that mirrors all features of the training function.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import time
import typing
import json

import torch

from . import arepo_processing as ap
if typing.TYPE_CHECKING:
    from . import hyper as hp
    from . import reversion
    from . import observation


def train_reverter(reverter: reversion.Reverter,
                   dataset: ap.Dataset | ap.ConcatDataset,
                   noise: observation.Noise | None = None,
                   litter: ap.InfiniteDataset | None = None,
                   observer: observation.Observer | None = None,
                   hyper: hp.Hyper | None = None,
                   checkpoint_directory: str | None = None,
                   checkpoint_name: str | None = None,
                   auto_startup: bool = True,
                   auto_cleanup: bool = True) -> tuple[torch.nn.Module, int]:
    """
    Implements the training setup for a [`Reverter`][iris.reversion.Reverter].

    The training setup is a simple supervised training scheme. A pair of an input
    [observation][iris.observation.Observer] and output [top-down density image] make up the ground
    truth the neural network is trained to reproduce. At training time, a batch of observations
    is fed into the network, which produces a corresponding batch of top-down predictions. A
    [physical loss function][iris.training.PhysicalLoss] compares the predicted images to the true
    top-down images, yielding a loss reduced loss on which a backwards pass is executed. The
    `Reverter` parameters are stepped based on the gradient of this physical loss. Because the
    predictions exist in a space imbued with physical units of density, the loss function must
    contain a units normalization in order to be units-invariant, so that loss scores can be compared
    between datasets with differing units. See [`PhysicalLoss`][iris.training.PhysicalLoss] for
    details on specifying a physical loss function.

    This setup is designed to operate with both [preobserved][iris.arepo_processing.PreObservedDataset]
    and [standard][iris.arepo_processing.StandardDataset] datasets, although it is recommended that
    only preobserved datasets be used for `Reverter` training, as they reduce disk usage and load
    latency by orders of magnitude and eliminate redundant observation at runtime, which introduces
    a large computational overhead. The setup is also configured to enable an optional addition of
    noise and litter. Noise is a random observational defect added to the input observations by
    a [`Noise`][iris.observation.Noise] object, in order to innoculate the `Reverter` to noise
    expected in the true observations on which it will be applied.

    Litter is a separate data augmentation that addresses the other source of confounding information
    in a true observation--foreground and background features that do not reflect actual structures
    in the CMZ itself. If full-cone observations are used in constructing the training dataset, then
    foreground and background features will naturally be present. Full-cone observations, however,
    require orders of magnitude more time and memory to compute than observations of a small, central
    cutout of the CMZ region, and so were not found to be pragmatic for constructing a training dataset
    in the IRIS paper. Instead, litter allows a separate dataset of only foreground/background features
    that are added to the `Reverter` inputs randomly during training. This also introduces an additional
    source of regularization, since the litter is applied randomly as opposed to being matched with
    specific observations via a fixed one-to-one pairing.

    Of course, the use of litter assumes the task of the `Reverter` regarding foreground and background
    features is merely to learn to ignore them, i.e. that there is no useful information that can be
    extracted from them regarding the CMZ structure, which may not be precisely true in theory.
    Litter must be constructed as an [`InfiniteDataset`][iris.arepo_processing.InfiniteDataset],
    which can be accomplished by a [`Reader`][iris.arepo_processing.Reader]. Like the training dataset,
    litter may be sourced from either a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset]
    or a [`StandardDataset`][iris.arepo_processing.StandardDataset], but it is recommended that only
    a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] be used, for optimal performance.

    This training setup is also designed to enable a fully distributed, multi-node, multi-GPU,
    data parallel configuration. A scalable set of CPU workers load datapoints from the disk
    while a single manager process per GPU asynchronously manages the training process. Model gradients
    are automatically synced across GPUs via `torch.nn.parallel.DistributedDataParallel`. This provides
    infinitely scalable batch sizes. In practice, however, the IRIS paper found that small batches
    provide a critical source of regularization. Moreover, the primary training bottleneck was found to
    be the latency in loading a training point from the disk into memory--in particular, because the
    IRIS paper performed training on a large dataset stored on a locally networked drive in an HPC
    environment. The GPU latency of the forward and backward passes and step computations was found to be
    small. Instead, it was found that the best practical setup involved training on a single GPU with a large
    number of CPU workers for asynchronous data loading. See the IRIS paper for more details and discussion
    (subsec: Implementation of Reversion: Training Hyperparameters, Overfitting, and Regularization).

    All training hyperparameters are specified in the [`training_hyper`][iris.hyper.TrainingHyper]
    of a [hyperparameters object][iris.hyper.Hyper]. These include:

    * `validation_data_fraction`: How much of the training data is segregated for validation.
    * `epochs`: How many epochs to train for.
    * `batch_size`: The batch size. Specified per-GPU, i.e. if `batch_size=8` and training on two GPUs,
    the actual batch size is 16.
    * `batches_per_update`: How many batches over which to accumulate gradients before computing
    an optimizer step.
    * `physical_loss`: Type of the specific [`PhysicalLoss`][iris.training.PhysicalLoss] to be
    instantiated and used during training.
    * `density_normalization`: The units normalization to be applied to the true and predicted
    densities in computing a units-invariant physical loss.
    * `optimizer`: A callable that accepts the `Reverter` parameters and returns a tuple
    `optimizer, scheduler` of a `torch.optim.optimizer.Optimizer`, and
    `torch.optim.lr_scheduler.LRScheduler` initialized on these parameters.

    Args:
        reverter: The reverter to be trained.
        dataset: A dataset on which to train the reverter.
        noise: An object that adds random noise to each input observation.
        litter: A dataset of foreground/background features to be added randomly to observations
            during training.
        observer: An observer with which to generate observations of
            [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor], if training with a
            [`StandardDataset`][iris.arepo_processing.StandardDataset].
        hyper: A hyperparameters object.
        checkpoint_directory: If not `None`, will save a model checkpoint to this directory at the end
            of each epoch. Will save each checkpoint inside a subdirectory of this directory of name
            `checkpoint_name`. If no such subdirectory exists, will create this subdirectory automatically.
        checkpoint_name: The subdirectory name in which to save model checkpoints. Must be specified
            if `checkpoint_directory` is not `None`.
        auto_startup: If `True`, will create a new `torch.distributed` process group for distributed
            training. If training with a single GPU, leave this argument as `True`. Only specify
            `False` if manually creating a process group.
        auto_cleanup: If `True`, will destroy the `torch.distributed` process group for distributed
            training. If training with a single GPU, leave this argument as `True`. Only specify
            `False` if manually destroying the process group.

    Returns:
        A tuple `reverter, rank` containing the trained module (on the CPU, in eval mode),
        and the integer rank of the current process in the `torch.distributed` process group.

    Raises:
        ValueError: If `dataset` is not a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset]
            or [`ConcatDataset`][iris.arepo_processing.ConcatDataset] of preobserved datasets,
            and no `observer` is provided.
        ValueError: If `litter` is not either `None` or an
            [`InfiniteDataset`][iris.arepo_processing.InfiniteDataset].
        ValueError: If `litter` is specified, and not a constructed from a
            [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset] of preobserved datasets,
            but no `observer` is provided.
        ValueError: If one but not both of `checkpoint_directory` and `checkpoint_name` are specified.
    """
    start_time = time.time()
    if hyper is None:
        hyper = dataset.hyper
    # Initialize the distributed process group and get SLURM job parameters.
    if auto_startup:
        torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    cpus_per_slurm_process = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    gpus_per_slurm_process = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    cpus_per_gpu = cpus_per_slurm_process // gpus_per_slurm_process
    device = torch.device('cuda', local_rank)

    # Get the training data. This won't load the entire dataset into memory at once.
    # It will only load at most num_workers * prefetch_factor datapoints.
    training_sampler, training, validation_sampler, validation = dataset.make_training_and_validation_dataloaders(cpus_per_gpu)

    # Ensure the reverter is ready for a distributed implementation.
    # SyncBatchNorm makes sure batchnorm is updated with the statistics of the entire distributed batch.
    # DistributedDataParallel ensures reverter gradients are efficiently synced across the process group.
    reverter = torch.nn.SyncBatchNorm.convert_sync_batchnorm(reverter)
    reverter = torch.nn.parallel.DistributedDataParallel(reverter.to(device), device_ids=[device])
    reverter.train()

    # Determine if the dataset is preobserved or if an observer is required.
    preobserved = False
    if isinstance(dataset, ap.PreObservedDataset):
        preobserved = True
    elif isinstance(dataset, ap.ConcatDataset):
        if issubclass(dataset.from_type, ap.PreObservedDataset):
            preobserved = True
    if not preobserved:
        if observer is not None:
            observer = observer.to(device)
            observer.eval()
        else:
            raise ValueError('Must provide Observer or PreObservedDataset to train Reverter.')

    # Configure noise.
    if noise is not None:
        noise.to(device)
        noise.eval()
    # Configure litter. Determine if the litter is preobserved or if an observer is required.
    if litter is None:
        litter_training = None
        litter_validation = None
    else:
        if not isinstance(litter, ap.InfiniteDataset):
            raise ValueError('Must specify litter as either None or an InfiniteDataset.')
        litter_training, litter_validation = litter.make_infinite_training_validation_dataloaders(cpus_per_gpu)
        litter_training = iter(litter_training)
        litter_validation = iter(litter_validation)
        litter_preobserved = False
        if issubclass(litter.from_type, ap.PreObservedDataset):
            litter_preobserved = True
        else:
            if observer is not None:
                observer = observer.to(device)
                observer.eval()
            else:
                raise ValueError('Must provide Observer or PreObservedDataset as litter to train Reverter with litter.')

    # Initialize the physical loss function, optimizer, scheduler, and stats history.
    physical_loss_fn = hyper.training_hyper.physical_loss(hyper)
    physical_loss_fn.to(device)
    optimizer, scheduler = hyper.training_hyper.optimizer(reverter.parameters())
    batches_per_update = hyper.training_hyper.batches_per_update
    training_physical_losses = []
    validation_physical_losses = []

    # Iterate over all training epochs.
    for epoch in range(hyper.training_hyper.epochs):
        if rank == 0:
            print(f'Epoch {epoch + 1}/{hyper.training_hyper.epochs}', flush=True)
            training_physical_loss = 0

        # Compute the training epoch.
        training_sampler.set_epoch(epoch)
        optimizer.zero_grad()
        n = 0
        for item in training:
            # Get the batch of preobserved pairs or physical tensors.
            # Observe the physical tensors if necessary.
            # As in a preobserved dataset, any in-blur is applied to both the
            # observation and the top-down map.
            if preobserved:
                columnized, observed = item
                columnized = columnized.to(device, non_blocking=True)
                observed = observed.to(device, non_blocking=True)
            else:
                physical_tensor = item.to(device, non_blocking=True)
                if observer.in_blur is not None:
                    with torch.no_grad():
                        physical_tensor = observer.in_blur(physical_tensor, inplace=True)
                columnized = ap.columnize_physical_tensor(physical_tensor, hyper)
                with torch.no_grad():
                    observed = reverter.module.reduction(observer(physical_tensor, bypass_blur_in=True))

            # Add noise and litter to the observations if applicable.
            if noise is not None:
                observed = noise(observed, inplace=True, mode='lv')
            if litter_training is not None:
                item = next(litter_training)
                if litter_preobserved:
                    _, litter_observed = item
                    litter_observed = litter_observed.to(device, non_blocking=True)
                else:
                    litter_physical_tensor = item.to(device, non_blocking=True)
                    litter_observed = reverter.module.reduction(observer(litter_physical_tensor, bypass_blur_in=True))
                observed += litter_observed

            # Compute the model forward pass, physical loss function, and backward pass.
            physical_prediction = reverter(observed.detach(), reduce=False)
            physical_loss = physical_loss_fn(physical_prediction, columnized)
            physical_loss.backward()

            # Reduce loss scores across the distributed cluster for metrics.
            torch.distributed.all_reduce(physical_loss.detach(), op=torch.distributed.ReduceOp.AVG)
            if rank == 0:
                training_physical_loss += physical_loss.item()

            # If applicable, step the optimizer.
            n += 1
            if n % batches_per_update == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Step the scheduler to update optimizer learning rates if necessary.
        if scheduler is not None:
            scheduler.step()

        # Print the training metrics.
        if rank == 0:
            training_physical_loss /= n
            training_physical_losses.append(training_physical_loss)
            print(f'\tTraining physical loss:\t\t{training_physical_loss:.8e}', flush=True)

        # Compute the validation epoch.
        if validation is not None:
            validation_sampler.set_epoch(epoch)
            reverter.eval()
            if rank == 0:
                validation_physical_loss = 0
                n = 0
            for item in validation:
                with torch.no_grad():
                    # Get the batch of preobserved pairs or physical tensors.
                    # Observe the physical tensors if necessary.
                    # As in a preobserved dataset, any in-blur is applied to both the
                    # observation and the top-down map.
                    if preobserved:
                        columnized, observed = item
                        columnized = columnized.to(device, non_blocking=True)
                        observed = observed.to(device, non_blocking=True)
                    else:
                        physical_tensor = item.to(device, non_blocking=True)
                        if observer.in_blur is not None:
                            physical_tensor = observer.in_blur(physical_tensor, inplace=True)
                        columnized = ap.columnize_physical_tensor(physical_tensor, hyper)
                        observed = reverter.module.reduction(observer(physical_tensor, bypass_blur_in=True))

                    # Add noise and litter to the observations if applicable.
                    if noise is not None:
                        observed = noise(observed, inplace=True, mode='lv')
                    if litter_validation is not None:
                        item = next(litter_validation)
                        if litter_preobserved:
                            _, litter_observed = item
                            litter_observed = litter_observed.to(device, non_blocking=True)
                        else:
                            litter_physical_tensor = item.to(device, non_blocking=True)
                            litter_observed = reverter.module.reduction(observer(litter_physical_tensor, bypass_blur_in=True))
                        observed += litter_observed

                    # Compute the model forward pass and physical loss function.
                    physical_prediction = reverter(observed, reduce=False)
                    physical_loss = physical_loss_fn(physical_prediction, columnized)

                    # Reduce loss scores across the distributed cluster for metrics.
                    torch.distributed.all_reduce(physical_loss, op=torch.distributed.ReduceOp.AVG)
                    if rank == 0:
                        validation_physical_loss += physical_loss.item()
                        n += 1

            # Print the validation metrics.
            if rank == 0:
                validation_physical_loss /= n
                validation_physical_losses.append(validation_physical_loss)
                print(f'\tValidation physical loss:\t{validation_physical_loss:.8e}', flush=True)

            reverter.train()

        # Save a reverter checkpoint to the disk.
        if rank == 0 and checkpoint_directory is not None or checkpoint_name is not None:
            if checkpoint_directory is None or checkpoint_name is None:
                raise ValueError('If checkpointing is enabled, must specify both '
                                 'checkpoint_directory and checkpoint_name.')
            directory = os.path.join(os.path.expanduser(checkpoint_directory), checkpoint_name)
            if not os.path.exists(directory):
                os.mkdir(directory)
            path = os.path.join(directory, f'chp_{epoch + 1}.pt')
            torch.save(reverter.module.state_dict(), path)

    # Compute and print final training statistics.
    if rank == 0:
        cpu_memory_usage = ap.gauge_cpu_memory()
        gpu_memory_usage = torch.cuda.max_memory_allocated(device=device) / 1024 ** 3
        print(f'\nCPU memory usage:\t\t\t{cpu_memory_usage:.2f} GiB TOTAL')
        print(f'GPU memory usage:\t\t\t{gpu_memory_usage:.2f} GiB/GPU')
        end_time = time.time()
        training_time_hours = (end_time - start_time) / 3600
        print(f'Total training time:\t\t\t{training_time_hours:.2f} hours', flush=True)

        if checkpoint_directory is not None and checkpoint_name is not None:
            stats = {'training_physical_losses': training_physical_losses,
                     'validation_physical_losses': validation_physical_losses,
                     'cpu_memory_usage': cpu_memory_usage,
                     'gpu_memory_usage': gpu_memory_usage,
                     'training_time_hours': training_time_hours}
            directory = os.path.join(os.path.expanduser(checkpoint_directory), checkpoint_name)
            path = os.path.join(directory, 'stats.json')
            with open(path, 'w') as file:
                json.dump(stats, file)

    reverter.eval()
    reverter.cpu()

    if auto_cleanup:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()

    return reverter.module, rank

def test_reverter(reverter: reversion.Reverter,
                  dataset: ap.Dataset | ap.ConcatDataset,
                  noise: observation.Noise | None = None,
                  litter: ap.InfiniteDataset | None = None,
                  observer: observation.Observer | None = None,
                  hyper: hp.Hyper | None = None,
                  auto_startup: bool = True,
                  auto_cleanup: bool = True) -> None:
    """
    Tests [`Reverter`][iris.reversion.Reverter] performance over a dataset.

    Mirrors all configurations and functionalities of [`train_reverter`][iris.training.train_reverter]
    other than model training. Instead, the `Reverter` is tested over a single epoch over the
    entire dataset. No gradients or parameter steps are computed and the model is called in eval
    mode in order to record physical loss scores, as in validation. See
    [`train_reverter`][iris.training.train_reverter] for all details regarding physical losses,
    hyperparameters, and the multi-node, multi-GPU distributed setup.

    The specific hyperparameters in [`TrainingHyper`][iris.hyper.TrainingHyper] that still apply
    during `Reverter` testing are:

    * `batch_size`: The batch size. Specified per-GPU, i.e. if `batch_size=8` and training on two GPUs,
    the actual batch size is 16.
    * `physical_loss`: Type of the specific [`PhysicalLoss`][iris.training.PhysicalLoss] to be
    instantiated and used during testing.
    * `density_normalization`: The units normalization to be applied to the true and predicted
    densities in computing a units-invariant physical loss.

    Args:
        reverter: The reverter to be trained.
        dataset: A dataset on which to test the reverter.
        noise: An object that adds random noise to each input observation.
        litter: A dataset of foreground/background features to be added randomly to observations
            during testing.
        observer: An observer with which to generate observations of
            [physical tensors][iris.arepo_processing.Snapshot.make_physical_tensor], if testing with a
            [`StandardDataset`][iris.arepo_processing.StandardDataset].
        hyper: A hyperparameters object.
        auto_startup: If `True`, will create a new `torch.distributed` process group for distributed
            testing. If testing with a single GPU, leave this argument as `True`. Only specify
            `False` if manually creating a process group.
        auto_cleanup: If `True`, will destroy the `torch.distributed` process group for distributed
            testing. If testing with a single GPU, leave this argument as `True`. Only specify
            `False` if manually destroying the process group.

    Raises:
        ValueError: If `dataset` is not a [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset]
            or [`ConcatDataset`][iris.arepo_processing.ConcatDataset] of preobserved datasets,
            and no `observer` is provided.
        ValueError: If `litter` is not either `None` or an
            [`InfiniteDataset`][iris.arepo_processing.InfiniteDataset].
        ValueError: If `litter` is specified, and not a constructed from a
            [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset] or
            [`ConcatDataset`][iris.arepo_processing.ConcatDataset] of preobserved datasets,
            but no `observer` is provided.
    """
    start = time.time()
    if hyper is None:
        hyper = dataset.hyper
    # Initialize the distributed process group and get SLURM job parameters.
    if auto_startup:
        torch.distributed.init_process_group(backend='nccl')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    cpus_per_slurm_process = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    gpus_per_slurm_process = int(os.environ.get('LOCAL_WORLD_SIZE', '1'))
    cpus_per_gpu = cpus_per_slurm_process // gpus_per_slurm_process
    device = torch.device('cuda', local_rank)

    # Get the test data. This won't load the entire dataset into memory at once.
    # It will only load at most num_workers * prefetch_factor datapoints.
    test_sampler, test = dataset.make_test_dataloader(cpus_per_gpu)

    reverter.to(device)
    reverter.eval()

    # Determine if the dataset is preobserved or if an observer is required.
    preobserved = False
    if isinstance(dataset, ap.PreObservedDataset):
        preobserved = True
    elif isinstance(dataset, ap.ConcatDataset):
        if issubclass(dataset.from_type, ap.PreObservedDataset):
            preobserved = True
    if not preobserved:
        if observer is not None:
            observer = observer.to(device)
            observer.eval()
        else:
            raise ValueError('Must provide Observer or PreObservedDataset to test Reverter.')

    # Configure noise.
    if noise is not None:
        noise.to(device)
        noise.eval()
    # Configure litter. Determine if the litter is preobserved or if an observer is required.
    if litter is None:
        litter_test = None
    else:
        if not isinstance(litter, ap.InfiniteDataset):
            raise ValueError('Must specify litter as either None or an InfiniteDataset.')
        litter_test = litter.make_infinite_test_dataloader(cpus_per_gpu)
        litter_test = iter(litter_test)
        litter_preobserved = False
        if issubclass(litter.from_type, ap.PreObservedDataset):
            litter_preobserved = True
        else:
            if observer is not None:
                observer = observer.to(device)
                observer.eval()
            else:
                raise ValueError('Must provide Observer or PreObservedDataset as litter to test Reverter with litter.')

    # Configure the physical loss function.
    physical_loss_fn = hyper.training_hyper.physical_loss(hyper)
    physical_loss_fn.to(device)

    test_sampler.set_epoch(0)
    if rank == 0:
        print('Running test...', flush=True)
        test_physical_loss = 0
        n = 0

    # Compute the test epoch.
    for item in test:
        with torch.no_grad():
            # Get the batch of preobserved pairs or physical tensors.
            # Observe the physical tensors if necessary.
            # As in a preobserved dataset, any in-blur is applied to both the
            # observation and the top-down map.
            if preobserved:
                columnized, observed = item
                columnized = columnized.to(device, non_blocking=True)
                observed = observed.to(device, non_blocking=True)
            else:
                physical_tensor = item.to(device, non_blocking=True)
                if observer.in_blur is not None:
                    with torch.no_grad():
                        physical_tensor = observer.in_blur(physical_tensor, inplace=True)
                columnized = ap.columnize_physical_tensor(physical_tensor, hyper)
                with torch.no_grad():
                    observed = reverter.module.reduction(observer(physical_tensor, bypass_blur_in=True))

            # Add noise and litter to the observations if applicable.
            if noise is not None:
                observed = noise(observed, inplace=True, mode='lv')
            if litter_test is not None:
                item = next(litter_test)
                if litter_preobserved:
                    _, litter_observed = item
                    litter_observed = litter_observed.to(device, non_blocking=True)
                else:
                    litter_physical_tensor = item.to(device, non_blocking=True)
                    litter_observed = reverter.module.reduction(observer(litter_physical_tensor, bypass_blur_in=True))
                observed += litter_observed

            # Compute the model forward pass and physical loss function.
            physical_prediction = reverter(observed, reduce=False)
            physical_loss = physical_loss_fn(physical_prediction, columnized)

            # Reduce loss scores across the distributed cluster for metrics.
            torch.distributed.all_reduce(physical_loss, op=torch.distributed.ReduceOp.AVG)
            if rank == 0:
                test_physical_loss += physical_loss.item()
                n += 1

    # Compute and print final test statistics.
    if rank == 0:
        test_physical_loss /= n
        print(f'\tTest physical loss:\t\t{test_physical_loss:.8e}', flush=True)

        cpu_memory_usage = ap.gauge_cpu_memory()
        gpu_memory_usage = torch.cuda.max_memory_allocated(device=device) / 1024 ** 3
        print(f'\nCPU memory usage:\t\t\t{cpu_memory_usage:.2f} GiB TOTAL')
        print(f'GPU memory usage:\t\t\t{gpu_memory_usage:.2f} GiB/GPU')

        end = time.time()
        elapsed = (end - start) / 3600
        print(f'Total test time:\t\t\t{elapsed:.2f} hours', flush=True)

    reverter.cpu()

    if auto_cleanup:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()

    return

class PhysicalLoss(torch.nn.Module):
    """
    The abstract base class for a physical loss function to be used in
    [`Reverter` training][iris.training.train_reverter].

    Takes a batch of predicted [top-down density images][iris.arepo_processing.columnize_physical_tensor]
    and corresponding true top-down density images and computes a scalar comparison metric to
    minimize during `Reverter` training.

    Attributes:
        normalization: The normalization constant, in units of density, to be applied to true and
            predicted densities. Is set as the conversion of
            `hyper.training_hyper.density_normalization` into IRIS units.

    Args:
        hyper: A hyperparameters object.
    """

    normalization: torch.nn.Parameter

    def __init__(self, hyper: hp.Hyper) -> None:
        super().__init__()
        mass = hyper.dataset_hyper._mass_iris_per_SI
        length = hyper.dataset_hyper._length_iris_per_SI
        volume = length * length * length
        density = mass / volume
        normalization = hyper.training_hyper.density_normalization
        self.normalization = torch.nn.Parameter(torch.tensor(normalization * density, dtype=torch.float32),
                                        requires_grad=False)
        return

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        The abstract signature of the forward pass.

        Args:
            pred: The predicted top-down density image.
            true: The true top-down density image.

        Returns:
            The scalar loss metric.
        """
        pass

    def normed_residual(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Computes a unitless residual by normalizing by the density units constant.

        Args:
            pred: The predicted top-down density image.
            true: The true top-down density image.

        Returns:
            The units-normalized residual.
        """
        return (pred - true) / self.normalization

class ScaledDensityLoss(PhysicalLoss):
    """
    The specific [`PhysicalLoss`][iris.training.PhysicalLoss] used in the IRIS paper.
    """
    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """
        Injects an arc-hyperbolic-sine nonlinearity inside a units-normalized
        mean square error. Experiments conducted for the IRIS paper indicated that
        nonlinearizing the loss function was essential to enabling effective `Reverter`
        training by expanding the dynamic range of the prediction space in the target
        regime and enabling the training process to better "see" inaccuracies in the
        model predictions.

        Args:
            pred: The predicted top-down density image.
            true: The true top-down density image.

        Returns:
            The scalar loss metric.
        """
        return torch.mean(torch.asinh(torch.square(self.normed_residual(pred, true))))
