# Copyright (c) 2026 University of Connecticut
# Created by B.L. DuBois
# SPDX-License-Identifier: MIT
# See the LICENSE file for details
"""
The parallel computing framework for AREPO processing and dataset construction.

Segregates all MPI-based logic for AREPO processing into a separate module from
[`arepo_processing`][iris.arepo_processing] so that all non-writing functions
are MPI-optional. This is important since when importing `mpi4py`, Python will
raise an error if an OpenMPI module is not loaded into the cluster environment.
Also imports core functionalities from [`arepo_processing`][iris.arepo_processing]
so they can be aliased from this module.

Authors:
    B.L. DuBois (brendan@bldubois.com)
"""

from __future__ import annotations

import os
import re
import typing
import subprocess
import random
import time
from pathlib import Path

from mpi4py import MPI
import torch
import numpy as np

if typing.TYPE_CHECKING:
    from . import hyper as hp
    from . import observation
from . import arepo_processing as ap
from .arepo_processing import (Reader,
                               Dataset,
                               ConcatDataset,
                               StandardDataset,
                               SyntheticallyObservedDataset,
                               CPUBatchObservedDataset,
                               SimplyObservedDataset,
                               PreObservedDataset)


class Writer(ap.Processor):
    r"""
    A class for writing [`Dataset`][iris.arepo_processing.Dataset] objects to the disk.

    Handles all CPU multiprocessing/parallelism with MPI via `mpi4py`.
    Depending on the world and node ranks, launches one of three types of processes:

    * Manager: The world rank 0 process. Coordinates workers and GPU managers,
    Acts as a consolidation point for data.
    * GPU Manager: The highest node-rank process on any node, if that node contains
    multiple processes and GPU support is both available and required.
    Manages access keys for each GPU allocated to its node, and issues these
    keys to workers, ensuring that only one worker can access the GPU at a time.
    This prevents memory overflow on the GPU.
    * Worker: All other processes. Workers receive and execute tasks from the manager.
    The first set of tasks is data generation. Each worker works independently
    to produce one physical tensor and/or observed pair from a snapshot file
    on the disk. The second set of tasks is data normalization. If applicable,
    each worker applies the conversion from processing units to IRIS units
    over its own data points. Upon completion of all tasks, merges its
    dataset back into the manager dataset.

    The manager task first determines which AREPO snapshots will be targeted for data
    production. If `snapshot_paths` is specified, the manager will produce
    `self.hyper.writer_hyper.points_per_snapshot` physical tensors
    for each snapshot. If `snapshot_directory` is specified, the manager will produce
    `self.hyper.writer_hyper.points_per_snapshot` physical tensors
    from `self.hyper.writer_hyper.total_snapshots` snapshots randomly
    selected from this directory. In either case, the `points_per_snapshot` points will
    be symmetrically distributed around the galactic center at the vertices of a regular $n$-gon
    in the galactic disk. If `remote_address` and `local_cache` are not `None`,
    will interpret either `snapshot_paths` or `snapshot_directory` as a path
    on a remote device, and will automatically handle file copy from the remote to the cache
    via SCP, as well as deletion of local copies when complete. If `ssh_key_path` is not `None`,
    will use the key for server access. Otherwise, will use the default SSH behavior of
    the current user session, i.e. search for default keys. Only copies one file at a time
    and deletes when the worker processes are complete and before copying the next file.
    This allows storage of large simulation databases on remote servers and avoids the need
    for manual file transfer to the HPC environment. In conjunction with SLURM job arrays,
    this system can automate weeks of data generation from snapshots stored on the remote,
    while maximizing usage of storage in the HPC environment for processed datasets.

    Attributes:
        hyper: A hyperparameters object.
        path: The path at which to write a [`Dataset`][iris.arepo_processing.Dataset].
            If a readable [`Dataset`][iris.arepo_processing.Dataset] of the same `dataset_type`
            is already present at this path, will open and extend the existing dataset. If
            an unreadable directory is present at this path, will search for an available
            amended path. (See [`_load`][iris.arepo_processing.Dataset._load] and
            [`_make`][iris.arepo_processing.Dataset._make] for details.)
        dataset: The working dataset object.
        verbose: If `True`, will print progress updates.
        world_comm: The MPI intracomm used for communicating with all processes.
        node_comm: The MPI intracomm used for communicating with same-node processes.
        rank: The world rank of this process.
        world_size: The total number of all processes.
        node_rank: The node rank of this process.
        node_size: The number of processes on this node.
        workers: A list of worker processes, identified by world ranks.
        gpu_managers: A list of GPU manager processes, identified by world ranks.
        gpu_interpolate: If `True`, each worker task will ask [`Snapshot`][iris.arepo_processing.Snapshot]
            to interpolate with CuPy on GPU (if available). See
            [`Snapshot.gpu_interpolate`][iris.arepo_processing.Snapshot] for details.
        gpu_normalize: If `True`, each worker task will normalize on GPU.
            (A legacy feature, not a substantial performance gain.)
        snapshot_paths: A list of paths to AREPO snapshots (HDF5 files) from which to generate data.
        snapshot_directory: The `Writer` will randomly select AREPO snapshots (HDF5 files)
            from this directory until all data is exhausted.
        remote_source: Will be set to `True` if `remote_address` and `local_cache` are both not `None`.
        ssh_key_path: Path, on the local device, to the SSH private key file,
            for access to the remote server. If none is specified, will search for default keys.
        remote_address: Address of a remote server, e.g. 'user@remote.server.edu'. If not `None`,
            will interpret `snapshot_paths` and `snapshot_directory` as paths on this remote.
        local_cache: A path to a local cache directory. Will copy snapshots from the remote server
            to this cache as temporary files, deleted after data generation is complete.
            For best performance, create a RAM directory in `/dev/shm/`.
        abundance: An abundance function to supply to the
            [`SyntheticObserver`][iris.observation.SyntheticObserver] of a
            [`SyntheticallyObservedDataset`][iris.arepo_processing.SyntheticallyObservedDataset].
            If `None`, the observer will use the default abundance.
        units_from: If not `None`, will adopt IRIS units from this dataset.
        observer_kwargs: Extra keyword args to pass to an observer, if writing a
            [`PreObservedDataset`][iris.arepo_processing.PreObservedDataset]. Ignored otherwise.

    Args:
        path: Sets `self.path`.
        snapshot_paths: Sets `self.snapshot_paths`.
        snapshot_directory: Sets `self.snapshot_directory`.
        ssh_key_path: Sets `self.ssh_key_path`.
        remote_address: Sets `self.remote_address`.
        local_cache: Sets `self.local_cache`.
        hyper: Sets `self.hyper`.
        dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to write.
        abundance: Sets `self.abundance`.
        units_from: Sets `self.units_from`.
        gpu_interpolate: Sets `self.gpu_interpolate`.
        gpu_normalize: Sets `self.gpu_normalize`.
        verbose: Sets `self.verbose`.
        observer_kwargs: Sets `self.observer_kwargs`.
        
    Raises:
        ValueError: If not one and only one of `snapshot_paths` and `snapshot_directory` is not `None`.
    """
    
    verbose: bool
    world_comm: MPI.Intracomm
    node_comm: MPI.Intracomm
    rank: int
    world_size: int
    node_rank: int
    node_size: int
    workers: list[int]
    gpu_managers: list[int]
    gpu_interpolate: bool
    gpu_normalize: bool
    snapshot_paths: list[str] | None
    snapshot_directory: str | None
    remote_source: bool
    ssh_key_path: str | None
    remote_address: str | None
    local_cache: str | None
    abundance: observation.Abundance | None
    units_from: Dataset | ConcatDataset | None
    observer_kwargs: dict
    
    def __init__(self,
                 path: str,
                 snapshot_paths: typing.Sequence[str] | None = None,
                 snapshot_directory: str | None = None,
                 ssh_key_path: str | None = None,
                 remote_address: str | None = None,
                 local_cache: str | None = None,
                 hyper: hp.Hyper | None = None,
                 dataset_type: type[Dataset] = PreObservedDataset,
                 abundance: observation.Abundance | None = None,
                 units_from: Dataset | ConcatDataset | None = None,
                 gpu_interpolate: bool = True,
                 gpu_normalize: bool = False,
                 verbose: bool = True,
                 **observer_kwargs: any) -> None:
        self.verbose = verbose
        self.world_comm = MPI.COMM_WORLD
        self.node_comm = self.world_comm.Split_type(MPI.COMM_TYPE_SHARED)
        self.rank = self.world_comm.Get_rank()
        self.world_size = self.world_comm.Get_size()
        self.node_rank = self.node_comm.Get_rank()
        self.node_size = self.node_comm.Get_size()
        self.workers = list(range(1, self.world_size))
        self.gpu_managers = []
        self.gpu_interpolate = gpu_interpolate
        self.gpu_normalize = gpu_normalize

        self.snapshot_paths = snapshot_paths
        self.snapshot_directory = snapshot_directory
        if snapshot_directory is None and snapshot_paths is None:
            raise ValueError('Must provide Writer with either snapshot_paths or snapshot_directory.')
        if snapshot_directory is not None and snapshot_paths is not None:
            raise ValueError('Cannot provide Writer with both snapshot_paths and snapshot_directory.')
        if remote_address is not None and local_cache is not None:
            self.remote_source = True
        else:
            self.remote_source = False
        self.ssh_key_path = ssh_key_path
        self.remote_address = remote_address
        self.local_cache = local_cache
        self._get_processing_units(hyper)
        self.abundance = abundance
        self.units_from = units_from
        self.observer_kwargs = observer_kwargs
        super().__init__(path, hyper, dataset_type)

        if self.rank == 0:
            self._manage()
        elif self.node_rank == self.node_size - 1:
            self._manage_gpu(dataset_type)
        else:
            self._work(dataset_type)

        self.world_comm.Barrier()
        return

    def _load_dataset(self, path: str, dataset_type: type[Dataset]) -> None:
        """
        Loads or makes a dataset for extension or writing.

        Args:
            path: A free path or a path to an existing dataset directory of the same `dataset_type`.
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to write.

        Raises:
            RuntimeError: If no [hyperparameters][iris.hyper.Hyper] were provided to the `Writer`.
        """
        if self.rank == 0:
            self.dataset = dataset_type.spawn_parent(path=path,
                                                     hyper=self.hyper,
                                                     abundance=self.abundance,
                                                     node_comm=self.node_comm,
                                                     observer_kwargs=self.observer_kwargs)
            if self.hyper is None:
                if self.dataset.hyper is None:
                    raise Exception('No hyper provided to Processor object.')
                self.hyper = self.dataset.hyper
        return

    def _manage(self) -> None:
        """
        The rank-zero MPI task. Coordinates all task parallelism.
        """
        start = time.time()
        self._census_gpu_managers()
        self._issue_generation_tasks()
        self._receive_datasets()
        if self.units_from is None:
            if self.verbose:
                print(f'Calculating units...', flush=True)
            iris_processing_units_different = self.dataset.calculate_iris_units()
        else:
            if self.verbose:
                print(f'Copying units...', flush=True)
            iris_processing_units_different = self.dataset.take_units(self.units_from.hyper)
        self._issue_normalization_tasks(iris_processing_units_different)
        self._kill_gpu_managers()
        if self.verbose:
            print(f'Shuffling dataset...', flush=True)
        self.dataset.shuffle()
        if self.verbose:
            print(f'Saving dataset...', flush=True)
        self.dataset.save()
        if self.verbose:
            cpu_memory_usage = ap.gauge_cpu_memory()
            print(f'Total CPU memory usage:  {cpu_memory_usage:.2f} GiB', flush=True)
            end = time.time()
            elapsed = (end - start) / 3600
            print(f'Total processing time: {elapsed:.2f} hours', flush=True)
            print('Complete.', flush=True)
        return

    def _census_gpu_managers(self) -> None:
        """
        Checks if GPU managers have been designated.
        """
        if self.verbose:
            print(f'Checking for GPUs...', flush=True)
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES'))
        if self.node_size == 1:
            num_nodes -= 1
        for _ in range(num_nodes):
            rank = self.world_comm.recv(source=MPI.ANY_SOURCE, tag=1)
            if rank is not None:
                self.gpu_managers.append(rank)
        return

    def _kill_gpu_managers(self) -> None:
        """
        Kills GPU manager processes.
        """
        if self.verbose:
            if len(self.gpu_managers) > 0:
                print(f'Killing GPU managers...', flush=True)
            else:
                print(f'No GPU managers to kill...', flush=True)
        for rank in self.gpu_managers:
            self.world_comm.Send(np.array([1], dtype=np.int32), dest=rank, tag=3)
        return

    def _get_processing_units(self, hyper: hp.Hyper) -> None:
        """
        Computes the parsecs per processing length written to `self.hyper.WriterHyper._length_parsec_per_processing`.

        Args:
            hyper: The hyperparameters object from which to pull units.
        """
        cm_per_parsec = 100.0 * hyper.dataset_hyper.meters_per_parsec
        length_cm_per_processing = hyper.writer_hyper.length_cm_per_processing
        hyper.writer_hyper._length_parsec_per_processing = length_cm_per_processing / cm_per_parsec
        return

    @staticmethod
    def _get_snapshot_index(path: str | Path) -> int | None:
        """
        Gets the final integer index from an AREPO snapshot filename.

        Args:
            path: The snapshot path whose filename should be inspected.

        Returns:
            The final integer index from the filename stem, or `None` if no such index exists.
        """
        filename = re.split(r'[\\/]', str(path))[-1]
        stem, _ = os.path.splitext(filename)
        match = re.search(r'(?:^|\D)(\d+)$', stem)
        if match is None:
            return None
        return int(match.group(1))

    def _filter_snapshot_paths_by_index(self, paths: typing.Sequence[str | Path]) -> list[str | Path]:
        """
        Filters snapshot paths by the configured minimum and maximum snapshot index.

        Args:
            paths: Snapshot paths to filter.

        Returns:
            Snapshot paths with final filename indices inside the configured range. If neither
            range bound is configured, returns all paths as a list.
        """
        min_snapshot_index = self.hyper.writer_hyper.min_snapshot_index
        max_snapshot_index = self.hyper.writer_hyper.max_snapshot_index
        if min_snapshot_index is None and max_snapshot_index is None:
            return list(paths)

        filtered_paths = []
        for path in paths:
            snapshot_index = self._get_snapshot_index(path)
            if snapshot_index is None:
                continue
            if min_snapshot_index is not None and snapshot_index < min_snapshot_index:
                continue
            if max_snapshot_index is not None and snapshot_index > max_snapshot_index:
                continue
            filtered_paths.append(path)
        return filtered_paths

    def _issue_generation_tasks(self) -> None:
        r"""
        Issues data generation tasks to worker processes.

        If `self.remote_source`, copies
        single AREPO snapshots at a time from the remote server to the local machine.
        If `self.snapshot_paths` is specified, works through all snapshots one at a time.
        If `self.snapshot_directory` is specified,
        randomly chooses one snapshot from the directory at a time until a total of
        `self.hyper.writer_hyper.total_snapshots` snapshots has been reached.
        For each snapshot, creates `self.hyper.writer_hyper.points_per_snapshot` unique
        data generation tasks. (A unique observation looking in towards the galactic center
        from each vertex of a regular $n$-gon centered on the galactic center. Additional
        uniqueness is added by the perturbations added by [`Snapshot`][iris.arepo_processing.Snapshot].)
        Issues each unique task to a different worker process.

        Raises:
            RuntimeError: If processing from `self.snapshot_directory`,
                but no HDF5 snapshots are found there.
            RuntimeError: If a snapshot download fails.
        """
        if self.verbose:
            print('Issuing generation tasks...', flush=True)
        total_snapshots = self.hyper.writer_hyper.total_snapshots
        points_per_snapshot = self.hyper.writer_hyper.points_per_snapshot

        if self.snapshot_paths is None:
            if self.verbose:
                print('Finding all snapshots at directory...', flush=True)
            if self.remote_source:
                remote_command = f"find {self.snapshot_directory} -maxdepth 1 -type f \\( -name '*.h5' -o -name '*.hdf5' \\)"
                ssh_cmd = ['ssh', '-o', 'BatchMode=yes']
                if self.ssh_key_path is not None:
                    ssh_cmd.extend(['-i', self.ssh_key_path])
                ssh_cmd.extend([self.remote_address, remote_command])
                result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
                paths = [path for path in result.stdout.strip().split('\n') if path != '']
            else:
                self.snapshot_directory = os.path.expanduser(self.snapshot_directory)
                p = Path(self.snapshot_directory)
                paths = list(p.glob('*.hdf5')) + list(p.glob('*.h5'))
            if len(paths) == 0:
                raise RuntimeError(f'No HDF5 snapshots found at {self.snapshot_directory}.')
            paths = self._filter_snapshot_paths_by_index(paths)
            if len(paths) == 0:
                raise RuntimeError(f'HDF5 snapshots found at {self.snapshot_directory}, '
                                   f'but none with indices ranging between '
                                   f'hyper.writer_hyper.min_snapshot_index='
                                   f'{self.hyper.writer_hyper.min_snapshot_index} and '
                                   f'hyper.writer_hyper.max_snapshot_index='
                                   f'{self.hyper.writer_hyper.max_snapshot_index}.')
        else:
            paths = self.snapshot_paths
            total_snapshots = len(paths)
        n = min(total_snapshots, len(paths))
        completed_snapshot = None
        workers = []

        for s in range(n):
            snapshot = paths.pop(random.randint(0, len(paths) - 1))
            if self.remote_source:
                if self.verbose:
                    print(f'Downloading snapshot {s + 1} of {n}\t({os.path.basename(snapshot)})...', flush=True)

                standby = len(workers)
                for _ in range(len(self.workers) - len(self.gpu_managers) - standby):
                    rank = self.world_comm.recv(source=MPI.ANY_SOURCE, tag=2)
                    workers.append(rank)
                snapshot_name = os.path.basename(snapshot)
                local_snapshot = os.path.join(self.local_cache, snapshot_name)
                if completed_snapshot is not None:
                    os.remove(completed_snapshot)
                completed_snapshot = local_snapshot

                with open(local_snapshot, 'wb') as f:
                    ssh_cmd = ['ssh', '-o', 'BatchMode=yes']
                    if self.ssh_key_path is not None:
                        ssh_cmd.extend(['-i', self.ssh_key_path])
                    ssh_cmd.extend([self.remote_address, f'cat "{snapshot}"'])
                    try:
                        subprocess.run(ssh_cmd, stdout=f, stderr=subprocess.PIPE, check=True)
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Snapshot download failed: {e.stderr.decode().strip()}")
                snapshot = local_snapshot

            if self.verbose:
                print(f'Processing snapshot {s + 1} of {n}\t({os.path.basename(snapshot)})...', flush=True)
            for i in range(points_per_snapshot):
                if len(workers) > 0:
                    rank = workers.pop()
                else:
                    rank = self.world_comm.recv(source=MPI.ANY_SOURCE, tag=2)
                theta = 2 * np.pi / points_per_snapshot * i
                self.world_comm.send((self.hyper,
                                      self.dataset.path,
                                      snapshot,
                                      theta),
                                     dest=rank,
                                     tag=3)
        standby = len(workers)
        for _ in range(len(self.workers) - len(self.gpu_managers) - standby):
            rank = self.world_comm.recv(source=MPI.ANY_SOURCE, tag=2)
            workers.append(rank)
        for rank in workers:
            self.world_comm.send(None, dest=rank, tag=3)
        return

    def _receive_datasets(self) -> None:
        """
        Collects all worker [`DatasetChild`][iris.arepo_processing.DatasetChild] objects
        and merges them into the manager [`DatasetParent`][iris.arepo_processing.DatasetParent].
        """
        if self.verbose:
            print('Joining worker data...', flush=True)
        for rank in self.workers:
            if rank not in self.gpu_managers:
                dataset = self.world_comm.recv(source=rank, tag=4)
                if dataset is not None:
                    self.dataset.merge(dataset)
        return

    def _issue_normalization_tasks(self, iris_processing_units_different: bool) -> None:
        """
        Informs each worker process that data generation is complete and instructs
        each process to convert its [`Dataset`][iris.arepo_processing.Dataset]
        from processing units to the newly calculated or adopted IRIS units.

        If no conversion is necessary, instructs each worker process to skip normalization.

        Args:
            iris_processing_units_different: The `bool` passed to each worker process
                determining whether normalization is computed or skipped.
        """
        if self.verbose:
            if iris_processing_units_different:
                print('Normalizing tensors...', flush=True)
            else:
                print('Skipping normalization...', flush=True)
        for rank in self.workers:
            if rank not in self.gpu_managers:
                self.world_comm.send((self.hyper, iris_processing_units_different), dest=rank, tag=5)
        for rank in self.workers:
            if rank not in self.gpu_managers:
                self.world_comm.recv(source=rank, tag=6)
        return

    def _manage_gpu(self, dataset_type: type[Dataset]) -> None:
        """
        Manages access keys for each GPU allocated to its node, and issues these
        keys to workers, ensuring that only one worker can access the GPU at a time.

        Prevents memory overflow on the GPU. If GPU support is not available or required,
        or if the lone process on its node, reverts to a worker process.

        Args:
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to write.
        """
        if (self.node_size == 1 or
                not torch.cuda.is_available() or
                not (issubclass(dataset_type, PreObservedDataset) or self.gpu_interpolate or self.gpu_normalize)):
            self.world_comm.send(None, dest=0, tag=1)
            if self.verbose and torch.cuda.is_available():
                print(f'GPU manager {self.rank} reverted to worker', flush=True)
            self._work(dataset_type)
            return
        gpus = [i for i in range(torch.cuda.device_count())]

        self.world_comm.send(self.rank, dest=0, tag=1)
        kill_signal = np.empty((1,), dtype=np.int32)
        kill_request = self.world_comm.Irecv(kill_signal, source=0, tag=3)
        worker_rank = np.empty((1,), dtype=np.int32)
        worker_request = self.node_comm.Irecv(worker_rank, source=MPI.ANY_SOURCE, tag=7)
        gpu_task_complete = np.empty((3,), dtype=np.int64)
        completion_signal = self.node_comm.Irecv(gpu_task_complete, source=MPI.ANY_SOURCE, tag=9)
        torch_memory_usage = 0
        cupy_memory_usage = 0
        while True:
            complete, _ = worker_request.test()
            if complete:
                rank = int(worker_rank[0])
                if len(gpus) == 0:
                    completion_signal.wait()
                    gpu = int(gpu_task_complete[0])
                    torch_memory_usage = max(torch_memory_usage, int(gpu_task_complete[1]))
                    cupy_memory_usage = max(cupy_memory_usage, int(gpu_task_complete[2]))
                    completion_signal = self.node_comm.Irecv(gpu_task_complete, source=MPI.ANY_SOURCE, tag=9)
                else:
                    gpu = gpus.pop()
                self.node_comm.send(gpu, dest=rank, tag=8)
                worker_request = self.node_comm.Irecv(worker_rank, source=MPI.ANY_SOURCE, tag=7)
            complete, _ = completion_signal.test()
            if complete:
                gpu = int(gpu_task_complete[0])
                torch_memory_usage = max(torch_memory_usage, int(gpu_task_complete[1]))
                cupy_memory_usage = max(cupy_memory_usage, int(gpu_task_complete[2]))
                gpus.append(gpu)
                completion_signal = self.node_comm.Irecv(gpu_task_complete, source=MPI.ANY_SOURCE, tag=9)
            complete, _ = kill_request.test()
            if complete:
                if self.verbose:
                    torch_memory_usage /= 1024 ** 3
                    cupy_memory_usage /= 1024 ** 3
                    print(f'GPU manager {self.rank} max memory usage'
                          f'\n\tPyTorch: {torch_memory_usage:.2f} GiB\tCuPy: {cupy_memory_usage:.2f} GiB',
                          flush=True)
                return

    def _work(self, dataset_type: type[Dataset]) -> None:
        """
        The worker process. Executes two steps: data generation and normalization.

        Args:
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to write.
        """
        self._generate(dataset_type)
        self._normalize()
        return

    def _generate(self, dataset_type: type[Dataset]) -> None:
        """
        The worker data generation task.

        Listens for the task assignment from the manager process. Then reads the task,
        creates a [`Snapshot`][iris.arepo_processing.Snapshot] object, and
        makes a physical tensor and adds it or an observed pair to the dataset
        by calling [`make_physical_tensor`][iris.arepo_processing.Snapshot.make_physical_tensor].
        Upon receiving a null task from the manager, transmits its accumulated
        [`Dataset`][iris.arepo_processing.Dataset] to the manager for merging.
        The data tensors for each `Dataset` are stored on-disk rather than in memory,
        so only the `Dataset` metadata is transmitted.

        Args:
            dataset_type: The type of [`Dataset`][iris.arepo_processing.Dataset] to write.
        """
        while True:
            self.world_comm.send(self.rank, dest=0, tag=2)
            task = self.world_comm.recv(source=0, tag=3)
            if task is None:
                self.world_comm.send(self.dataset, dest=0, tag=4)
                return
            hyper, parent_path, snapshot_path, theta = task
            if self.dataset is None:
                self.hyper = hyper
                name = 'node_{}'.format(self.rank)
                self.dataset = dataset_type.spawn_child(name=name,
                                                        parent_path=parent_path,
                                                        hyper=self.hyper,
                                                        abundance=self.abundance,
                                                        node_comm=self.node_comm,
                                                        observer_kwargs=self.observer_kwargs)
            snapshot = ap.Snapshot(snapshot_path, hyper, gpu_interpolate=self.gpu_interpolate)
            snapshot.make_physical_tensor(self.dataset,
                                          self.node_comm,
                                          theta=theta)

    def _normalize(self) -> None:
        """
        The worker normalization task.

        Listens for the normalization task from the manager process,
        which includes `iris_processing_units_different` and a
        [`Hyper`][iris.hyper.Hyper] object containing the IRIS units
        computed by the manager process. If `iris_processing_units_different`,
        calls the `normalize` method of its [`Dataset`][iris.arepo_processing.Dataset].
        """
        hyper, iris_processing_units_different = self.world_comm.recv(source=0, tag=5)
        if iris_processing_units_different and self.dataset is not None:
            self.dataset.normalize(hyper=hyper,
                                   node_comm=self.node_comm,
                                   gpu_normalize=self.gpu_normalize)
        self.world_comm.send(None, dest=0, tag=6)
        return
