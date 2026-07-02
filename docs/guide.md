# IRIS Documentation

**Author:** B.L. DuBois
**Updated:** 1 July 2026

Welcome to the official documentation for Imagery Reversion Informed by Simulation (IRIS).
For a detailed description of the IRIS project, see the README at the
official GitHub repository (https://github.com/bldubois/IRIS) as well as the
preprint publication ([LINK COMING SOON]). 

In the sections below, you can find a guide for installation and usage of the IRIS code.
This guide is intended for those aiming to reproduce the results of the IRIS paper,
or for those collaborating on future research relating to the IRIS project.
In the other tabs on this page, you can find a detailed description of the
entire IRIS sourcecode for future developers.

## 1. Installation

To begin, clone the IRIS code onto your local machine by running:

```bash
git clone https://github.com/bldubois/IRIS.git
```

Now we'll want to install IRIS as a Python package into a dedicated virtual environment.
Note that IRIS is only suitable for installation onto a high-performance cluster (HPC)
with CUDA GPU support and MPI. A detailed list of requirements can be found in `pyproject.toml`.
By default, GPU support is required for all PyTorch code elements, including
synthetic observation and ML components. Optionally, GPU support can be activated
for interpolation of physical tensors during simulation processing through CuPy.
These CuPy-enabled options make use of prerelease features in CuPy, which requires editing
a machine-specific CuPy `.whl` URL into `pyproject.toml`, and is thus enabled only as an optional
add-on.

First, add the paths to your target Python and OpenMPI distributions into `create_venv.sh`.
Then run:

```bash
bash create_venv.sh
```

IRIS is now on your machine.

## 2. Data Generation

Now that we have IRIS on our machine, we will want to produce data for training
the IRIS reversion model, testing the trained model, and visualizing results.
In all the following data-generation routines, be sure to update the relevant scripts
with paths to local AREPO snapshots or snapshot directories.
Also be sure to update the SLURM batch files according to the needs of your HPC.

Lastly, note that the default scripts will run full data-generation routines
according to the specifications described in the IRIS paper.
These include generation of a ~100k-datapoint training dataset by running a
SLURM job array of 80 separate jobs (~1 week of queued compute running two jobs at a time).
Adjust the job-array specifications in the SLURM files and the dataset parameters in the
`iris.hyper.Hyper` objects if these are not your requirements.

To begin, build a training dataset by navigating to `jobs/` and running:

```bash
sbatch gen_data_training.sh
```

Then generate a litter (foreground/background) dataset, as well as test and visualization
datasets by running:

```bash
sbatch gen_data_litter.sh
sbatch gen_data_test.sh
sbatch gen_data_full_cone.sh
sbatch gen_data_sims_overview.sh
sbatch gen_data_wrong_physics.sh
```

Note that these scripts are configured to adopt the units from the primary training dataset.
Units consistency is not strictly necessary, however, since units conversion
between datasets is managed dynamically by the IRIS code.

## 3. A Note on Synthetic Observation

The IRIS code provides a highly general synthetic-observation capability for AREPO simulations.
See the documentation and IRIS paper for more details on capabilities of the synthetic observation
capabilities of IRIS. In theory, users can build custom observation pipelines for their AREPO
simulations by using `iris.observation` in combination with `iris.arepo_processing.StandardDataset`.
Users may also wish to synthetically observe custom fields by building bespoke processing pipelines
from their simulations to `iris.arepo_processing.StandardDataset` objects. The developers of the
IRIS project leave open the possibility of releasing a more general version of the IRIS
synthetic observation code as its own Python package in the future.

## 4. Model Training

Now that we have a training dataset, let's train the IRIS reversion model.
Be sure to edit the `jobs/train.py` script to include paths to your training data,
and edit `jobs/train.sh` according to the needs of your local HPC.
Note, by default, that `jobs/train.sh` will run six separate instances of the training script,
training six separate models. If this is not your intention, adjust the job-array configuration
in the SLURM file. Once ready, navigate to `jobs/` and run:

```bash
sbatch train.sh
```

## 5. Visualizing Results

We now have a trained IRIS model. To generate all the figures included in the IRIS paper,
first update all the figure scripts and SLURM files in `jobs/` to include the relevant paths
to your trained IRIS models and test/visualization data. Then navigate to `jobs/` and run:

```bash
sbatch fig_cmz_overview.sh
sbatch fig_sims_overview.sh
sbatch fig_optically_thin_vs_thick.sh
sbatch fig_no_dust_vs_dust.sh
sbatch fig_continuum_temperature.sh
sbatch fig_OT_background.sh
sbatch fig_formal_vs_smooth.sh
sbatch fig_simple_vs_synth.sh
sbatch fig_loss_trajectory.sh
sbatch fig_synthetic_reversions.sh
sbatch fig_failure_modes.sh
```

For the following figures, RADMC-3D and POLARIS must be installed on your local machine,
and the relevant scripts must be updated to include paths to the RADMC-3D and POLARIS binaries:

```bash
sbatch fig_radmc_vs_iris.sh
sbatch fig_polaris_vs_iris.sh
sbatch fig_balance_OT_vs_ALT.sh
```

To run the speed test of the IRIS synthetic observation code against RADMC-3D, first run:

```bash
sbatch speed_test.sh
```

This script will run a battery of 81 separate speed comparisons, each twice, 
across a SLURM array of 82 separate jobs. This will take several days of compute,
queueing two jobs at a time. Note that you will also have to update this script with
the path of your RADMC-3D binary. Once complete, run:

```bash
sbatch fig_speed_test.sh
```

To apply IRIS to real observations, as in the IRIS paper via application to the SEDIGISM data,
you will need a FITS file for the observation on your local machine, and you will need
to update the `iris.hyper.Hyper` object with the appropriate path to this FITS file.
Then run:

```bash
sbatch fig_true_reversions.sh
```

## 6. Conclusion

You've now reproduced all the results of the IRIS paper.
For more information on the IRIS project, or on contacting its creators regarding
questions or proposals for collaboration, please see the `README.md` file at the
repository homepage.