# IRIS: Imagery Reversion Informed by Simulation

Imagery Reversion Informed by Simulation (IRIS) is a large machine-learning project for astrophysics.
Its purpose is in leveraging ML to infer 3D structure in the
Milky Way Galaxy's Central Molecular Zone (CMZ) through the intelligent comparison of
observational data and galaxy simulations. The IRIS code serves three primary functionalities:
(i) processing AREPO galaxy simulations into training data;
(ii) producing synthetic, non-LTE spectral-line observations of processed simulations; and
(iii) training a custom-designed, deep convolutional neural network to "revert"
observations into top-down density maps of the observed field.
The method and results of the project are described in full in the IRIS publication,
currently in preprint at [LINK]. 

## Installation, Usage, and Documentation

A complete documentation of the IRIS code is provided at [LINK].
Included in the documentation is an installation and usage guide,
for those interested in reproducing the results of the IRIS paper,
or in collaborating on future development of the IRIS project.

## Authorship

IRIS was created by B.L. DuBois at the University of Connecticut's Milky Way Lab
(https://battersby.physics.uconn.edu/), under the advisement of Dr. Cara Battersby.
All design and implementation of the primary code is due to B.L. DuBois.

Jonah Baade, Jack Sullivan, and Stefan Reissl&mdash;co-authors on the
IRIS paper&mdash;helped write code for one of the paper's figures, comparing
synthetic observations produced by IRIS to those produced by the synthetic observation
code POLARIS (https://github.com/polaris-MCRT/POLARIS). This figure code is contained
in the scripts `jobs/fig_polaris_vs_iris.py` and `jobs/fig_polaris_vs_iris_no_dust.py`.

## Questions and Contributions

For more information on B.L. DuBois, who can be reached via email at brendan@bldubois.com,
please see https://bldubois.com. Future research relating to the IRIS project will continue to
be housed at the Milky Way Laboratory under the direction of Dr. Cara Battersby,
who can be reached via email at cara.battersby@uconn.edu.
Both B.L. DuBois and Dr. Battersby are happy to respond to any questions relating to the IRIS
code as able. B.L. DuBois, Dr. Battersby, and the entire team of authors on the IRIS paper
also enthusiastically invite all researchers interested in collaborations related to the
IRIS project to direct their inquiries to Dr. Battersby.

## Citation and License

All IRIS code is under the copyright of the University of Connecticut, released
for open-source usage under the MIT license (see LICENSE.md). We ask that any research
referencing any element of the IRIS code or results cite the primary IRIS paper,
currently in preprint at [LINK]. 
