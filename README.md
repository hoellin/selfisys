# Assessing the impact of systematics in galaxy surveys with `pySELFI`

## General information

This repository contains a toy model of galaxy surveys to investigate the impact of model misspecification using the [SELFI](10.5281/zenodo.7576130) algorithm (see also: [arXiv:1902.10149](arXiv:1902.10149) and [arXiv:2209.11057](arXiv:2209.11057)).

## Requirements

The code is written in Python 3.10 and requires the following packages:

- [`pySELFI`](https://github.com/florent-leclercq/pyselfi): a Python implementation of the Simulator Expansion for Likelihood-Free Inference.
- [`Simbelmynë`](https://simbelmyne.readthedocs.io/en/latest/): a hierarchical probabilistic simulator to generate synthetic galaxy survey data
- [`ELFI`](https://github.com/elfi-dev/elfi): a statistical software package for likelihood-free inference such as ABC-PMC

along with their dependencies.

A `yaml` file with all the required packages will be provided in the near future.