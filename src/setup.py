#!/usr/bin/env python
#-------------------------------------------------------------------------------------
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# The text of the license is located in the root directory of the source package.
#-------------------------------------------------------------------------------------

"""
Module to perform implicit likelihood inference of cosmological parameters based on
mocks of a toy model of galaxy surveys mimicking a few systematic effects, while using
the SELFI method to check for model misspecification. The approach works in 2 steps:
1) a. Use SELFI to infer the initial matter power spectrum after inflation using the
      full complex forward model to generate mocks. The model is treated as a blackbox.
   b. Use the inferred power spectrum to diagnose model misspecification and assess the
      impact of known sources of systematics on the inference.
2) Use implicit likelihood inference (e.g. ABC-PMC or more advanced methods) to infer
   cosmological parameters. To that purpose, all simulations performed for step 1) can
   and should be recycled for optimal data compression.

This code is intended to demonstrate how the SELFI package
(https://github.com/florent-leclercq/pyselfi/) can be used together with any custom
user-defined blackbox forward model, in order to diagnose know sources of systematic
sat play in the survey. 

The gravitational evolution in the forward model used in this package relies on COLA
(COmoving Lagrangian Acceleration) using the Simbelmynë probabilistic simulator
(https://florent-leclercq.eu/data.php?page=2).
"""

__author__  = "Tristan Hoellinger"
__version__ = "0.1"
__date__    = "2024"
__license__ = "GPLv3"

from setuptools import setup

setup(name='selfi_sys_public',
   version='0.1',
   author='Tristan Hoellinger',
   packages=['selfisys'])