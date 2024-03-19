#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------------
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
# -------------------------------------------------------------------------------------

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"

"""Step 0b of the SELFI pipeline.
Run all the Simbelmynë simulations required to normalize the blackbox.

This script uses the pySbmy wrapper to run Simbelmynë simulations.
"""
import os
from numpy import array
from selfisys.utils.parser import *

parser = ArgumentParser(description="Run Simbelmynë for step 0b of the SELFI pipeline.")
parser.add_argument("--pool_path", type=str, help="Path to the pool of simulations.")
parser.add_argument("--ii", type=int, nargs="+", help="Indices of simulations to run.")
parser.add_argument(
    "--npar", type=int, help="Number of simulations to run in parallel.", default=4
)
parser.add_argument(
    "--force", type=bool_sh, help="Force the computations.", default=False
)

args = parser.parse_args()
pool_path = args.pool_path
npar = args.npar
force = args.force
ii = array(args.ii, dtype=int)
if len(ii) == 1 and ii[0] == -1:
    ii = array(
        [
            int(f.split("norm__")[1].split("_")[0])
            for f in os.listdir(pool_path)
            if f.startswith("sim_norm") and f.endswith(".sbmy")
        ],
        dtype=int,
    )
nsim = len(ii)


def worker_norm(i):
    from pysbmy import pySbmy
    from selfisys.utils.low_level import stdout_redirector, stderr_redirector
    from io import BytesIO

    file_prefix = "sim_norm__" + str(i)
    suffix = [str(f) for f in os.listdir(pool_path) if f.startswith(file_prefix)][0]
    fname_simparfile = pool_path + suffix
    fname_output = (
        pool_path + "output_density_" + suffix.split(".")[0].split("sim_")[1] + ".h5"
    )
    fname_simlogs = pool_path + file_prefix + ".txt"
    if os.path.isfile(fname_output) and not force:
        print("> Output file {} already exists, skipping...".format(fname_output))
    else:
        f = BytesIO()
        g = BytesIO()
        with stdout_redirector(f):
            with stderr_redirector(g):
                pySbmy(fname_simparfile, fname_simlogs)
            g.close()
        f.close()


if __name__ == "__main__":
    from tqdm import tqdm
    from multiprocessing import Pool

    print("Running the simulations to normalize the blackbox...")
    with Pool(processes=npar) as pool:
        for _ in tqdm(pool.imap(worker_norm, ii), total=nsim):
            pass
    print("Running the simulations to normalize the blackbox done.")
