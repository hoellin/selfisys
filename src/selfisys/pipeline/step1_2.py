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

"""Run all the Simbelmyne simulations based on the sbmy files written at step 0.
"""

import os
import gc
import numpy as np
from selfisys.utils.parser import *

parser = ArgumentParser(
    description="Run the Simbelmyne simulations required to linearize the blackbox."
)
parser.add_argument("--pool_path", type=str, help="Path to the pool of simulations.")
parser.add_argument("--directions", type=int, nargs="+", help="List of directions.")
parser.add_argument("--pp", type=int, nargs="+", help="List of simulation indices p.")
parser.add_argument("--Npop", type=int, help="Number of populations.", default=None)
parser.add_argument("--npar", type=int, help="# of sim to run in //.", default=8)
parser.add_argument("--force", type=bool_sh, help="Force computations.", default=False)

args = parser.parse_args()
pool_path = args.pool_path
force = args.force
directions = np.array(args.directions, dtype=int)
pp = np.array(args.pp, dtype=int)
if len(pp) == 1 and pp[0] == -1:
    import os

    pp = np.array(
        [
            int(f.split("_")[2].split(".")[0][1:])
            for f in os.listdir(pool_path + "d" + str(directions[0]))
            if f.startswith("sim_d") and f.endswith(".sbmy")
        ],
        dtype=int,
    )
Npop = args.Npop
npar = args.npar
ndir = len(directions)
nsim_per_dir = len(pp)


def run_sim(val):
    from pysbmy import pySbmy

    d, p = val
    dir = pool_path + "d" + str(d) + "/"
    fname_simparfile = dir + "sim_d" + str(d) + "_p" + str(p) + ".sbmy"
    if Npop is not None:
        fname_simparfile = dir + "sim_d{}_p{}_{}.sbmy".format(d, p, Npop)
    else:
        fname_simparfile = dir + "sim_d{}_p{}.sbmy".format(d, p)
    fname_output = dir + "output_density_d" + str(d) + "_p" + str(p) + ".h5"
    fname_simlogs = dir + "logs_sim_d" + str(d) + "_p" + str(p) + ".txt"
    if os.path.isfile(fname_output) and not force:
        gc.collect()
        print("> Output file {} already exists, skipping...".format(fname_output))
    else:
        from io import BytesIO
        from selfisys.utils.low_level import stdout_redirector, stderr_redirector

        # print("> Running Simbelmyne for direction d = {} and p = {}...".format(d, p))
        # sys.stdout.flush()
        f = BytesIO()
        g = BytesIO()
        with stdout_redirector(f):
            with stderr_redirector(g):
                pySbmy(fname_simparfile, fname_simlogs)
            g.close()
        # sys.stdout.flush()
        gc.collect()
        # print("> Running Simbelmyne for direction d = {} and p = {} done.".format(d, p))


if __name__ == "__main__":
    from itertools import product
    import tqdm.auto as tqdm

    vals = list(product(directions, pp))
    nsim = len(vals)
    from multiprocessing import Pool
    with Pool(processes=npar) as pool:
        for _ in tqdm.tqdm(pool.imap(run_sim, vals), total=nsim):
            pass
        print("> Running Simbelmyne done.")
