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

"""Step 0c of the SELFI pipeline.

Compute the normalization constants based on the simulations performed in step 0b.
"""

import numpy as np
from selfisys.utils.parser import *
from selfisys.global_parameters import *

parser = ArgumentParser(
    description="Step 0c of the SELFI pipeline.\
                 Compute the normalization constants based on the simulations performed\
                 in step 0b."
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory")
parser.add_argument(
    "--npar_norm",
    type=int,
    help="Number of simulations to load in parallel to compute the summaries.\
          Note that the overdensity fields were already computed at step 0b.",
    default=1,
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    help="Path to the survey mask for the well specified model.",
    default=None,
)
parser.add_argument(
    "--effective_volume",
    type=bool_sh,
    help="Use the effective volume to compute alpha_cv.",
    default=False,
)
parser.add_argument(
    "--force",
    type=bool_sh,
    help="Force the recomputation of the mocks.",
    default=False,
)
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity level. 0: low verbosity, 1: only important outputs, 2: all outputs",
    default=1,
)

args = parser.parse_args()

wd = args.wd
npar_norm = args.npar_norm
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
force = args.force
verbosity = args.verbosity

if __name__ == "__main__":
    from os.path import exists
    from pysbmy.timestepping import *
    import pickle
    from selfisys.sbmy_blackbox import blackbox

    print("################################")
    print("## Setting up main parameters ##")
    print("################################")
    modeldir = wd + "model/"

    print("Loading main parameters...")
    with open(modeldir + "other_params.pkl", "rb") as f:
        other_params = pickle.load(f)
    size = other_params["size"]
    Np0 = other_params["Np0"]
    Npm0 = other_params["Npm0"]
    L = other_params["L"]
    S = other_params["S"]
    total_steps = other_params["total_steps"]
    aa = other_params["aa"]
    P = other_params["P"]
    G_sim_path = other_params["G_sim_path"]
    G_ss_path = other_params["G_ss_path"]
    P_ss_obj_path = other_params["P_ss_obj_path"]
    Nnorm = other_params["Nnorm"]
    sim_params = other_params["sim_params"]

    radial_selection = np.load(modeldir + "radial_selection.npy", allow_pickle=True)
    if radial_selection == None:
        radial_selection = None
    selection_params = np.load(modeldir + "selection_params.npy")
    lin_bias = np.load(modeldir + "lin_bias.npy")
    obs_density = np.load(modeldir + "obs_density.npy", allow_pickle=True)
    if obs_density == None:
        obs_density = None
    noise = np.load(modeldir + "noise.npy")

    k_max = int(1e3 * np.pi * size * np.sqrt(3) / L + 1) * 1e-3
    # Cosmo at the expansion point:
    params_planck_EH = params_planck_kmax_missing.copy()
    params_planck_EH["k_max"] = k_max

    Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
    Pbins = np.load(modeldir + "Pbins.npy")
    k_s = np.load(modeldir + "k_s.npy")
    P_0 = np.load(modeldir + "P_0.npy")
    print("Done.")

    def theta2P(theta):
        return theta * P_0

    print("Setting up the time-stepping...")
    nsteps = [
        round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
        for i in range(len(aa) - 1)
    ]
    if sum(nsteps) != total_steps:
        nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)
    indices_steps_cumul = list(np.cumsum(nsteps) - 1)
    merged_path = modeldir + "merged.h5"
    TS_merged = read_timestepping(merged_path)
    if sim_params[:6] == "custom":
        TimeStepDistribution = merged_path
        eff_redshifts = 1 / aa[-1] - 1
    else:
        raise NotImplementedError("Chosen time-stepping strategy not yet implemented.")

    print("###############################")
    print("## Initializing the blackbox ##")
    print("###############################")

    print("> Instantiating the blackbox...")
    BB_selfi = blackbox(
        k_s=k_s,
        P_ss_path=P_ss_obj_path,
        Pbins_bnd=Pbins_bnd,
        theta2P=theta2P,
        P=P * np.shape(selection_params)[1],  # (P * Npop
        size=size,
        L=L,
        G_sim_path=G_sim_path,
        G_ss_path=G_ss_path,
        Np0=Np0,
        Npm0=Npm0,
        fsimdir=wd[:-1],
        noise_std=noise,
        radial_selection=radial_selection,
        selection_params=selection_params,
        observed_density=obs_density,
        linear_bias=lin_bias,
        norm_csts=None,
        survey_mask_path=survey_mask_path,
        sim_params=sim_params,
        TimeStepDistribution=TimeStepDistribution,
        TimeSteps=indices_steps_cumul,
        eff_redshifts=eff_redshifts,
        seedphase=BASELINE_SEEDPHASE,
        seednoise=BASELINE_SEEDNOISE,
        fixnoise=False,
        seednorm=BASELINE_SEEDNORM,
        reset=False,
        save_frequency=5,
        verbosity=verbosity,
    )
    print("> Done.")
    if not exists(modeldir + "norm_csts.npy") or force:
        if force:
            BB_selfi.switch_recompute_pool()
        norm_csts = BB_selfi.define_normalization(
            Pbins,
            params_planck_EH,
            Nnorm,
            min_k_norma=min_k_norma,
            npar=npar_norm,
        )
        if force:
            BB_selfi.switch_recompute_pool()
        np.save(modeldir + "norm_csts.npy", norm_csts)
    else:
        print("> Normalization constants already exist.")
        norm_csts = np.load(modeldir + "norm_csts.npy")
    print("> norm_csts = {}".format(norm_csts))
