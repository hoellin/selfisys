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

"""Step 0a of the SELFI pipeline.

The first step consists in generating all the Simbelmyne parameter files required to
normalize the blackbox, and computing the white noise fields. At this point, no
simulations are run except to compute the ground truth spectrum.
"""
import gc
from pathlib import Path
import numpy as np

from selfisys.utils.parser import *
from selfisys.global_parameters import *

parser = ArgumentParser(
    description="Run the first step of the SELFI pipeline.\
                 The first step consists in generating all the Simbelmyne parameter \
                 files required to normalize the blackbox."
)
parser.add_argument(
    "--wd_ext",
    type=str,
    help="Name of the working directory (relative to ROOT_PATH defined in \
          `../global_parameters.py`), ending with a slash.",
)
parser.add_argument(
    "--name",
    type=str,
    help="Suffix to the working directory for this run.\
          Only the white noise fields are shared between runs with different names.",
    default="std",
)
parser.add_argument(
    "--total_steps", type=int, help="Number of timesteps.", default=None
)
parser.add_argument(
    "--aa",
    type=float,
    nargs="*",
    help="List of scale factor at which to synchronise the kicks and drifts.",
    default=None,
)
parser.add_argument(
    "--size",
    type=int,
    help="Number of elements of the simulation grid in each direction.",
    default=512,
)
parser.add_argument(
    "--Np0", type=int, help="Number of DM particles in each direction.", default=1024
)
parser.add_argument(
    "--Npm0",
    type=int,
    help="Number of elements in each direction of the particle-mesh grid.",
    default=1024,
)
parser.add_argument(
    "--L", type=int, help="Size of the simulation box in Mpc/h.", default=3600
)
parser.add_argument(
    "--S",
    type=int,
    help="Number of support wavenumbers to define the initial matter power spectrum.",
    default=64,
)
parser.add_argument(
    "--Pinit",
    type=int,
    help="Maximum number of bins for the summaries.\
          The actual number of bins may be smaller since it is tuned to ensure that \
          each bin contains a sufficient number of modes (see setup_model.py).",
    default=50,
)
parser.add_argument(
    "--Nnorm",
    type=int,
    help="Number of simulations used to compute the normalization of the summaries.",
    default=10,
)
parser.add_argument(
    "--Ne",
    type=int,
    help="Number of simulations at the expansion point to linearize the blackbox.",
    default=300,
)
parser.add_argument(
    "--Ns",
    type=int,
    help="Number of simulations for each component of the gradient at expansion point.",
    default=20,
)
parser.add_argument(
    "--Delta_theta",
    type=float,
    help="Step size to compute the gradient by finite differences.",
    default=1e-2,
)
parser.add_argument("--OUTDIR", type=str, help="Absolute path of the output directory.")
parser.add_argument(
    "--prior",
    type=str,
    help='Prior for the parameters. Possible values are:\
            - "selfi2019": prior used in the SELFI 2019 article\
            - "planck2018": prior obtained by sampling from Planck 2018 cosmology\
            - "planck2018_cv": same as "planck2018" but accounting for cosmic variance\
          Note that "selfi2019" and "planck2018_cv" have not been checked with the\
          current version of the code. Use at your own risk.',
    default="planck2018",
)
parser.add_argument(
    "--nsamples_prior",
    type=int,
    help="Number of samples (drawn from the prior on cosmology) to compute the prior \
          on the initial power spectrum after inflation (when using planck2018[_cv]).",
    default=int(5e4),
)
parser.add_argument(
    "--radial_selection",
    type=none_or_bool_or_str,
    help='Use a radial selection function. Available options are:\
            - None: no radial selection\
            - "multiple_lognormal": lognormal selection w/ multiple populations',
    default="multiple_lognormal",
)
parser.add_argument(
    "--selection_params",
    type=float,
    nargs="*",
    help="Parameters for the selection function of the well specified model.\
          See `sbmy_blackbox.py` for more details.",
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    help="Absolute path of the extinction mask.",
    default=None,
)
parser.add_argument(
    "--sim_params",
    type=none_or_bool_or_str,
    help="Parameters for the simulations.",
    default=None,
)
parser.add_argument(
    "--effective_volume",
    type=bool_sh,
    help='Use the effective volume to compute alpha_cv.\
          Only used when prior="planck2018_cv", ignored otherwise.',
    default=False,
)
parser.add_argument(
    "--lin_bias",
    type=float,
    nargs="*",
    help="Linear bias for the well specified model.",
)
parser.add_argument(
    "--obs_density",
    type=none_or_bool_or_str,
    help="Observation density for the well specified model.",
    default=None,
)
parser.add_argument(
    "--noise", type=float, help="Noise for the well specified model.", default=0.1
)
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity level. 0: low verbosity, 1: only important outputs, 2: all outputs",
    default=1,
)
parser.add_argument(
    "--force", type=bool_sh, help="Force the computations.", default=False
)

args = parser.parse_args()

wd_ext = args.wd_ext
name = args.name
total_steps = args.total_steps
aa = args.aa
size = args.size
Np0 = args.Np0
Npm0 = args.Npm0
L = args.L
S = args.S
Pinit = args.Pinit
Nnorm = args.Nnorm
Ne = args.Ne
Ns = args.Ns
Delta_theta = args.Delta_theta
OUTDIR = args.OUTDIR
prior_type = args.prior
nsamples_prior = int(args.nsamples_prior)
radial_selection = args.radial_selection
selection_params = np.reshape(np.array(args.selection_params), (3, -1))
survey_mask_path = args.survey_mask_path
sim_params = args.sim_params
effective_volume = args.effective_volume
lin_bias = (
    np.array(args.lin_bias) if type(args.lin_bias) is not float else args.lin_bias
)
obs_density = args.obs_density
noise = args.noise
verbosity = args.verbosity
force = args.force

wd_noname = OUTDIR + wd_ext + str(size) + str(int(L)) + str(Pinit) + str(Nnorm) + "/"
wd = wd_noname + name + "/"
modeldir = wd + "model/"

k_max = int(1e3 * np.pi * size * np.sqrt(3) / L + 1) * 1e-3

# Cosmo at the expansion point:
params_planck = params_planck_kmax_missing.copy()
params_planck["k_max"] = k_max
# BBKS spectrum with fiducial cosmology, for normalization:
params_BBKS = params_BBKS_kmax_missing.copy()
params_BBKS["k_max"] = k_max
# Observed cosmology:
params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
params_cosmo_obs["k_max"] = k_max

Path(wd + "RESULTS/").mkdir(parents=True, exist_ok=True)
Path(modeldir).mkdir(exist_ok=True)
Path(wd_noname + "wn/").mkdir(exist_ok=True)
Path(wd + "data/").mkdir(parents=True, exist_ok=True)
figuresdir = wd + "Figures/"
Path(figuresdir).mkdir(exist_ok=True)
Path(wd + "pool/").mkdir(exist_ok=True)
Path(wd + "score_compression/").mkdir(exist_ok=True)
for d in range(S + 1):
    dirsims = wd + "pool/d" + str(d) + "/"
    Path(dirsims).mkdir(parents=True, exist_ok=True)

# Save parameters:
np.save(modeldir + "radial_selection.npy", radial_selection)
np.save(modeldir + "selection_params.npy", selection_params)
np.save(modeldir + "lin_bias.npy", lin_bias)
np.save(modeldir + "obs_density.npy", obs_density)
np.save(modeldir + "noise.npy", noise)

if __name__ == "__main__":

    print("################################")
    print("## Setting up main parameters ##")
    print("################################")
    from os.path import exists
    from pysbmy.timestepping import *
    from selfisys.sbmy_blackbox import blackbox
    from selfisys.setup_model import setup_model
    from selfisys.utils.timestepping import merge_nTS
    import pickle

    print("Setting up the binning and main parameters...")

    params = setup_model(
        workdir=modeldir,
        params_planck=params_planck,
        params_P0=params_BBKS,
        size=size,
        L=L,
        S=S,
        Pinit=Pinit,
        force=force,
    )
    gc.collect()
    (
        size,
        L,
        P,
        S,
        G_sim_path,
        G_ss_path,
        Pbins_bnd,
        Pbins,
        k_s,
        P_ss_obj_path,
        P_0,
        planck_Pk_EH,
    ) = params
    other_params = {
        "size": size,
        "P": P,
        "Np0": Np0,
        "Npm0": Npm0,
        "L": L,
        "S": S,
        "total_steps": total_steps,
        "aa": aa,
        "G_sim_path": G_sim_path,
        "G_ss_path": G_ss_path,
        "P_ss_obj_path": P_ss_obj_path,
        "Pinit": Pinit,
        "Nnorm": Nnorm,
        "Ne": Ne,
        "Ns": Ns,
        "Delta_theta": Delta_theta,
        "sim_params": sim_params,
    }

    with open(modeldir + "other_params.pkl", "wb") as f:
        pickle.dump(other_params, f)

    with open(wd + "params.txt", "w") as f:
        f.write("Parameters for this run:\n")
        f.write("size: " + str(size) + "\n")
        f.write("Np0: " + str(Np0) + "\n")
        f.write("Npm0: " + str(Npm0) + "\n")
        f.write("L: " + str(L) + "\n")
        f.write("S: " + str(S) + "\n")
        f.write("Pinit: " + str(Pinit) + "\n")
        f.write("P: " + str(P) + "\n")
        f.write("Nnorm: " + str(Nnorm) + "\n")
        f.write("total_steps: " + str(total_steps) + "\n")
        f.write("aa: " + str(aa) + "\n")
        f.write("Ne: " + str(Ne) + "\n")
        f.write("Ns: " + str(Ns) + "\n")
        f.write("Delta_theta: " + str(Delta_theta) + "\n")
        f.write("OUTDIR: " + OUTDIR + "\n")
        f.write("prior_type: " + prior_type + "\n")
        f.write("nsamples_prior: " + str(nsamples_prior) + "\n")
        f.write("radial_selection: " + str(radial_selection) + "\n")
        f.write("selection_params:\n" + str(selection_params) + "\n")
        f.write("survey_mask_path: " + str(survey_mask_path) + "\n")
        f.write("effective_volume: " + str(effective_volume) + "\n")
        f.write("lin_bias: " + str(lin_bias) + "\n")
        f.write("obs_density: " + str(obs_density) + "\n")
        f.write("noise: " + str(noise) + "\n")
    print("Done.")

    print("Generating ground truth spectrum...")
    if (not exists(modeldir + "theta_gt.npy")) or force:
        from pysbmy.power import get_Pk

        theta_gt = get_Pk(k_s, params_cosmo_obs)
        np.save(modeldir + "theta_gt", theta_gt)
        del theta_gt
    print("Done.")

    def theta2P(theta):
        return theta * P_0

    print("Setting up the time-stepping...")
    print("> Creating 3 separate time-stepping objects...")
    TimeStepDistribution = 0
    nsteps = [
        round((aa[i + 1] - aa[i]) / (aa[-1] - aa[0]) * total_steps)
        for i in range(len(aa) - 1)
    ]
    if sum(nsteps) != total_steps:
        nsteps[nsteps.index(max(nsteps))] += total_steps - sum(nsteps)
    indices_steps_cumul = list(np.cumsum(nsteps) - 1)

    TSs = []
    TS_paths = []
    for i in range(len(nsteps)):
        ai = aa[i]
        af = aa[i + 1]
        snapshots = np.full((nsteps[i]), False)
        snapshots[-1] = True
        TS = StandardTimeStepping(ai, af, snapshots, TimeStepDistribution)
        TS_path = modeldir + "ts" + str(i + 1) + ".h5"
        TS_paths.append(TS_path)
        TS.write(TS_path)
        TSs.append(read_timestepping(TS_path))

    for i, TS in enumerate(TSs):
        TS.plot(path=figuresdir + "TS" + str(i) + ".png")

    print("> Merging time-stepping...")
    merged_path = modeldir + "merged.h5"
    merge_nTS(TS_paths, merged_path)
    TS_merged = read_timestepping(merged_path)
    TS_merged.plot(path=figuresdir + "TS_merged.png")
    if sim_params[:6] == "custom":
        TimeStepDistribution = merged_path
        eff_redshifts = 1 / aa[-1] - 1
    else:
        raise NotImplementedError("Chosen time-stepping strategy not yet implemented.")

    print("###############################")
    print("## Initializing the blackbox ##")
    print("###############################")

    print("Instantiating the blackbox...")
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
    print("Done. Creating Simbelmyne parameter files to compute the normalization...")
    print("> Putting the blackbox in setup_only mode...")
    BB_selfi.switch_setup()
    print(">> setup_only = {} (expected: True)".format(str(BB_selfi.setup_only)))
    for i in range(Nnorm):
        print("> Setting Simbelmynë file {}/{}...".format(i + 1, Nnorm))
        BB_selfi.worker_normalization_public(params_planck, Nnorm, i)
    print("> Putting the blackbox in normal mode (switching off setup_only)...")
    BB_selfi.switch_setup()
    print(">> setup_only = {} (expected: False)".format(str(BB_selfi.setup_only)))

    print("Done.")
