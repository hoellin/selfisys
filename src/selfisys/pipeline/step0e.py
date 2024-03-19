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

"""Step 0e of the SELFI pipeline.

Consists in generating all the Simbelmynë parameter files to run the simulations at the
expansion point, in all parameter space directions, in order to linearize the blackbox.

The only simulation actually ran in step 0e are the ones to generate the prior on the
initial spectrum (if using planck2018[_cv]), based on the cosmological parameters drawn
from the prior on the cosmology.
"""

from gc import collect


def worker_fct(params):
    from io import BytesIO
    from selfisys.utils.low_level import stdout_redirector

    x = params[0]
    index = params[1]
    selfi_object = params[2]
    f = BytesIO()
    with stdout_redirector(f):
        selfi_object.run_simulations(d=x, p=index)
    f.close()
    del selfi_object
    collect()
    return 0


import pickle
import numpy as np
from selfisys.utils.parser import *
from selfisys.global_parameters import *


parser = ArgumentParser(
    description="Step 0e of the SELFI pipeline.\
                 Generate all the required Simbelmyne parameter files\
                 for the simulations at the expansion point\
                 and compute the prior on the initial spectrum."
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory")
parser.add_argument(
    "--N_THREADS",
    type=int,
    help="Number of threads for the computation of the prior\
          Will also serve as the number of parameter files to generate in parallel\
          (note that a distinct blackbox object has to be instanciated for each one).",
    default=64,
)
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
    help="Number of samples (drawn from the prior on cosmology) used to compute the\
          prior on the primordial power spectrum (when using planck2018[_cv]).",
    default=int(1e4),
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
    help='Use the effective volume to compute alpha_cv.\
          Only used when prior="planck2018_cv", ignored otherwise.',
    default=False,
)
parser.add_argument(
    "--force_recompute_prior",
    type=bool_sh,
    help="Force overwriting the prior.",
    default=False,
)
parser.add_argument(
    "--verbosity",
    type=int,
    help="Verbosity level. 0: low verbosity, 1: only important outputs, 2: all outputs",
    default=1,
)
parser.add_argument(
    "--Ne",
    type=int,
    help="Number of simulations to keep at the expansion point.\
          If None, all simulations from step 1 are kept.",
    default=None,
)
parser.add_argument(
    "--Ns",
    type=int,
    help="Number of simulations to keep for each component of the gradient.\
          If None, all simulations from step 1 are kept.",
    default=None,
)

args = parser.parse_args()

wd = args.wd
N_THREADS = args.N_THREADS
prior_type = args.prior
nsamples_prior = int(args.nsamples_prior)
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
force_recompute_prior = args.force_recompute_prior
verbosity = args.verbosity

if __name__ == "__main__":
    from os.path import exists
    from pysbmy.timestepping import *
    from selfisys.sbmy_blackbox import blackbox

    modeldir = wd + "model/"

    print("################################")
    print("## Setting up main parameters ##")
    print("################################")
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
    Ne = other_params["Ne"] if args.Ne is None else args.Ne
    Ns = other_params["Ns"] if args.Ns is None else args.Ns
    Delta_theta = other_params["Delta_theta"]
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

    print("Loading main parameters...")
    Pbins_bnd = np.load(modeldir + "Pbins_bnd.npy")
    Pbins = np.load(modeldir + "Pbins.npy")
    k_s = np.load(modeldir + "k_s.npy")
    P_0 = np.load(modeldir + "P_0.npy")
    planck_Pk_EH = np.load(modeldir + "theta_planck.npy")
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

    print("> Loading normalization constants...")
    if not exists(modeldir + "norm_csts.npy"):
        raise ValueError(
            "Normalization constants not found. Please run steps 0c and 0d before 0e."
        )
    else:
        norm_csts = np.load(modeldir + "norm_csts.npy")
    print("> Loading normalization constants done.")

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
        norm_csts=norm_csts,
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
    print("> Instantiating the blackbox done.")

    print("> Loading ground truth spectrum...")
    try:
        theta_gt = np.load(modeldir + "theta_gt.npy")
    except:
        raise ValueError(
            "Ground truth cosmology not found. Please run step 0d step 0e."
        )
    print("> Loading ground truth spectrum done.")

    print("> Loading observations...")
    if exists(modeldir + "phi_obs.npy"):
        phi_obs = np.load(modeldir + "phi_obs.npy")
    else:
        raise ValueError("Observations not found. Please run step 0d 0e.")
    print("> Loading observations done.")

    print("> Setting up the prior and instantiating the selfi object...")
    fname_results = wd + "RESULTS/res.h5"
    pool_prefix = wd + "pool/pool_res_dir_"
    pool_suffix = ".h5"
    from pyselfi.power_spectrum.selfi import power_spectrum_selfi

    if prior_type == "selfi2019":
        from pyselfi.power_spectrum.prior import power_spectrum_prior

        theta_0 = np.ones(S)
        if effective_volume:
            alpha_cv = np.load(modeldir + "alpha_cv.npy")
        else:
            alpha_cv = np.load(modeldir + "alpha_cv.npy")
        prior = power_spectrum_prior(k_s, theta_0, theta_norm, k_corr, alpha_cv, False)
        selfi = power_spectrum_selfi(
            fname_results,
            pool_prefix,
            pool_suffix,
            prior,
            BB_selfi,
            theta_0,
            Ne,
            Ns,
            Delta_theta,
            phi_obs,
        )
        selfi.prior.theta_norm = theta_norm
        selfi.prior.k_corr = k_corr
        selfi.prior.alpha_cv = alpha_cv
    elif prior_type[:10] == "planck2018":
        from selfisys.prior import planck_prior

        theta_planck = np.load(modeldir + "theta_planck.npy")
        theta_0 = theta_planck / P_0
        prior = planck_prior(
            planck_mean,
            planck_cov,
            k_s,
            P_0,
            k_max,
            nsamples=nsamples_prior,
            nthreads=N_THREADS,
            filename=ROOT_PATH
            + "data/stored_priors/planck_prior_S"
            + str(S)
            + "_L"
            + str(L)
            + "_size"
            + str(size)
            + "_"
            + str(nsamples_prior)
            + "_"
            + str(WhichSpectrum)
            + ".npy",
        )
        selfi = power_spectrum_selfi(
            fname_results,
            pool_prefix,
            pool_suffix,
            prior,
            BB_selfi,
            theta_0,
            Ne,
            Ns,
            Delta_theta,
            phi_obs,
        )
    print("> Done.")

    from selfisys.utils.plot_utils import *

    print("Plotting the observed summaries...")
    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(k_s, theta_gt / P_0, label=r"$\theta_{\mathrm{gt}}$", color="C0")
    ax1.set_xscale("log")
    ax1.semilogx(
        k_s,
        planck_Pk_EH / P_0,
        label=r"$P_{\mathrm{Planck}}(k)/P_0(k)$",
        color="C1",
        lw=0.5,
    )
    plt.xlabel("$k\quad[h/Mpc]$")
    plt.ylabel("$[{\\rm Mpc/h}]^3$")
    ax1.grid(which="both", axis="y", linestyle="dotted", linewidth=0.6)
    for k in k_s[:-1]:
        plt.axvline(x=k, color="green", linestyle="dotted", linewidth=0.6)
    plt.axvline(
        x=k_s[-1],
        color="green",
        linestyle="dotted",
        linewidth=0.6,
        label=r"$\theta$-bins boundaries",
    )
    plt.axvline(x=Pbins[0], color="red", linestyle="dashed", linewidth=0.5)
    plt.axvline(x=Pbins[-1], color="red", linestyle="dashed", linewidth=0.5)
    for k in Pbins[1:-2]:
        plt.axvline(x=k, ymax=0.167, color="red", linestyle="dashed", linewidth=0.5)
    ax1.legend(loc=2)
    plt.xlim(max(1e-4, k_s.min() - 2e-4), k_s.max())
    plt.ylim(7e-1, 1.6e0)

    ax2 = ax1.twinx()
    plt.axvline(
        x=Pbins[-2],
        ymax=0.333,
        color="red",
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\psi$-bins centers",
    )
    len_obs = len(phi_obs) // np.shape(selection_params)[1]
    cols = ["C4", "C5", "C6", "C7"]
    for i in range(np.shape(selection_params)[1]):
        ax2.plot(
            Pbins,
            phi_obs[i * len_obs : (i + 1) * len_obs],
            marker="x",
            label=r"Summary $\psi_{\mathrm{obs}}$, " + str(i),
            linewidth=0.5,
            color=cols[i],
        )
    ax2.legend(loc=1)
    plt.title(
        "Observations generated with the ground truth cosmology and the well specified models"
    )
    plt.savefig(wd + "Figures/summary_obs_step0e.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print("Done.")

    print("##############################")
    print("## Computing/loading priors ##")
    print("##############################")

    error_str_prior = (
        "Error while computing the prior. For OOM issues, a simple fix"
        + "might be to set os.environ['OMP_NUM_THREADS'] = '1'. Otherwise, refer to"
        + "the error message."
    )

    if prior_type != "selfi2019":
        if not force_recompute_prior:
            try:
                selfi.prior = selfi.prior.load(selfi.fname)
                print("Prior loaded from file")
            except:
                print("Prior not found in {}, recomputing...".format(selfi.fname))
                try:
                    selfi.compute_prior()
                    selfi.save_prior()
                    selfi.prior = selfi.prior.load(selfi.fname)
                except:
                    print(error_str_prior)
                    exit(1)
        else:
            print("Forcing recomputation of the prior (asked by user)...")
            selfi.compute_prior()
            selfi.save_prior()
            selfi.prior = selfi.prior.load(selfi.fname)
    else:
        selfi.compute_prior()
        selfi.save_prior()
        selfi.load_prior()

    print("#############################")
    print("## Writing parameter files ##")
    print("#############################")
    from os import cpu_count
    import tqdm.auto as tqdm
    from multiprocessing import Pool

    # TODO: use the blackbox object once first here to make sure it is correctly set up
    #       and avoid any access issues in the Pool below

    BB_selfi.switch_recompute_pool()
    BB_selfi.switch_setup()

    list_part_1 = [[0, idx, selfi] for idx in range(Ne)]
    list_part_2 = [[x, None, selfi] for x in range(1, S + 1)]

    ncors = cpu_count()
    nprocess = min(N_THREADS, ncors, len(list_part_1) + len(list_part_2))
    print("Using {} processes to generate Simbelmynë parameter files".format(nprocess))
    collect()
    # WARNING: it is mandatory to start with list_part_1  to avoid file access issues
    print("> Generating parameter files for the estimation of f0...")
    with Pool(processes=nprocess) as mp_pool:
        pool = mp_pool.map(worker_fct, list_part_1)
        for contrib_to_grad in tqdm.tqdm(pool, total=len(list_part_1)):
            pass
    print("> Generating parameter files for the estimation of the gradient...")
    with Pool(processes=nprocess) as mp_pool:
        pool = mp_pool.map(worker_fct, list_part_2)
        for contrib_to_grad in tqdm.tqdm(pool, total=len(list_part_2)):
            pass

    BB_selfi.switch_setup()
    BB_selfi.switch_recompute_pool()

    print("Done.")
