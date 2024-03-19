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

"""Third step of the SELFI pipeline using PM/COLA simulations with Simbelmyne.
The third step consists in running the actual inference of the initial matter power
spectrum using SELFI, based on the simulations performed in the previous steps.
"""

import gc


def worker_fct(params):
    from io import BytesIO
    from selfisys.utils.low_level import stdout_redirector, stderr_redirector

    x = params[0]
    index = params[1]
    selfi_object = params[2]
    f = BytesIO()
    g = BytesIO()
    with stdout_redirector(f):
        with stderr_redirector(g):
            selfi_object.run_simulations(d=x, p=index)
    g.close()
    f.close()

    del selfi_object
    gc.collect()
    return 0


from pathlib import Path
import numpy as np
from selfisys.utils.parser import *
from selfisys.global_parameters import *

parser = ArgumentParser(
    description="Run the SELFI pipeline based on the simulations"
    + "performed in the previous steps."
)
parser.add_argument("--wd", type=str, help="Absolute path of the working directory")
parser.add_argument("--N_THREADS", type=int, help="1 direction / thread", default=64)
parser.add_argument("--N_THREADS_PRIOR", type=int, help="See step0e.py", default=64)
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
    help="Number of samples (drawn from the prior on cosmology) used to compute the"
    + "prior on primordial power spectrum (when using planck2018[_cv]).",
    default=int(1e4),
)
parser.add_argument(
    "--survey_mask_path",
    type=none_or_bool_or_str,
    help="Path to the survey mask.",
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
    "--params_obs",
    type=none_or_bool_or_str,
    help="Recompute the observations with the specified parameters.",
    default=None,
)
parser.add_argument(
    "--force_recompute_prior",
    type=bool_sh,
    help="Force overwriting the prior.",
    default=False,
)
parser.add_argument(
    "--update_obs_phase",
    type=bool_sh,
    help="Change the phase for observations.",
    default=False,
)
parser.add_argument(
    "--recompute_mocks",
    type=none_or_bool_or_str,
    help="Recompute the all the mocks in the inference phase."
    + "This leaves the DM density fields unchanged.",
    default=False,
)
parser.add_argument(
    "--perform_score_compression",
    type=bool_sh,
    help="Perform score compression.",
    default=False,
)
parser.add_argument(
    "--force_score_compression",
    type=bool_sh,
    help="Force recomputation of the score compression.",
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
parser.add_argument(
    "--prefix_mocks_mis",
    type=none_or_bool_or_str,
    help="Prefix for the mock files.",
    default=None,
)
parser.add_argument(
    "--selection_params",
    type=float,
    nargs="*",
    help="Parameters for the selection function of the well specified model.\
                        See the content of `GRF_blackbox.py` for more details.",
    default=None,
)
parser.add_argument(
    "--obs_density",
    type=float,
    help="Observed density.",
    default=None,
)
parser.add_argument(
    "--lin_bias",
    type=float,
    nargs="*",
    help="Linear bias to use recompute the mocks or the observation.",
    default=None,
)
parser.add_argument(
    "--noise_dbg",
    type=float,
    help="Specify noise amount by hand instead of using what was specified at step0a.\
          Only for dbg purpose because the normalization constants will be wrong.",
    default=None,
)

args = parser.parse_args()

wd = args.wd
N_THREADS = args.N_THREADS
N_THREADS_PRIOR = args.N_THREADS_PRIOR
prior_type = args.prior
nsamples_prior = int(args.nsamples_prior)
survey_mask_path = args.survey_mask_path
effective_volume = args.effective_volume
force_recompute_prior = args.force_recompute_prior
update_obs_phase = args.update_obs_phase
recompute_mocks = args.recompute_mocks
prefix_mocks_mis = args.prefix_mocks_mis
params_obs = args.params_obs
force_obs = params_obs is not None
perform_score_compression = args.perform_score_compression
force_score_compression = args.force_score_compression
verbosity = args.verbosity
noise_dbg = args.noise_dbg

modeldir = wd + "model/"
if prefix_mocks_mis is not None:
    modeldir_refined = modeldir + prefix_mocks_mis + "/"
else:
    modeldir_refined = modeldir
if args.selection_params is None:
    selection_params = np.load(modeldir + "selection_params.npy")
else:
    selection_params = np.reshape(np.array(args.selection_params), (3, -1))

print("selection_params", selection_params)

radial_selection = np.load(modeldir + "radial_selection.npy", allow_pickle=True)
if radial_selection == None:
    radial_selection = None
Npop = np.shape(selection_params)[1]

if args.obs_density is None:
    obs_density = np.load(modeldir + "obs_density.npy", allow_pickle=True)
    if obs_density == None:
        obs_density = None
else:
    obs_density = args.obs_density
if args.lin_bias is None:
    lin_bias = np.load(modeldir + "lin_bias.npy")
else:
    if not (recompute_mocks or force_obs):
        raise ValueError(
            "lin_bias shouldn't be specified if recompute_mocks & force_obs are False."
        )
    lin_bias = args.lin_bias
    if type(lin_bias) is not float:
        lin_bias = np.array(args.lin_bias)

Path(wd + "Figures/").mkdir(exist_ok=True)
if prefix_mocks_mis is not None and recompute_mocks:
    resultsdir = wd + "RESULTS/" + prefix_mocks_mis + "/"
else:
    resultsdir = wd + "RESULTS/"
figuresdir = (
    wd + "Figures/" + prefix_mocks_mis + "/"
    if prefix_mocks_mis is not None
    else wd + "Figures/"
)
resultsdir_obs = (
    wd + "RESULTS/" + prefix_mocks_mis + "/"
    if prefix_mocks_mis is not None
    else wd + "RESULTS/"
)
modeldir_obs = wd + "model/"
if prefix_mocks_mis is not None:
    scoredir = wd + "score_compression/" + prefix_mocks_mis + "/"
else:
    scoredir = wd + "score_compression/"
Path(resultsdir).mkdir(parents=True, exist_ok=True)
Path(figuresdir).mkdir(parents=True, exist_ok=True)
Path(resultsdir_obs).mkdir(parents=True, exist_ok=True)
Path(modeldir_obs).mkdir(parents=True, exist_ok=True)
Path(modeldir_refined).mkdir(parents=True, exist_ok=True)
Path(scoredir).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from pickle import load
    from pysbmy.timestepping import *
    from selfisys.sbmy_blackbox import blackbox
    from os.path import exists

    print("################################")
    print("## Setting up main parameters ##")
    print("################################")

    print("Loading main parameters...")
    with open(modeldir + "other_params.pkl", "rb") as f:
        other_params = load(f)
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
    Pinit = other_params["Pinit"]
    Ne = other_params["Ne"] if args.Ne is None else args.Ne
    Ns = other_params["Ns"] if args.Ns is None else args.Ns
    Delta_theta = other_params["Delta_theta"]
    sim_params = other_params["sim_params"]

    noise = np.load(modeldir + "noise.npy") if noise_dbg is None else noise_dbg
    k_max = int(1e3 * np.pi * size * np.sqrt(3) / L + 1) * 1e-3
    # Cosmo at the expansion point:
    params_planck_EH = params_planck_kmax_missing.copy()
    params_planck_EH["k_max"] = k_max
    # Observed cosmology:
    params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
    params_cosmo_obs["k_max"] = k_max

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
            "Normalization constants not found. Please run steps 0, 1, 2 before step 3."
        )
    else:
        norm_csts = np.load(modeldir + "norm_csts.npy")

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
        modeldir=modeldir_refined,
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

    print("> Loading ground truth spectrum...")
    if force_obs:
        from pysbmy.power import get_Pk

        theta_gt = get_Pk(k_s, params_cosmo_obs)
        np.save(modeldir_obs + "theta_gt", theta_gt)
    elif not exists(modeldir_obs + "theta_gt.npy"):
        raise ValueError(
            "Ground truth cosmology not found. Please run steps 0, 1, 2 before step 3."
        )
    else:
        theta_gt = np.load(modeldir_obs + "theta_gt.npy")

    print("> Loading observations...")
    if not exists(modeldir + "phi_obs.npy") and not force_obs:
        raise ValueError(
            "Observations not found. Please run steps 0, 1, 2 before step 3."
        )
    elif force_obs:
        if exists(modeldir_obs + "phi_obs.npy") and not (update_obs_phase):
            d_obs = -2
        else:
            d_obs = -1
        BB_selfi.update(sim_params=params_obs)
        BB_selfi.switch_recompute_pool()
        phi_obs = BB_selfi.make_data(
            cosmo=params_cosmo_obs,
            id="obs",
            seedphase=SEEDPHASE_OBS,
            seednoise=SEEDNOISE_OBS,
            d=d_obs,
            force_powerspectrum=force_obs,
            force_parfiles=force_obs,
            force_sim=force_obs,
            force_cosmo=force_obs,
        )
        BB_selfi.switch_recompute_pool()
        BB_selfi.update(sim_params=sim_params)
        np.save(modeldir_obs + "phi_obs.npy", phi_obs)
    else:
        phi_obs = np.load(modeldir_obs + "phi_obs.npy")

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
            nthreads=N_THREADS_PRIOR,
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

    print("##############################")
    print("## Computing/loading priors ##")
    print("##############################")

    if prior_type != "selfi2019":
        if not force_recompute_prior:
            try:
                selfi.prior = selfi.prior.load(selfi.fname)
            except:
                print(
                    "WARNING: prior not found in {}, computing it from scratch.\
                        Are you sure nothing went wrong during step 1?".format(
                        selfi.fname
                    )
                )
                try:
                    print(
                        "Prior not found in {}, computing it from scratch".format(
                            selfi.fname
                        )
                    )
                    selfi.compute_prior()
                    selfi.save_prior()
                    selfi.prior = selfi.prior.load(selfi.fname)
                except:
                    print(
                        'Error while computing the prior.\
                            For OOM issues, a simple fix might be to set\
                            os.environ["OMP_NUM_THREADS"] = "1".\
                            Otherwise, refer to the error message.'
                    )
        else:
            try:
                print(
                    "WARNING: forcing recomputation of the prior from scratch.\
                        selfi.fname={}".format(
                        selfi.fname
                    )
                )
                selfi.compute_prior()
                selfi.save_prior()
                selfi.prior = selfi.prior.load(selfi.fname)
            except:
                print(
                    'Error while computing the prior.\
                        For OOM issues, a simple fix might be to set\
                        os.environ["OMP_NUM_THREADS"] = "1".\
                        Otherwise, refer to the error message.'
                )
    else:
        selfi.compute_prior()
        selfi.save_prior()
        selfi.load_prior()

    if recompute_mocks:
        print("#####################")
        print("## Computing mocks ##")
        print("#####################")
        from multiprocessing import Pool
        from os import cpu_count

        if recompute_mocks is True:
            list_part_1 = [[0, idx, selfi] for idx in range(Ne)]
            list_part_2 = [[x, None, selfi] for x in range(1, S + 1)]
            liste = list_part_1 + list_part_2
        elif recompute_mocks == "gradients":
            liste = [[x, None, selfi] for x in range(1, S + 1)]
        else:
            raise ValueError("recompute_mocks can't be {}".format(recompute_mocks))

        ncors = cpu_count()
        nprocess = min(N_THREADS, ncors, 2 * (S + 1))
        print("Using {} processes to compute the mocks".format(nprocess))
        gc.collect()
        BB_selfi.switch_recompute_pool(prefix_mocks_mis=prefix_mocks_mis)
        with Pool(processes=nprocess) as mp_pool:
            import tqdm.auto as tqdm

            pool = mp_pool.imap(worker_fct, liste)
            for contrib_to_grad in tqdm.tqdm(pool, total=2 * (S + 1)):
                pass
        BB_selfi.switch_recompute_pool(prefix_mocks_mis=None)

    print("#########################")
    print("## Loading simulations ##")
    print("#########################")

    BB_selfi.update(_prefix_mocks_mis=prefix_mocks_mis)
    selfi.likelihood.switch_bar()
    selfi.run_simulations()  # actually load the simulations
    selfi.likelihood.switch_bar()
    BB_selfi.update(_prefix_mocks_mis=None)
    print("Done.")

    from selfisys.utils.plot_utils import *  # WARNING: this increase the memory load

    print("Plotting the observed summaries...")
    plot_observations(
        k_s, theta_gt, planck_Pk_EH, P_0, Pbins, phi_obs, selection_params, figuresdir
    )
    gc.collect()
    print("Done.")

    print("###########################")
    print("## Running the inference ##")
    print("###########################")

    print("Computing likelihoods and posteriors...")
    selfi.compute_likelihood()
    selfi.save_likelihood()
    selfi.load_likelihood()
    selfi.compute_posterior()
    selfi.save_posterior()
    selfi.load_posterior()

    print("Done. Preparing SELFI outputs for plotting...")
    C_0 = selfi.likelihood.C_0
    grad_f = selfi.likelihood.grad_f
    Phi_0 = selfi.likelihood.Phi_0.Phi
    f_0 = selfi.likelihood.f_0
    f_16 = selfi.likelihood.f_16
    f_32 = selfi.likelihood.f_32
    f_48 = selfi.likelihood.f_48
    grad_f_16 = (f_16 - f_0) / Delta_theta
    grad_f_32 = (f_32 - f_0) / Delta_theta
    grad_f_48 = (f_48 - f_0) / Delta_theta
    X0, Y0 = np.meshgrid(Pbins, Pbins)
    X1, Y1 = np.meshgrid(k_s, Pbins)
    N = Ne

    np.save(resultsdir + "Phi_0.npy", Phi_0)
    np.save(resultsdir + "grad_f.npy", grad_f)
    np.save(resultsdir + "f_0.npy", f_0)
    np.save(resultsdir + "f_16.npy", f_16)
    np.save(resultsdir + "f_32.npy", f_32)
    np.save(resultsdir + "f_48.npy", f_48)
    np.save(resultsdir + "C_0.npy", C_0)

    print("Done. Plotting mock data and covariance matrices...")
    plot_mocks(
        1,
        N,
        P,
        Pbins,
        phi_obs,
        Phi_0,
        np.mean(Phi_0, axis=0),
        C_0,
        X0,
        Y0,
        CovarianceMap,
        suptitle="Mock observations and their covariance matrix",
        savepath=figuresdir + "covariance_matrix.png",
    )

    print("Done. Plotting full covariance matrix...")
    plot_C(
        C_0,
        X0,
        Y0,
        Pbins,
        FullCovarianceMap,
        binning=False,
        suptitle="Full covariance matrix",
        savepath=figuresdir + "full_covariance_matrix.png",
    )

    print("Done. Plotting the gradients...")
    plot_gradients(
        Pbins,
        P,
        grad_f_16,
        grad_f_32,
        grad_f_48,
        grad_f,
        k_s,
        X1,
        Y1,
        fixscale=True,
        suptitle="Estimated gradients at expansion point for all populations of galaxies",
        savepath=figuresdir + "gradients.png",
    )
    print("Done. Preparing posterior for the plots...")

    prior_theta_mean = selfi.prior.mean
    prior_theta_covariance = selfi.prior.covariance
    # Keep all scales:
    Nbin_min, Nbin_max = 0, len(k_s)
    k_s = k_s[Nbin_min:Nbin_max]
    P_0 = P_0[Nbin_min:Nbin_max]
    prior_theta_mean = prior_theta_mean[Nbin_min:Nbin_max]
    prior_theta_covariance = prior_theta_covariance[
        Nbin_min:Nbin_max, Nbin_min:Nbin_max
    ]
    posterior_theta_mean, posterior_theta_covariance, posterior_theta_icov = (
        selfi.restrict_posterior(Nbin_min, Nbin_max)
    )

    X2, Y2 = np.meshgrid(k_s, k_s)
    prior_covariance = np.diag(P_0).dot(prior_theta_covariance).dot(np.diag(P_0))
    np.save(resultsdir + "prior_theta_mean.npy", prior_theta_mean)
    np.save(resultsdir + "prior_theta_covariance.npy", prior_theta_covariance)
    np.save(resultsdir_obs + "posterior_theta_mean.npy", posterior_theta_mean)
    np.save(
        resultsdir_obs + "posterior_theta_covariance.npy", posterior_theta_covariance
    )

    print("Done. Plotting the prior and posterior...")
    plot_prior_and_posterior_covariances(
        X2,
        Y2,
        k_s,
        prior_theta_covariance,
        prior_covariance,
        posterior_theta_covariance,
        P_0,
        suptitle="Prior and posterior covariance matrices",
        savepath=figuresdir + "prior_and_posterior_covariances.png",
    )
    print("Done. Plotting the reconstruction...")
    plot_reconstruction(
        k_s,
        Pbins,
        prior_theta_mean,
        prior_theta_covariance,
        posterior_theta_mean,
        posterior_theta_covariance,
        theta_gt,
        P_0,
        suptitle="Posterior primordial matter power spectrum",
        savepath=figuresdir + "reconstruction.png",
    )
    print("Done.")

    if perform_score_compression:
        print("#######################")
        print("## Score compression ##")
        print("#######################")

        from selfisys.utils.tools import *
        from selfisys.utils.workers import evaluate_gradient_of_Symbelmyne

        print("Compute the gradient of CLASS wrt the cosmological parameters...")
        delta = 1e-3
        if (
            not exists(wd + "score_compression/grads_class.npy")
            or force_score_compression
        ):
            coeffs = [4 / 5, -1 / 5, 4 / 105, -1 / 280]

            grad = np.zeros((len(planck_mean), len(k_s)))
            for i in range(len(planck_mean)):
                if i == 0:  # workaround to correctly evaluate the gradient wrt h
                    delta *= 10
                print("Evaluating gradient of CLASS wrt parameter %d" % i)
                deltas_x = delta * np.linspace(1, len(coeffs), len(coeffs))
                grad[i, :] = evaluate_gradient_of_Symbelmyne(
                    planck_mean,
                    i,
                    k_s,
                    coeffs=coeffs,
                    deltas_x=deltas_x,
                    delta=delta,
                    kmax=max(k_s),
                )
                if i == 0:
                    delta /= 10
            np.save(wd + "score_compression/grads_class.npy", grad)
        else:
            grad = np.load(wd + "score_compression/grads_class.npy")

        grad_class = grad.T

        if (
            not exists(wd + "score_compression/planck_Pk.npy")
            or force_score_compression
        ):
            from pysbmy.power import get_Pk

            print("Computing the Planck spectrum...")
            planck_Pk = get_Pk(k_s, params_planck_EH)
            np.save(wd + "score_compression/planck_Pk", planck_Pk)
        else:
            planck_Pk = np.load(wd + "score_compression/planck_Pk.npy")

        print("Done. Plotting the gradients of CLASS...")
        plt.figure(figsize=(14, 10))
        names_of_parameters = [
            r"$h$",
            r"$\Omega_b$",
            r"$\Omega_m$",
            r"$n_s$",
            r"$\sigma_8$",
        ]
        fig, ax = plt.subplots(3, 2, figsize=(14, 15))
        u, v = (-1, -1)
        ax[u, v].loglog(k_s, planck_Pk)
        ax[u, v].set_xlabel(r"$k$ [h/Mpc]")
        ax[u, v].set_ylabel(r"$P(k)$")
        ax[u, v].set_title(r"$P=\mathcal{T}(\omega_{\rm Planck})$")
        for k in k_s:
            ax[u, v].axvline(k, color="k", alpha=0.1, linewidth=0.5)
        for i in range(len(planck_mean)):
            u = i // 2
            v = i % 2
            for k in k_s:
                ax[u, v].axvline(k, color="k", alpha=0.1, linewidth=0.5)
            ax[u, v].plot(k_s, grad[i])
            ax[u, v].set_xscale("log")
            ax[u, v].set_xlabel(r"$k$ [h/Mpc]")
            ax[u, v].set_ylabel(r"$\partial P(k)/\partial$" + names_of_parameters[i])
            ax[u, v].set_title("Gradient wrt " + names_of_parameters[i])
        plt.suptitle("Gradient of Simbelmyne wrt cosmological parameters", fontsize=20)
        plt.tight_layout()
        plt.savefig(
            figuresdir + "gradient_class.png",
            bbox_inches="tight",
            dpi=300,
            transparent=True,
        )
        plt.savefig(figuresdir + "gradient_class.pdf", bbox_inches="tight", dpi=300)
        plt.close()

        if prefix_mocks_mis is None:
            print("Done. Computing Fisher matrix for the well specified model...")
        else:
            print(
                "Done. Computing Fisher matrix for the misspecified model {}...".format(
                    prefix_mocks_mis
                )
            )
        params_ids_fisher = np.linspace(0, 4, 5, dtype=int)

        dw_f0 = selfi.likelihood.grad_f.dot(grad_class)[:, params_ids_fisher]
        C0_inv = np.linalg.inv(selfi.likelihood.C_0)
        F0 = dw_f0.T.dot(C0_inv).dot(dw_f0)

        if not exists(scoredir + "dw_f0.npy") or force_score_compression:
            np.save(scoredir + "dw_f0.npy", dw_f0)
        if not exists(scoredir + "C0_inv.npy") or force_score_compression:
            np.save(scoredir + "C0_inv.npy", C0_inv)
        if not exists(scoredir + "F0.npy") or force_score_compression:
            np.save(scoredir + "F0.npy", F0)

        f0 = selfi.likelihood.f_0
        np.save(scoredir + "f0_expansion.npy", f0)

        plot_fisher(
            F0,
            names_of_parameters,
            title="Fisher matrix",
            path=figuresdir + "fisher.png",
        )

        print("Done.")

    print("#####################################")
    print("## Computing additional statistics ##")
    print("#####################################")

    print("Computing the Mahalanobis distances...")
    prior_theta_icov = selfi.prior.inv_covariance
    diff = posterior_theta_mean - prior_theta_mean
    Mahalanobis_distance = np.sqrt(diff.dot(prior_theta_icov).dot(diff))
    print(
        "Done. Mahalanobis distance between the posterior and the prior: {}".format(
            Mahalanobis_distance
        )
    )
    np.savetxt(resultsdir_obs + "Mahalanobis_distances.txt", [Mahalanobis_distance])
