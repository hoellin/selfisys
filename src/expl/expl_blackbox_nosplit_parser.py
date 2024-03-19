#!/usr/bin/env python
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

"""Script to play with the blackbox forward model.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"

from os.path import exists
from pathlib import Path
from pysbmy.field import read_basefield
from selfisys.setup_model import *
from selfisys.global_parameters import *
from selfisys.sbmy_blackbox import blackbox
from selfisys.utils.plot_utils import *

from selfisys.utils.parser import *

parser = ArgumentParser(
    description="Perform tests with the blackbox to be used for the final run."
)
parser.add_argument("--OUTDIR", type=str, help="Absolute path of the output directory.")
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
    default=256,
)
parser.add_argument(
    "--Np0", type=int, help="Number of DM particles in each direction.", default=512
)
parser.add_argument(
    "--Npm0",
    type=int,
    help="Number of elements in each direction of the particle-mesh grid.",
    default=512,
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
          The actual number of bins may be smaller since it is automatically tuned to \
          ensure that each bin contains a sufficient number of modes.",
    default=50,
)
parser.add_argument(
    "--Nnorm",
    type=int,
    help="Number of simulations used to compute the normalization of the summaries.",
    default=10,
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
                        See the content of `GRF_blackbox.py` for more details.",
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
    "--lin_bias",
    type=float,
    nargs="*",
    help="Linear bias for the well specified model.",
    default=1.1,
)
parser.add_argument(
    "--obs_density",
    type=none_or_bool_or_str,
    help="Observation density for the well specified model.",
    default=None,
)
parser.add_argument(
    "--noise", type=float, help="Noise for the well specified model.", default=0
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
OUTDIR = args.OUTDIR
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
radial_selection = args.radial_selection
selection_params = args.selection_params
survey_mask_path = args.survey_mask_path
sim_params = args.sim_params
lin_bias = args.lin_bias
obs_density = args.obs_density
noise = args.noise
verbosity = args.verbosity
force = args.force

print("Loading parameters...")
if aa is not None and total_steps is not None:
    from pysbmy.timestepping import *
    from selfisys.utils.timestepping import merge_nTS

    custom_timestepping = True
else:
    custom_timestepping = False
selection_params = np.reshape(np.array(selection_params), (3, -1))
wd_noname = OUTDIR + wd_ext + str(size) + str(int(L)) + str(Pinit) + str(Nnorm) + "/"
wd = wd_noname + name + "/"
modeldir = wd + "model/"
datadir = wd + "data/"
figuresdir = wd + "Figures/"
k_max = int(1e3 * np.pi * size * np.sqrt(3) / L + 1) * 1e-3
# Cosmo at the expansion point:
params_planck_EH = params_planck_kmax_missing.copy()
params_planck_EH["k_max"] = k_max
# BBKS spectrum with fiducial cosmology, for normalization:
params_BBKS = params_BBKS_kmax_missing.copy()
params_BBKS["k_max"] = k_max
# Observed cosmology:
params_cosmo_obs = params_cosmo_obs_kmax_missing.copy()
params_cosmo_obs["k_max"] = k_max

Path(modeldir).mkdir(parents=True, exist_ok=True)
Path(datadir).mkdir(parents=True, exist_ok=True)
Path(figuresdir).mkdir(exist_ok=True)

print("Setting up the model...")
params = setup_model(
    workdir=modeldir,
    params_planck=params_planck_EH,
    params_P0=params_BBKS,
    size=size,
    L=L,
    S=S,
    Pinit=Pinit,
    force=force,
)
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

if custom_timestepping:
    print("Setting up the time-stepping...")
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

    print("Merging time-stepping...")
    merged_path = modeldir + "merged.h5"
    merge_nTS(TS_paths, merged_path)
    TS_merged = read_timestepping(merged_path)
    TS_merged.plot(path=figuresdir + "TS_merged.png")
    if sim_params[:5] == "split":
        TimeStepDistribution = TS_paths
        eff_redshifts = [1 / a - 1 for a in aa[1:]]
    else:
        TimeStepDistribution = merged_path
        eff_redshifts = 1 / aa[-1] - 1
else:
    merged_path = None
    indices_steps_cumul = None
    eff_redshifts = None

print("Loading the groundtruth power spectrum...")
if (not exists(modeldir + "theta_gt.npy")) or force:
    from pysbmy.power import get_Pk

    theta_gt = get_Pk(k_s, params_cosmo_obs)
    np.save(modeldir + "theta_gt", theta_gt)
    del theta_gt


def theta2P(theta):
    return theta * P_0


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
    reset=True,
    save_frequency=5,
    verbosity=verbosity,
)

if not exists(modeldir + "norm_csts.npy") or force:
    print("Computing normalization constants...")
    BB_selfi.switch_recompute_pool()
    norm_csts = BB_selfi.define_normalization(
        Pbins,
        params_planck_EH,
        Nnorm,
        min_k_norma=min_k_norma,
        npar=1,
        force=force,
    )
    BB_selfi.switch_recompute_pool()
    np.save(modeldir + "norm_csts.npy", norm_csts)
else:
    print("Normalization constants already exist. Loading...")
    norm_csts = np.load(modeldir + "norm_csts.npy")
print("Updating the blackbox with norm_csts = ", norm_csts)
BB_selfi.update(norm_csts=norm_csts)

force_obs = force
if not exists(modeldir + "phi_obs.npy") or force_obs:
    print("Making observed data...")
    d_obs = -1
    BB_selfi.switch_recompute_pool()
    phi_obs, g_obs = BB_selfi.make_data(
        cosmo=params_cosmo_obs,
        id="obs",
        seedphase=SEEDPHASE_OBS,
        seednoise=SEEDNOISE_OBS,
        d=d_obs,
        force_powerspectrum=force_obs,
        force_parfiles=force_obs,
        force_sim=force_obs,
        force_cosmo=force_obs,
        return_g=True,
        RSDs=True,
    )
    BB_selfi.switch_recompute_pool()
    g_obs = np.array(g_obs)
    np.save(modeldir + "phi_obs.npy", phi_obs)
    np.save(modeldir + "g_obs.npy", g_obs)
else:
    print("Observed data already exist.")
    phi_obs = np.load(modeldir + "phi_obs.npy")
    g_obs = np.load(modeldir + "g_obs.npy")

if custom_timestepping:
    print("Plotting slices through dark matter overdensity fields...")
    print("> Loading the dark matter overdensity fields...")
    g1_path = datadir + "output_realdensity_obs_{}.h5".format(indices_steps_cumul[0])
    g2_path = datadir + "output_realdensity_obs_{}.h5".format(indices_steps_cumul[1])
    g3_path = datadir + "output_realdensity_obs_{}.h5".format(indices_steps_cumul[2])
    g1 = read_basefield(g1_path).data
    g2 = read_basefield(g2_path).data
    g3 = read_basefield(g3_path).data
    g1RSDs_path = datadir + "output_density_obs_{}.h5".format(indices_steps_cumul[0])
    g2RSDs_path = datadir + "output_density_obs_{}.h5".format(indices_steps_cumul[1])
    g3RSDs_path = datadir + "output_density_obs_{}.h5".format(indices_steps_cumul[2])
    g1RSDs = read_basefield(g1RSDs_path).data
    g2RSDs = read_basefield(g2RSDs_path).data
    g3RSDs = read_basefield(g3RSDs_path).data

    print("> Plotting...")
    zz_plot = np.round([1 / a - 1 for a in aa[1:]], 3)
    fig, axs = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.04, 1, 1, 0.04], "width_ratios": [1, 1, 1]},
    )
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i, gi in enumerate([g1, g2, g3]):
        sns.heatmap(
            np.log(2 + gi[:, :, int(size // 12)]),
            cmap="YlGnBu_r",
            ax=axs[1, i],
            cbar=False,
            square=True,
            rasterized=True,
        )
        cbar = fig.colorbar(
            axs[1, i].get_children()[0], cax=axs[0, i], orientation="horizontal"
        )
        cbar.ax.tick_params(labelsize=15)
        axs[1, i].set_aspect("equal", adjustable="box")
        axs[0, i].set_title(f"z={zz_plot[i]}", fontsize=20)
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_yticklabels([])
    for i, gi in enumerate([g1RSDs, g2RSDs, g3RSDs]):
        sns.heatmap(
            np.log(2 + gi[:, :, int(size // 12)]),
            cmap="YlGnBu_r",
            ax=axs[2, i],
            cbar=False,
            square=True,
            rasterized=True,
        )
        cbar = fig.colorbar(
            axs[2, i].get_children()[0], cax=axs[3, i], orientation="horizontal"
        )
        cbar.set_label(r"$\log(2+\delta_z)$", fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        axs[2, i].set_aspect("equal", adjustable="box")
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
        axs[2, i].set_xticklabels([])
        axs[2, i].set_yticklabels([])

    axs[1, 0].set_ylabel("w/o RSDs", fontsize=20)
    axs[2, 0].set_ylabel("with RSDs", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        figuresdir + "different_reshifts_woRSDs_withRSDs.png",
        bbox_inches="tight",
        dpi=300,
        format="png",
        transparent=True,
    )
    plt.savefig(
        figuresdir + "different_reshifts_woRSDs_withRSDs.pdf",
        bbox_inches="tight",
        dpi=300,
        format="pdf",
    )
    plt.show()

fig, axs = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(14, 5),
    gridspec_kw={"height_ratios": [1, 0.04], "width_ratios": [1, 1, 1]},
)
plt.subplots_adjust(wspace=0.0, hspace=0.0)
for i, gi in enumerate(g_obs):
    slice = gi[:, :, int(size // 12)]
    ax = axs[0, i]

    limits = "minmax"  # available options are "truncate", "max"
    if limits == "max":
        maxcol = np.max(np.abs(slice))
        mincol = -maxcol
        cmap = GalaxyMap
    elif limits == "truncate":
        maxcol = np.min([np.max(-slice), np.max(slice)])
        mincol = -maxcol
        cmap = "PiYG"
    elif limits == "minmax":
        maxcol = np.max(slice)
        mincol = np.min(slice)
        cmap = GalaxyMap
    divnorm = colors.TwoSlopeNorm(vmin=mincol, vcenter=0, vmax=maxcol)
    im = ax.imshow(slice, norm=divnorm, cmap=cmap, rasterized=True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    cbar = fig.colorbar(im, ax=ax, cax=axs[1, i], orientation="horizontal")
    cbar.outline.set_visible(False)
    ticks = [mincol, mincol / 2, 0, maxcol / 3, 2 * maxcol / 3, maxcol]
    ticks_labels = [f"{x:.2f}" for x in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks_labels)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(r"$\delta_\textrm{g}$", size=20)
    ax.set_xticks(
        [size * i / 4.0 for i in range(5)], [f"{L*1e-3*i/4:.1f}" for i in range(5)]
    )
    ax.set_yticks(
        [size * i / 4.0 for i in range(5)], [f"{L*1e-3*i/4:.1f}" for i in range(5)]
    )
    ax.set_xlabel("Gpc/h", size=16)
    ax.set_ylabel("Gpc/h", size=16)

fig.suptitle("Observed galaxy overdensity fields", fontsize=20)
plt.tight_layout()
plt.savefig(
    figuresdir + "g_obs.png",
    bbox_inches="tight",
    dpi=300,
    format="png",
    transparent=True,
)
plt.savefig(
    figuresdir + "g_obs.pdf",
    bbox_inches="tight",
    dpi=300,
    format="pdf",
)

print("Plotting the observed spectra...")
theta_gt = np.load(modeldir + "theta_gt.npy")
plot_observations(
    k_s,
    theta_gt,
    planck_Pk_EH,
    P_0,
    Pbins,
    phi_obs,
    selection_params,
    figuresdir=figuresdir,
)
