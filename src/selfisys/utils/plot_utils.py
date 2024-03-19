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

"""Useful plotting utilities for this project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors, cm, gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({"lines.linewidth": 2})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update(
    {
        "text.latex.preamble": r"\usepackage{amsmath,amsfonts,amssymb,amsthm} \usepackage{upgreek}"
    }
)
plt.rcParams.update({"font.family": "serif"})
import plotly.graph_objects as go


def get_contours(Z, nBins, confLevels=(0.3173, 0.0455, 0.0027)):
    Z /= Z.sum()
    nContourLevels = len(confLevels)
    chainLevels = np.ones(nContourLevels + 1)
    histOrdered = np.sort(Z.flat)
    histCumulative = np.cumsum(histOrdered)
    nBinsFlat = np.linspace(0.0, nBins**2, nBins**2)

    for l in range(nContourLevels):
        # Find location of contour level in 1d histCumulative
        temp = np.interp(confLevels[l], histCumulative, nBinsFlat)
        # Find "height" of contour level
        chainLevels[nContourLevels - 1 - l] = np.interp(temp, nBinsFlat, histOrdered)

    return chainLevels


# Create the color map to plot the galaxies
Ndots = 2**13
stretch_top = 0.5
truncate_bottom = 0.0
stretch_bottom = 1.0

top = cm.get_cmap("RdPu", Ndots)
top = colors.LinearSegmentedColormap.from_list("", ["white", top(0.5), top(1.0)])
bottom = cm.get_cmap("Greens_r", Ndots)
bottom = colors.LinearSegmentedColormap.from_list("", [bottom(0), bottom(0.5), "white"])

interp_top = np.linspace(0, 1, Ndots) ** stretch_top
interp_bottom = np.linspace(truncate_bottom, 1, Ndots) ** stretch_bottom
cols_galaxy = np.vstack((bottom(interp_bottom), top(interp_top)))
GalaxyMap = colors.ListedColormap(cols_galaxy, name="GradientMap")

# Create the color map to plot the gradient matrix
Ndots = 2**13
stretch_bottom = 6.0
stretch_top = 1 / 2.5
truncate_bottom = 0.35

bottom = cm.get_cmap("BuGn_r", Ndots)
top = cm.get_cmap("RdPu", Ndots)

interp_top = np.linspace(0, 1, Ndots) ** stretch_top
interp_bottom = np.linspace(truncate_bottom, 1, Ndots) ** stretch_bottom
newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
GradientMap = colors.ListedColormap(newcolors, name="GradientMap")

# Create the color map to plot the diagonal blocs of the covariance matrix
Ndots = 2**15
stretch_top_1 = 0.3
stretch_top_2 = 1
stretch_bottom = 0.2
middle = 0.4  # "middle" of the positive scale, between 0 and 1!
top = colormaps["BrBG"]
bottom = colormaps["BrBG"]

# split interp_top in two parts that are stretched differently:
interp_top = np.concatenate(
    (
        middle * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_1 + 0.5,
        (1 - middle) * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_2 + 0.5 + middle,
    )
)
interp_bottom = np.linspace(0.0, 1.0, Ndots) ** stretch_bottom - 0.5
newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
CovarianceMap = colors.ListedColormap(newcolors, name="CovarianceMap")

# Create the color map to plot the full covariance matrix:
Ndots = 2**15
stretch_top_1 = 0.3
stretch_top_2 = 1
middle_top = 0.4  # "middle" of the positive scale, between 0 and 1!

stretch_bottom_1 = 1
stretch_bottom_2 = 5
middle_bottom = 0.7  # "middle" of the negative scale, between 0 and 1!
colname = "PRGn_r"  # "PRGn", "PRGn_r", "BrBG", "PuOr"
top = colormaps[colname]
bottom = colormaps[colname]

# split interp_top in two parts that are stretched differently:
interp_top = np.concatenate(
    (
        middle_top * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_1 + 0.5,
        (1 - middle_top) * np.linspace(0.0, 1, Ndots // 2) ** stretch_top_2
        + 0.5
        + middle_top,
    )
)
interp_bottom = np.concatenate(
    (
        middle_bottom * np.linspace(0.0, 1, Ndots // 2) ** stretch_bottom_1 - 0.5,
        (1 - middle_bottom) * np.linspace(0.0, 1, Ndots // 2) ** stretch_bottom_2
        - 0.5
        + middle_bottom,
    )
)
newcolors = np.vstack((bottom(interp_bottom), top(interp_top)))
FullCovarianceMap = colors.ListedColormap(newcolors, name="CovarianceMap")

top = cm.get_cmap("Reds_r", 128)
bottom = cm.get_cmap("Blues", 128)
newcolors = np.vstack((top(np.linspace(0.7, 1, 128)), bottom(np.linspace(0, 1, 128))))
Blues_Reds = colors.ListedColormap(newcolors, name="Blues_Reds")

top = cm.get_cmap("Oranges_r", 128)
bottom = cm.get_cmap("Purples", 128)
newcolors = np.vstack((top(np.linspace(0.7, 1, 128)), bottom(np.linspace(0, 1, 128))))
Purples_Oranges = colors.ListedColormap(newcolors, name="Purples_Oranges")


def plot_observations(
    k_s,
    theta_gt,
    planck_Pk_EH,
    P_0,
    Pbins,
    phi_obs,
    selection_params,
    title=None,
    figuresdir=None,
):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(k_s, theta_gt / P_0, label=r"$\uptheta_{\mathrm{gt}}$", color="C0")
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
        label=r"$\uptheta$ support wavenumbers",
    )
    plt.axvline(x=Pbins[0], color="red", linestyle="dashed", linewidth=0.5)
    plt.axvline(x=Pbins[-1], color="red", linestyle="dashed", linewidth=0.5)
    for k in Pbins[1:-2]:
        plt.axvline(x=k, ymax=0.167, color="red", linestyle="dashed", linewidth=0.5)
    ax1.legend(loc=3, fontsize=18)
    plt.xlim(max(1e-4, k_s.min() - 2e-4), k_s.max())
    plt.ylim(7e-1, 1.6e0)

    ax2 = ax1.twinx()
    plt.axvline(
        x=Pbins[-2],
        ymax=0.333,
        color="red",
        linestyle="dashed",
        linewidth=0.5,
        label=r"$\phi$-bins centers",
    )
    if selection_params is None:
        Npop = 1
    else:
        Npop = np.shape(selection_params)[1]
    len_obs = len(phi_obs) // Npop
    cols = ["C4", "C5", "C6", "C7"]
    for i in range(Npop):
        ax2.plot(
            Pbins,
            phi_obs[i * len_obs : (i + 1) * len_obs],
            marker="x",
            label=r"$\phi_{\mathrm{obs}}$, galaxy pop " + str(i),
            linewidth=0.5,
            color=cols[i],
        )
    ax2.legend(loc=1, fontsize=17)
    if title is not None:
        plt.title(title, fontsize=17)
    if figuresdir is None:
        plt.show()
    else:
        plt.savefig(figuresdir + "summary_obs.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_mocks(
    NORM, N, P, Pbins, phi_obs, Phi_0, f_0, C_0, X, Y, CMap, suptitle="", savepath=None
):
    Phi_0_full = Phi_0.copy()
    phi_obs_full = phi_obs.copy()
    f_0_full = f_0.copy()
    C_0_full = C_0.copy()
    idx = 0
    Phi_0 = Phi_0[:, idx * P : (idx + 1) * P]
    phi_obs = phi_obs[idx * P : (idx + 1) * P]
    f_0 = f_0[idx * P : (idx + 1) * P]
    C_0 = C_0[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]

    color_list = ["C4", "C5", "C6"]

    fig = plt.figure(figsize=(16, 10))
    gs0 = gridspec.GridSpec(
        3,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.2,
        hspace=0.0,
    )
    gs0.update(right=1.0, left=0.0)
    ax0 = plt.subplot(gs0[0, 0])
    ax0b = plt.subplot(gs0[0, 1])
    ax01 = plt.subplot(gs0[1, 0], sharex=ax0)
    ax01b = plt.subplot(gs0[1, 1])
    ax02 = plt.subplot(gs0[2, 0], sharex=ax0)
    ax02b = plt.subplot(gs0[2, 1])
    axx0x = [[ax01, ax01b], [ax02, ax02b]]
    gs1 = gridspec.GridSpec(
        3, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0, 1.0], hspace=0.0
    )
    gs1.update(top=0.881, bottom=0.112)
    ax1a = plt.subplot(gs1[0, 2])
    ax1b = plt.subplot(gs1[1, 2], sharex=ax1a)
    ax1c = plt.subplot(gs1[2, 2], sharex=ax1a)
    axx1x = [ax1a, ax1b, ax1c]

    ax0.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])

    # Observed power spectrum:
    ax0.semilogx(
        Pbins,
        phi_obs * NORM,
        linewidth=2,
        color="black",
        label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
        zorder=3,
    )

    # Ne mock realizations of the blackbox at the expansion point:
    for i in range(N - 1):  # N=Ne
        ax0.semilogx(Pbins, Phi_0[i], color="C7", alpha=0.35, linewidth=0.7)
    ax0.semilogx(
        Pbins,
        Phi_0[N - 1],
        color="C7",
        alpha=0.35,
        linewidth=0.7,
        label=r"$\boldsymbol{\Phi}_{\theta_0}$",
    )

    # Mean of the blackbox at the expansion point:
    ax0.semilogx(
        Pbins,
        f_0,
        linewidth=2,
        color=color_list[idx],
        linestyle="--",
        label=r"$\textbf{f}_0$ (pop 1)",
        zorder=2,
    )

    # 2-sigma intervals around f_0:
    ax0.fill_between(
        Pbins,
        f_0 - 2 * np.sqrt(np.diag(C_0)),
        f_0 + 2 * np.sqrt(np.diag(C_0)),
        color=color_list[idx],
        alpha=0.4,
        label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$ (pop 1)",
        zorder=2,
    )

    # Plot the binning:
    (ymin, ymax) = ax0.get_ylim()
    ax0.set_ylim([ymin, ymax])
    for i in range(len(Pbins)):
        ax0.plot(
            (Pbins[i], Pbins[i]),
            (ymin, ymax),
            linestyle="--",
            linewidth=0.8,
            color="red",
            alpha=0.5,
            zorder=1,
        )
    ax0.yaxis.grid(linestyle=":", color="grey")
    ax0.set_ylabel(r"$\boldsymbol{\Phi}$ (population 1)", size=16)
    ax0.legend(fontsize=18, loc="upper right")
    ax0.xaxis.set_ticks_position("both")
    ax0.yaxis.set_ticks_position("both")
    ax0.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax0.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    for axis in ["top", "bottom", "left", "right"]:
        ax0.spines[axis].set_linewidth(1.0)
    ax0.xaxis.set_tick_params(which="major", length=6)
    ax0.xaxis.set_tick_params(which="minor", length=4)
    ax0.yaxis.set_tick_params(which="major", length=6)

    ax0b.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
    ax0b.yaxis.grid(linestyle=":", color="grey")

    normalization = phi_obs
    # normalization = f_0

    # Observed power spectrum (normalized):
    ax0b.semilogx(
        Pbins,
        NORM * phi_obs / normalization,
        linewidth=2,
        color="black",
        label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
        zorder=3,
    )

    # Ne mock realizations of the blackbox at the expansion point (normalized):
    for i in range(N - 1):  # N=Ne
        ax0b.semilogx(
            Pbins, Phi_0[i] / normalization, color="C7", alpha=0.35, linewidth=0.7
        )
    ax0b.semilogx(
        Pbins,
        Phi_0[N - 1] / normalization,
        color="C7",
        alpha=0.35,
        linewidth=0.7,
        label=r"$\boldsymbol{\Phi}_{\theta_0}$",
    )

    # Mean of the blackbox at the expansion point (normalized):
    ax0b.semilogx(
        Pbins,
        f_0 / normalization,
        linewidth=2,
        color=color_list[idx],
        linestyle="--",
        label=r"$\textbf{f}_0$ (pop 1)",
        zorder=2,
    )

    # 2-sigma intervals (normalized):
    ax0b.fill_between(
        Pbins,
        f_0 / normalization - 2 * np.sqrt(np.diag(C_0)) / normalization,
        f_0 / normalization + 2 * np.sqrt(np.diag(C_0)) / normalization,
        color=color_list[idx],
        alpha=0.4,
        label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$ (pop 1)",
        zorder=2,
    )

    # Plot the binning:
    (ymin, ymax) = ax0b.get_ylim()
    ax0b.set_ylim([ymin, ymax])
    for i in range(len(Pbins)):
        ax0b.plot(
            (Pbins[i], Pbins[i]),
            (ymin, ymax),
            linestyle="--",
            linewidth=0.8,
            color="red",
            alpha=0.5,
            zorder=1,
        )
    ax0b.set_ylabel(
        r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$ (population 1)", size=16
    )
    ax0b.set_xlabel(r"$k$ [$h$/Mpc]", size=16)
    ax0b.xaxis.set_ticks_position("both")
    ax0b.yaxis.set_ticks_position("both")
    ax0b.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax0b.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    for axis in ["top", "bottom", "left", "right"]:
        ax0b.spines[axis].set_linewidth(1.0)
    ax0b.xaxis.set_tick_params(which="major", length=6)
    ax0b.xaxis.set_tick_params(which="minor", length=4)
    ax0b.yaxis.set_tick_params(which="major", length=6)

    # Now deal with the other axes:
    for ax, axb in axx0x:
        idx += 1
        Phi_0 = Phi_0_full[:, idx * P : (idx + 1) * P]
        phi_obs = phi_obs_full[idx * P : (idx + 1) * P]
        f_0 = f_0_full[idx * P : (idx + 1) * P]
        C_0 = C_0_full[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]
        ax.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        ax.yaxis.grid(linestyle=":", color="grey")

        # Observed power spectrum:
        ax.semilogx(Pbins, phi_obs, linewidth=2, color="black", zorder=3)

        # Ne mock realizations of the blackbox at the expansion point:
        for i in range(N - 1):  # N=Ne
            ax.semilogx(Pbins, Phi_0[i], color="C7", alpha=0.35, linewidth=0.7)
        ax.semilogx(Pbins, Phi_0[N - 1], color="C7", alpha=0.35, linewidth=0.7)
        ax.semilogx(
            Pbins,
            f_0,
            linewidth=2,
            color=color_list[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$ (pop " + str(idx + 1) + ")",
            zorder=2,
        )

        # 2-sigma intervals:
        ax.fill_between(
            Pbins,
            f_0 - 2 * np.sqrt(np.diag(C_0)),
            f_0 + 2 * np.sqrt(np.diag(C_0)),
            color=color_list[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$ (pop " + str(idx + 1) + ")",
            zorder=2,
        )

        # Plot the binning:
        (ymin, ymax) = ax.get_ylim()
        ax.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            ax.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=0.5,
                zorder=1,
            )
        ax.set_ylabel(r"$\boldsymbol{\Phi}$ (population " + str(idx + 1) + ")", size=16)
        ax.set_xlabel(r"$k$ [$h$/Mpc]", size=16)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.0)
        ax.xaxis.set_tick_params(which="major", length=6)
        ax.xaxis.set_tick_params(which="minor", length=4)
        ax.yaxis.set_tick_params(which="major", length=6)
        ax.legend(loc="upper right", fontsize=14)  # ,frameon=False

        normalization = phi_obs  # np.mean(Phi_0, axis=0)
        axb.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        axb.yaxis.grid(linestyle=":", color="grey")

        # Observed power spectrum (normalized):
        axb.semilogx(
            Pbins,
            phi_obs / normalization,
            linewidth=2,
            color="black",
            label=r"$\boldsymbol{\Phi}_\mathrm{O}$",
            zorder=3,
        )

        # Ne mock realizations of the blackbox at the expansion point (normalized):
        for i in range(N - 1):  # N=Ne
            axb.semilogx(
                Pbins, Phi_0[i] / normalization, color="C7", alpha=0.35, linewidth=0.7
            )
        axb.semilogx(
            Pbins,
            Phi_0[N - 1] / normalization,
            color="C7",
            alpha=0.35,
            linewidth=0.7,
            label=r"$\boldsymbol{\Phi}_{\theta_0}$",
        )

        # Mean of the blackbox at the expansion point (normalized):
        axb.semilogx(
            Pbins,
            f_0 / normalization,
            linewidth=2,
            color=color_list[idx],
            linestyle="--",
            label=r"$\textbf{f}_0$",
            zorder=2,
        )

        # 2-sigma intervals (normalized):
        axb.fill_between(
            Pbins,
            f_0 / normalization - 2 * np.sqrt(np.diag(C_0)) / normalization,
            f_0 / normalization + 2 * np.sqrt(np.diag(C_0)) / normalization,
            color=color_list[idx],
            alpha=0.4,
            label=r"2 $\sqrt{\mathrm{diag}(\textbf{C}_0)}$",
            zorder=2,
        )

        # Plot the binning:
        (ymin, ymax) = axb.get_ylim()
        axb.set_ylim([ymin, ymax])
        for i in range(len(Pbins)):
            axb.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=0.5,
                zorder=1,
            )
        axb.set_ylabel(
            r"$\boldsymbol{\Phi}/\boldsymbol{\Phi}_\mathrm{O}$ (population "
            + str(idx + 1)
            + ")",
            size=16,
        )
        axb.set_xlabel(r"$k$ [$h$/Mpc]", size=16)
        axb.xaxis.set_ticks_position("both")
        axb.yaxis.set_ticks_position("both")
        axb.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        axb.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            axb.spines[axis].set_linewidth(1.0)
        axb.xaxis.set_tick_params(which="major", length=6)
        axb.xaxis.set_tick_params(which="minor", length=4)
        axb.yaxis.set_tick_params(which="major", length=6)
        # axb.legend(loc="upper right",frameon=False,fontsize=12)

    # Plot the diagonal blocks of the covariance matrix
    # (corresponding to the intra-population covariance matrices):
    idx = 0
    for ax1 in axx1x:
        Phi_0 = Phi_0_full[:, idx * P : (idx + 1) * P]
        phi_obs = phi_obs_full[idx * P : (idx + 1) * P]
        f_0 = f_0_full[idx * P : (idx + 1) * P]
        C_0 = C_0_full[idx * P : (idx + 1) * P, idx * P : (idx + 1) * P]
        idx += 1

        ax1.set_aspect("equal")  # make the plot square
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.xaxis.set_ticks_position("both")
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.xaxis.set_tick_params(which="major", length=6)
        ax1.xaxis.set_tick_params(which="minor", length=4)
        ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.yaxis.set_tick_params(which="major", length=6)
        ax1.yaxis.set_tick_params(which="minor", length=4)

        C0min = C_0.min()
        C0max = C_0.max()
        if C0min < 0 and C0max > 0:
            centerval = 0
        else:
            centerval = np.mean(C_0)
        divnorm = colors.TwoSlopeNorm(vmin=C_0.min(), vcenter=centerval, vmax=C_0.max())
        im1 = ax1.pcolormesh(
            X, Y, C_0[:-1, :-1], shading="flat", norm=divnorm, cmap=CMap
        )  # CovarianceMap or "PiYG"

        # Explicitly plot the binning:
        for i in range(len(Pbins)):
            ax1.plot(
                (Pbins[i], Pbins[i]),
                (Pbins.min(), Pbins.max()),
                linestyle="--",
                linewidth=0.5,
                color="red",
                alpha=0.5,
            )
        for i in range(len(Pbins)):
            ax1.plot(
                (Pbins.min(), Pbins.max()),
                (Pbins[i], Pbins[i]),
                linestyle="--",
                linewidth=0.5,
                color="red",
                alpha=0.5,
            )
        if idx == 1:
            ax1.set_title(r"diagonal blocks of $\textbf{C}_0$", size=18, y=1.05)
        ax1.set_xlabel(r"$k$ [$h$/Mpc]", size=16)
        ax1.set_ylabel(r"$k$ [$h$/Mpc]", size=14)

        cbar1 = fig.colorbar(im1, shrink=0.9, pad=0.02, format="%.1e")
        cbar1.ax.tick_params(axis="y", direction="in", width=1.0, length=4)
        cbar1.update_normal(im1)
        cbar1.mappable.set_clim(vmin=C_0[:-1, :-1].min(), vmax=C_0[:-1, :-1].max())
        ticks = np.concatenate(
            [np.linspace(C_0.min(), 0, 5), np.linspace(0, C_0.max(), 5)[1:]]
        )
        cbar1.set_ticks(ticks)
        cbar1.draw_all()  # avoid "posx and posy should be finite values" warning

    plt.suptitle(suptitle, y=0.95, fontsize=20)
    if savepath is None:
        plt.show()
    if savepath is not None:
        fig.savefig(
            savepath, bbox_inches="tight", transparent=True, dpi=300, format="png"
        )
        fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def plot_C(
    C_0, X, Y, Pbins, CMap, binning=True, suptitle=None, savepath=None, force=False
):
    from itertools import product

    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    fs = 22
    P = len(Pbins)
    quantity = C_0
    vmin = quantity.min()
    vmax = quantity.max()
    if vmin < 0 and vmax > 0:
        centerval = 0
    else:
        centerval = np.mean(quantity)
    divnorm = colors.TwoSlopeNorm(vmin=C_0.min(), vcenter=centerval, vmax=C_0.max())
    for i, j in product(range(3), range(3)):
        C_0_ij = C_0[i * P : (i + 1) * P, j * P : (j + 1) * P]
        imat = 2 - i
        ax = axs[imat, j]
        ax.set_aspect("equal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")

        if i == 0:
            ax.xaxis.set_tick_params(
                which="both", direction="in", width=1.0, labelsize=fs
            )
            ax.xaxis.set_tick_params(which="major", length=6)
            ax.xaxis.set_tick_params(which="minor", length=4)
        else:
            ax.set_xticks([])
        if j == 0:
            ax.yaxis.set_tick_params(
                which="both", direction="in", width=1.0, labelsize=fs
            )
            ax.yaxis.set_tick_params(which="major", length=6)
            ax.yaxis.set_tick_params(which="minor", length=4)
        else:
            ax.set_yticks([])

        im1 = ax.pcolormesh(
            X, Y, C_0_ij[:-1, :-1], shading="flat", norm=divnorm, cmap=CMap
        )

        if binning:
            for n in range(len(Pbins)):
                ax.plot(
                    (Pbins[n], Pbins[n]),
                    (Pbins.min(), Pbins.max()),
                    linestyle="--",
                    linewidth=0.5,
                    color="red",
                    alpha=0.5,
                )
            for n in range(len(Pbins)):
                ax.plot(
                    (Pbins.min(), Pbins.max()),
                    (Pbins[n], Pbins[n]),
                    linestyle="--",
                    linewidth=0.5,
                    color="red",
                    alpha=0.5,
                )

    suptitle = (
        r"$\textbf{C}_0$" if suptitle is None else r"$\textbf{C}_0$ (" + suptitle + ")"
    )

    plt.suptitle(suptitle, y=0.94, x=0.45, size=fs + 8)
    plt.subplots_adjust(wspace=0, hspace=0)

    cbar = fig.colorbar(
        im1,
        ax=axs.ravel().tolist(),
        shrink=1,
        pad=0.009,
        aspect=40,
        orientation="vertical",
    )
    cbar.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=fs + 2)
    cbar.update_normal(im1)
    cbar.mappable.set_clim(vmin=C_0[:-1, :-1].min(), vmax=C_0[:-1, :-1].max())
    loc_xticks = np.concatenate(
        [np.linspace(C_0.min(), 0, 5), np.linspace(0, C_0.max(), 5)[1:]]
    )
    val_xticks = np.round(loc_xticks, 2)
    cbar.set_ticks(loc_xticks, labels=val_xticks)

    fig.text(0.45, 0.04, r"$k$ [$h$/Mpc]", ha="center", size=fs + 4)
    fig.text(
        0.04, 0.5, r"$k'$ [$h$/Mpc]", va="center", rotation="vertical", size=fs + 4
    )

    if savepath is None or force:
        plt.show()
    if savepath is not None:
        fig.savefig(
            savepath, bbox_inches="tight", dpi=300, format="png", transparent=True
        )
        fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close(fig)


def plot_gradients(
    Pbins,
    P,
    df_16_full,
    df_32_full,
    df_48_full,
    df_full,
    k_s,
    X,
    Y,
    fixscale=False,
    force=False,
    suptitle="",
    savepath=None,
):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(suptitle, y=0.95, fontsize=22)

    gs0 = gridspec.GridSpec(
        3,
        2,
        width_ratios=[1.0, 0.5],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.0,
        wspace=0.2,
    )
    gs0.update(right=1.0, left=0.0)
    ax00 = plt.subplot(gs0[0, 0])
    ax01 = plt.subplot(gs0[1, 0], sharex=ax00)
    ax02 = plt.subplot(gs0[2, 0], sharex=ax00)

    gs1 = gridspec.GridSpec(
        3,
        2,
        width_ratios=[1.0, 0.5],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.0,
        wspace=0.2,
    )
    gs1.update(top=0.881, bottom=0.112)
    ax10 = plt.subplot(gs1[0, 1])
    ax11 = plt.subplot(gs1[1, 1], sharex=ax10)
    ax12 = plt.subplot(gs1[2, 1], sharex=ax10)

    axx = [(ax00, ax10), (ax01, ax11), (ax02, ax12)]
    for axs, idx in zip(axx, range(3)):
        ax = axs[0]
        df_16 = np.copy(df_16_full[idx * P : (idx + 1) * P])
        df_32 = np.copy(df_32_full[idx * P : (idx + 1) * P])
        df_48 = np.copy(df_48_full[idx * P : (idx + 1) * P])
        df = df_full[idx * P : (idx + 1) * P]

        # Plot 3 components of the derivative:
        ax.set_xlim([Pbins.min() - 0.0001, Pbins.max() + 0.01])
        ax.semilogx(Pbins, np.zeros_like(Pbins), linestyle=":", color="black")
        ax.semilogx(
            Pbins,
            df_16,
            linewidth=2,
            linestyle="-",
            color="C4",
            label=r"$(\nabla \mathbf{f}_0)^\intercal_{16}$",
            zorder=2,
        )
        ax.semilogx(
            Pbins,
            df_32,
            linewidth=2,
            linestyle="-",
            color="C0",
            label=r"$(\nabla \mathbf{f}_0)^\intercal_{32}$",
            zorder=2,
        )
        ax.semilogx(
            Pbins,
            df_48,
            linewidth=2,
            linestyle="-",
            color="C2",
            label=r"$(\nabla \mathbf{f}_0)^\intercal_{48}$",
            zorder=2,
        )
        if fixscale:
            ymin = np.min([df_16, df_32, df_48]) - 1e-2
            ymax = np.max([df_16, df_32, df_48]) + 1e-2
        else:
            (ymin, ymax) = ax.get_ylim()
        ax.set_ylim([ymin, ymax])

        # Binning:
        for i in range(len(Pbins)):
            ax.plot(
                (Pbins[i], Pbins[i]),
                (ymin, ymax),
                linestyle="--",
                linewidth=0.8,
                color="red",
                alpha=0.5,
                zorder=1,
            )
        ax.yaxis.grid(linestyle=":", color="grey")
        ax.set_ylabel("population " + str(idx + 1), size=21)
        ax.xaxis.set_ticks_position("both")
        ax.yaxis.set_ticks_position("both")
        ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(1.0)
        ax.xaxis.set_tick_params(which="major", length=6)
        ax.xaxis.set_tick_params(which="minor", length=4)
        ax.yaxis.set_tick_params(which="major", length=6)

        # Plot the full gradient:
        ax1 = axs[1]
        ax1.set_xlim([k_s.min(), k_s.max()])
        ax1.set_ylim([Pbins.min(), Pbins.max()])
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.xaxis.set_ticks_position("both")
        ax1.yaxis.set_ticks_position("both")
        ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.xaxis.set_tick_params(which="major", length=6)
        ax1.xaxis.set_tick_params(which="minor", length=4)
        ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
        ax1.yaxis.set_tick_params(which="major", length=6)
        ax1.yaxis.set_tick_params(which="minor", length=4)
        quantity = df
        vmin = quantity.min()
        vmax = quantity.max()
        if vmin < 0 and vmax > 0:
            centerval = 0
        else:
            centerval = np.mean(quantity)
        norm_grad = colors.TwoSlopeNorm(vmin=df.min(), vcenter=centerval, vmax=df.max())
        im1 = ax1.pcolormesh(
            X, Y, df[:-1, :-1], cmap=GradientMap, shading="flat", norm=norm_grad
        )
        ax1.plot(k_s, k_s, color="grey", linestyle="--")
        for i in range(len(k_s)):
            ax1.plot(
                (k_s[i], k_s[i]),
                (Pbins.min(), Pbins.max()),
                linestyle=":",
                linewidth=0.5,
                color="green",
                alpha=0.5,
            )
        for i in range(len(Pbins)):
            ax1.plot(
                (k_s.min(), k_s.max()),
                (Pbins[i], Pbins[i]),
                linestyle="--",
                linewidth=0.5,
                color="red",
                alpha=0.5,
            )
        ax1.set_ylabel(r"$k$ [$h$/Mpc]", size=17)
        cbar1 = fig.colorbar(im1, shrink=0.9, pad=0.01)  # , format='%.1e')
        cbar1.ax.tick_params(axis="y", direction="in", width=1.0, length=6)
        ticks = np.concatenate(
            [np.linspace(df.min(), 0, 5), np.linspace(0, df.max(), 5)[1:]]
        )
        # Replace ticks by corresponding strings in scientific notation:
        # ticks_labels = ["{:.1e}".format(tick) for tick in ticks]
        cbar1.set_ticks(ticks)
        # cbar1.set_ticklabels(ticks_labels)

        if idx == 0:
            ax.legend(fontsize=21, loc="upper left")
            ax1.set_title(r"Full gradient $\nabla \mathbf{f}_0$", size=21)
        elif idx == 2:
            ax.set_xlabel(r"$k$ [$h$/Mpc]", size=21)
            ax1.set_xlabel(r"$k$ [$h$/Mpc]", size=21)

    if savepath is None or force:
        plt.show()
    if savepath is not None:
        fig.savefig(
            savepath, bbox_inches="tight", dpi=300, format="png", transparent=True
        )
        fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
        fig.suptitle("")
        fig.savefig(
            savepath[:-4] + "_notitle.png",
            bbox_inches="tight",
            dpi=300,
            format="png",
            transparent=True,
        )
        fig.savefig(
            savepath[:-4] + "_notitle.pdf", bbox_inches="tight", dpi=300, format="pdf"
        )
    plt.close(fig)


def plot_prior_and_posterior_covariances(
    X,
    Y,
    k_s,
    prior_theta_covariance,
    prior_covariance,
    posterior_theta_covariance,
    P_0,
    force=False,
    suptitle="",
    savepath=None,
):
    fs = 18
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(figsize=(15, 14), nrows=2, ncols=2)
    fig.suptitle(suptitle, fontsize=fs + 4, y=0.99)

    # Covariance matrix of the prior (wrt normalized spectra theta):
    ax0.set_aspect("equal")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.xaxis.set_ticks_position("both")
    ax0.yaxis.set_ticks_position("both")
    ax0.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax0.xaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax0.xaxis.set_tick_params(which="minor", length=4)
    ax0.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax0.yaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax0.yaxis.set_tick_params(which="minor", length=4)
    divider = make_axes_locatable(ax0)
    ax0_cb = divider.new_horizontal(size="5%", pad=0.10)
    im0 = ax0.pcolormesh(
        X, Y, prior_theta_covariance[:-1, :-1], cmap="Blues", shading="flat"
    )
    for i in range(len(k_s)):
        ax0.plot(
            (k_s[i], k_s[i]),
            (k_s.min(), k_s.max()),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    for i in range(len(k_s)):
        ax0.plot(
            (k_s.min(), k_s.max()),
            (k_s[i], k_s[i]),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    ax0.set_title("$\\textbf{S}$", size=fs + 4)
    ax0.set_xlabel("$k$ [$h$/Mpc]", size=fs + 4)
    ax0.set_ylabel("$k$ [$h$/Mpc]", size=fs + 4)
    fig.add_axes(ax0_cb)
    cbar0 = fig.colorbar(im0, cax=ax0_cb)
    cbar0.ax.tick_params(
        axis="y", direction="in", width=1.0, length=6, labelsize=fs + 1
    )

    # Covariance matrix of the prior (wrt unnormalized power spectra):
    ax1.set_aspect("equal")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.set_ticks_position("both")
    ax1.yaxis.set_ticks_position("both")
    ax1.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax1.xaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax1.xaxis.set_tick_params(which="minor", length=4)
    ax1.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax1.yaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax1.yaxis.set_tick_params(which="minor", length=4)
    divider = make_axes_locatable(ax1)
    ax1_cb = divider.new_horizontal(size="5%", pad=0.10)
    im1 = ax1.pcolormesh(
        X, Y, prior_covariance[:-1, :-1], cmap="Purples", shading="flat"
    )
    for i in range(len(k_s)):
        ax1.plot(
            (k_s[i], k_s[i]),
            (k_s.min(), k_s.max()),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    for i in range(len(k_s)):
        ax1.plot(
            (k_s.min(), k_s.max()),
            (k_s[i], k_s[i]),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    ax1.set_title(
        "$\mathrm{diag}(\\textbf{P}_0) \cdot \\textbf{S} \cdot \mathrm{diag}(\\textbf{P}_0)$",
        size=fs + 4,
    )
    ax1.set_xlabel("$k$ [$h$/Mpc]", size=fs + 4)
    ax1.set_ylabel("$k$ [$h$/Mpc]", size=fs + 4)
    fig.add_axes(ax1_cb)
    cbar1 = fig.colorbar(im1, cax=ax1_cb)
    cbar1.ax.tick_params(
        axis="y", direction="in", width=1.0, length=6, labelsize=fs + 1
    )

    posterior_covariance = (
        np.diag(P_0).dot(posterior_theta_covariance).dot(np.diag(P_0))
    )

    # Covariance matrix of the posterior (wrt normalized spectra theta):
    ax2.set_aspect("equal")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.xaxis.set_ticks_position("both")
    ax2.yaxis.set_ticks_position("both")
    ax2.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax2.xaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax2.xaxis.set_tick_params(which="minor", length=4)
    ax2.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax2.yaxis.set_tick_params(which="major", length=6, labelsize=fs + 1)
    ax2.yaxis.set_tick_params(which="minor", length=4)
    divider = make_axes_locatable(ax2)
    ax2_cb = divider.new_horizontal(size="5%", pad=0.10)
    quantity = posterior_theta_covariance
    vmin = quantity.min()
    vmax = quantity.max()
    if vmin < 0 and vmax > 0:
        centerval = 0
    else:
        centerval = np.mean(quantity)
    norm_posterior = colors.TwoSlopeNorm(
        vmin=posterior_theta_covariance.min(),
        vcenter=centerval,
        vmax=posterior_theta_covariance.max(),
    )
    im2 = ax2.pcolormesh(
        X,
        Y,
        posterior_theta_covariance[:-1, :-1],
        cmap=Blues_Reds,
        norm=norm_posterior,
        shading="flat",
    )
    for i in range(len(k_s)):
        ax2.plot(
            (k_s[i], k_s[i]),
            (k_s.min(), k_s.max()),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    for i in range(len(k_s)):
        ax2.plot(
            (k_s.min(), k_s.max()),
            (k_s[i], k_s[i]),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    ax2.set_title("$\\boldsymbol{\Gamma}$", size=fs + 4)
    ax2.set_xlabel("$k$ [$h$/Mpc]", size=fs + 4)
    ax2.set_ylabel("$k$ [$h$/Mpc]", size=fs + 4)
    fig.add_axes(ax2_cb)
    cbar2 = fig.colorbar(im2, cax=ax2_cb)
    cbar2.ax.tick_params(
        axis="y", direction="in", width=1.0, length=6, labelsize=fs + 1
    )
    cbar2.mappable.set_clim(
        vmin=posterior_theta_covariance.min(), vmax=posterior_theta_covariance.max()
    )
    ticks = np.concatenate(
        [
            np.linspace(posterior_theta_covariance.min(), 0, 5),
            np.linspace(0, posterior_theta_covariance.max(), 5)[1:],
        ]
    )
    # Replace ticks by corresponding strings in scientific notation:
    ticks_labels = ["{:.1e}".format(tick) for tick in ticks]
    cbar2.set_ticks(ticks)
    cbar2.set_ticklabels(ticks_labels)

    # Covariance matrix of the posterior (unnormalized spectra theta):
    ax3.set_aspect("equal")
    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.xaxis.set_ticks_position("both")
    ax3.yaxis.set_ticks_position("both")
    ax3.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax3.xaxis.set_tick_params(which="major", length=6, labelsize=19)
    ax3.xaxis.set_tick_params(which="minor", length=4)
    ax3.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax3.yaxis.set_tick_params(which="major", length=6, labelsize=19)
    ax3.yaxis.set_tick_params(which="minor", length=4)
    divider = make_axes_locatable(ax3)
    ax3_cb = divider.new_horizontal(size="5%", pad=0.10)
    quantity = posterior_covariance
    vmin = quantity.min()
    vmax = quantity.max()
    if vmin < 0 and vmax > 0:
        centerval = 0
    else:
        centerval = np.mean(quantity)
    norm_posterior_spectrum = colors.TwoSlopeNorm(
        vmin=posterior_covariance.min(),
        vcenter=centerval,
        vmax=posterior_covariance.max(),
    )
    im3 = ax3.pcolormesh(
        X,
        Y,
        posterior_covariance[:-1, :-1],
        cmap=Purples_Oranges,
        norm=norm_posterior_spectrum,
        shading="flat",
    )
    for i in range(len(k_s)):
        ax3.plot(
            (k_s[i], k_s[i]),
            (k_s.min(), k_s.max()),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    for i in range(len(k_s)):
        ax3.plot(
            (k_s.min(), k_s.max()),
            (k_s[i], k_s[i]),
            linestyle=":",
            linewidth=0.5,
            color="green",
            alpha=0.5,
        )
    ax3.set_title(
        "$\mathrm{diag}(\\textbf{P}_0) \cdot \\boldsymbol{\Gamma} \cdot \mathrm{diag}(\\textbf{P}_0)$",
        size=22,
    )
    ax3.set_xlabel("$k$ [$h$/Mpc]", size=22)
    ax3.set_ylabel("$k$ [$h$/Mpc]", size=22)
    fig.add_axes(ax3_cb)
    cbar3 = fig.colorbar(im3, cax=ax3_cb)
    cbar3.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=19)
    cbar3.ax.tick_params(axis="y", direction="in", width=1.0, length=6, labelsize=19)
    cbar3.mappable.set_clim(
        vmin=posterior_covariance.min(), vmax=posterior_covariance.max()
    )
    ticks = np.concatenate(
        [
            np.linspace(posterior_covariance.min(), 0, 5),
            np.linspace(0, posterior_covariance.max(), 5)[1:],
        ]
    )
    # Replace ticks by corresponding strings in scientific notation:
    ticks_labels = ["{:.1e}".format(tick) for tick in ticks]
    cbar3.set_ticks(ticks)
    cbar3.set_ticklabels(ticks_labels)

    fig.tight_layout()
    if savepath is None or force:
        plt.show()
    if savepath is not None:
        fig.savefig(
            savepath, bbox_inches="tight", dpi=300, format="png", transparent=True
        )
        fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
        fig.suptitle("")
        fig.savefig(
            savepath[:-4] + "_notitle.png",
            bbox_inches="tight",
            dpi=300,
            format="png",
            transparent=True,
        )
        fig.savefig(
            savepath[:-4] + "_notitle.pdf", bbox_inches="tight", dpi=300, format="pdf"
        )
    plt.close(fig)


def plot_reconstruction(
    k_s,
    Pbins,
    prior_theta_mean,
    prior_theta_covariance,
    posterior_theta_mean,
    posterior_theta_covariance,
    theta_gt,
    P_0,
    phi_obs=None,
    suptitle="",
    theta_fid=None,
    savepath=None,
):
    fig, (ax) = plt.subplots(figsize=(14, 5))
    fs = 14  # fontsize
    ax.set_ylim([0.55, 1.45])
    # ax.set_ylim([0.65,1.35])
    ax.set_xlim([k_s.min() - 0.0001, k_s.max()])
    ax.set_xscale("log")
    ax.yaxis.grid(linestyle=":", color="grey")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax.xaxis.set_tick_params(which="major", length=6, labelsize=fs)
    ax.xaxis.set_tick_params(which="minor", length=4)
    ax.yaxis.set_tick_params(which="both", direction="in", width=1.0)
    ax.yaxis.set_tick_params(which="major", length=6, labelsize=fs)

    # Prior:
    ax.plot(
        k_s,
        prior_theta_mean,
        linestyle="-",
        color="gold",
        label="$\\boldsymbol{\\uptheta}_0$ (prior)",
        zorder=3,
    )

    # 2-sigma around the prior:
    ax.fill_between(
        k_s,
        prior_theta_mean - 2 * np.sqrt(np.diag(prior_theta_covariance)),
        prior_theta_mean + 2 * np.sqrt(np.diag(prior_theta_covariance)),
        color="gold",
        alpha=0.20,
    )

    # theta_fid:
    if theta_fid is not None:
        ax.plot(
            k_s,
            theta_fid,
            linestyle="--",
            color="C1",
            label="$\\boldsymbol{\\uptheta}_{\mathrm{fid}}$ (fiducial for hyperparameter optimisation)",
            zorder=3,
        )

    # Posterior:
    ax.plot(
        k_s,
        posterior_theta_mean,
        color="C2",
        label="$\\boldsymbol{\\upgamma}$ (reconstruction)",
        zorder=3,
    )

    # 2-sigma around the posterior:
    ax.fill_between(
        k_s,
        posterior_theta_mean - 2 * np.sqrt(np.diag(posterior_theta_covariance)),
        posterior_theta_mean + 2 * np.sqrt(np.diag(posterior_theta_covariance)),
        color="C2",
        alpha=0.35,
    )

    # Groundtruth:
    ax.plot(
        k_s,
        theta_gt / P_0,
        color="C0",
        label="$\\boldsymbol{\\uptheta}_\mathrm{gt}$ (groundtruth)",
        zorder=3,
    )

    # Binning:
    (ymin, ymax) = ax.get_ylim()
    ax.set_ylim([ymin, ymax])
    for i in range(len(k_s)):
        ax.plot(
            [k_s[i], k_s[i]],
            [ymin, ymax],
            linestyle=":",
            linewidth=0.8,
            color="green",
            alpha=0.5,
        )
    for i in range(1, len(Pbins) - 1):
        ax.plot(
            [Pbins[i], Pbins[i]],
            [ymin, 0.8],
            linestyle="--",
            linewidth=0.8,
            color="red",
            alpha=0.5,
        )
    ax.plot(
        [Pbins[0], Pbins[0]],
        [ymin, ymax],
        linestyle="--",
        linewidth=0.8,
        color="red",
        alpha=0.5,
    )
    ax.plot(
        [Pbins[len(Pbins) - 1], Pbins[len(Pbins) - 1]],
        [ymin, ymax],
        linestyle="--",
        linewidth=0.8,
        color="red",
        alpha=0.5,
    )
    ax.set_xlabel("$k$ [$h$/Mpc]", size=fs + 2)
    ax.set_ylabel("$\\theta(k) = P(k)/P_0(k)$", size=fs + 2)
    ax.legend(loc="upper right", fontsize=fs + 3)

    if phi_obs is not None:
        # Observations:
        ax2 = ax.twinx()
        ax2.plot(
            Pbins,
            phi_obs,
            color="C3",
            marker=".",
            linestyle="--",
            linewidth=1,
            label=r"$\boldsymbol{\Phi}_{\textrm{O}}$ (observations)",
            zorder=3,
            dashes=(5, 5),
        )
        ax2.legend(loc="lower right", fontsize=fs + 3)
    plt.suptitle(suptitle, fontsize=fs + 4)
    if savepath is None:
        plt.show()
    if savepath is not None:
        fig.savefig(
            savepath, bbox_inches="tight", dpi=300, format="png", transparent=True
        )
        fig.savefig(savepath[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
        fig.suptitle("")
        fig.savefig(
            savepath[:-4] + "_notitle.png",
            bbox_inches="tight",
            dpi=300,
            format="png",
            transparent=True,
        )
        fig.savefig(
            savepath[:-4] + "_notitle.pdf",
            bbox_inches="tight",
            dpi=300,
            format="pdf",
        )
    plt.close(fig)


def plot_fisher(F0, params_names_fisher, title=None, path=None):
    from seaborn import heatmap

    plt.figure(figsize=(10, 10))
    F0min = F0.min()
    F0max = F0.max()
    if F0min < 0 and F0max > 0:
        center = 0
    else:
        center = np.mean(F0)
    divnorm = colors.TwoSlopeNorm(vmin=F0min, vcenter=center, vmax=F0max)
    heatmap(F0, annot=True, fmt=".2e", cmap="RdBu_r", norm=divnorm, square=True)
    plt.xticks(
        np.arange(len(params_names_fisher)) + 0.5, params_names_fisher, rotation=0
    )
    plt.yticks(
        np.arange(len(params_names_fisher)) + 0.5, params_names_fisher, rotation=0
    )
    if title is not None:
        plt.title(title)
    else:
        plt.title("Fisher matrix")
    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches="tight", dpi=300, format="png", transparent=True)
        plt.savefig(path[:-4] + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
    plt.close()


def plotly_3d(field, size=128, L=None, colormap="RdYlBu", limits="max"):
    volume = field.T
    r, c = volume[0].shape
    if limits == "max":
        maxcol = np.max(np.abs(volume))
        mincol = -maxcol
    elif limits == "truncate":
        maxcol = np.min([np.max(-volume), np.max(volume)])
        mincol = -maxcol
    else:
        maxcol = np.max(volume)
        mincol = np.min(volume)
    midcol = np.mean(volume)

    # I followed this: https://plotly.com/python/visualizing-mri-volume-slices/
    nb_frames = int(size)

    fig = go.Figure(
        frames=[
            go.Frame(
                data=go.Surface(
                    z=(size - k) * np.ones((r, c)),
                    surfacecolor=np.flipud(volume[c - 1 - k]),
                    cmin=mincol,
                    cmid=midcol,
                    cmax=maxcol,
                ),
                name=str(
                    k
                ),  # one need to name the frame for the animation to behave properly
            )
            for k in range(nb_frames)
        ]
    )

    # Add data to be displayed before animation starts
    fig.add_trace(
        go.Surface(
            z=size * np.ones((r, c)),
            surfacecolor=np.flipud(volume[int(c / 2)]),
            colorscale=colormap,  # 'grey'
            cmin=mincol,
            cmid=midcol,
            cmax=maxcol,
            colorbar=dict(thickness=20, ticklen=4),
        )
    )

    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]
    if L is not None:
        fig.update_layout(
            title="Slices in density field",
            width=600,
            height=600,
            scene=dict(
                zaxis=dict(
                    range=[0, size - 1],
                    autorange=False,
                    ticktext=[0, L / 2, L],
                    tickvals=[0, size / 2, size],
                    title="x [Mpc/h]",
                ),
                xaxis=dict(
                    ticktext=[0, L / 2, L],
                    tickvals=[0, size / 2, size],
                    title="x [Mpc/h]",
                ),
                yaxis=dict(
                    ticktext=[0, L / 2, L],
                    tickvals=[0, size / 2, size],
                    title="y [Mpc/h]",
                ),
                # xaxis_title='x [Mpc/h]',
                # yaxis_title='y [Mpc/h]',
                # zaxis_title='z [Mpc/h]',
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
        )
    else:
        fig.update_layout(
            title="Slices in density field",
            width=600,
            height=600,
            scene=dict(
                zaxis=dict(range=[0, size - 1], autorange=False),
                xaxis_title="x [Mpc/h]",
                yaxis_title="y [Mpc/h]",
                zaxis_title="z [Mpc/h]",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
            ],
            sliders=sliders,
        )
    fig.show()


def plot_selection_functions(
    x,
    res,
    res_mis,
    z_means,
    redshifts,
    D,
    params,
    L,
    corner,
    axis="com",
    zz=None,
    zcorner=None,
    path=None,
):
    fs = 22

    colors_list = ["C4", "C5", "C6"]
    plt.figure(figsize=(10, 5))
    for i in range(len(res) - 1):
        r = res[i]
        plt.plot(x, r, color=colors_list[i])
    plt.plot(x, res[-1], color=colors_list[-1])
    plt.plot(x, res[-1], color="black", alpha=0, label="Model A")
    if res_mis is not None:
        for i in range(len(res_mis) - 1):
            r = res_mis[i]
            plt.plot(x, r, linestyle="--", color=colors_list[i])
        plt.plot(x, res_mis[-1], linestyle="--", color=colors_list[-1])
        plt.plot(
            x, res_mis[-1], linestyle="--", color="black", alpha=0, label="Model B"
        )
    xticks = [0, corner]
    if axis == "com":
        xticks_labels = [
            r"$0$",
            r"$\sqrt 3\,L\simeq{}$".format(np.round(corner, 2)),
        ]
    elif axis == "redshift":
        xticks_labels = [
            r"$0$",
            r"$z(\sqrt 3\,L)\simeq{}$".format(np.round(corner, 3)),
        ]
    plt.axvline(L, color="black", linestyle="-", linewidth=1, zorder=0)
    for i in range(len(res)):
        if i == 0:
            bias = 1.47
        else:
            z = z_means[i]
            z_index_D = np.argmin(np.abs(redshifts - z))
            bias = 1.55 / D[z_index_D]
        mean = params[1][i]
        mean_plt = x[np.argmin(np.abs(zz - mean))] if zz is not None else mean
        plt.axvline(mean_plt, color=colors_list[i], linestyle="-.", linewidth=1)

        std = params[0][i]
        mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
        sig2 = np.log(1 + std**2 / mean**2)
        mode = np.exp(mu - sig2)
        mode_plt = x[np.argmin(np.abs(zz - mode))] if zz is not None else mode

        xticks += [mean_plt]
        xticks_labels += [np.round(mean_plt, 2)]

        plt.axvline(mode_plt, color=colors_list[i], linestyle="-", linewidth=1)
        plt.axvline(
            mode_plt,
            color=colors_list[i],
            linestyle="-",
            alpha=0,
            linewidth=1,
            label=r"$b_{} = {:.2f}$".format(i + 1, bias),
        )
    if axis == "com":
        xlabel = r"$r\,[{\rm Gpc}/h$]"
        ylabel = r"$R_i(r)$"
    elif axis == "redshift":
        xlabel = r"$z$"
        ylabel = r"$R_i(z)$"
    plt.xlabel(xlabel, fontsize=fs + 2)
    plt.ylabel(ylabel, fontsize=fs + 2)
    plt.tick_params(axis="y", which="major", labelsize=fs - 2)
    plt.locator_params(axis="x", nbins=fs - 4)
    plt.locator_params(axis="y", nbins=fs - 4)
    plt.grid(
        which="both", axis="both", linestyle="-", linewidth=0.4, color="gray", alpha=0.5
    )
    plt.xticks(xticks, xticks_labels)

    maxs = [np.max(r) for r in res]
    yticks = [0] + maxs
    plt.yticks(yticks)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    plt.tick_params(axis="x", which="major", labelsize=fs - 2)

    leg = plt.legend(fontsize=fs + 2, frameon=True, loc="upper right")
    leg.get_frame().set_edgecolor("white")
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    if zcorner is not None:
        xlim = plt.xlim()
        ax2 = plt.gca().twiny()
        ax2.set_xlabel(r"$z$", fontsize=fs + 2)
        ax2.set_xticks(
            xticks,
            [
                r"${:.2f}$".format(z)
                for z in np.concatenate([[0], [zcorner], params[1]])
            ],
        )
        ax2.tick_params(axis="x", which="major", labelsize=fs - 2)
        ax2.grid(
            which="both",
            axis="both",
            linestyle="-",
            linewidth=0.4,
            color="gray",
            alpha=0.5,
        )
        # Make sure the two axes are aligned:
        ax2.set_ylim(plt.ylim())
        ax2.set_yticks(plt.yticks()[0])
        ax2.set_xlim(xlim)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.savefig(path[:-4] + ".pdf", bbox_inches="tight", dpi=300)
    else:
        plt.show()
    plt.close()
