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

"""Set up parameters related to the grid and the fiducial power spectrum.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def setup_model(
    workdir,  # Name of the directory where the results will be stored
    params_planck,
    params_P0,
    size=256,  # Number of elements in each direction of the box
    L=3600,  # Comoving length of the box in Mpc/h
    S=100,  # Number of support wavenumbers to define the input power spectra theta
    N_exact=8,  # Number of support wavenumbers taken to match the Fourier grid
    Pinit=50,  # Maximum number of bins for the summaries
    trim_threshold=100,  # Min number of modes required per bin for the summaries
    minval=None,  # min k for the summaries. Default (None) -> np.min(knorms[knorms!=0])
    maxval=None,  # max k for the summaries. Default (None) -> 2*np.pi*(size/2)/L
    force=False,  # Force recomputation of the inputs
):
    from os.path import exists
    from gc import collect
    import numpy as np
    from pysbmy.power import PowerSpectrum, FourierGrid, get_Pk

    # Define the full Fourier grid
    if not exists(workdir + "G_sim.h5") or force:
        print("> Computing Fourier grid...")
        G_sim = FourierGrid(L, L, L, size, size, size)
        G_sim.write(workdir + "G_sim.h5")
    else:
        print("> Loading Fourier grid...")
        G_sim = FourierGrid.read(workdir + "G_sim.h5")
    G_sim_path = workdir + "G_sim.h5"

    if minval is None:
        minval = np.min(G_sim.k_modes[G_sim.k_modes != 0])
    if maxval is None:
        maxval = np.pi * size / L  # 1d Nyquist frequency

    Pbins_left_bnds_init = np.logspace(
        np.log10(minval), np.log10(maxval), Pinit + 1, dtype=np.float32
    )
    k_ss_max_offset = Pbins_left_bnds_init[-1] - Pbins_left_bnds_init[-2]
    np.save(workdir + "k_ss_max_offset.npy", k_ss_max_offset)
    print("> k_ss_max_offset =", k_ss_max_offset)
    Pbins_left_bnds_init = Pbins_left_bnds_init[:-1]

    # Support wavenumbers for the input power spectrum
    if not exists(workdir + "k_s.npy") or force:
        print("> Done. Computing input power spectrum support wavenumbers...")
        k_s = np.zeros(S)
        sorted_knorms = np.sort(G_sim.k_modes.flatten())
        which_unique = np.unique(np.round(sorted_knorms, 5), return_index=True)[1]
        sorted_knorms_corrected = sorted_knorms[which_unique]
        del sorted_knorms
        k_s[:N_exact] = sorted_knorms_corrected[1 : N_exact + 1]
        k_s_max = int(1e3 * np.sqrt(3) * np.pi * size / L + 1) * 1e-3
        k_s[N_exact:] = np.linspace(
            sorted_knorms_corrected[N_exact], k_s_max, S - N_exact + 1
        )[1:]
        k_s[N_exact:] = np.logspace(
            np.log10(sorted_knorms_corrected[N_exact]),
            np.log10(k_s_max),
            S - N_exact + 1,
        )[1:]
        del sorted_knorms_corrected
        np.save(workdir + "k_s.npy", k_s)
    else:
        print("> Done. Loading input power spectrum support wavenumbers...")
        k_s = np.load(workdir + "k_s.npy")
    print("> Done.")

    # Fourier grid for the summaries
    if not exists(workdir + "G_ss.h5") or force:
        G_ss = FourierGrid(
            L,
            L,
            L,
            size,
            size,
            size,
            k_modes=Pbins_left_bnds_init,
            kmax=maxval,
            trim_bins=True,
            trim_threshold=trim_threshold,
        )
        G_ss.write(workdir + "G_ss.h5")
    else:
        G_ss = FourierGrid.read(workdir + "G_ss.h5")
    G_ss_path = workdir + "G_ss.h5"

    P = G_ss.NUM_MODES

    # BBKS spectrum for normalization
    if not exists(workdir + "P_ss.npy") or not exists(workdir + "P_0.npy") or force:
        P_0 = get_Pk(k_s, params_P0)  # normalization for inputs theta
        np.save(workdir + "P_0.npy", P_0)
    else:
        P_0 = np.load(workdir + "P_0.npy")

    if not exists(workdir + "P_ss_obj.h5") or force:
        P_0_ss = get_Pk(G_ss.k_modes, params_P0)
        P_ss_obj = PowerSpectrum.from_FourierGrid(
            G_ss, powerspectrum=P_0_ss, cosmo=params_P0
        )
        P_ss_obj.write(workdir + "P_ss_obj.h5")
    else:
        P_ss_obj = PowerSpectrum.read(workdir + "P_ss_obj.h5")
    P_ss_obj_path = workdir + "P_ss_obj.h5"

    # Planck power spectrum
    if not exists(workdir + "theta_planck.npy") or force:
        planck_Pk = get_Pk(k_s, params_planck)
        np.save(workdir + "theta_planck.npy", planck_Pk)
    else:
        planck_Pk = np.load(workdir + "theta_planck.npy")

    if (
        not exists(workdir + "Pbins.npy")
        or not exists(workdir + "Pbins_bnd.npy")
        or force
    ):
        Pbins_bnd = G_ss.k_modes
        Pbins_bnd = np.concatenate([Pbins_bnd, [Pbins_bnd[-1] + k_ss_max_offset]])
        Pbins = (Pbins_bnd[1:] + Pbins_bnd[:-1]) / 2
        np.save(workdir + "Pbins.npy", Pbins)
        np.save(workdir + "Pbins_bnd.npy", Pbins_bnd)
    else:
        Pbins = np.load(workdir + "Pbins.npy")
        Pbins_bnd = np.load(workdir + "Pbins_bnd.npy")

    del G_sim
    del G_ss
    del P_ss_obj
    del Pbins_left_bnds_init

    collect()

    return (
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
        planck_Pk,
    )
