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


"""This module contains tools for the selfisys package.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def custom_stat(vec):
    """Custom statistic used to compute the power spectrum with
    `scipy.stats.binned_statistic`. The factor 1/(N-2) arises from the assumption that
    the data power spectrum is inverse-Gamma distributed (see eg Jasche et al.,
    https://arxiv.org/abs/0911.2493), which amounts to using Jeffreys prior.

    """
    if sum(vec) == 0 or len(vec) <= 2:
        return "NaN"  # `scipy.stats.binned_statistic` expects NaN when len(vec)<=2
    else:
        res = sum(vec) / (len(vec) - 2)
        return res


def cosmo_vector_to_Simbelmyne_dict(x, kmax=1.4):
    """Converts a vector x of cosmological parameters into a dictionary of Simbelmyne
    parameters.

    Parameters:
    -----------
    x: array
        Cosmological parameters.
    kmax: float
        Maximum wavenumber.

    Returns:
    --------
    dict
        Dictionary of Simbelmyne parameters.

    """
    from selfisys.global_parameters import WhichSpectrum

    return {
        "h": x[0],
        "Omega_r": 0.0,
        "Omega_q": 1.0 - x[2],
        "Omega_b": x[1],
        "Omega_m": x[2],
        "m_ncdm": 0.0,
        "Omega_k": 0.0,
        "tau_reio": 0.066,
        "n_s": x[3],
        "sigma8": x[4],
        "w0_fld": -1.0,
        "wa_fld": 0.0,
        "k_max": kmax,
        "WhichSpectrum": WhichSpectrum,
    }


def cosmo_vector_to_class_dict(x, lmax=2500, kmax=1.4):
    """Converts a vector x of cosmological parameters into a dictionary of CLASS
    parameters.
    """
    return {
        "output": "lCl mPk",
        "l_max_scalars": lmax,
        "lensing": "no",
        "N_ncdm": 0,
        "P_k_max_h/Mpc": kmax,
        "h": x[0],
        "Omega_b": x[1],
        "Omega_m": x[2],
        "n_s": x[3],
        "sigma8": x[4],
    }


def params_ids_to_Simbelmyne_dict(params_vals, params_ids, fixed, kmax):
    """Converts a list of cosmological parameters into a dictionary of CLASS parameters.
    The other parameters are fixed to the values given in the `fixed` input vector.
    """
    from numpy import copy

    x = copy(fixed)
    x[params_ids] = params_vals
    return cosmo_vector_to_Simbelmyne_dict(x, kmax=kmax)


def get_summary(
    params_vals, params_ids, Omegas_fixed, bins, normalization=None, kmax=1.4
):
    """Compute the summary of the power spectrum.

    Parameters:
    -----------
    params_vals: array
        Values of the cosmological parameters to be varied.
    params_ids: array
        Indices of the cosmological parameters to be varied.
    Omegas_fixed: array
        Vector of the fixed cosmological parameters.
    bins: array
        Bins for the power spectrum.
    normalization: array
        Normalization for the power spectrum.
    kmax: float
        Maximum wavenumber for the power spectrum.

    Returns:
    --------
    array
        (Normalized) power spectrum.

    """
    from pysbmy.power import get_Pk
    from numpy import array

    phi = get_Pk(
        bins, params_ids_to_Simbelmyne_dict(params_vals, params_ids, Omegas_fixed, kmax)
    )

    if normalization is not None:
        return array(phi) / normalization
    else:
        return array(phi)


def summary_to_score(params_ids, omega0, F0, F0_inv, f0, dw_f0, C0_inv, phi):
    """Compute the score \Tilde{\omega} of a summary \phi."""
    # TODO: remove F0 from the inputs. Needs to update ABC-PMC consequently.
    return omega0[params_ids] + F0_inv.dot(dw_f0.T).dot(C0_inv).dot(phi - f0)


def fisher_rao(Com, Com_obs, F0):
    """Compute the Fisher-Rao distance between two summaries."""
    from numpy import sqrt

    diff = Com - Com_obs
    return sqrt(diff.T.dot(F0).dot(diff))


def sample_omega_from_prior(nsample, omega_mean, omega_cov, params_ids, seed=None):
    """Sample from the prior distribution of the cosmological parameters."""
    from numpy import array, ix_, clip
    from numpy.random import default_rng

    if seed is not None:
        rng = default_rng(seed)
    else:
        raise ValueError(
            "seednoise must be provided by user. None is not an acceptable value."
        )
    OO_unbounded = rng.multivariate_normal(
        array(omega_mean)[params_ids],
        array(omega_cov)[ix_(params_ids, params_ids)],
        nsample,
    )
    eps = (
        1e-5  # ensure physical values (at the cost of slighlty losing the gaussianity)
    )
    return clip(OO_unbounded, eps, 1 - eps)
