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

"""Workers to use Simbelmyne together with python multiprocessing.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def worker_gradient_Symbelmyne(coeff, delta_x, omega, param_index, k_s, delta, kmax):
    """Worker to compute the gradient of CLASS wrt the cosmological parameters at a
    given point in the parameter space using arbitrary order central finite differences.

    Parameters:
    -----------
    coeff: float
        Coefficient defining the finite difference scheme together with delta_x.
    delta_x: float
        Coefficient defining the finite difference scheme together with coeff.
    omega: array
        Cosmological parameters.
    param_index: int
        Index of the parameter to be varied.
    k_s: array
        Support wavenumbers for the power spectrum.
    delta: float
        Step size for the finite difference approximation.
    kmax: float
        Maximum wavenumber to be used in the power spectrum computation.

    Returns:
    --------
    contrib_to_grad: array
        Contribution to the gradient.

    """
    from numpy import array
    from pysbmy.power import get_Pk
    from selfisys.utils.tools import cosmo_vector_to_Simbelmyne_dict

    omega_new = omega.copy()

    omega_new[param_index] += delta_x
    contrib_to_grad = (
        coeff
        * get_Pk(k_s, cosmo_vector_to_Simbelmyne_dict(omega_new, kmax=kmax))
        / delta
    )
    return array(contrib_to_grad)


def evaluate_gradient_of_Symbelmyne(
    omega,
    param_index,
    k_s,
    coeffs=[2 / 3.0, -1 / 12.0],
    deltas_x=[0.01, 0.02],
    delta=1e-2,
    kmax=1.4,
):
    """Estimate the gradient of CLASS wrt the cosmological parameters at a given point
    in the parameter space, using arbitrary order central finite differences.

    Parameters:
    -----------
    omega: array
        Cosmological parameters.
    param_index: int
        Index of the parameter to be varied.
    k_s: array
        Support wavenumbers for the power spectrum.
    coeffs: list
        Coefficients defining the finite difference scheme together with deltas_x.
    deltas_x: list
        Coefficients defining the finite difference scheme together with coeffs.
    delta: float
        Step size for the finite difference approximation.
    kmax: float
        Maximum wavenumber to be used in the power spectrum computation.

    Returns:
    --------
    grad: array
        Gradient of the power spectrum wrt the cosmological parameters.

    """
    from numpy import zeros, array, concatenate
    from multiprocessing import Pool

    grad = zeros(len(k_s))
    full_coeffs = concatenate((-array(coeffs)[::-1], coeffs))
    deltas_x_full = concatenate((-array(deltas_x)[::-1], deltas_x))
    list = [
        (coeff, delta_x, omega, param_index, k_s, delta, kmax)
        for coeff, delta_x in zip(full_coeffs, deltas_x_full)
    ]
    with Pool() as mp_pool:
        pool = mp_pool.starmap(worker_gradient_Symbelmyne, list)
        for contrib_to_grad in pool:
            grad += contrib_to_grad

    return grad
