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

"""Set up global parameters for this project.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"

from numpy import array, diag

computer = "infinity"
possible_paths = {"infinity": "/home/hoellinger/selfi_example_sys/"}
ROOT_PATH = possible_paths[computer]

WhichSpectrum = "class"
cosmo_params_name_to_idx = {"h": 0, "Omega_b": 1, "Omega_m": 2, "nS": 3, "sigma8": 4}
cosmo_params_names = [r"$h$", r"$\Omega_b$", r"$\Omega_m$", r"$n_S$", r"$\sigma_8$"]

BASELINE_SEEDNORM = 100030898
BASELINE_SEEDNOISE = 200030898
BASELINE_SEEDPHASE = 300030898
SEEDPHASE_OBS = 100030896
SEEDNOISE_OBS = 100030897

h_planck = 0.6766
Omega_b_planck = 0.02242 / h_planck**2
Omega_m_planck = 0.3111
nS_planck = 0.9665
sigma8_planck = 0.8102
planck_mean = array(
    [h_planck, Omega_b_planck, Omega_m_planck, nS_planck, sigma8_planck]
)
planck_cov = diag(
    array(((0.0042) ** 2, (0.00030) ** 2, (0.0056) ** 2, (0.0038) ** 2, (0.0060) ** 2))
)

# Unknown truth cosmology:
h_obs = 0.679187146124996
Omega_b_obs = 0.0487023481098232
Omega_m_obs = 0.2983845329810192
nS_obs = 0.9689584319027499
sigma8_obs = 0.8159826491039461
omegas_gt = array([h_obs, Omega_b_obs, Omega_m_obs, nS_obs, sigma8_obs])

min_k_norma = 4e-2  # min k value to be used to normalize the summaries

params_planck_kmax_missing = {
    "h": h_planck,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_planck,
    "Omega_b": Omega_b_planck,
    "Omega_m": Omega_m_planck,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_planck,
    "sigma8": sigma8_planck,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": WhichSpectrum,
}

params_BBKS_kmax_missing = {
    "h": h_planck,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_planck,
    "Omega_b": Omega_b_planck,
    "Omega_m": Omega_m_planck,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_planck,
    "sigma8": sigma8_planck,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": "BBKS",
}

params_cosmo_obs_kmax_missing = {
    "h": h_obs,
    "Omega_r": 0.0,
    "Omega_q": 1.0 - Omega_m_obs,
    "Omega_b": Omega_b_obs,
    "Omega_m": Omega_m_obs,
    "m_ncdm": 0.0,
    "Omega_k": 0.0,
    "tau_reio": 0.066,
    "n_s": nS_obs,
    "sigma8": sigma8_obs,
    "w0_fld": -1.0,
    "wa_fld": 0.0,
    "WhichSpectrum": WhichSpectrum,
}

# Default hyperparameters for the selfi2019 prior:
theta_norm = 0.05
k_corr = 0.01
