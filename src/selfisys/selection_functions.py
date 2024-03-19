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

"""Lognormal selection functions.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def r_grid(L, size):
    from numpy import linspace, meshgrid, sqrt

    range1d = linspace(0, L, size, endpoint=False)
    xx, yy, zz = meshgrid(range1d, range1d, range1d)
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    return sqrt((xx - x0) ** 2 + (yy - y0) ** 2 + (zz - z0) ** 2) + 1e-10


def one_lognormal(x, std, mean, rescale=None):
    from numpy import sqrt, log, exp, pi, max

    mu = log(mean**2 / sqrt(std**2 + mean**2))
    sig2 = log(1 + std**2 / mean**2)
    lognorm = (1 / (sqrt(2 * pi) * sqrt(sig2) * x)) * exp(
        -((log(x) - mu) ** 2 / (2 * sig2))
    )
    return lognorm / max(lognorm) if rescale is None else lognorm * rescale


def multiple_lognormal(x, mask, ss, ll, rr):
    if mask is None:
        from numpy import ones_like

        mask = ones_like(x)
    return [one_lognormal(x, s, l, r) * mask for s, l, r in zip(ss, ll, rr)]


def one_lognormal_z(x, sig2, mu, rescale=None):
    from numpy import sqrt, log, exp, pi

    lognorm = (1 / (sqrt(2 * pi) * sqrt(sig2) * x)) * exp(
        -((log(x) - mu) ** 2 / (2 * sig2))
    )
    return lognorm * rescale if rescale is not None else lognorm


def multiple_lognormal_z(x, mask, ss, mm, rr):
    from numpy import sqrt, log, exp, pi, max

    if mask is None:
        from numpy import ones_like

        mask = ones_like(x)

    res = []
    maxs = []
    for s, m, r in zip(ss, mm, rr):
        mu = log(m**2 / sqrt(s**2 + m**2))
        sig2 = log(1 + s**2 / m**2)
        res.append(one_lognormal_z(x, sig2, mu, rescale=r) * mask)
        maxs.append(exp(sig2 / 2 - mu) / (sqrt(2 * pi * sig2)))
    max = max(maxs)
    return [r / max for r in res]


def lognormals_z_to_x(xx, mask, params, spline):
    from numpy import maximum

    ss, mm, rr = params
    zs = maximum(1e-4, spline(xx))
    res = multiple_lognormal_z(zs, mask, ss, mm, rr)
    return zs, res
