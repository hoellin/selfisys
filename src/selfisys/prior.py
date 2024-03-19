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

"""Planck2018 prior for pySELFI.

The prior is based on the template provided by Florent Leclercq in the pySELFI package
(https://github.com/florent-leclercq/pyselfi/) and compatible with pySELFI.
"""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


def get_summary(x, bins, normalization=None, kmax=1.4):
    """Computes the summary of the power spectrum.

    Parameters:
    -----------
    x : list
        list of cosmological parameters (h, Omega_b, Omega_m, n_s, sigma_8)
    bins : array
        array of wavenumbers
    normalization : float, optional
        normalization factor
    kmax : float, optional
        max wavenumber
    """
    from pysbmy.power import get_Pk
    from selfisys.utils.tools import cosmo_vector_to_Simbelmyne_dict
    from numpy import array

    theta = get_Pk(bins, cosmo_vector_to_Simbelmyne_dict(x, kmax=kmax))
    if normalization is not None:
        theta /= normalization

    return array(theta)


def worker_class(params):
    """Worker function to compute power spectra with CLASS."""
    x, bins, normalization, kmax = params
    return get_summary(x, bins, normalization, kmax)


class planck_prior(object):
    """This class represents a SELFI prior based on the Planck2018 cosmology."""

    def __init__(
        self,
        Omega_mean,
        Omega_cov,
        bins,
        normalization,
        kmax,
        nsamples=10000,
        nthreads=-1,
        EPS_K=1e-7,
        EPS_residual=1e-3,
        filename=None,
    ):
        from numpy import where

        self.Omega_mean = Omega_mean
        self.Omega_cov = Omega_cov
        self.bins = bins
        self.normalization = normalization
        self.kmax = kmax
        self.nsamples = nsamples
        self.EPS_K = EPS_K
        self.EPS_residual = EPS_residual
        self.filename = filename
        if nthreads == -1:
            from multiprocessing import cpu_count

            self.nthreads = cpu_count() - 1 or 1
        else:
            self.nthreads = nthreads
        self._Nbin_min = where(self.bins >= 0.01)[0].min()
        self._Nbin_max = where(self.bins <= self.kmax)[0].max() + 1

    @property
    def Nbin_min(self):
        """Returns the index of the minimal wavenumber."""
        return self._Nbin_min

    @property
    def Nbin_max(self):
        """Returns the index of the maximal wavenumber."""
        return self._Nbin_max

    def compute(self):
        """Computes the prior (mean, covariance matrix and its inverse)."""
        import numpy as np
        from os.path import exists
        from pyselfi.utils import regular_inv

        if exists(self.filename):
            self.thetas = np.load(self.filename)
        else:
            # Samples from the prior distribution of the cosmological parameters:
            OO = np.random.multivariate_normal(
                np.array(self.Omega_mean), np.array(self.Omega_cov), self.nsamples
            )
            eps = 1e-5
            OO = np.clip(OO, eps, 1 - eps)
            list = [(o, self.bins, self.normalization, self.kmax) for o in OO]

            from multiprocessing import Pool
            import tqdm.auto as tqdm
            from time import time

            start = time()
            thetas = []
            pool = Pool(self.nthreads)
            for theta in tqdm.tqdm(pool.imap(worker_class, list), total=len(list)):
                thetas.append(theta)
            thetas = np.array(thetas)
            pool.close()
            pool.join()
            end = time()
            print("\nDone! (in {:.2f} seconds)".format(end - start))

            self.thetas = thetas
            np.save(self.filename, thetas)

        # Compute the mean and covariance matrix:
        self.mean = np.mean(self.thetas, axis=0)
        self.covariance = np.cov(self.thetas.T)
        self.inv_covariance = regular_inv(
            self.covariance, self.EPS_K, self.EPS_residual
        )

    def logpdf(self, theta, theta_mean, theta_covariance, theta_icov):
        """Return the log prior probability at a given point in parameter space.
        See equation (23) in |Leclercqetal2019|_.

        Parameters
        ----------
        theta : array, double, dimension=S
            evaluation point in parameter space
        theta_mean : array, double, dimension=S
            prior mean
        theta_covariance : array, double, dimension=(S,S)
            prior covariance
        theta_icov : array, double, dimension=(S,S)
            inverse prior covariance

        Returns
        -------
        logpdf : double
            log prior probability

        """
        import numpy as np

        diff = theta - theta_mean
        return (
            -diff.dot(theta_icov).dot(diff) / 2.0
            - np.linalg.slogdet(2 * np.pi * theta_covariance)[1] / 2.0
        )

    def sample(self, seedsample=None):
        """Draw a random sample from the prior.

        Parameters
        ----------
        seedsample : int, optional
            seed for the random number generator

        Returns
        -------
        theta : array, double, dimension=S
            Value in parameter space sampled from the prior

        """
        from numpy.random import seed, multivariate_normal

        if seedsample is not None:
            seed(seedsample)
        return multivariate_normal(self.mean, self.covariance)

    def save(self, fname):
        """Save the prior to an output file.

        Parameters
        ----------
        fname : str
            output filename

        """
        from h5py import File
        from ctypes import c_double
        from pyselfi.utils import PrintMessage, save_replace_dataset, save_replace_attr

        PrintMessage(3, "Writing prior in data file '{}'...".format(fname))

        with File(fname, "r+") as hf:

            def save_to_hf(name, data, **kwargs):
                save_replace_dataset(
                    hf, "/prior/{}".format(name), data, dtype=c_double, **kwargs
                )

            # Save "hyperparameters":
            save_to_hf(
                "thetas",
                self.thetas,
                maxshape=(
                    None,
                    None,
                ),
            )
            save_to_hf("Omega_mean", self.Omega_mean, maxshape=(None,))
            save_to_hf(
                "Omega_cov",
                self.Omega_cov,
                maxshape=(
                    None,
                    None,
                ),
            )
            save_to_hf("bins", self.bins, maxshape=(None,))
            save_replace_attr(
                hf, "/prior/normalization", self.normalization, dtype=c_double
            )
            save_replace_attr(hf, "/prior/kmax", self.kmax, dtype=c_double)

            # Save mandatory attributes: mean, covariance and inv_covariance
            save_to_hf("mean", self.mean, maxshape=(None,))
            save_to_hf("covariance", self.covariance, maxshape=(None, None))
            save_to_hf("inv_covariance", self.inv_covariance, maxshape=(None, None))

        PrintMessage(3, "Writing prior in data file '{}' done.".format(fname))

    @classmethod
    def load(cls, fname):
        """Loads the prior from an input file.

        Parameters
        ----------
        fname : str
            input filename

        Returns
        -------
        prior : :obj:`prior`
            loaded prior object

        """
        from h5py import File
        from numpy import array
        from ctypes import c_double
        from pyselfi.utils import PrintMessage

        PrintMessage(3, "Reading prior in data file '{}'...".format(fname))

        with File(fname, "r") as hf:
            # Load mandatory parameter to call the class constructor:
            Omega_mean = array(hf.get("/prior/Omega_mean"), dtype=c_double)
            Omega_cov = array(hf.get("/prior/Omega_cov"), dtype=c_double)
            bins = array(hf.get("/prior/bins"), dtype=c_double)
            normalization = hf.attrs["/prior/normalization"]
            kmax = hf.attrs["/prior/kmax"]
            # Call the class constructor
            prior = cls(Omega_mean, Omega_cov, bins, normalization, kmax)
            # Load the mandatory attributes (mean, covariance and inv_covariance):
            prior.mean = array(hf.get("prior/mean"), dtype=c_double)
            prior.covariance = array(hf.get("/prior/covariance"), dtype=c_double)
            prior.inv_covariance = array(
                hf.get("/prior/inv_covariance"), dtype=c_double
            )

        PrintMessage(3, "Reading prior in data file '{}' done.".format(fname))
        return prior
