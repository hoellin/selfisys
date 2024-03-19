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

"""Custom forward data model of galaxy surveys to be used as a blackbox for the
inference of cosmological parameters in 2 steps with pySELFI+ILI."""

__author__ = "Tristan Hoellinger"
__version__ = "0.1"
__date__ = "2024"
__license__ = "GPLv3"


class blackbox(object):
    """Custom forward data model to be used as a blackbox for the inference of
    cosmological parameters in 2 steps with pySELFI+ILI.

    Attributes
    ----------
    # > Mandatory arguments:
    k_s : array, double, dimension=S
        vector of input support wavenumbers
    P_ss_path : str
        path to BBKS spectrum at expansion point to normalize outputs. dim(P_ss) = P
    Pbins_bnd : array, double, dimension=P+2
        vector of bin boundaries for the summaries
    theta2P : func
        function to go from normalized theta=spectum/P_ss to spectrum=theta*P_ss
    P : int
        number of output summary statistics. P = #bins x Npop where Npop = #populations.
        /!\ Mandatory for the blackbox to be compatible with pySELFI, even if unused.
    size : int
        size of the simulation box (number of voxels per dimension)
    L : double
        size of the simulation box (Mpc/h)
    G_sim_path : str
        path to the simulation grid
    G_ss_path : str
        path to the summary grid
    Np0 : int
        number of dark matter particles per dimension
    Npm0 : int
        particle-mesh grid size
    fsimdir : str
        directory for outputs

    # > Optional arguments:
    noise_std : double
        standard deviation for the Gaussian noise. If None, no noise is added.
        The noise is added only at the locations where the survey mask is non-zero.
    radial_selection: str
        if not None, apply a radial selection mask before computing the summaries.
        Available options are:
            - 'multiple_lognormal': apply Npop distinct lognormal radial selection to
                                    simulate Npop populations of galaxies.
    selection_params : list
        list of list of parameters for the radial selection mask for each population.
        For the 'multiple_lognormal' radial selection masks, the list should contain:
            [[selection_std]*Npop, [selection_mean]*Npop, [selection_rescale]*Npop]
            These parameters correspond to:
                selection_std : double, example: 0.2
                    standard deviation of the distribution wrt z, e.g. cst * (1+z)
                selection_mean : double, example: 0.5
                    Mean of the distribution in redshift space
                selection_rescale : double, default=None
                    If None, the distributions are only rescaled so that their global
                    maximum is 1. If not None, also individually rescale by these values
    observed_density : double
        if not None, work with the observed density field rather than the overdensity
        contrast field
    linear_bias : double
        if not None, apply a linear bias to the observed field
    norm_csts : double or list of double
        If not None, normalize the output of the blackbox so that for instance the
        observed wiggles are ~1. For the 'multiple_lognormal' radial selection mask,
        should be a list containing exactly Npop values
    survey_mask_path : str
        if not None, apply the given survey mask to the observed field
    sim_params : str
        string to select the set of simulation parameters to be used for Simbelmyne.
        If "std19COLA6RSD", LPT->z=19 then COLA w/ 6 steps->z=0, RSDs.
            19 can be replaced by any initial redshift.
            COLA can be either COLA or PM. The number of timesteps can be any integer.
            If the string does not end with "RSD", RSDs are not applied.
        If "custom19COLA20RSD", LPT->z=19 then COLA w/ 20 steps->z=0, RSDs.
            When using "custom" instead of "std", the user must provide the timestepping
            object in the TimeStepDistribution attribute.
        If "split...", use as many Simbelmynë data cards as there are populations.
            For debugging purposes only.
    TimeStepDistribution : str
        if not None, provide the timestepping object for the custom Simbelmynë card.
    TimeSteps : list
        if not None, provide the list of number of timesteps between all intermediate
        redshifts for the custom Simbelmynë card.
    eff_redshifts : list
        if not None, provide the list of effective redshifts for the Simbelmynë card.
    seedphase : int
        value of the seed to generate the "initial" white noise realization
    seednoise : int
        value of the seed to generate the noise sequence (initial state of the RNG).
        If fixnoise is True, the seed used in `compute_pool` will always be seednoise,
        If fixnoise is False, will be seednoise + i (w/ i = index of the realization).
    fixnoise : bool
        fix the noise realization? if True the seed used will always be seednoise
    seednorm : int
        value of the seed to generate the normalization sequence
    reset : bool
        if True, overwrite the existing files when initializing the blackbox
    save_frequency : int
        saves the output on the blackbox on disk each save_frequency evaluations
    verbosity : int
        verbosity level of the blackbox. 0 is silent, 1 is minimal, 2 is verbose

    """

    # Using __slots__ saves memory by avoiding the creation of __dict__ and __weakref__:
    __slots__ = [
        "k_s",
        "P_ss_path",
        "Pbins_bnd",
        "theta2P",
        "P",
        "size",
        "L",
        "G_sim_path",
        "G_ss_path",
        "Np0",
        "Npm0",
        "fsimdir",
        "modeldir",
        "noise_std",
        "radial_selection",
        "selection_params",
        "observed_density",
        "linear_bias",
        "norm_csts",
        "survey_mask_path",
        "sim_params",
        "TimeStepDistribution",
        "TimeSteps",
        "eff_redshifts",
        "seedphase",
        "seednoise",
        "fixnoise",
        "seednorm",
        "reset",
        "save_frequency",
        "verbosity",
        "_Npop",
        "_Ntimesteps",
        "_force_recompute_mocks",
        "_setup_only",
        "_prefix_mocks_mis",
        "_modified_selfi",
        "_Psingle",
        "S",
    ]

    def __init__(
        self,
        k_s,
        P_ss_path,
        Pbins_bnd,
        theta2P,
        P,
        size,
        L,
        G_sim_path,
        G_ss_path,
        Np0,
        Npm0,
        fsimdir,
        **kwargs,
    ):
        """Initializes the blackbox object."""
        for attr in self.__slots__[:12]:
            setattr(self, attr, eval(attr))
        for key, value in kwargs.items():
            setattr(self, key, value)
        if "modeldir" not in kwargs:
            self.modeldir = fsimdir + "/model/"

        self._Npop = (
            len(self.selection_params[0])
            if self.radial_selection == "multiple_lognormal"
            else 1
        )
        self._force_recompute_mocks = False
        self._setup_only = False
        self._prefix_mocks_mis = None
        self._modified_selfi = True
        self._Ntimesteps = (
            len(self.TimeStepDistribution) if self.sim_params[:5] == "split" else None
        )

        # Create the window function W(n,r) = C(n) * R(r)
        self._init_survey_mask()
        self._init_radial_selection()

        self.S = len(self.k_s)

    def _init_survey_mask(self):
        """Create the survey mask C(n) and write it to disk."""
        from os.path import exists

        if not exists(self.modeldir + "survey_mask_binary.h5") or self.reset:
            from gc import collect
            from numpy import array, load

            if self.survey_mask_path is not None:
                survey_mask = load(self.survey_mask_path)
                survey_mask_binary = array(survey_mask > 0, dtype=int)
            else:
                from numpy import ones

                survey_mask = ones([self.size] * 3, dtype=int)
                survey_mask_binary = survey_mask
            from h5py import File

            with File(self.modeldir + "survey_mask_binary.h5", "w") as f:
                f.create_dataset("survey_mask_binary", data=survey_mask_binary)
            del survey_mask
            del survey_mask_binary
            collect()

    def _init_radial_selection(self):
        """Create the radial selection function R(r) and write it to disk."""
        if self.radial_selection is not None:
            if self.radial_selection == "multiple_lognormal":
                self.norm_csts = (
                    [1.0 for _ in range(len(self.selection_params[0]))]
                    if self.norm_csts is None
                    else self.norm_csts
                )
                self._Psingle = self.P // len(self.selection_params[0])
                from os.path import exists

                if not exists(self.modeldir + "select_fct.h5") or self.reset:
                    from gc import collect
                    from h5py import File
                    from numpy import load, linspace
                    from astropy.cosmology import FlatLambdaCDM
                    from scipy.interpolate import UnivariateSpline
                    from selfisys.global_parameters import omegas_gt
                    from selfisys.selection_functions import r_grid, lognormals_z_to_x

                    redshifts_upper_bound = 3.0
                    zz = linspace(0, redshifts_upper_bound, 10000)
                    cosmo = FlatLambdaCDM(
                        H0=100 * omegas_gt[0], Ob0=omegas_gt[1], Om0=omegas_gt[2]
                    )
                    d = cosmo.comoving_distance(zz) / 1e3
                    d_np = d.to_value()
                    spline = UnivariateSpline(d_np, zz, k=5)

                    self._plot_selection_functions(zz, d_np, spline, omegas_gt)

                    if self.survey_mask_path is not None:
                        survey_mask = load(self.survey_mask_path)
                    else:
                        survey_mask = None
                    r = r_grid(self.L / 1e3, self.size)
                    _, select_fct = lognormals_z_to_x(
                        r,
                        survey_mask,
                        self.selection_params,
                        spline,
                    )

                    del survey_mask, r, d, d_np, spline, _
                    with File(self.modeldir + "select_fct.h5", "w") as f:
                        f.create_dataset("select_fct", data=select_fct)
                    del select_fct
                    collect()
            else:
                raise ValueError(
                    "Unknown/unimplemented selection function: " + self.radial_selection
                )
        else:
            self._Psingle = self.P
            if self.norm_csts is None:
                self.norm_csts = 1.0

    @property
    def force_recompute_mocks(self):
        return self._force_recompute_mocks

    @property
    def setup_only(self):
        return self._setup_only

    @property
    def Ntimesteps(self):
        return self._Ntimesteps

    @property
    def modified_selfi(self):
        return self._modified_selfi

    def update(self, **kwargs):
        """Updates the given parameter(s) of the blackbox with the given value(s).

        Parameters
        ----------
        **kwargs : dict
            dictionary of parameters to update

        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _save_cosmo(self, cosmo, fname_cosmo, force_cosmo=False):
        """Saves cosmological parameters in json format.

        Parameters
        ----------
        cosmo : dictionary
            cosmological parameters (+ some infrastructure parameters) to be saved
        fname_cosmo : str
            name of the output json file
        force_cosmo : bool, optional, default=False
            overwrite if the file already exists?

        """
        from os.path import exists

        if not exists(fname_cosmo) or force_cosmo:
            from json import dump

            with open(fname_cosmo, "w") as fp:
                dump(cosmo, fp)

    def _plot_selection_functions(self, zz, d_np, spline, omega):
        """Plot the selection functions and write the plot to disk.

        Parameters
        ----------
        zz : array, double, dimension=N
            redshifts
        d_np : array, double, dimension=N
            comoving distances
        spline : UnivariateSpline
            spline interpolating the comoving distances
        omega : array, double
            cosmological parameters

        """

        from numpy import linspace, sqrt, argmin, abs
        from classy import Class
        from selfisys.utils.tools import cosmo_vector_to_class_dict
        from selfisys.selection_functions import lognormals_z_to_x
        from selfisys.utils.plot_utils import plot_selection_functions

        L = self.L / 1e3
        Lcorner = sqrt(3) * L
        zcorner = zz[argmin(abs(d_np - Lcorner))]

        # Get linear growth factor from CLASS:
        cosmo_dict = cosmo_vector_to_class_dict(omega)
        cosmo_class = Class()
        cosmo_class.set(cosmo_dict)
        cosmo_class.compute()
        Dz = cosmo_class.get_background()["gr.fac. D"]
        redshifts = cosmo_class.get_background()["z"]
        cosmo_class.struct_cleanup()
        cosmo_class.empty()

        # Define the axis for the plot:
        xx = linspace(1e-5, Lcorner, 1000)
        zz, res = lognormals_z_to_x(
            xx,
            None,
            self.selection_params,
            spline,
        )

        # Call auxiliary plotting routine:
        plot_selection_functions(
            xx,
            res,
            None,
            self.selection_params[1],
            redshifts,
            Dz,
            self.selection_params,
            L,
            Lcorner,
            zz=zz,
            zcorner=zcorner,
            path=self.modeldir + "selection_functions.png",
        )

    def _get_powerspectrum_from_cosmo(self, cosmo, fname_powerspectrum, force=False):
        """Load or compute the power spectrum from input cosmological parameters and
        write it to disk.

        Parameters
        ----------
        cosmo : dictionary
            cosmological parameters (and some infrastructure parameters)
        fname_powerspectrum : str
            name of input/output power spectrum file
        force : bool, optional, default=False
            force recomputation?

        """
        from os.path import exists

        if (not exists(fname_powerspectrum)) or force:
            from pysbmy.power import PowerSpectrum

            L = self.L
            N = self.size
            P = PowerSpectrum(L, L, L, N, N, N, cosmo)
            P.write(fname_powerspectrum)

    def _add_noise(self, g, seednoise, field=None):
        """Adds noise to a realization in physical space.

        Parameters
        ----------
        g : array, double, dimension=(size,size,size)
            field to which the noise is applied /!\ g is modified in place
        seednoise : int or list of int
            value of the seed to generate the noise realization
        field : array, double, dimension=(size,size,size), optional, default=None
            selection function to apply to the input field

        Returns
        -------
        g : array, double, dimension=(size,size,size)
            realization in physical space with added noise
        """
        if self.noise_std > 0:
            from gc import collect
            from numpy import random, sqrt
            from h5py import File

            if seednoise is not None:
                rng = random.default_rng(seednoise)
            else:
                raise ValueError(
                    "seednoise must be provided by user. None is not acceptable."
                )
            if field is None:
                from numpy import ones_like

                field = ones_like(g)
            N = self.observed_density if self.observed_density is not None else 1.0
            field = rng.normal(
                size=[self.size] * 3,
                scale=self.noise_std * sqrt(N * field),
            )

            with File(self.modeldir + "survey_mask_binary.h5", "r") as f:
                field *= f["survey_mask_binary"][:]
            g += field
            del field

            collect()

    def _get_density_field(self, delta_g_dm, bias):
        """Apply galaxy bias to a dark matter overdensity field.

        Parameters
        ----------
        delta_g_dm : array, double, dimension=(size,size,size)
            dark matter density contrast in physical space
        bias : float
            linear bias

        Returns
        -------
        delta_g : array, double, dimension=(size,size,size)
            galaxy density/overdensity
        """
        if bias is None:
            bias = 1.0
        if not isinstance(bias, float):
            raise TypeError("bias must be a float")
        if self.observed_density is None:
            delta_g = bias * delta_g_dm
        else:
            delta_g = self.observed_density * (1 + bias * delta_g_dm)
        return delta_g

    def _compute_Phi(self, g_obj, norm, AliasingCorr=True):
        """Compute the summary statistics from a field object.

        Parameters
        ----------
        g_obj : :obj:`Field`
            input field object
        norm : array, double, dimension=P
            normalization constants for the summary statistics
        AliasingCorr : bool, optional, default=True
            apply the aliasing correction to the power spectrum?

        Returns
        -------
        Phi : array, c_double, dimension=P
            vector of summary statistics

        """
        from gc import collect
        from pysbmy.correlations import get_autocorrelation
        from pysbmy import c_double
        from io import BytesIO

        from pysbmy.power import FourierGrid

        G_ss = FourierGrid.read(self.G_ss_path)
        if self.verbosity > 1:
            Pk, _ = get_autocorrelation(g_obj, G_ss, AliasingCorr=AliasingCorr)
        else:
            from selfisys.utils.low_level import stdout_redirector

            f = BytesIO()
            with stdout_redirector(f):
                Pk, _ = get_autocorrelation(g_obj, G_ss, AliasingCorr=AliasingCorr)
            f.close()
        from pysbmy.power import PowerSpectrum

        P_ss = PowerSpectrum.read(self.P_ss_path)
        Phi = Pk / (norm * P_ss.powerspectrum)
        del G_ss
        del P_ss
        collect()
        return Phi.astype(c_double)

    def _repaint_and_get_Phi(
        self,
        g_obj,
        norm,
        seednoise,
        bias=None,
        field=None,
        return_g=False,
        AliasingCorr=True,
    ):
        """Repaint a realization in physical space, and compute its summary statistics.

        Parameters
        ----------
        g_obj : :obj:`Field`
            input field object
        norm : array, double, dimension=P
            normalization constants for the summary statistics
        seednoise : int or list of int
            value of the seed to generate the noise realization
        bias : float, optional, default=None
            value of the bias to apply to the input field
        field : array, double, dimension=(size,size,size), optional, default=None
            Selection function. Reused as output to save memory allocations.
        return_g : bool, optional, default=False
            if True, returns the full field in addition to the summary
        AliasingCorr : bool, optional, default=True
            apply the aliasing correction to the power spectrum?

        Returns
        -------
        Phi : array, c_double, dimension=P
            vector of summary statistics
        delta_g : array, double, dimension=(size,size,size)
            realization in physical space

        """
        from gc import collect
        import copy

        g_obj_local = copy.deepcopy(g_obj)
        g_obj_local.data = self._get_density_field(g_obj_local.data, bias)
        if field is not None:
            g_obj_local.data *= field
        self._add_noise(g_obj_local.data, seednoise=seednoise, field=field)  # /!\ i/o
        Phi = self._compute_Phi(g_obj_local, norm, AliasingCorr=AliasingCorr)
        field = copy.deepcopy(g_obj_local.data) if return_g else None
        del g_obj_local
        collect()
        return Phi, field

    def _apply_selection(
        self, fnames_outputdensity, seednoise, return_g=False, AliasingCorr=True
    ):
        """Apply the selection function to a realization in physical space.

        Parameters
        ----------
        fnames_outputdensity : str
            name of the output density field file
        seednoise : int or list of int
            value of the seed to generate the noise realization
        return_g : bool, optional, default=False
            if True, returns the full field in addition to the summary
        AliasingCorr : bool, optional, default=True
            apply the aliasing correction to the power spectrum?

        Returns
        -------
        result : tuple
            Phi_tot : array, double, dimension = P*Npop
                summary statistics obtained for each population
            gs : list of array, double, dimension=[(size,size,size)]*Npop
                full field in physical space if return_g=True, None otherwise

        """
        from gc import collect
        from pysbmy.field import read_basefield

        split = self.sim_params[:5] == "split" or self.sim_params[:6] == "custom"
        if not split:
            g_obj = read_basefield(fnames_outputdensity[0])
        if self.radial_selection is not None:
            from numpy import concatenate, array
            from h5py import File

            Phi_tot = []
            gs = []
            for ifct in range(len(self.selection_params[0])):
                if split:
                    g_obj = read_basefield(fnames_outputdensity[ifct])
                with File(self.modeldir + "select_fct.h5", "r") as f:
                    field = f["select_fct"][:][ifct]
                    field = array(field, dtype=float)
                bias = self.linear_bias[ifct] if self.linear_bias is not None else None
                Phi, g_obs = self._repaint_and_get_Phi(
                    g_obj,
                    self.norm_csts[ifct],
                    seednoise=seednoise,
                    bias=bias,
                    field=field,
                    return_g=return_g,
                    AliasingCorr=AliasingCorr,
                )
                Phi_tot = concatenate([Phi_tot, Phi])
                gs.append(g_obs) if return_g else None
            result = Phi_tot, gs
            del g_obj
            del g_obs
            del field
            collect()
        else:
            bias = self.linear_bias if self.linear_bias is not None else None
            if not split:
                result = self._repaint_and_get_Phi(
                    g_obj,
                    self.norm_csts,
                    seednoise=seednoise,
                    bias=bias,
                    field=None,
                    return_g=return_g,
                    AliasingCorr=AliasingCorr,
                )
            else:
                from numpy import concatenate

                Phi_tot = []
                gs = []
                for ifct in range(len(self.TimeSteps)):
                    g_obj = read_basefield(fnames_outputdensity[ifct])
                    Phi, g_obs = self._repaint_and_get_Phi(
                        g_obj,
                        self.norm_csts,
                        seednoise=seednoise,
                        bias=bias,
                        field=None,
                        return_g=return_g,
                        AliasingCorr=AliasingCorr,
                    )
                    Phi_tot = concatenate([Phi_tot, Phi])
                    gs.append(g_obs) if return_g else None
                result = Phi_tot, gs
                del g_obj
                del g_obs
                collect()
        return result

    def _check_files_exist(self, files):
        from os.path import exists

        return all([exists(f) for f in files])

    def _setup_parfiles(
        self,
        d,
        cosmology,
        fname_simparfile,
        fname_powerspectrum,
        fname_whitenoise,
        fname_outputinitialdensity,
        fnames_outputrealspacedensity,
        fnames_outputdensity,
        force=False,
    ):
        """Sets up Simbelynë parameter file given the necessary inputs
        (please refer to the Simbelynë documentation for more details).

        Parameters
        ----------
        d : int
            index giving the direction in parameter space:
            -1 for mock data, 0 for the expansion point, or from 1 to S
        cosmology : array, double, dimension=5
            cosmological parameters
        fname_simparfile : str
            name of output simulation parameter file
        fname_powerspectrum : str
            name of input power spectrum file
        fname_whitenoise : str
            name of output white noise file
        fname_outputinitialdensity : str
            name of output initial density field file
        fnames_outputrealspacedensity : str
            names of output real-space density field files
        fnames_outputdensity : str
            names of output redshift-space density field files
        force : bool, optional, default=False
            overwrite if files already exists?

        """
        from os.path import exists

        if not exists(fname_simparfile + "_{}.sbmy".format(self.Npop)) or force:
            from pysbmy import param_file
            from re import search

            regex = r"([a-zA-Z]+)(\d+)?([a-zA-Z]+)?(\d+)?([a-zA-Z]+)?"
            m = search(regex, self.sim_params)
            if m.group(1) == "std":
                # Single LPT+COLA/PM Simbelmynë data card with linear time stepping
                RedshiftLPT = int(m.group(2))
                RedshiftFCs = 0.0
                match m.group(3):
                    case None:
                        ModulePMCOLA = 0
                        EvolutionMode = 2
                        NumberOfTimeSteps = 0
                        ModuleRSD = 0
                    case "RSD":
                        ModulePMCOLA = 0
                        EvolutionMode = 2
                        NumberOfTimeSteps = 0
                        ModuleRSD = 1
                    case "PM":
                        ModulePMCOLA = 1
                        EvolutionMode = 1
                        NumberOfTimeSteps = m.group(4)
                        ModuleRSD = 1 if m.group(5)[:3] == "RSD" else 0
                    case "COLA":
                        ModulePMCOLA = 1
                        EvolutionMode = 2
                        NumberOfTimeSteps = m.group(4)
                        ModuleRSD = 1 if m.group(5)[:3] == "RSD" else 0
                    case _:
                        raise ValueError(
                            "sim_params = {} not valid".format(self.sim_params)
                        )
                NumberOfTimeSteps = int(m.group(4)) if m.group(4) is not None else 0
            elif m.group(1) == "custom":
                # Single LPT+COLA/PM Simbelmynë card w/ user-provided time stepping obj:
                RedshiftLPT = int(m.group(2))
                match m.group(3):
                    case "PM":
                        ModulePMCOLA = 1
                        EvolutionMode = 1
                        ModuleRSD = 1 if m.group(5)[:3] == "RSD" else 0
                    case "COLA":
                        ModulePMCOLA = 1
                        EvolutionMode = 2
                        ModuleRSD = 1 if m.group(5)[:3] == "RSD" else 0
                    case _:
                        raise ValueError(
                            "sim_params = {} not valid".format(self.sim_params)
                        )
                if self.TimeStepDistribution is None:
                    raise ValueError(
                        "TimeStepDistribution must be provided for 'custom'."
                    )
            elif m.group(1) == "split":
                # Use as many Simbelmynë data cards as there are populations:
                if self.TimeStepDistribution is None:
                    raise ValueError(
                        "TimeStepDistribution must be provided for 'custom'."
                    )
                if self.eff_redshifts is None:
                    raise ValueError("eff_redshifts must be provided for 'split'.")
                elif len(self.eff_redshifts) != self.Ntimesteps:
                    raise ValueError("len(eff_redshifts) != Ntimesteps")

                RedshiftLPT = int(m.group(2))
                match m.group(3):
                    case None:
                        ModulePMCOLA = 1
                        EvolutionMode = 2
                        ModuleRSD = 0
                    case "RSD":
                        ModulePMCOLA = 1
                        EvolutionMode = 2
                        ModuleRSD = 1
                    case _:
                        raise ValueError(
                            "sim_params = {} not valid".format(self.sim_params)
                        )
                NumberOfTimeSteps = int(m.group(4)) if m.group(4) is not None else 0

            else:
                raise ValueError("sim_params = {} not valid" + self.sim_params)

            if self.sim_params[-3:] != "obs":
                from selfisys.global_parameters import (
                    h_obs as h,
                    Omega_b_obs as Omega_b,
                    Omega_m_obs as Omega_m,
                    nS_obs as nS,
                    sigma8_obs as sigma8,
                )
            else:
                if self.modified_selfi:
                    h, Omega_b, Omega_m, nS, sigma8 = cosmology
                else:
                    # Fix the fiducial cosmology to the best guess of the prior:
                    from selfisys.global_parameters import (
                        h_planck as h,
                        Omega_b_planck as Omega_b,
                        Omega_m_planck as Omega_m,
                        nS_planck as nS,
                        sigma8_planck as sigma8,
                    )

            if d < 0:  # -1 for mock data or -2 to recompute the observations
                WriteInitialConditions = 1
                WriteDensities = 1  # write also density fields in real space
            else:  # d==0, expansion point, or d>0, points needed for gradients
                WriteInitialConditions = 0
                WriteDensities = 1  # write also density fields in real space

            if m.group(1) == "std":
                S = param_file(  ## Module LPT ##
                    ModuleLPT=1,
                    # Basic setup:
                    Particles=self.Np0,
                    Mesh=self.size,
                    BoxSize=self.L,
                    corner0=0.0,
                    corner1=0.0,
                    corner2=0.0,
                    # Initial conditions:
                    ICsMode=1,
                    WriteICsRngState=0,
                    WriteInitialConditions=WriteInitialConditions,
                    InputWhiteNoise=fname_whitenoise,
                    OutputInitialConditions=fname_outputinitialdensity,  # a=1e-3
                    # Power spectrum:
                    InputPowerSpectrum=fname_powerspectrum,
                    # Final conditions for LPT:
                    RedshiftLPT=RedshiftLPT,
                    WriteLPTSnapshot=0,
                    WriteLPTDensity=0,
                    ####################
                    ## Module PM/COLA ##
                    ####################
                    ModulePMCOLA=ModulePMCOLA,
                    EvolutionMode=EvolutionMode,  # 1 for PM, 2 for COLA
                    ParticleMesh=self.Npm0,
                    NumberOfTimeSteps=NumberOfTimeSteps,
                    # Final snapshot:
                    RedshiftFCs=RedshiftFCs,
                    WriteFinalSnapshot=0,
                    WriteFinalDensity=WriteDensities,
                    OutputFinalDensity=fnames_outputrealspacedensity[0],
                    #########
                    ## RSD ##
                    #########
                    ModuleRSD=ModuleRSD,
                    WriteIntermediaryRSD=0,
                    DoNonLinearMapping=0,  # whether (0) or not (1) to use the linear
                    # approximation for the RSDs
                    WriteRSDensity=1,
                    OutputRSDensity=fnames_outputdensity[0],
                    #############################
                    ## Cosmological parameters ##
                    #############################
                    h=h,
                    Omega_q=1.0 - Omega_m,
                    Omega_b=Omega_b,
                    Omega_m=Omega_m,
                    Omega_k=0.0,
                    n_s=nS,
                    sigma8=sigma8,
                    w0_fld=-1.0,
                    wa_fld=0.0,
                )
                S.write(fname_simparfile + "_{}.sbmy".format(self.Npop))
            elif m.group(1) == "custom":
                RedshiftFCs = self.eff_redshifts
                fname_outputdensity = (
                    fnames_outputdensity[0][: fnames_outputdensity[0].rfind("_")]
                    + ".h5"
                )
                S = param_file(  ## Module LPT ##
                    ModuleLPT=1,
                    # Basic setup:
                    Particles=self.Np0,
                    Mesh=self.size,
                    BoxSize=self.L,
                    corner0=0.0,
                    corner1=0.0,
                    corner2=0.0,
                    # Initial conditions:
                    ICsMode=1,
                    WriteICsRngState=0,
                    WriteInitialConditions=WriteInitialConditions,
                    InputWhiteNoise=fname_whitenoise,
                    OutputInitialConditions=fname_outputinitialdensity,  # a=1e-3
                    # Power spectrum:
                    InputPowerSpectrum=fname_powerspectrum,
                    # Final conditions for LPT:
                    RedshiftLPT=RedshiftLPT,
                    WriteLPTSnapshot=0,
                    WriteLPTDensity=0,
                    ####################
                    ## Module PM/COLA ##
                    ####################
                    ModulePMCOLA=ModulePMCOLA,
                    EvolutionMode=EvolutionMode,  # 1 for PM, 2 for COLA
                    ParticleMesh=self.Npm0,
                    OutputKickBase=self.fsimdir + "/data/cola_kick_",
                    # Final snapshot:
                    RedshiftFCs=RedshiftFCs,
                    WriteFinalSnapshot=0,
                    WriteFinalDensity=0,
                    OutputFinalDensity=fnames_outputrealspacedensity[0],
                    # Intermediate snapshots:
                    WriteSnapshots=0,
                    WriteDensities=WriteDensities,
                    OutputDensitiesBase=fnames_outputrealspacedensity[0][:-3] + "_",
                    OutputDensitiesExt=".h5",
                    ############################
                    ## Time step distribution ##
                    ############################
                    TimeStepDistribution=self.TimeStepDistribution,
                    ModifiedDiscretization=1,
                    n_LPT=-2.5,  # Ansatz in modified Kick&Drift operator for COLA
                    #########
                    ## RSD ##
                    #########
                    ModuleRSD=ModuleRSD,
                    WriteIntermediaryRSD=1,
                    DoNonLinearMapping=0,  # whether (0) or not (1) to use the linear
                    # approximation for the RSDs
                    WriteRSDensity=1,
                    OutputRSDensity=fname_outputdensity,
                    #############################
                    ## Cosmological parameters ##
                    #############################
                    h=h,
                    Omega_q=1.0 - Omega_m,
                    Omega_b=Omega_b,
                    Omega_m=Omega_m,
                    Omega_k=0.0,
                    n_s=nS,
                    sigma8=sigma8,
                    w0_fld=-1.0,
                    wa_fld=0.0,
                )
                S.write(fname_simparfile + "_{}.sbmy".format(self.Npop))
            elif m.group(1) == "split":
                datadir = self.fsimdir + "/data/"
                RedshiftFCs = self.eff_redshifts[0]

                S = param_file(
                    ################
                    ## Module LPT ##
                    ################
                    ModuleLPT=1,
                    # Basic setup:
                    Particles=self.Np0,
                    Mesh=self.size,
                    BoxSize=self.L,
                    corner0=0.0,
                    corner1=0.0,
                    corner2=0.0,
                    # Initial conditions:
                    ICsMode=1,
                    WriteICsRngState=0,
                    WriteInitialConditions=WriteInitialConditions,
                    InputWhiteNoise=fname_whitenoise,
                    OutputInitialConditions=fname_outputinitialdensity,
                    # Power spectrum:
                    InputPowerSpectrum=fname_powerspectrum,
                    # Final conditions for LPT:
                    RedshiftLPT=RedshiftLPT,
                    WriteLPTSnapshot=0,
                    WriteLPTDensity=0,
                    ####################
                    ## Module PM/COLA ##
                    ####################
                    ModulePMCOLA=ModulePMCOLA,
                    EvolutionMode=EvolutionMode,
                    ParticleMesh=self.Npm0,
                    OutputKickBase=datadir + "cola_kick_0_",
                    # Final snapshot:
                    RedshiftFCs=RedshiftFCs,
                    WriteFinalSnapshot=1,
                    OutputFinalSnapshot=datadir + "cola_snapshot_0.gadget3",
                    WriteFinalDensity=1,
                    OutputFinalDensity=fnames_outputrealspacedensity[0],
                    WriteLPTDisplacements=1,
                    OutputPsiLPT1=datadir + "lpt_psi1_0.h5",
                    OutputPsiLPT2=datadir + "lpt_psi2_0.h5",
                    ############################
                    ## Time step distribution ##
                    ############################
                    TimeStepDistribution=self.TimeStepDistribution[0],
                    ModifiedDiscretization=1,
                    #########
                    ## RSD ##
                    #########
                    ModuleRSD=ModuleRSD,
                    WriteIntermediaryRSD=0,
                    DoNonLinearMapping=0,
                    WriteRSDensity=1,
                    OutputRSDensity=fnames_outputdensity[0],
                    #############################
                    ## Cosmological parameters ##
                    #############################
                    h=h,
                    Omega_q=1.0 - Omega_m,
                    Omega_b=Omega_b,
                    Omega_m=Omega_m,
                    Omega_k=0.0,
                    n_s=nS,
                    sigma8=sigma8,
                    w0_fld=-1.0,
                    wa_fld=0.0,
                )
                S.write(fname_simparfile + "_pop0.sbmy")

                for i in range(1, self.Ntimesteps):
                    RedshiftFCs = self.eff_redshifts[i]

                    S = param_file(
                        ModuleLPT=0,
                        # Basic setup:
                        Particles=self.Np0,
                        Mesh=self.size,
                        BoxSize=self.L,
                        corner0=0.0,
                        corner1=0.0,
                        corner2=0.0,
                        InputPsiLPT1=datadir + "lpt_psi1_0.h5",
                        InputPsiLPT2=datadir + "lpt_psi2_0.h5",
                        ####################
                        ## Module PM/COLA ##
                        ####################
                        ModulePMCOLA=ModulePMCOLA,
                        InputPMCOLASnapshot=datadir
                        + "cola_snapshot_{:d}.gadget3".format(i - 1),
                        EvolutionMode=EvolutionMode,
                        ParticleMesh=self.Npm0,
                        OutputKickBase=datadir + "cola_kick_{:d}_".format(i),
                        # Final snapshot:
                        RedshiftFCs=RedshiftFCs,
                        WriteFinalSnapshot=1,
                        OutputFinalSnapshot=datadir
                        + "cola_snapshot_{:d}.gadget3".format(i),
                        WriteFinalDensity=1,
                        OutputFinalDensity=fnames_outputrealspacedensity[i],
                        WriteLPTDisplacements=0,
                        ############################
                        ## Time step distribution ##
                        ############################
                        TimeStepDistribution=self.TimeStepDistribution[i],
                        ModifiedDiscretization=1,
                        #########
                        ## RSD ##
                        #########
                        ModuleRSD=ModuleRSD,
                        WriteIntermediaryRSD=0,
                        DoNonLinearMapping=0,
                        WriteRSDensity=1,
                        OutputRSDensity=fnames_outputdensity[i],
                        #############################
                        ## Cosmological parameters ##
                        #############################
                        h=h,
                        Omega_q=1.0 - Omega_m,
                        Omega_b=Omega_b,
                        Omega_m=Omega_m,
                        Omega_k=0.0,
                        n_s=nS,
                        sigma8=sigma8,
                        w0_fld=-1.0,
                        wa_fld=0.0,
                    )
                    S.write(fname_simparfile + "_pop{}.sbmy".format(i))

    def _run_sim(
        self,
        fname_simparfile,
        fname_simlogs,
        fnames_outputdensity,
        force_sim=False,
        check_output=False,
    ):
        """Runs a simulation with Simbelynë.

        Parameters
        ----------
        fname_simparfile : str
            name of the input parameter file
        fname_simlogs : str
            name of the output Simbelynë logs
        fnames_outputdensity : str
            names of the output density fields to be written
        force_sim : bool, optional, default=False
            force recomputation if output density already exists?
        check_output : bool, optional, default=True
            check the integrity of the output file and recomputes it if corrupted

        """
        from os.path import exists
        from os import remove
        from glob import glob

        if not self._check_files_exist(fnames_outputdensity) or force_sim:
            from pysbmy import pySbmy

            if self.sim_params[:5] != "split":
                pySbmy(fname_simparfile + "_{}.sbmy".format(self.Npop), fname_simlogs)
            else:
                for i in range(self.Ntimesteps):
                    pySbmy(fname_simparfile + "_pop{}.sbmy".format(i), fname_simlogs)
        elif check_output:
            from pysbmy.field import read_basefield

            try:
                for fname in fnames_outputdensity:
                    g = read_basefield(fname)
                del g
            except:
                from os import remove
                from pysbmy import pySbmy

                for fname in fnames_outputdensity:
                    remove(fname)
                pySbmy(fname_simparfile, fname_simlogs)

        for f in glob(self.fsimdir + "/data/cola_kick_*.h5"):
            remove(f)  # workaround to deal with another workaround in Simbelynë
        if self.sim_params[:5] == "split":
            for f in glob(self.fsimdir + "/data/cola_snapshot_*.h5"):
                remove(f)
            for f in glob(self.fsimdir + "/data/lpt_psi*.h5"):
                remove(f)

    def _compute_mocks(
        self,
        seednoise,
        fnames_outputdensity,
        fname_mocks,
        return_g,
        fname_g_obs=None,
        AliasingCorr=True,
    ):
        """Apply galaxy bias and observational effects to compute the summary statistics
        from a dark matter overdensity field.

        Parameters
        ----------
        seednoise : int or list of int
            value of the seed to generate the noise realization
        fnames_outputdensity : str
            names of the output density fields
        fname_mocks : str
            name of the output mock files
        return_g : bool
            if True, returns the full field in addition to the summary
        fname_g_obs : str, optional, default=None
            name of the output g_obs file
        AliasingCorr : bool, optional, default=True
            apply the aliasing correction to the power spectrum?
        """
        from os.path import exists
        from h5py import File

        if (not exists(fname_mocks)) or self.force_recompute_mocks:
            Phi, g_obs = self._apply_selection(
                fnames_outputdensity,
                seednoise=seednoise,
                return_g=return_g,
                AliasingCorr=AliasingCorr,
            )
            with File(fname_mocks, "w") as f:
                f.create_dataset("Phi", data=Phi)
            if return_g:
                with File(fname_g_obs, "w") as f:
                    f.create_dataset("g", data=g_obs)
        else:
            self._PrintMessage(1, "fname_mocks = " + fname_mocks)
            with File(fname_mocks, "r") as f:
                Phi = f["Phi"][:]
            if return_g:
                with File(fname_g_obs, "r") as f:
                    g_obs = f["g"][:]
            else:
                g_obs = None

        return Phi, g_obs

    def _generate_white_noise_Field(
        self, seedphase, fname_whitenoise, seedname_whitenoise, force_phase=False
    ):
        """Generates a white noise realization in physical space using SeedSequence and
        default_rng from numpy.random.

        Parameters
        ----------
        seedphase : int or list of int
            user-provided seed to generate the "initial" white noise realization
        fname_whitenoise : str
            name of the output white noise file
        seedname_whitenoise : str
            name of the output white noise seed file
        force_phase : bool, optional, default=False
            force recomputation of the phase?

        """
        from os.path import exists

        if not exists(fname_whitenoise) or force_phase:
            from gc import collect
            from numpy import random, save
            from pysbmy.field import BaseField

            rng = random.default_rng(seedphase)
            save(seedname_whitenoise, rng.bit_generator.state)
            with open(seedname_whitenoise + ".txt", "w") as f:
                f.write(str(rng.bit_generator.state))
            data = rng.standard_normal(size=self.size**3)
            L = self.L
            N = self.size
            wn = BaseField(L, L, L, 0, 0, 0, 1, N, N, N, data)
            del data
            wn.write(fname_whitenoise)
            del wn
            collect()

    def _sample_omega(self, seed):
        """Samples the cosmological parameters from the prior.

        Parameters
        ----------
        seed : int
            seed to generate the cosmological parameters

        Returns
        -------
        omega : array, double, dimension=5
            sampled cosmological parameters

        """
        from selfisys.utils.tools import sample_omega_from_prior
        from selfisys.global_parameters import planck_mean, planck_cov

        ids = list(range(len(planck_mean)))
        return sample_omega_from_prior(1, planck_mean, planck_cov, ids, seed=seed)[0]

    def _aux_blackbox(
        self,
        d,
        seedphase,
        seednoise,
        fname_powerspectrum,
        fname_simparfile,
        fname_whitenoise,
        seedname_whitenoise,
        fname_outputinitialdensity,
        fnames_outputrealspacedensity,
        fnames_outputdensity,
        fname_simlogs,
        fname_mocks,
        force_parfiles=False,
        force_sim=False,
        force_phase=False,
        return_g=False,
        fname_g_obs=None,
        check_output=False,
        RSDs=True,
    ):
        """
        Generates observations from an input primordial matter power spectrum after
        inflation, and return the estimated power spectrum of the galaxy count field.

        Parameters
        ----------
        d : int
            index giving the direction in parameter space: -1 for mock data, 0 for the
            expansion point, or from 1 to S for the gradient directions
        seedphase : int or list of int
            user-provided seed to generate "initial" white noise in Fourier space
        seednoise : int or list of int
            user-provided seed to generate the observational noise realization
        force_parfiles : bool, optional, default=False
            force recomputation of the parameter files?
        force_sim : bool, optional, default=False
            force recomputation of the simulation?
        force_phase : bool, optional, default=False
            force recomputation of the phase?
        return_g : bool, optional, default=False
            if True, returns the full realization in physical space. For dbg purposes.
        check_output : bool, optional, default=True
            check the integrity of the output file and recomputes it if corrupted
        RSDs : bool, optional, default=True
            if True/False, use the redshift/real -space density fields
        # > For the other parameters, please refer to the corresponding methods.

        Returns
        -------
        result: tuple
            tuple containing the following elements:
            - Phi : array, double, dimension = P * N_pop
                vector of summary statistics (concatenation summaries for each pop)
            - g : array, double, dimension=[(size,size,size)]*N_pop
                  list of observel fields if return_g=True, None otherwise

        """
        current_samples = self._sample_omega(seedphase)

        self._setup_parfiles(
            d,
            current_samples,
            fname_simparfile,
            fname_powerspectrum,
            fname_whitenoise,
            fname_outputinitialdensity,
            fnames_outputrealspacedensity,
            fnames_outputdensity,
            force_parfiles,
        )
        self._generate_white_noise_Field(
            seedphase, fname_whitenoise, seedname_whitenoise, force_phase
        )

        if not self.setup_only:
            self._run_sim(
                fname_simparfile,
                fname_simlogs,
                fnames_outputdensity,
                force_sim,
                check_output,
            )
            fnames = fnames_outputdensity if RSDs else fnames_outputrealspacedensity
            result = self._compute_mocks(
                seednoise, fnames, fname_mocks, return_g, fname_g_obs
            )
        else:
            result = [], []

        return result

    def _PrintMessage(self, required_verbosity, message, verbosity=None):
        """Print a message to standard output using PrintMessage from pyselfi.utils."""
        from pyselfi.utils import PrintMessage

        if verbosity is None:
            verbosity = self.verbosity
        if verbosity >= required_verbosity:
            PrintMessage(3, message)

    def _indent(self):
        """Indents the standard output using INDENT from pyselfi.utils."""
        from pyselfi.utils import INDENT

        INDENT()

    def _unindent(self):
        """Unindents the standard output using UNINDENT from pyselfi.utils."""
        from pyselfi.utils import UNINDENT

        UNINDENT()

    def make_data(
        self,
        cosmo,
        id,
        seedphase,
        seednoise,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        force_cosmo=False,
        force_phase=False,
        d=-1,
        remove_sbmy=False,
        verbosity=None,
        return_g=False,
        RSDs=True,
    ):
        """
        Parameters
        ----------
        cosmo : dictionary
            cosmological parameters (and some infrastructure parameters)
        id : int or string, optional, default=0
            identifier of the realization. It is used as a suffix in every filename
        seedphase : int or list of int
            user-provided seed to generate the "initial" white noise in Fourier space
        seednoise : int or list of int
            user-provided seed to generate the observational noise realization
        force_powerspectrum : bool, optional, default=False
            force recomputation of the input power spectrum?
        force_parfiles : bool, optional, default=False
            force recomputation of the parameter files?
        force_sim : bool, optional, default=False
            force recomputation of the simulation?
        force_cosmo : bool, optional, default=False
            force recomputation of the cosmological parameters?
        force_phase : bool, optional, default=False
            force recomputation of the phase?
        d : int, optional, default=-1
        remove_sbmy : bool, optional, default=False
            remove the Simbelynë files after the simulation? WARNING: use with caution
        verbosity : int, optional, default=None
            verbosity level. If None, the value of self.verbosity is used
        return_g : bool, optional, default=False
            if True, returns the full field in addition to the summary
        RSDs : bool, optional, default=True
            if True/False, use the redshift/real -space density fields
        """

        self._PrintMessage(0, "Making mock data...", verbosity)
        self._indent()

        datadir = self.fsimdir + "/data/"
        fname_cosmo = datadir + "input_cosmo_" + str(id) + ".json"
        # TODO: fname_powerspectrum should depend on the cosmological parameters instead
        #       of the id, so that it does not need to be recomputed for a given omega.
        fname_powerspectrum = datadir + "input_power_" + str(id) + ".h5"
        fname_simparfile = datadir + "sim_" + str(id)  # + ".sbmy"
        fname_whitenoise = datadir + "initial_density_white_noise_" + str(id) + ".h5"
        seedname_whitenoise = datadir + "initial_density_wn_" + str(id) + "_seed"
        fname_outputinitialdensity = datadir + "initial_density_" + str(id) + ".h5"
        if self.sim_params[:5] == "split":
            fnames_outputdensity = [
                datadir + "output_density_" + str(id) + "_{}.h5".format(i)
                for i in self.TimeSteps
            ]
            fnames_outputrealspacedensity = [
                datadir + "output_realdensity_" + str(id) + "_{}.h5".format(i)
                for i in self.TimeSteps
            ]
        elif self.sim_params[:6] == "custom":
            fnames_outputdensity = [
                datadir + "output_density_" + str(id) + "_{}.h5".format(i)
                for i in self.TimeSteps
            ]
            fnames_outputrealspacedensity = [
                datadir + "output_realdensity_" + str(id) + ".h5"
            ]
        else:
            fnames_outputdensity = [datadir + "output_density_" + str(id) + ".h5"]
            fnames_outputrealspacedensity = [
                datadir + "output_realdensity_" + str(id) + ".h5"
            ]
        fname_g_obs = datadir + "gobs_" + str(id) + ".h5" if return_g else None
        fname_simlogs = datadir + "logs_sim_" + str(id) + ".txt"
        if self._prefix_mocks_mis is None:
            fname_mocks = datadir + "mocks_" + str(id) + ".h5"
        else:
            fname_mocks = datadir + self._prefix_mocks_mis + "_mocks_" + str(id) + ".h5"

        # Save cosmological parameters
        self._save_cosmo(cosmo, fname_cosmo, force_cosmo)

        # Generate input initial matter power spectrum
        self._PrintMessage(1, "Computing initial power spectrum...", verbosity)
        self._get_powerspectrum_from_cosmo(
            cosmo, fname_powerspectrum, force_powerspectrum
        )
        self._PrintMessage(1, "Computing initial power spectrum done.", verbosity)

        # Generate the mock data
        self._PrintMessage(
            1,
            "Generating a noisy realization of the input power spectrum...",
            verbosity,
        )
        Phi, g = self._aux_blackbox(
            d,
            seedphase,
            seednoise,
            fname_powerspectrum,
            fname_simparfile,
            fname_whitenoise,
            seedname_whitenoise,
            fname_outputinitialdensity,
            fnames_outputrealspacedensity,
            fnames_outputdensity,
            fname_simlogs,
            fname_mocks,
            force_parfiles=force_parfiles,
            force_sim=force_sim,
            force_phase=force_phase,
            return_g=return_g,
            fname_g_obs=fname_g_obs,
            RSDs=RSDs,
        )
        self._PrintMessage(
            1,
            "Generating a noisy realization of the input power spectrum done.",
            verbosity,
        )

        self._clean_output(fname_outputinitialdensity)
        if remove_sbmy:
            for fname in fnames_outputrealspacedensity:
                self._clean_output(fname)
            for fname in fnames_outputdensity:
                self._clean_output(fname)
            self._clean_output(fname_powerspectrum)
            self._clean_output(fname_whitenoise)

        self._unindent()
        self._PrintMessage(1, "Making mock data done.", verbosity)

        if return_g:
            return Phi, g
        else:
            return Phi

    def _worker_normalization(self, params):
        """Worker function to compute the normalization constants w/ multiprocessing."""
        cosmo, seedphase, seednoise, force = params
        name = (
            "norm"
            + "__"
            + "_".join([str(int(s)) for s in seedphase])
            + "__"
            + "_".join([str(int(s)) for s in seednoise])
        )

        if self.verbosity > 1:
            self._PrintMessage(1, "Running simulation...")
            self._indent()
            phi = self.make_data(
                cosmo, name, seedphase, seednoise, force, force, force, force
            )
            self._unindent()
        else:
            from selfisys.utils.low_level import stdout_redirector
            from io import BytesIO

            f = BytesIO()
            with stdout_redirector(f):
                phi = self.make_data(
                    cosmo, name, seedphase, seednoise, force, force, force, force
                )
            f.close()

        return phi

    def worker_normalization_public(self, cosmo, N, i):
        """Run the i-th simulation required to compute the normalization constants.
        This method can be substituted to the direct use of `define_normalization`
        e.g. if one wants to use multiprocessing or to first create the required
        Simbelmynë parameter files before actually running the simulations.

        Parameters
        ----------
        cosmo : dictionary
            cosmological+infrastructure parameters
        N : int
            number of realizations required
        i : int
            index of the simulation to be computed
        """

        params = (
            cosmo,
            [i, self.__global_seednorm],
            [i + N, self.__global_seednorm],
            False,
        )
        self._worker_normalization(params)

    @property
    def Npop(self):
        """Number of populations."""
        return self._Npop

    def define_normalization(
        self, Pbins, cosmo, N, min_k_norma=4e-2, npar=1, force=False
    ):
        """Defines the normalization constants of the blackbox.

        Parameters
        ----------
        Pbins : array, double, dimension=P
        cosmo : dictionary
            cosmological+infrastructure parameters
        N : int
            number of realizations required
        min_k_norma : float, optional, default=4e-2
            minimum k value taken into account to compute the normalization constants
        npar : int, optional, default=1
            number of parallel processes to use
        force : bool, optional, default=False
            force recomputation of the normalization constants?
        """
        from os import cpu_count
        from numpy import where, zeros, array, mean
        import tqdm.auto as tqdm
        from multiprocessing import Pool

        self._PrintMessage(0, "Defining normalization constants...")
        self._indent()
        indices = where(Pbins > min_k_norma)
        list = [
            (cosmo, [i, self.__global_seednorm], [i + N, self.__global_seednorm], force)
            for i in range(0, N)
        ]

        ncors = cpu_count()
        nprocs = min(npar, ncors)

        norm_csts_list = zeros((self.Npop, N))
        if npar > 1:
            with Pool(nprocs) as p:
                for j, val in enumerate(
                    tqdm.tqdm(p.imap(self._worker_normalization, list), total=N)
                ):
                    norm_csts_list[:, j] = array(
                        [
                            mean(
                                val[i * self._Psingle : (i + 1) * self._Psingle][
                                    indices
                                ]
                            )
                            for i in range(self.Npop)
                        ]
                    )
        else:
            for j, val in enumerate(
                tqdm.tqdm(map(self._worker_normalization, list), total=N)
            ):
                norm_csts_list[:, j] = array(
                    [
                        mean(val[i * self._Psingle : (i + 1) * self._Psingle][indices])
                        for i in range(self.Npop)
                    ]
                )
        norm_csts = mean(norm_csts_list, axis=1)
        self._unindent()
        self._PrintMessage(0, "Defining normalization constants done.")

        return norm_csts

    def _powerspectrum_from_theta(
        self, theta, fname_powerspectrum, thetaIsP=False, force=False
    ):
        """Get the values of a given power spectrum by using a spline interpolation.

        Parameters
        ----------
        theta : array, double, dimension=S
            vector of power spectrum values at the support wavenumbers
        fname_powerspectrum : str
            name of input/output power spectrum file
        thetaIsP : bool, optional, default=False
            if True, theta is already an unnormalized power spectrum
        force : bool, optional, default=False
            force recomputation?

        """
        from os.path import exists
        from pysbmy.power import PowerSpectrum

        if (not exists(fname_powerspectrum)) or force:
            from gc import collect
            from scipy.interpolate import InterpolatedUnivariateSpline
            from pysbmy.power import FourierGrid

            PP = theta if thetaIsP else self.theta2P(theta)
            Spline = InterpolatedUnivariateSpline(self.k_s, PP, k=5)
            G_sim = FourierGrid.read(self.G_sim_path)
            powerspectrum = Spline(G_sim.k_modes)
            powerspectrum[0] = 0.0
            P = PowerSpectrum.from_FourierGrid(G_sim, powerspectrum=powerspectrum)
            P.write(fname_powerspectrum)
            del G_sim
            collect()

    def _clean_output(self, fname_output):
        """Remove a file from disk.

        Parameters
        ----------
        fname_output : str
            name of the file to remove

        """
        from os import path, remove

        if path.exists(fname_output):
            remove(fname_output)

    def evaluate(
        self,
        theta,
        d,
        seedphase,
        seednoise,
        i=0,
        N=0,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        remove_sbmy=False,
        thetaIsP=False,
        simspath=None,
        check_output=False,
        RSDs=True,
    ):
        """Evaluates the blackbox for a given input power spectrum.
        The result is determinstic (the phase is fixed), except as it is modified by the
        nuisance if any.

        Parameters
        ----------
        theta : array, double, dimension=S
            vector of power spectrum values at the support wavenumbers
        d : int
            direction in parameter space, from 0 to S /!\ Unused but andatory for SELFI.
        seedphase : int or list of int
            user-provided seed to generate the "initial" white noise in Fourier space
        seednoise : int list of int
            user-provided seed to generate the gaussian noise realization
        i : int, optional, default=0
            current evaluation index of the blackbox
        N : int, optional, default=0
            total number of evaluations of the blackbox
        force_powerspectrum : bool, optional, default=False
            force recomputation of the power spectrum?
        force_parfiles : bool, optional, default=False
            overwrite if parameter files already exists?
        force_sim : bool, optional, default=False
            force recomputation if output density already exists?
        remove_sbmy : bool, optional, default=False
            remove Simbelynë output files from disk? (to save disk space)
            WARNING: if you want to keep the fields, eg to try different selection
            functions, make sure to leave remove_sbmy=False in the first place!
        thetaIsP : bool, optional, default=False
            if True, the input vector must be the unnormalized power spectrum. This
            argument is used for ABC, and may be used for debugging purposes as well.
        simspath : str, optional, default=None
            path to the directory containing the simulations
        check_output : bool, optional, default=True
            check the integrity of the output file and recomputes it if corrupted
        RSDs : bool, optional, default=True
            if True/False, use the redshift/real -space density fields

        Returns
        -------
        Phi : array, double, dimension=P
            vector of summary statistics

        """
        if not thetaIsP:
            self._PrintMessage(
                1,
                "Direction {}. Evaluating blackbox (index {}/{})...".format(
                    d, i, N - 1
                ),
            )
            self._indent()

        simdir = simspath if simspath is not None else self.fsimdir
        simdir_d = simdir + "/pool/d" + str(d) + "/"
        fname_powerspectrum = simdir_d + "input_power_d" + str(d) + ".h5"
        fname_simparfile = simdir_d + "sim_d" + str(d) + "_p" + str(i)
        # Important note: For each i, a single white noise and initial density files are
        # used for all directions d (including d=0).
        # Therefore, to avoid conflicts when using multiprocessing, one shall start the
        # computations for direction d=0 first, and only then for the other d, while
        # ensuring that the former are advancing at least as fast as the latter.
        fname_whitenoise = simdir + "/../wn/initial_density_white_p" + str(i) + ".h5"
        seedname_whitenoise = simdir + "/../wn/initial_density_white_p" + str(i)
        fname_outputinitialdensity = simdir + "/initial_density_p" + str(i) + ".h5"
        if self.sim_params[:5] == "split":
            fnames_outputdensity = [
                simdir_d + "output_density_d{}_p{}_{}.h5".format(d, i, j)
                for j in self.TimeSteps
            ]
            fnames_outputrealspacedensity = [
                simdir_d + "output_realdensity_d{}_p{}_{}.h5".format(d, i, j)
                for j in self.TimeStep
            ]
        elif self.sim_params[:6] == "custom":
            fnames_outputdensity = [
                simdir_d + "output_density_d{}_p{}_{}.h5".format(d, i, j)
                for j in self.TimeSteps
            ]
            fnames_outputrealspacedensity = [
                simdir_d + "output_realdensity_d{}_p{}_{}.h5".format(d, i, j)
                for j in self.TimeSteps
            ]
        else:
            fnames_outputdensity = [
                simdir_d + "output_density_d{0}_p{1}.h5".format(d, i)
            ]
            fnames_outputrealspacedensity = [
                simdir_d + "output_realdensity_d{0}_p{1}.h5".format(d, i)
            ]
        fname_simlogs = simdir_d + "logs_sim_d" + str(d) + "_p" + str(i) + ".txt"
        if self._prefix_mocks_mis is None:
            fname_mocks = simdir_d + "mocks_d" + str(d) + "_p" + str(i) + ".h5"
        else:
            fname_mocks = (
                simdir_d
                + self._prefix_mocks_mis
                + "_mocks_d"
                + str(d)
                + "_p"
                + str(i)
                + ".h5"
            )

        self._PrintMessage(1, "Interpolating power spectrum over the Fourier grid...")
        self._powerspectrum_from_theta(
            theta, fname_powerspectrum, thetaIsP, force=force_powerspectrum
        )
        self._PrintMessage(
            1, "Interpolating power spectrum over the Fourier grid done."
        )

        self._PrintMessage(
            1, "Generating an observed summary from the input power spectrum..."
        )
        Phi, _ = self._aux_blackbox(
            d,
            seedphase,
            seednoise,
            fname_powerspectrum,
            fname_simparfile,
            fname_whitenoise,
            seedname_whitenoise,
            fname_outputinitialdensity,
            fnames_outputrealspacedensity,
            fnames_outputdensity,
            fname_simlogs,
            fname_mocks,
            force_parfiles,
            force_sim,
            force_phase=False,
            return_g=False,
            fname_g_obs=None,
            check_output=check_output,
            RSDs=RSDs,
        )
        self._PrintMessage(
            1, "Generating an observed summary from the input power spectrum done."
        )

        self._clean_output(fname_outputinitialdensity)
        if remove_sbmy:
            for fname in fnames_outputrealspacedensity:
                self._clean_output(fname)
            for fname in fnames_outputdensity:
                self._clean_output(fname)
            self._clean_output(fname_powerspectrum)
            self._clean_output(fname_whitenoise)

        if not thetaIsP:
            self._unindent()
            self._PrintMessage(
                1,
                "Direction {}. Evaluating blackbox (index {}/{}) done.".format(
                    d, i, N - 1
                ),
            )

        return Phi

    def switch_recompute_pool(self, prefix_mocks_mis=None):
        """Forces recomputing the pool for future calls of the compute_pool method.

        Parameters
        ----------
        file_prefix : str, optional, default=None
            prefix for the new mock files.
        """
        self._force_recompute_mocks = not self.force_recompute_mocks
        self._prefix_mocks_mis = (
            prefix_mocks_mis if prefix_mocks_mis is not None else None
        )

    def switch_setup(self):
        self._setup_only = not self.setup_only

    @property
    def __global_seedphase(self):
        return self.seedphase

    @property
    def __global_seednoise(self):
        return self.seednoise

    @property
    def __global_seednorm(self):
        return self.seednorm

    def _get_current_seed(self, parent_seed, fixed_seed, i):
        """Returns the current seed for the i-th realization.

        Do NOT change this method. It ensures safe enough seeding for our purposes.
        """
        if fixed_seed:
            this_seed = parent_seed
        else:
            this_seed = [i, parent_seed]
        return this_seed

    def compute_pool(
        self,
        theta,
        d,
        pool_fname,
        N,
        index=None,
        force_powerspectrum=False,
        force_parfiles=False,
        force_sim=False,
        remove_sbmy=False,
        thetaIsP=False,
        simspath=None,
        bar=False,
    ):
        """Computes a pool of realizations of the blackbox.

        Parameters
        ----------
        # > Mandatory:
        theta : array, double, dimension=S
            vector of power spectrum values at the support wavenumbers
        d : int
            direction in parameter space, from 0 to S
        pool_fname : str
            pool file name
        N : int
            number of realizations required

        # > Optional:
        index : int, optional, default=class value
            index of a single simulation to run or load. If set to None, all simulations
            are run or loaded. This option is useful for simple use of multiprocessing
            when Ns is much smaller than Ne.
        force_powerspectrum : bool, optional, default=False
            force recomputation of the power spectrum?
        force_parfiles : bool, optional, default=False
            overwrite if parameter files already exists?
        force_sim : bool, optional, default=False
            force recomputation if output density already exists?
        remove_sbmy : bool, optional, default=False
            remove Simbelynë output files from disk? (to save disk space)
            WARNING: if you want to keep the fields, eg to try different selection
            functions, make sure to leave remove_sbmy=False in the first place!
        thetaIsP : bool, optional, default=False
            if True, the input vector must be the unnormalized power spectrum
        simspath : str, optional, default=None
            path where the simulations will be stored. Shall be None for use with the
            SELFI pipeline (-> stored in self.fsimdir). For any other use, it should be
            set to a well chosen path as to avoid the simulations being reused at each
            call of the compute_pool method.
        bar : bool, optional, default=False
            if True, displays a progress bar

        Returns
        -------
        p : :obj:`pool`
            simulation pool

        """
        import tqdm.auto as tqdm
        from pyselfi.pool import pool

        self._PrintMessage(1, "Computing a pool of realizations of the blackbox...")

        pool_fname = str(pool_fname)
        if self.force_recompute_mocks:
            from os import path, remove

            if path.exists(pool_fname):
                remove(pool_fname)

        p = pool(pool_fname, N, retro=False)
        indices = list(range(N)) if index is None else [index]

        def worker(i):
            this_seedphase = self._get_current_seed(self.__global_seedphase, False, i)
            this_seednoise = self._get_current_seed(
                self.__global_seednoise, self.fixnoise, i
            )
            Phi = self.evaluate(
                theta,
                d,
                this_seedphase,
                this_seednoise,
                i=i,
                N=N,
                force_powerspectrum=force_powerspectrum,
                force_parfiles=force_parfiles,
                force_sim=force_sim,
                remove_sbmy=remove_sbmy,
                thetaIsP=thetaIsP,
                simspath=simspath,
            )
            p.add_sim(Phi, i)

        if bar:
            for i in tqdm.tqdm(indices, desc="Direction {}/{}".format(d, self.S)):
                worker(i)
        else:
            for i in indices:
                worker(i)

        if index is None:
            p.load_sims()
            p.save_all()
        return p

    def load_pool(self, pool_fname, N):
        """Loads a pool of realizations of the blackbox."""
        from pyselfi.pool import pool

        pool_fname = str(pool_fname)
        p = pool(pool_fname, N, retro=False)
        p.load_sims()
        p.save_all()
        return p
