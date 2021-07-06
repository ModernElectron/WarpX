"""Framework that sets up a 1D or 2D run based on input parameters, with
defaults available.
"""
import sys

import numpy as np

# [[[TODO]]] These imports should mostly be removed, but if they can be changed
# to current functionality that would be great.
from metools import cs_ionization, diags, emission, gentools
from metools import init_restart_util, mcc, merging
from metools import particlescraper1d, poisson1d, reflection, resultsinfo
from metools import runinfo, runtools, util, warputil
from metools.warputil import e


# [[[TODO]]] I would keep the overall structure here, but raise
# NotImplementedError("Not yet implemented in mewarpx") for functions that
# can't be run yet.
class DiodeRun_V1(object):

    """A combination of settings and initialization functions to standardize
    unit tests (hopefully).

    Explicitly versioned with V1 so that if API is changed this can be updated,
    but if "Best practices" change in a way that would change results, a new
    version can be created while existing unit tests use this.
    """

    # ### ELECTRODES ###
    # Cathode temperature in K
    CATHODE_TEMP = 1550
    # Richardson constant in A/m^2/K^2
    CATHODE_A = 120.17e4
    # Work function in eV.
    CATHODE_PHI = 2.4
    # Anode temperature in K
    ANODE_TEMP = 773
    # Work Function of Anode.
    ANODE_PHI = 1.4
    # Anode vacuum bias relative to cathode
    V_ANODE_CATHODE = 2.0

    # ### VACUUM ###
    # Cathode anode distance
    D_CA = 1e-3

    # Gas pressure
    P_AR = 4.0  # in Torr
    # Determine neutral temperature and density
    T_AR = 0.5 * (CATHODE_TEMP + ANODE_TEMP)  # rough approximation

    # [[[TODO]]] Do go ahead and remove all Cs-ionization-related
    # functionality.
    # ### CS IONIZATION ###
    P_CS = 1.0  # in Torr
    T_CS = 0.5 * (CATHODE_TEMP + ANODE_TEMP)  # rough approximation
    SUBCYCLE_CS = 20
    # Number of bins in Z
    ZBINS_CS = 30
    # Number of bins in X
    XBINS_CS = 3
    # Number of Cs particles to inject per timestep
    NPARTPERSTEP_CS = 100
    # Number of particles to sample from for Cs injection
    SAMPLES_CS = 1000

    # ### PHYSICS MODEL SETTINGS ###
    # Beam velocity spread in transverse vs longitudinal
    TRANSVERSE_FAC = 1.1

    # Reflection physics
    #   Type of reflection (specular or diffuse), or None to use a template.
    REFLECTION_TYPE = None
    #   Type of scattering model (pure/uniform/cos for specular or
    #   elastic/partialelastic/inelastic for diffuse). If using a reflection
    #   template, set to None, or use to override the default diffuse low energy
    #   choice.
    SCATTERING_MODEL = None
    #   Probability of specular reflection from cathode/anode. Set to None if
    #   using a prebuilt model. Or if not None and using a prebuilt model, it is
    #   transformed into a scale factor by approximating the probability of
    #   reflection at 0.5 eV.
    REFLECTION_PROB = 0.4

    # For templates only:
    #   Additional scaling used on the diffuse component vs the specular
    #   component.
    DIFFUSE_FACTOR = 1.0
    #   Energy at which specular reflection starts decreasing, leaving only
    #   diffuse reflection. Set to > 20 to use only specular at low energies.
    DIFFUSE_START = 1.0
    #   Energy at which specular reflection goes to 0, leaving only diffuse
    #   reflection. Set to < 0 to use only diffuse at low energies. Must be
    #   greater than DIFFUSE_START.
    SPECULAR_CUTOFF = 3.0
    #   Loss fraction for partial elastic reflection at low energies. Set to 0
    #   if using standard elastic reflection.
    LOW_E_LOSS_FRACTION = 0.

    # ### INJECTION SETTINGS ###
    # Number of particles to inject TOTAL per timestep
    NPARTPERSTEP = 10
    # Whether to use Schottky injection
    USE_SCHOTTKY = True
    # 2D only: Normal thermionic injection - number of particles to inject per
    # cell per timestep
    NPPC = 2
    # 2D only: Normal thermionic injection - specify PERIOD instead to ignore
    # NPARTPERSTEP and force this period
    PERIOD = None
    # Noninteracting run - only inject trace particles and do field solve once
    NONINTERAC = False
    # Number of particles injected each noninteracting wave
    NONINTERAC_WAVENUM = 50000

    # ### RUN SETTINGS ###
    # Integer number multiplied by crossing time to give total running time
    TOTAL_CROSSINGS = 8.0
    # Integer number multiplied by crossing time to give time between each
    # diagnostic output (and check for early run termination)
    DIAG_CROSSINGS = 2.0
    # P_CUTOFF is the value at which the simulation is terminated at a
    # diagnostic interval, if power output is below the cutoff
    P_CUTOFF = -1e6
    # J_TOLERANCE is the percentage change allowed in currents between
    # diag_steps results for which the run will terminate rather than continue
    # (ie the acceptable percentage deviation for convergence). Set to None to
    # disallow.
    J_TOLERANCE = None
    # J_TOTAL_TOLERANCE disallows termination due to J convergence if there's
    # still a net current (eg emitted currents != collected currents) above a
    # given threshold. Presently defaults to 1e-3 if not specified here.
    J_TOTAL_TOLERANCE = None
    # CFL_FACTOR determines the time step at which 99% of particles should
    # satisfy the CFL condition. 1.0 would be typical; usually we use 0.4 for a
    # safety margin
    CFL_FACTOR = 0.4
    # RES_LENGTH is the cell size in meters
    RES_LENGTH = 0.8e-06
    # Check Debye Length allows disabling this check to run with large
    # resolutions
    CHECK_DEBYE_LENGTH = True
    # OFFSET pads the cathode and anode by extending the simulation limits.
    OFFSET = 2.5 * RES_LENGTH
    # ANODE_OFFSET shifts the anode away from its standard position. If it's
    # normally right on a cell boundary, this may want to be 1e-10 rather than
    # 0. However, setting OFFSET to a non-multiple of RES_LENGTH should mitigate
    # most of this concern in standard runs.
    ANODE_OFFSET = 0.
    # NSCALE sets how fine the grid used for scraping particles is relative to
    # the standard field-solve grid. Optimal is uncertain right now (unlikely to
    # matter much for diode in any case).
    NSCALE = 1
    # Use the 2D/3D scraper in 1D, for testing purposes or particle histograms
    # (not currently implemented in 1D scraper) only
    FORCE_2D_SCRAPER = False
    # DT allows the auto-calculated DT to be overridden. CFL_FACTOR is ignored
    # in this case.
    DT = None
    # CHECK_CHARGE_CONSERVATION is a heuristic only for detecting (usually)
    # violations of CFL. However, it can also be triggered during short
    # diagnostic periods that cover transient behavior, and should thus often be
    # disabled in short unit tests.
    CHECK_CHARGE_CONSERVATION = True

    # MERGING_DV controls dv bin width
    MERGING_DV = 1e5
    # MERGING_PERPERIOD controls number of merges per diag_steps
    MERGING_PERPERIOD = 5
    # MERGING_DXFAC controls how many cells are in a merging bin
    MERGING_DXFAC = 1.
    # MERGING_XYFAC controls XY binning vs Z binning
    MERGING_XYFAC = 10.

    # Number of trace particles injected
    NTRACE = 400
    # Record timings and print timing diagnostics
    PROFILE_DIAG = False

    # Parallelization options - for testing old parallelization types
    # Are particles kept through the whole domain?
    PART_FULL_DECOMP = True
    # Is decomposition of the field solver skipped?
    NO_FS_DECOMP = True

    def __init__(self, dim=1, rz=False, **kwargs):
        for kw in list(kwargs.keys()):
            setattr(self, kw, kwargs[kw])

        self.dim = dim
        if self.dim not in [1, 2, 3]:
            raise ValueError("Unavailable dimension {}".format(self.dim))
        self.rz = rz
        if self.dim != 2 and self.rz:
            raise ValueError("dim=2 required for RZ")

        if self.dim > 1:
            if self.PERIOD is None:
                self.PERIOD = (
                    np.ceil(float(self.NPARTPERSTEP)/self.NPPC)
                    * self.RES_LENGTH
                )

        # V_CATHODE is set so that its Fermi level is at V=0.
        self.V_CATHODE = -self.CATHODE_PHI
        self.V_ANODE = self.V_CATHODE + self.V_ANODE_CATHODE

    def setup_run(
        self,
        init_base=True,
        init_solver=True,
        init_conductors=True,
        init_scraper=True,
        init_injectors=True,
        init_reflection=False,
        init_cs_ionization=False,
        init_ar_ionization=False,
        init_merging=False,
        init_traceparticles=False,
        init_runinfo=False,
        init_fluxdiag=False,
        init_resultsinfo=False,
        generate=False
    ):
        """Perform each part of setup if requested.

        Ones marked True are generally necessary for a standard run. Reflection,
        ionization and generate are disabled by default, as is runinfo (so it
        can be tested separately).
        """
        if init_base:
            self.init_base()
        if init_solver:
            self.init_solver()
        if init_conductors:
            self.init_conductors()
        if init_scraper:
            self.init_scraper()
        if init_injectors:
            self.init_injectors()
        if init_cs_ionization:
            self.init_cs_ionization()
        if init_ar_ionization:
            self.init_ar_ionization()
        if init_reflection:
            self.init_reflection()
        if init_merging:
            self.init_merging()
        if init_traceparticles:
            self.init_traceparticles()
        if init_runinfo:
            self.init_runinfo()
        if init_fluxdiag:
            self.init_fluxdiag()
        if init_resultsinfo:
            self.init_resultsinfo()
        # [[[TODO]]] Probably change this to "init_warpx" both for the keyword,
        # and call the correct mwxrun initialize function.
        if generate:
            warputil.warp_generate()

    # [[[TODO]]] This part can be fully replaced with the WarpX version.
    def init_base(self):
        print('### Init Diode Base Setup ###')

        #######################################################################
        # Set up diags directory and clean old files                          #
        #######################################################################
        init_restart_util.init_run()

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        # Solver geometry
        if self.dim == 1:
            # Translational symmetry in x & y direction, 1D Poisson solver
            warp.w3d.solvergeom = warp.w3d.Zgeom
        elif self.dim == 2 and not self.rz:
            # Translational symmetry in y direction, 2D Poisson solver
            warp.w3d.solvergeom = warp.w3d.XZgeom
        elif self.rz:
            # 2D Poisson solver by using RZ for charge density & Poisson solver
            warp.w3d.solvergeom = warp.w3d.RZgeom
        elif self.dim == 3:
            # Fully 3D solver
            warp.w3d.solvergeom = warp.w3d.XYZgeom

        # Longitudinal conditions overriden by conducting plates
        warp.w3d.bound0 = warp.neumann
        warp.w3d.boundnz = warp.neumann
        warp.w3d.boundxy = warp.periodic
        # Particles boundary conditions
        warp.top.pbound0 = warp.absorb
        warp.top.pboundnz = warp.absorb

        if self.rz:
            warp.top.prwall = self.PERIOD/2.0
        else:
            warp.top.pboundxy = warp.periodic

        if self.dim < 3 and not self.rz:
            # Disable evolution of y for Z & XZ runs.
            warp.w3d.lpushy = False
        if self.dim == 1:
            # Disable evolution of x for 1D runs.
            warp.w3d.lpushx = False

        #######################################################################
        # Run dimensions                                                      #
        #######################################################################

        # Set grid boundaries
        if self.dim == 1:
            # Translational symmetry in x & y, 1D simulation in z. Note warp
            # immediately sets xmmin and ymmin to 0 when
            # warp.w3d.initdecompositionw3d() is called, so we work with that
            # offset here, though nothing should depend on it.
            warp.w3d.xmmin = 0
            warp.w3d.xmmax = 1.0
        else:
            warp.w3d.xmmin = -self.PERIOD/2.0
            warp.w3d.xmmax = self.PERIOD/2.0

        if self.dim < 3 and not self.rz:
            warp.w3d.ymmin = 0
            warp.w3d.ymmax = 1.0
        else:
            warp.w3d.ymmin = -self.PERIOD/2.0
            warp.w3d.ymmax = self.PERIOD/2.0

        warp.w3d.zmmin = -self.OFFSET
        warp.w3d.zmmax = self.D_CA + self.OFFSET

        # Grid parameters - set grid counts
        if self.dim == 1:
            warp.w3d.nx = 0
        else:
            warp.w3d.nx = int(round(self.PERIOD/self.RES_LENGTH))

        if self.dim < 3 and not self.rz:
            warp.w3d.ny = 0
        else:
            warp.w3d.ny = int(round(self.PERIOD/self.RES_LENGTH))

        warp.w3d.nz = int(round((warp.w3d.zmmax - warp.w3d.zmmin)
                                / self.RES_LENGTH))

        #######################################################################
        # Run setup calculations and print diagnostic info                    #
        #######################################################################

        # [[[TODO]]] Most of this can be commented out. For now diag_steps and
        # total_timesteps may need to be class input variables as other things
        # above are.

        # Must be run after warp boundaries (xmmin etc.) and cell counts (nx
        # etc.) are set.
        # Full decomp always used for 1D parallel runs
        # Particle decomposition - each processor keeps particles across full
        # domain
        # No field decomposition - each processor solves field itself
        self.setupinfo = gentools.SetupCalcs(
            cathode_temp=self.CATHODE_TEMP,
            cathode_phi=self.CATHODE_PHI,
            cathode_A=self.CATHODE_A,
            V_anode=self.V_ANODE_CATHODE,
            V_grid=self.V_ANODE_CATHODE,
            D_CG=self.D_CA,
            CFL_factor=self.CFL_FACTOR,
            transverse_fac=self.TRANSVERSE_FAC,
            diag_crossings=self.DIAG_CROSSINGS,
            total_crossings=self.TOTAL_CROSSINGS,
            check_debye_length=self.CHECK_DEBYE_LENGTH,
            dt=self.DT,
            part_full_decomp=self.PART_FULL_DECOMP,
            no_fs_decomp=self.NO_FS_DECOMP,
            use_B=False,
        )
        self.diag_steps = self.setupinfo.diag_steps
        self.total_timesteps = self.setupinfo.total_timesteps

    def init_solver(self):
        # [[[TODO]]] Init solver here should still make sense, with
        # WarpX/Directsolver options
        print('### Init Diode Solver Setup ###')
        if self.dim == 1:
            self.solver = poisson1d.PoissonSolver1D()
        elif self.dim == 2:
            self.solver = warp.MultiGrid2D()
            self.solver.mgverbose = 1 if self.NONINTERAC else -1
            self.solver.mgmaxiters = 10000
        elif self.dim == 3:
            self.solver = warp.MultiGrid3D()
            self.solver.mgverbose = 1 if self.NONINTERAC else -1
            self.solver.mgmaxiters = 10000

        warp.registersolver(self.solver)

    def init_conductors(self):
        # [[[TODO]]] Not implemented until thermionic emission task is merged
        # in
        print('### Init Diode Conductors Setup ###')
        # Create source conductors a.k.a the cathode
        self.cathode = warp.ZPlane(zcent=0., zsign=-1., voltage=self.V_CATHODE,
                                   name='cathode')
        self.cathode.WF = self.CATHODE_PHI
        self.cathode.temperature = self.CATHODE_TEMP
        warp.installconductor(self.cathode, dfill=warp.largepos)

        # Create ground plate a.k.a the anode
        # Subtract small z value for the anode plate not to land exactly on the
        # grid
        self.anode_plane = warp.ZPlane(voltage=self.V_ANODE,
                                       zcent=self.D_CA - self.ANODE_OFFSET,
                                       name='anode_plane')
        self.anode_plane.WF = self.ANODE_PHI
        self.anode_plane.temperature = self.ANODE_TEMP
        warp.installconductor(self.anode_plane, dfill=warp.largepos)

        self.surface_list = [self.cathode, self.anode_plane]

    def init_scraper(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode Scraper Setup ###')
        profile_decorator = None
        if self.PROFILE_DIAG:
            profile_decorator = util.get_runtime_profile_decorator()

        if self.dim == 1 and not self.FORCE_2D_SCRAPER:
            self.scraper = particlescraper1d.ParticleScraper1D(
                cathode=self.cathode, anode=self.anode_plane,
                profile_decorator=profile_decorator
            )

        else:
            # Small Aura required to make sure 1 cell (isnear) boundary is
            # computed properly for conductors that terminate on a grid line
            dmin = min(self.setupinfo.dx, self.setupinfo.dz)/100.
            # ltunnel to False because it requires a griddistance method that
            # ZSrfrv doesn't have
            self.scraper = warp.ParticleScraper(
                self.surface_list,
                lcollectlpdata=True,
                ltunnel=False,
                aura=dmin,
                nxscale=self.NSCALE,
                nyscale=self.NSCALE,
                profile_decorator=profile_decorator
            )

    def init_injectors(self):
        # [[[TODO]]] Not implemented until thermionic emission task is merged
        # in
        print('### Init Diode Injectors Setup ###')
        ####################################################################
        # Beam and injection setup, output RunInfo                         #
        ####################################################################

        # Setup emitter and injector.
        if self.rz:
            self.emitter = emission.ZDiscEmitter(
                conductor=self.cathode, T=self.CATHODE_TEMP,
                transverse_fac=self.TRANSVERSE_FAC,
                use_Schottky=self.USE_SCHOTTKY,
            )
        elif self.dim < 3:
            self.emitter = emission.ZPlaneEmitter(
                conductor=self.cathode, z=self.cathode.zcent,
                T=self.CATHODE_TEMP,
                ymin=0.0, ymax=0.0,
                transverse_fac=self.TRANSVERSE_FAC,
                use_Schottky=self.USE_SCHOTTKY,
            )
        else:
            self.emitter = emission.ZPlaneEmitter(
                conductor=self.cathode, z=self.cathode.zcent,
                T=self.CATHODE_TEMP,
                transverse_fac=self.TRANSVERSE_FAC,
                use_Schottky=self.USE_SCHOTTKY,
            )

        if self.NONINTERAC:
            runtools.set_noninteracting(solvefreq=self.diag_steps)
            weight = (gentools.J_RD(
                self.CATHODE_TEMP, self.CATHODE_PHI, self.CATHODE_A)
                    * self.emitter.area
                    * warp.top.dt * self.diag_steps
                    / e / self.NONINTERAC_WAVENUM)
            self.injector = emission.FixedNumberInjector(
                self.emitter, 'beam', npart=self.NONINTERAC_WAVENUM,
                injectfreq=self.diag_steps,
                weight=weight
            )
        else:
            if self.dim == 1:
                npart = self.NPARTPERSTEP
            else:
                npart = self.NPPC

            self.injector = emission.ThermionicInjector(
                self.emitter, 'beam', npart, self.CATHODE_TEMP,
                self.CATHODE_PHI, self.CATHODE_A
            )

    def init_cs_ionization(self):
        # [[[TODO]]] Remove entirely
        print('### Init Diode Cs Ionization Setup ###')

        # Set up ion and electron species
        self.cs_i = warp.Species(type=warp.Caesium, charge_state=1, name='cs_i')
        self.cs_i.sw = self.injector.species.sw

        # Now set up the Cs injector
        self.cs_injector = cs_ionization.CsMultistepIonizationInjector(
            self.injector.species, self.cs_i, self.P_CS, self.T_CS,
            subcycle=self.SUBCYCLE_CS,
            xbins=self.XBINS_CS, zbins=self.ZBINS_CS,
            npart_per_step=self.NPARTPERSTEP_CS,
            update_steps=None, nsteps_to_average=None
        )

    def init_ar_ionization(self):
        # [[[TODO]]] not implented for a while. Rename from "Ar" to "inert_gas"
        print('### Init Diode Ar Ionization Setup ###')

        # Set up ion and electron species
        self.ar_i = warp.Species(type=warp.Argon, charge_state=1, name='ar_i')
        self.ar_i.sw = self.injector.species.sw

        # Now set up the ion injector using MCC for ionization
        self.ion_injector = mcc.MCC(self.injector.species, self.ar_i,
                                    self.P_AR, self.T_AR)

    def init_reflection(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode Reflection Setup ###')
        exclude_particles = []

        if hasattr(self, 'ar_i'):
            exclude_particles.append(self.ar_i)
        if hasattr(self, 'cs_i'):
            exclude_particles.append(self.cs_i)
        # Set up reflection
        self.reflector = reflection.Reflection(
            scraper=self.scraper, shapelist=self.surface_list,
            exclude_particles=exclude_particles
        )
        #   Use constant probability reflection
        if self.REFLECTION_TYPE in ['specular', 'diffuse']:
            self.reflector.add_reflection_model(
                rprob=self.REFLECTION_PROB,
                reflection_type=self.REFLECTION_TYPE,
                scattering_model=self.SCATTERING_MODEL,
                loss_fraction=self.LOW_E_LOSS_FRACTION
            )

        # Use template-based reflection
        else:
            # Guessing at scale factor - if REFLECTION_PROB is specified, use
            # what is appropriate for that REFLECTION_PROB in specular-only at
            # 0.5 V
            if self.REFLECTION_PROB is None:
                scale_factor = 1.
            else:
                scale_factor = self.REFLECTION_PROB/0.58368848644593
            diffuse_scale_factor = scale_factor*self.DIFFUSE_FACTOR

            self.reflector.add_reflection_model(
                rtemplate='Spec_WCBa_20181001',
                scale=scale_factor,
                turnoff_e=(self.DIFFUSE_START, self.SPECULAR_CUTOFF),
            )
            self.reflector.add_reflection_model(
                rtemplate='Diffuse_WCBa_20181001',
                scale=diffuse_scale_factor,
                turnon_e=(self.DIFFUSE_START, self.SPECULAR_CUTOFF),
                scattering_model=self.SCATTERING_MODEL,
                loss_fraction=self.LOW_E_LOSS_FRACTION,
            )
            self.reflector.add_reflection_model(
                rtemplate='DiffusePartial_HighE_WCBa_20181001',
            )

    def init_merging(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode Merging ###')
        self.merger = merging.Merger(
            species=self.injector.species,
            period=self.diag_steps // self.MERGING_PERPERIOD,
            dv=self.MERGING_DV,
            dxfac=self.MERGING_DXFAC,
            xfac=self.MERGING_XYFAC,
            yfac=self.MERGING_XYFAC
        )

    def init_traceparticles(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode TraceParticles ###')
        self.trace_species = diags.TraceSpecies(
            numsteps=self.diag_steps-1, save_stride=2, write=True,
            name='trace', begin_step=1, type=warp.Electron,
        )
        self.trace_species.sw = 0.
        self.trace_injector = emission.FixedNumberInjector(
            emitter=self.emitter, speciesname=None, npart=self.NTRACE,
            injectfreq=None, species=self.trace_species
        )

    def init_runinfo(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode Runinfo Setup ###')
        # UNTESTED method of passing runvars at present!
        runvars = DiodeRun_V1.__dict__.copy()
        runvars.update(self.__dict__)

        injector_dict = {'cathode': self.injector}
        if hasattr(self, 'ion_injector'):
            injector_dict['ar_ionization'] = self.ion_injector
        if hasattr(self, 'cs_injector'):
            injector_dict['cs_ionization'] = self.cs_injector
        surface_dict = {'cathode': self.cathode, 'anode': self.anode_plane}

        # Output runinfo
        self.runinfo = runinfo.RunInfo(
            injector_dict=injector_dict,
            surface_dict=surface_dict,
            local_vars=runvars,
            run_file=__file__,
            run_param_dict=self.setupinfo.run_param_dict,
            electrode_params={"CATHODE_A": self.CATHODE_A,
                              "CATHODE_TEMP": self.CATHODE_TEMP,
                              "CATHODE_PHI": self.CATHODE_PHI,
                              "ANODE_PHI": self.ANODE_PHI},
        )
        self.runinfo.save()

    def init_fluxdiag(self):
        # [[[TODO]]] not implented for a while
        print('### Init Diode FluxDiag ###')
        self.fluxdiag = diags.FluxDiag(
            diag_steps=self.diag_steps,
            scraper=self.scraper,
            runinfo=self.runinfo,
            check_charge_conservation=self.CHECK_CHARGE_CONSERVATION,
            overwrite=False
        )

    def init_resultsinfo(self):
        # [[[TODO]]] This is where the stopping criteria defaults - at least
        # for total timesteps - should go. Most of the rest not implemented
        # yet.
        print('### Init Diode ResultsInfo ###')
        self.runresults = resultsinfo.ResultsInfo(
            fluxdiag=self.fluxdiag, diag_steps=self.diag_steps,
            noninterac=self.NONINTERAC, profile_diag=self.PROFILE_DIAG
        )
        self.runresults.setup_terminate(
            J_tolerance=self.J_TOLERANCE,
            J_total_tolerance=self.J_TOTAL_TOLERANCE,
            total_timesteps=self.total_timesteps,
            chargemon=None,
            P_cutoff=self.P_CUTOFF
        )
