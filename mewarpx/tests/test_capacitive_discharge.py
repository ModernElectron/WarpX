"""Test full 1D diode run with diagnostics."""
from __future__ import division

from builtins import range
import os
import sys
from picmistandard import base

import pytest
import numpy as np

from pywarpx import picmi
# import pywarpx
# import mewarpx

# Necessary to load Warp as library (because Warp processes arguments even when
# loaded as a module!!!)
# del sys.argv[1:]

# import warp
# from warp.data_dumping.openpmd_diag import ParticleDiagnostic

from mewarpx.setups_store import diode_setup
from mewarpx.mwxrun import mwxrun
##########################
# physics parameters
##########################

N_INERT = 9.64e20 # m^-3
T_INERT = 300.0 # K

FREQ = 13.56e6 # MHz

M_ION = 6.67e-27 # kg

PLASMA_DENSITY = 2.56e14 # m^-3
T_ELEC = 30000.0 # K

##########################
# numerics parameters
##########################

DT = 1.0 / (400 * FREQ)

# Total simulation time in seconds
TOTAL_TIME = 10.0 * DT # 1280 / FREQ
# Time (in seconds) between diagnostic evaluations
DIAG_INTERVAL = 2.0 * DT # 32 / FREQ

# --- Number of time steps
diag_steps = int(DIAG_INTERVAL / DT)
diagnostic_intervals = "::%i" % diag_steps #"%i:" % (max_steps - diag_steps + 1)


#mwxutil.init_libwarpx(ndim=2, rz=False)
@pytest.mark.parametrize(
    ("name"),
    [
        'Run1D',
        'Run1D_RZ',
        'Run2D'
        'Run2D_RZ',
    ]
)
def test_run1D_alldiags(capsys, name):
    basename = "Run"
    use_rz = 'RZ' in name
    dim = int(name.replace(basename, '')[0])
    # Include a random run number to allow parallel runs to not collide. Using
    # python randint prevents collisions due to numpy rseed below
    initialize_testingdir(name)

    # Initialize each run with consistent, randomly-chosen, rseed. Use a random
    # seed instead for initial dataframe generation.
    # np.random.seed()
    np.random.seed(92160881)

    # Histograms only work with 2D scraper at the moment so we test each
    # combination
    #use_2d_scraper = "2DScraper" in name
    constants = picmi.constants

    v_rms_elec = np.sqrt(constants.kb * T_ELEC / constants.m_e)
    v_rms_ion = np.sqrt(constants.kb * T_INERT / M_ION)

    uniform_plasma_elec = picmi.UniformDistribution(
        density = PLASMA_DENSITY,
        upper_bound = [None] * 3,
        rms_velocity = [v_rms_elec] * 3,
        directed_velocity = [0.] * 3
    )

    uniform_plasma_ion = picmi.UniformDistribution(
        density = PLASMA_DENSITY,
        upper_bound = [None] * 3,
        rms_velocity = [v_rms_ion] * 3,
        directed_velocity = [0.] * 3
    )

    electrons = picmi.Species(
        particle_type='electron', name='electrons',
        initial_distribution=uniform_plasma_elec
    )

    ions = picmi.Species(
        particle_type='He', name='he_ions',
        charge='q_e',
        initial_distribution=uniform_plasma_ion
    )

    # MCC collisions
    cross_sec_direc = '../../../warpx-data/MCC_cross_sections/He/'
    mcc_electrons = picmi.MCCCollisions(
        name='coll_elec',
        species=electrons,
        background_density=N_INERT,
        background_temperature=T_INERT,
        background_mass=ions.mass,
        scattering_processes={
            'elastic' : {
                'cross_section' : cross_sec_direc+'electron_scattering.dat'
            },
            'excitation1' : {
                'cross_section': cross_sec_direc+'excitation_1.dat',
                'energy' : 19.82
            },
            'excitation2' : {
                'cross_section': cross_sec_direc+'excitation_2.dat',
                'energy' : 20.61
            },
            'ionization' : {
                'cross_section' : cross_sec_direc+'ionization.dat',
                'energy' : 24.55,
                'species' : ions
            },
        }
    )

    mcc_ions = picmi.MCCCollisions(
        name='coll_ion',
        species=ions,
        background_density=N_INERT,
        background_temperature=T_INERT,
        scattering_processes={
            'elastic' : {
                'cross_section' : cross_sec_direc+'ion_scattering.dat'
            },
            'back' : {
                'cross_section' : cross_sec_direc+'ion_back_scatter.dat'
            },
            # 'charge_exchange' : {
            #    'cross_section' : cross_sec_direc+'charge_exchange.dat'
            # }
        }
    )

    # Specific numbers match older run for consistency
    run = diode_setup.DiodeRun_V1(
        dim=dim, rz=use_rz,
        CATHODE_TEMP=1473.15,
        CATHODE_A=3.0e5,
        CATHODE_PHI=2.11,
        ANODE_TEMP=400,
        ANODE_PHI=1.2,
        V_ANODE_CATHODE=1.21,
        D_CA=0.067,
        NPARTPERSTEP=200,
        TOTAL_CROSSINGS=2.0,
        DIAG_CROSSINGS=2.0,
        J_TOLERANCE=0.001,
        CFL_FACTOR=2.0,
        OFFSET=1e-06,
        MERGING=True,
        MERGING_DV=100000,
        MERGING_PERPERIOD=20,
        MERGING_DXFAC=2,
        MERGING_XYFAC=10,
        CHECK_CHARGE_CONSERVATION=False,
        NX=128,
        NZ=16,
        DT=DT,
        TOTAL_TIMESTEPS=int(TOTAL_TIME / DT),
        DIAG_STEPS=int(DIAG_INTERVAL / DT),
        DIAG_INTERVAL=DIAG_INTERVAL,
        SPECIES=[electrons, ions],
        NUMBER_PARTICLES_PER_CELL = [32, 16],
        FIELD_DIAG_DATA_LIST=['rho_electrons', 'rho_he_ions', 'phi']
    )
    # besides resultsinfo none of these are implemented in mewarpx
    run.setup_run(
        init_base=True,
        init_solver=True,
        init_simulation=True,
        init_conductors=False,
        init_scraper=False,
        init_injectors=False,
        init_reflection=False,
        init_inert_gas_ionization=False,
        init_merging=False,
        init_traceparticles=False,
        init_runinfo=False,
        init_fluxdiag=False,
        init_field_diag=True,
        init_resultsinfo=True,
        init_warpx=False
    )

    # Run the main WARP loop
    while run.control.check_criteria():
        run.sim.step()

    #######################################################################
    # Cleanup and final output                                            #
    #######################################################################

    #TODO: what can be kept here between these two lines?
    run.runresults.finalize_save()
    out, _ = capsys.readouterr()

    print(out)
    # make sure out isn't empty
    assert out

def initialize_testingdir(name):
    """Change into an appropriate directory for testing. This means placing it
    in util.temp_dir, and attaching "_XXXXXX", a random 6-digit integer, to the
    end of the path.
    Arguments:
        name (str): The base string used in the new directory.
    Returns:
        origwd, newdir (str, str): Original working directory, and current
        working directory.
    """
    wd = None
    if mwxrun.me == 0:
        # Use standard python random here, in case numpy rseed is being
        # used that might fix this randint call.
        wd = os.path.join('../tests/test_files', name + "_{:06d}".format(
            np.random.randint(0, 1000000)))

    #TODO: Use warpx functionality here instead.
    if warp.lparallel:
        wd = warp.mpibcast(wd)

    if wd is None:
        raise ValueError("Working directory not properly set or broadcast.")

    origwd = change_to_warpdir(wd)

    return origwd, os.getcwd()

def change_to_warpdir(wd):
    """Handle logic of moving to correct directory. The directory is created if
    needed.
    Arguments:
        wd (str): Path of desired working directory.
    Returns:
        origwd (str): Path of original working directory
    """
    origwd = os.getcwd()

    if mwxrun.me == 0:
        minutil.mkdir_p(wd)

    #TODO: Use warpx functionality here for these Barrier() calls instead.
    warp.comm_world.Barrier()

    os.chdir(wd)
    print("Change to working directory {}".format(os.getcwd()))

    warp.comm_world.Barrier()

    return origwd
