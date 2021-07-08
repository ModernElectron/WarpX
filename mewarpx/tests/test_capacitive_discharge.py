"""Test full 1D diode run with diagnostics."""
from __future__ import division

from builtins import range
# import os
# import sys

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

# from metools import analysis, diags
# from metools import init_restart_util, runtools, util, warputil
from mewarpx.setups_store import diode_setup


constants = picmi.constants

##########################
# physics parameters
##########################

N_INERT = 9.64e20 # m^-3
T_INERT = 300.0 # K

FREQ = 13.56e6 # MHz

VOLTAGE = 450.0

M_ION = 6.67e-27 # kg

PLASMA_DENSITY = 2.56e14 # m^-3
T_ELEC = 30000.0 # K

##########################
# numerics parameters
##########################

number_per_cell_each_dim = [32, 16]

DT = 1.0 / (400 * FREQ)

# Total simulation time in seconds
TOTAL_TIME = 10.0 * DT # 1280 / FREQ
# Time (in seconds) between diagnostic evaluations
DIAG_INTERVAL = 2.0 * DT # 32 / FREQ

# --- Number of time steps
diag_steps = int(DIAG_INTERVAL / DT)
diagnostic_intervals = "::%i" % diag_steps #"%i:" % (max_steps - diag_steps + 1)


# mwxutil.init_libwarpx(ndim=2, rz=False)
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
    # Include a random run number to allow parallel runs to not collide. Using
    # python randint prevents collisions due to numpy rseed below
    #initialize_testingdir(name)

    # Initialize each run with consistent, randomly-chosen, rseed. Use a random
    # seed instead for initial dataframe generation.
    # np.random.seed()
    np.random.seed(92160881)

    # Histograms only work with 2D scraper at the moment so we test each
    # combination
    #use_2d_scraper = "2DScraper" in name

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
        NUMBER_PARTICLES_PER_CELL = number_per_cell_each_dim,
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

    run.init_warpx()
    # if use_2d_scraper:
    #     runtools.PHistDiag(
    #         quantity_spec=[
    #             ('uxbirth', -run.setupinfo.vmax/4., run.setupinfo.vmax/4.),
    #             ('uzbirth', 0, run.setupinfo.vmax/4.),
    #         ],
    #         scraper=run.scraper,
    #         diag_steps=run.diag_steps,
    #         linres=30,
    #         jslist=0,
    #     )

    #     runtools.PHistDiag(
    #         quantity_spec=[
    #             ('zold', warp.w3d.zmmin, warp.w3d.zmmax),
    #         ],
    #         scraper=run.scraper,
    #         diag_steps=run.diag_steps,
    #         linres=50,
    #         jslist=0,
    #         name='particle_histogramcurrent'
    #     )

    # warputil.warp_generate()

    # # Run the main WARP loop
    # while not run.runresults.terminate_flag:
    #     warp.step(run.diag_steps)

    #######################################################################
    # Cleanup and final output                                            #
    #######################################################################

    # run.runresults.finalize_save()

    # out, _ = capsys.readouterr()

    # filelist = [
    #     "diags/fields/Barrier_index_0000000000.pdf",
    #     "diags/fields/Barrier_index_0000000000.png",
    #     "diags/fields/Barrier_index_0000000106.pdf",
    #     "diags/fields/Barrier_index_0000000106.png",
    #     "diags/fields/Electric_field_strength_0000000000.npy",
    #     "diags/fields/Electric_field_strength_0000000000.pdf",
    #     "diags/fields/Electric_field_strength_0000000000.png",
    #     "diags/fields/Electric_field_strength_0000000106.npy",
    #     "diags/fields/Electric_field_strength_0000000106.pdf",
    #     "diags/fields/Electric_field_strength_0000000106.png",
    #     "diags/fields/Electrostatic_potential_0000000000.npy",
    #     "diags/fields/Electrostatic_potential_0000000000.pdf",
    #     "diags/fields/Electrostatic_potential_0000000000.png",
    #     "diags/fields/Electrostatic_potential_0000000106.npy",
    #     "diags/fields/Electrostatic_potential_0000000106.pdf",
    #     "diags/fields/Electrostatic_potential_0000000106.png",
    #     "diags/fields/Net_charge_density_0000000000.npy",
    #     "diags/fields/Net_charge_density_0000000106.npy",
    #     "diags/fields/Net_charge_density_0000000106.pdf",
    #     "diags/fields/Net_charge_density_0000000106.png",
    #     "diags/traces/trace_0000000106.npz",
    #     "diags/xzsolver/hdf5/data0000000106.h5",
    #     "diags/fluxes/anode_plane_scraped.csv",
    #     "diags/fluxes/cathode_scraped.csv",
    #     "diags/results.dpkl",
    #     "diags/results.txt",
    #     "diags/runinfo.dpkl",
    # ]

    # if use_2d_scraper:
    #     filelist += [
    #         "diags/histograms/particle_histogram_0000000106.npy",
    #         "diags/histograms/particle_histogramcurrent_0000000106.npy",
    #         "diags/histograms/particle_histogramcurrent_setup.p",
    #         "diags/histograms/particle_histogramcurrent_zold_0000000106.pdf",
    #         "diags/histograms/particle_histogram_setup.p",
    #         "diags/histograms/particle_histogram_uxbirth_0000000106.pdf",
    #         "diags/histograms/particle_histogram_uxbirth_uzbirth_0000000106.pdf",
    #         "diags/histograms/particle_histogram_uzbirth_0000000106.pdf",
    #     ]

    # print(out)
    # assert "Step: 106; Diagnostic period: 1" in out

    # for filename in filelist:
    #     assert os.path.isfile(filename)

    # myrun = analysis.RunProcess(".", use_minerva=False)
    # myrun.print_analysis()
    # outtext, _ = capsys.readouterr()
    # print(outtext)

    # # Gather results to check.  The index call here ensures there's one row to
    # # assign into in the new DataFrame.
    # rdict = myrun.get_analysis()
    # df = pandas.DataFrame(index=list(range(1)))
    # df['cathode_J_emit'] = rdict['cathode_all']['emit_J_full']
    # df['cathode_J_abs'] = rdict['cathode_all']['abs_J_full']
    # df['anode_J_abs'] = rdict['anode_all']['abs_J_full']

    # assert util.test_df_vs_ref(basename, df)

# def initialize_testingdir(name):
#     """Change into an appropriate directory for testing. This means placing it
#     in util.temp_dir, and attaching "_XXXXXX", a random 6-digit integer, to the
#     end of the path.
#     Arguments:
#         name (str): The base string used in the new directory.
#     Returns:
#         origwd, newdir (str, str): Original working directory, and current
#         working directory.
#     """
#     wd = None
#     if warp.me == 0:
#         # Use standard python random here, in case numpy rseed is being
#         # used that might fix this randint call.
#         wd = os.path.join(util.temp_dir, name + "_{:06d}".format(
#             np.random.randint(0, 1000000)))

#     if warp.lparallel:
#         wd = warp.mpibcast(wd)

#     if wd is None:
#         raise ValueError("Working directory not properly set or broadcast.")

#     origwd = change_to_warpdir(wd)

#     return origwd, os.getcwd()

# def change_to_warpdir(wd):
#     """Handle logic of moving to correct directory. The directory is created if
#     needed.
#     Arguments:
#         wd (str): Path of desired working directory.
#     Returns:
#         origwd (str): Path of original working directory
#     """
#     origwd = os.getcwd()

#     if mewarpx.me == 0:
#         minutil.mkdir_p(wd)

#     warp.comm_world.Barrier()

#     os.chdir(wd)
#     print("Change to working directory {}".format(os.getcwd()))

#     warp.comm_world.Barrier()

#     return origwd