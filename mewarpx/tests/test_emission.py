"""Test functionality in mewarpx.emission.py"""
import os
import numpy as np
import pandas

from mewarpx import util as mwxutil

def test_thermionic_emission():
    name = "Thermionic Emission"
    dim = 2

    # Initialize and import only when we know dimension
    mwxutil.init_libwarpx(ndim=dim, rz=False)
    from mewarpx import testing_util
    from mewarpx.setups_store import diode_setup
    from mewarpx.mwxrun import mwxrun

    import mewarpx.mwxconstants as constants

    # Include a random run number to allow parallel runs to not collide.  Using
    # python randint prevents collisions due to numpy rseed below
    testing_util.initialize_testingdir(name)

    # Initialize each run with consistent, randomly-chosen, rseed, Use a random
    # seed instead for initial dataframe generation.
    # np.random.seed()
    np.random.seed(47239475)

    TOTAL_TIME = 1e-10 # s
    DIAG_INTERVAL = 5e-10 # s
    DT = 0.5e-12 # s

    P_INERT = 1 # torr
    T_INERT = 300 # K

    D_CA = 5e-4 # m
    VOLTAGE = 25 # V
    CATHODE_TEMP = 1100 + 273.15 # K
    CATHODE_PHI = 2.1 # work function in eV
    NX = 8
    NZ = 128

    DIRECT_SOLVER = True

    max_steps = int(TOTAL_TIME / DT)
    diag_steps = int(DIAG_INTERVAL / DT)

    run = diode_setup.DiodeRun_V1(
        dim=dim,
        CATHODE_TEMP=CATHODE_TEMP,
        CATHODE_PHI=CATHODE_PHI,
        V_ANODE_CATHODE=VOLTAGE,
        D_CA=D_CA,
        P_INERT=P_INERT,
        T_INERT=T_INERT,
        NPPC=50,
        NX=NX,
        NZ=NZ,
        DIRECT_SOLVER=DIRECT_SOLVER,
        PERIOD=D_CA * NX / NZ,
        DT=DT,
        TOTAL_TIMESTEPS=max_steps,
        DIAG_STEPS=diag_steps,
        DIAG_INTERVAL=DIAG_INTERVAL
    )
    # Only the functions we change from defaults are listed here
    run.setup_run(
        init_conductors=True,
        init_scraper=False,
        init_warpx=True
    )

    mwxrun.simulation.step(max_steps)

    net_rho_grid = np.array(mwxrun.get_gathered_rho_grid()[0][:, :, 0])
    ref_path = os.path.join(testing_util.test_dir,
                            "thermionic_emission",
                            "thermionic_emission.npy")
    ref_rho_grid = np.load(ref_path)

    assert np.allclose(net_rho_grid, ref_rho_grid)


def test_circle_emitter():

    mwxutil.init_libwarpx(ndim=2, rz=False)
    from pywarpx import picmi
    from mewarpx.mcc_wrapper import MCC
    from mewarpx import assemblies, emission, mepicmi, testing_util

    from mewarpx.mwxrun import mwxrun

    # Initialize each run with consistent, randomly-chosen, rseed. Use a random
    # seed instead for initial dataframe generation.
    np.random.seed(671237741)

    #####################################
    # embedded boundary, grid, and solver
    #####################################

    T_cylinder = 2173.15 # K
    cylinder = assemblies.Cylinder(
        center_x=0.2, center_z=0.4, radius=0.1, V=0, T=T_cylinder,
        WF=1.2, name='circle'
    )

    mwxrun.init_grid(0, 0.5, 0, 1, 16, 16)
    solver = picmi.ElectrostaticSolver(
        grid=mwxrun.grid, method='Multigrid', required_precision=1e-6
    )

    #################################
    # physics components
    ################################

    electrons = mepicmi.Species(particle_type='electron', name='electrons')

    #################################
    # simulation setup
    ################################

    mwxrun.simulation.solver = solver
    mwxrun.init_run()

    ######################################
    # Add ME emission
    #####################################

    emitter = emission.ArbitraryEmitter2D(
        conductor=cylinder, T=T_cylinder, use_Schottky=False, res_fac=10
    )

    res_dict = emitter.get_newparticles(
        10000, 1, electrons.sq, electrons.sm, randomdt=False, velhalfstep=False
    )
    df = pandas.DataFrame(index=list(range(1)))

    # Compare main results, leave out E_total, since variation is too high
    for label in ['vx', 'vy', 'vz', 'x', 'y', 'z']:
        df[label + '_min'] = np.min(res_dict[label])
        df[label + '_max'] = np.max(res_dict[label])
        df[label + '_mean'] = np.mean(res_dict[label])
        df[label + '_std'] = np.std(res_dict[label])

    assert testing_util.test_df_vs_ref(
        testname="circle_emitter", df=df, margin=0.4
    )
