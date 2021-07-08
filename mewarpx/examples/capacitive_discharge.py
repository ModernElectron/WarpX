"""
Monte-Carlo Collision script based on case 1 from
Turner et al. (2013) - https://doi.org/10.1063/1.4775084
"""

from mewarpx import util as mwxutil
mwxutil.init_libwarpx(ndim=2, rz=False)

from mewarpx.mwxrun import mwxrun
from mewarpx.poisson_pseudo_1d import PoissonSolverPseudo1D
from mewarpx.diags_store import diag_base

from pywarpx import picmi
import pywarpx

import numpy as np

constants = picmi.constants

##########################
# physics parameters
##########################

D_CA = 0.067 # m

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

# --- Grid
nx = 8
nz = 128

xmin = 0.0
zmin = 0.0
xmax = D_CA / nz * nx
zmax = D_CA

number_per_cell_each_dim = [32, 16]

DT = 1.0 / (400 * FREQ)

# Total simulation time in seconds
TOTAL_TIME = 500.0 * DT # 1280 / FREQ
# Time (in seconds) between diagnostic evaluations
# DIAG_INTERVAL = 100.0 * DT # 32 / FREQ

# --- Number of time steps
max_steps = int(TOTAL_TIME / DT)
diag_steps = 100
diagnostic_intervals = "400::10"

print('Setting up simulation with')
print('  dt = %.3e s' % DT)
print('  Total time = %.3e s (%i timesteps)' % (TOTAL_TIME, max_steps))

##########################
# physics components
##########################

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

##########################
# numerics components
##########################

grid = picmi.Cartesian2DGrid(
    number_of_cells = [nx, nz],
    lower_bound = [xmin, zmin],
    upper_bound = [xmax, zmax],
    lower_boundary_conditions=['periodic', 'dirichlet'],
    upper_boundary_conditions=['periodic', 'dirichlet'],
    lower_boundary_conditions_particles=['periodic', 'absorbing'],
    upper_boundary_conditions_particles=['periodic', 'absorbing'],
    # warpx_potential_hi_z = "%.1f*sin(2*pi*%.5e*t)" % (VOLTAGE, FREQ),
    moving_window_velocity = None,
    warpx_max_grid_size = nz//4
)

solver = picmi.ElectrostaticSolver(
    grid=grid, method='Multigrid', required_precision=1e-12
)

##########################
# diagnostics
##########################

field_diag = picmi.FieldDiagnostic(
    name = 'diags',
    grid = grid,
    period = diagnostic_intervals,
    data_list = ['rho_electrons', 'rho_he_ions', 'phi'],
    write_dir = 'diags/',
)

##########################
# simulation setup
##########################

sim = picmi.Simulation(
    solver = solver,
    time_step_size = DT,
    max_steps = max_steps,
    warpx_collisions=[mcc_electrons, mcc_ions],
    verbose=0
)

sim.add_species(
    electrons,
    layout = picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)
sim.add_species(
    ions,
    layout = picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)

sim.add_diagnostic(field_diag)
# sim.add_diagnostic(restart_dumps)

##########################
# WarpX and mewarpx initialization
##########################

mwxrun.init_run(simulation=sim)

##########################
# Add direct solver
##########################

anode_voltage = lambda t: VOLTAGE * np.sin(2.0 * np.pi * FREQ * t)
# comment line below to use the multigrid solver
my_solver = PoissonSolverPseudo1D(right_voltage=anode_voltage)

##########################
# Add ME diagnostic
##########################

diag_base.TextDiag(diag_steps=diag_steps, preset_string='perfdebug')

##########################
# simulation run
##########################

sim.step()

##########################
# collect diagnostics
##########################

if mwxrun.me == 0:
    import glob
    import yt

    data_dirs = glob.glob('diags/diags*')

    for ii, data in enumerate(data_dirs):

        datafolder = data
        print('Reading ', datafolder, '\n')
        ds = yt.load( datafolder )
        grid_data = ds.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )
        if ii == 0:
            rho_data = np.mean(
                grid_data['rho_he_ions'].to_ndarray()[:,:,0], axis=0
            ) / constants.q_e
        else:
            rho_data += np.mean(
                grid_data['rho_he_ions'].to_ndarray()[:,:,0], axis=0
            ) / constants.q_e
    rho_data /= (ii + 1)
    np.save('direct_solver_avg_rho_data.npy', rho_data)
