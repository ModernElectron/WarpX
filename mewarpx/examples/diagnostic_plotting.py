"""
Monte-Carlo Collision script based on case 1 from
Turner et al. (2013) - https://doi.org/10.1063/1.4775084
"""

from mewarpx import util as mwxutil

mwxutil.init_libwarpx(ndim=2, rz=False)

from mewarpx import mepicmi

from mewarpx.mwxrun import mwxrun
from mewarpx.mcc_wrapper import MCC
from mewarpx.diags_store import diag_base, field_diagnostic

from pywarpx import picmi

import numpy as np
import matplotlib.pyplot as plt

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

number_per_cell_each_dim = [16, 32]

DT = 1.0 / (400 * FREQ)

# Total simulation time in seconds
TOTAL_TIME = 10.0 * DT # 1280 / FREQ
# Time (in seconds) between diagnostic evaluations
# DIAG_INTERVAL = 100.0 * DT # 32 / FREQ

# --- Number of time steps
max_steps = int(TOTAL_TIME / DT)
diag_steps = 2
diagnostic_intervals = "::1"

print('Setting up simulation with')
print('  dt = %.3e s' % DT)
print('  Total time = %.3e s (%i timesteps)' % (TOTAL_TIME, max_steps))

##########################
# physics components
##########################

anode_voltage = lambda t: VOLTAGE * np.sin(2.0 * np.pi * FREQ * t)

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

electrons = mepicmi.Species(
    particle_type='electron', name='electrons',
    initial_distribution=uniform_plasma_elec
)
ions = mepicmi.Species(
    particle_type='He', name='he_ions',
    charge='q_e',
    initial_distribution=uniform_plasma_ion
)

# MCC collisions
mcc_wrapper = MCC(
    electrons, ions, T_INERT=T_INERT, N_INERT=N_INERT,
    exclude_collisions=['charge_exchange']
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
    warpx_potential_hi_z = "%.1f*sin(2*pi*%.5e*t)" % (VOLTAGE, FREQ),
    moving_window_velocity = None,
    warpx_max_grid_size = nz//4
)

##########################
# declare solver
##########################

solver = picmi.ElectrostaticSolver(
   grid=grid, method='Multigrid', required_precision=1
)
#solver = PoissonSolverPseudo1D(grid=grid)

##########################
# simulation setup
##########################

mwxrun.simulation.solver = solver
mwxrun.simulation.time_step_size = DT
mwxrun.simulation.max_steps = max_steps

mwxrun.simulation.add_species(
    electrons,
    layout = picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)
mwxrun.simulation.add_species(
    ions,
    layout = picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)

##########################
# diagnostics
##########################

field_diag = field_diagnostic.FieldDiagnostic(
    name = 'diags',
    grid = grid,
    diag_steps = diag_steps,
    diag_data_list = ['rho_electrons', 'rho_he_ions', 'phi'],
    write_dir = 'diags/',
    plot_on_diag_step = True,
    plot_data_list = ['rho', 'phi']
)

# sim.add_diagnostic(restart_dumps)

##########################
# WarpX and mewarpx initialization
##########################

mwxrun.init_run()

##########################
# Add ME diagnostic
##########################

diag_base.TextDiag(diag_steps=diag_steps, preset_string='perfdebug')

##########################
# simulation run
##########################

mwxrun.simulation.step()
