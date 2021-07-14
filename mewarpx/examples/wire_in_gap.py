"""
Monte-Carlo Collision script based on case 1 from
Turner et al. (2013) - https://doi.org/10.1063/1.4775084
"""

from mewarpx import util as mwxutil
mwxutil.init_libwarpx(ndim=2, rz=False)

from mewarpx.mwxrun import mwxrun
from mewarpx.mcc_wrapper import MCC
from mewarpx.diags_store import diag_base

from pywarpx import picmi
import pywarpx

import numpy as np

constants = picmi.constants

##########################
# physics parameters
##########################

D_CA = 0.5e-3 # m

P_INERT = 2 # Torr
T_INERT = 300.0 # K

VOLTAGE = 20.0

M_ION = 6.67e-27 # kg

PLASMA_DENSITY = 2.56e14 # m^-3
T_ELEC = 30000.0 # K

##########################
# numerics parameters
##########################

# --- Grid
nx = 128
nz = 128

xmin = 0.0
zmin = 0.0
xmax = D_CA / nz * nx
zmax = D_CA

number_per_cell_each_dim = [10, 10]

DT = 1.0e-12

# Total simulation time in seconds
TOTAL_TIME = 5.0 * DT # 1280 / FREQ
# Time (in seconds) between diagnostic evaluations
# DIAG_INTERVAL = 100.0 * DT # 32 / FREQ

# --- Number of time steps
max_steps = int(TOTAL_TIME / DT)
diag_steps = 5
diagnostic_intervals = f"::{diag_steps}"

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
mcc_wrapper = MCC(
    electrons, ions, T_INERT=T_INERT, P_INERT=P_INERT,
    exclude_collisions=['charge_exchange']
)

# Embedded Boundary
boundary = picmi.EmbeddedBoundary(
    geom_type="cylinder", cylinder_center="0.25e-3 0.25e-3",
    cylinder_radius=100e-6, cylinder_height=1, cylinder_direction=2,
    has_fluid_inside=False, potential=-2.0
)
mwxrun.simulation.embedded_boundary = boundary

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
    warpx_potential_lo_z = VOLTAGE,
    warpx_potential_hi_z = VOLTAGE,
    warpx_max_grid_size = nz//4
)

##########################
# declare solver
##########################

solver = picmi.ElectrostaticSolver(
   grid=grid, method='Multigrid', required_precision=1e-6,
   warpx_self_fields_verbosity=0
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

mwxrun.simulation.add_diagnostic(field_diag)
# sim.add_diagnostic(restart_dumps)

##########################
# WarpX and mewarpx initialization
##########################

mwxrun.init_run()

##########################
# Add ME diagnostic
##########################

#diag_base.TextDiag(diag_steps=diag_steps, preset_string='perfdebug')

##########################
# simulation run
##########################

mwxrun.simulation.step()
#mwxrun.simulation.write_input_file()

##########################
# collect diagnostics
##########################

if mwxrun.me == 0:
    import glob
    import yt
    import matplotlib.pyplot as plt

    try:
        datafolder = sorted(glob.glob('diags/diags*'))[-1]

        print('Reading ', datafolder, '\n')
        ds = yt.load( datafolder )
        grid_data = ds.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )
        phi_data = grid_data['phi'].to_ndarray()[:,:,0]

        c = plt.imshow(phi_data,
            extent=[
                ds.domain_left_edge[0], ds.domain_right_edge[0],
                ds.domain_left_edge[1], ds.domain_right_edge[1]
            ],
            aspect='auto'
        )
        plt.colorbar(c)
        plt.show()

    except IndexError:
        print('No datafiles found.')
