from mewarpx import util as mwxutil
mwxutil.init_libwarpx(ndim=2, rz=False)


import numpy as np

from minerva import util as minutil

from pywarpx import picmi

from mewarpx import assemblies, emission, mepicmi

from mewarpx.mwxrun import mwxrun
from mewarpx.diags_store import diag_base

constants = picmi.constants

##############################
# physics parameters
##############################

PLASMA_DENSITY = 2.56e14 #m^-3
N_INERT = 9.6e14 #m^-3
T_ELEC = 30000.0 # K
T_INERT = 300.0 # K

P_INERT = N_INERT * (minutil.kb_cgs * T_INERT) / minutil.torr_cgs # Torr

M_ION = 6.646475849910765e-27

# Cathode-Anode distance (to top of anode)
D_CA = 0.067 # m

FREQ = 13.56e6 # MHz

################################
# numerics parameters
################################

# --- Grid
nx = 8
ny = 128

xmin = 0.0
ymin = 0.0

xmax = D_CA / ny * nx
ymax = D_CA
number_per_cell_each_dim = [16, 16]

TOTAL_TIME = 64 / FREQ
DIAG_INTERVAL = 16 / FREQ
DT = 1.0 / (400 * FREQ)

# --- Number of time steps
max_steps = int(TOTAL_TIME / DT)
diag_steps = int(DIAG_INTERVAL / DT)
diagnostic_intervals = '::%i' % diag_steps

print('Setting up simulation with')
print('  dt = %.3e s' % DT)
print('  Total time = %.3e s (%i timesteps)' % (TOTAL_TIME, max_steps))
print('  Diag time = %.3e s (%i timesteps)' % (DIAG_INTERVAL, diag_steps))

################################
# physics components
################################

v_rms_elec = np.sqrt(constants.kb * T_ELEC / constants.m_e)
v_rms_ion = np.sqrt(constants.kb * T_INERT / M_ION)

uniform_plasma_elec = picmi.UniformDistribution(
    density=PLASMA_DENSITY,
    upper_bound=[None] * 3,
    rms_velocity=[v_rms_elec] * 3,
    directed_velocity=[0.] * 3
)

uniform_plasma_ion = picmi.UniformDistribution(
    density=PLASMA_DENSITY,
    upper_bound=[None] * 3,
    rms_velocity=[v_rms_ion] * 3,
    directed_velocity=[0.] * 3
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
        'elastic': {
            'cross_section': cross_sec_direc+'electron_scattering.dat'
        },
        'excitation1': {
            'cross_section': cross_sec_direc+'excitation_1.dat',
            'energy': 19.82
        },
        'excitation2': {
            'cross_section': cross_sec_direc+'excitation_2.dat',
            'energy': 20.16
        },
        'ionization': {
            'cross_section': cross_sec_direc+'ionization.dat',
            'energy': 24.55,
            'species': ions
        }
    }
)

mcc_ions = picmi.MCCCollisions(
    name='coll_ion',
    species=ions,
    background_density=N_INERT,
    background_temperature=T_INERT,
    scattering_processes={
        'elastic': {
            'cross_section': cross_sec_direc+'ion_scattering.dat'
        },
        'back': {
            'cross_section': cross_sec_direc+'ion_back_scatter.dat'
        }
    }
)

#####################################
# numerics components
#####################################

grid = picmi.Cartesian2DGrid(
    number_of_cells=[nx, ny],
    lower_bound=[xmin, ymin],
    upper_bound=[xmax, ymax],
    bc_xmin='periodic',
    bc_xmax='periodic',
    bc_ymin='dirichlet',
    bc_ymax='dirichlet',
    bc_xmin_particles='periodic',
    bc_xmax_particles='periodic',
    bc_ymin_particles='absorbing',
    bc_ymax_particles='absorbing',
    moving_window_velocity=None,
    warpx_max_grid_size=128
)
grid.potential_ymin = 0.0
grid.potential_ymax = '450.0*sin(2*pi*%.5e*t)' % FREQ

solver = picmi.ElectrostaticSolver(
    grid=grid, method='Multigrid', required_precision=1e-6
)

#####################################
# diagnostics
#####################################

field_diag = picmi.FieldDiagnostic(
    name='diags',
    grid=grid,
    period=diagnostic_intervals,
    data_list=['rho_electrons', 'rho_he_ions', 'phi'],
    write_dir='.',
    warpx_file_prefix = 'diags',
    #warpx_format='openpmd',
    #warpx_openpmd_backend='h5'
)

#####################################
# simulation setup
#####################################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=max_steps,
    warpx_collisions=[mcc_electrons, mcc_ions]
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)

sim.add_species(
    ions,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim, grid=grid
    )
)

sim.add_diagnostic(field_diag)

##################################
# WarpX and mewarpx initialization
##################################

mwxrun.init_run(simulation=sim)

####################################
# Add ME emission
####################################

cathode = assemblies.ZPlane(z=1e-10, zsign=-1, V=0, T=1323.15, WF=1.9,
                            name='cathode')
emitter = emission.ZPlaneEmitter(conductor=cathode, T=1323.15,
                                use_Schottky=False)
injector = emission.ThermionicInjector(emitter=emitter, species=electrons,
                                        npart_per_cellstep=50,
                                        T=1323.15, WF=1.9,
                                        A=6e5)
injector_2 = emission.FixedNumberInjector(emitter=emitter, species=ions,
                                        injectfreq=50,
                                        npart=100000, weight=1e4)


###############################
# Add ME diagnostic
###############################

diag_base.TextDiag(diag_steps=1, preset_string='perfdebug')

###############################
# Simulation run
###############################

sim.step(10)