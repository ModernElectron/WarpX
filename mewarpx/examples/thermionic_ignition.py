from mewarpx import util as mwxutil

from picmistandard import simulation
mwxutil.init_libwarpx(ndim=2, rz=False)

from minerva import util as minutil

from pywarpx import picmi

from mewarpx import assemblies, emission, mepicmi

from mewarpx.mwxrun import mwxrun
from mewarpx.diags_store import diag_base

constants = picmi.constants

####################################
# physics parameters
####################################

P_INERT = 2.0 # torr
T_INERT = 300.0 # K
N_INERT = (P_INERT * minutil.torr_cgs) / (minutil.kb_cgs * T_INERT) # m^-3

D_CA = 1e-4 # m
V_bias = 30 # V

###################################
# numerics parameters
##################################

# --- Grid
nx = 8
ny = 128

xmin = 0.0
ymin = 0.0

xmax = D_CA / ny * nx
ymax = D_CA
number_per_cell_eacg_dim = [16, 16]

TOTAL_TIME = 1.0e-9 # s
DIAG_INTERVAL = 1.0e-10
DT = 1.0e-12 # s

max_steps = int(TOTAL_TIME / DT)
diag_steps = int(DIAG_INTERVAL / DT)
diagnostic_intervals = '::%i' % diag_steps

print('Setting up simulation with')
print(' dt = %.3e s' % DT)
print(' Total tine = %.3e s (%i timesteps)' % (TOTAL_TIME, max_steps))
print(' Diag time = %.3e (%i timesteps)' % (DIAG_INTERVAL, diag_steps))

#################################
# physics components
################################

electrons = mepicmi.Species(
    particle_type='electron',
    name='electrons'
)

ions = mepicmi.Species(
    particle_type='Ar',
    name='ar_ions',
    charge='q_e',
)

# MCC Collisions
cross_sec_direc = '../../../warpx-data/MCC_cross_sections/Ar/'
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
        'ionization': {
            'cross_section': cross_sec_direc+'ionization.dat',
            'energy': 15.7596112,
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
# grid and solver
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
    warpx_max_grid_size=128,
    warpx_potential_lo_z=0.0,
    warpx_potential_hi_z=V_bias,
)

solver = picmi.ElectrostaticSolver(
    grid=grid, method='Multigrid', required_precision=1e-6
)

###################################
# diagnostics
##################################

field_diag = picmi.FieldDiagnostic(
    name='diags',
    grid=grid,
    period=diagnostic_intervals,
    data_list=['rho_electrons', 'rho_ar_ions', 'phi', 'J'],
    write_dir='.',
    warpx_file_prefix='diags'
)

#################################
# simulation setup
################################

sim = picmi.Simulation(
    solver=solver,
    time_step_size=DT,
    max_steps=max_steps,
    warpx_collisions=[mcc_electrons, mcc_ions]
)

sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_eacg_dim,
        grid=grid
    )
)

sim.add_species(
    ions,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_eacg_dim,
        grid=grid
    )
)

#sim.add_diagnostic(field_diag)

mwxrun.init_run(simulation=sim)

######################################
# Add ME emission
#####################################
T_cathode = 1100.0 # K
WF_cathode = 2.0 # eV

cathode = assemblies.ZPlane(z=1e-10, zsign=-1, V=0, T=T_cathode,
                            WF=WF_cathode,
                            name='cathode')
emitter = emission.ZPlaneEmitter(conductor=cathode, T=T_cathode,
                                use_Schottky=False)
injector = emission.ThermionicInjector(emitter=emitter, species=electrons,
                                        npart_per_cellstep=5,
                                        T=T_cathode, WF=WF_cathode,
                                        A=6e5)

####################################
# Add ME diagnostic
###################################

diag_base.TextDiag(diag_steps=diag_steps, preset_string='perfdebug')

##################################
# Simulation run
#################################
max_steps = 10
sim.step(1)