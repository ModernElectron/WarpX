from mewarpx import util as mwxutil

mwxutil.init_libwarpx(ndim=2, rz=False)


from pywarpx import picmi
from mewarpx.mcc_wrapper import MCC
from mewarpx import assemblies, emission, mepicmi, testing_util

from mewarpx.mwxrun import mwxrun
from pywarpx.picmi import constants

import numpy as np
import pandas

####################################
# physics parameters
####################################

P_INERT = 2.0 # torr
T_INERT = 300.0 # K
N_INERT = mwxutil.ideal_gas_density(P_INERT, T_INERT) # m^-3

D_CA = 1 # m
V_bias = 0 # V

###################################
# numerics parameters
##################################

# --- Grid
nx = 16
ny = 16

xmin = -0.3
ymin = 0

xmax = 0.3
ymax = D_CA
number_per_cell_each_dim = [16, 16]

TOTAL_TIME = 100.0e-12 # s
DIAG_INTERVAL = 1.0e-12
DT = 1.0e-12 # s

max_steps = int(TOTAL_TIME / DT)
diag_steps = int(DIAG_INTERVAL / DT)
diagnostic_intervals = '::%i' % diag_steps

print('Setting up simulation with')
print(' dt = %.3e s' % DT)
print(' Total time = %.3e s (%i timesteps)' % (TOTAL_TIME, max_steps))
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
mcc_wrapper = MCC(electrons, ions, T_INERT=T_INERT, N_INERT=N_INERT,
                    exclude_collisions=['charge_exchange'])

#####################################
# embedded boundary, grid, and solver
#####################################

T_cylinder = 2173.15 # K
WF_cylinder = 1.2 # eV
cylinder = assemblies.Cylinder(center_x=0.2, center_z=0.4, radius=0.1, V=0, T=T_cylinder,
                                WF=WF_cylinder, name='circle')

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
    warpx_max_grid_size=nx,
    warpx_potential_lo_z=0.0,
    warpx_potential_hi_z=V_bias,
)

solver = picmi.ElectrostaticSolver(
    grid=grid, method='Multigrid', required_precision=1e-3
)

###################################
# diagnostics
##################################

field_diag = picmi.FieldDiagnostic(
    name='diags',
    grid=grid,
    period=diagnostic_intervals,
    data_list=['rho_electrons', 'rho_ar_ions', 'phi', 'J'],
    write_dir='diags/',
    warpx_file_prefix='field_diags'
)


#################################
# simulation setup
################################

mwxrun.simulation.solver = solver
mwxrun.simulation.time_step_size = DT
mwxrun.simulation.max_steps = max_steps


mwxrun.simulation.add_species(
    electrons,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim,
        grid=grid
    )
)

mwxrun.simulation.add_species(
    ions,
    layout=picmi.GriddedLayout(
        n_macroparticle_per_cell=number_per_cell_each_dim,
        grid=grid
    )
)

# mwxrun.simulation.add_diagnostic(field_diag)

mwxrun.init_run()

######################################
# Add ME emission
#####################################

emitter = emission.ArbitraryEmitter2D(conductor=cylinder, T=T_cylinder, use_Schottky=False, res_fac=10)
injector = emission.ThermionicInjector(emitter=emitter, species=electrons,
                                        npart_per_cellstep=5,
                                        T=T_cylinder, WF=WF_cylinder,
                                        A=6e5)

res_dict = emitter.get_newparticles(10000, 1, electrons.sq, electrons.sm, randomdt=False, velhalfstep=False)
df = pandas.DataFrame(index=list(range(1)))

# Compare main results
for label in ['E_total', 'vx', 'vy', 'vz', 'x', 'y', 'z']:
    df[label + '_min'] = np.min(res_dict[label])
    df[label + '_max'] = np.max(res_dict[label])
    df[label + '_mean'] = np.mean(res_dict[label])
    df[label + '_std'] = np.std(res_dict[label])

assert testing_util.test_df_vs_ref(testname="circle_emitter", df=df, margin=0.4)
