"""
Dedalus script for 2D Rayleigh-Benard convection.

Usage:
    rb_with_S.py PR TAU RP

Arguments:
    PR		Prandtl number      (nu/k_T)
    TAU     diffusivity ratio   (k_s/k_T)
    RP      density ratio       (alpha T_z/beta S_z)

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ conda activate dedalus 		# activates dedalus
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5
    $ python3 create_gif.py

The 'mpiexec -n 4' can be ommited to run in series

The simulation should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# For adding arguments when running
from docopt import docopt

# Parameters
Lx, Lz = (4., 1.)
Prandtl = 7.
diffusivity_r = 1e-2
density_r = 1e2
f_or_d = -1.     # 1. for fingering case, -1. for diffusive case

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    Prandtl = float(arguments.get('PR'))
    print('Prandtl number = ', Prandtl)
    diffusivity_r = float(arguments.get('TAU'))
    print('Diffusivity ratio = ', diffusivity_r)
    density_r = float(arguments.get('RP'))
    print('Density ratio = ', density_r)

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','T', 'S','u','w','Tz', 'Sz','uz','wz'])
problem.meta['p','T','u','w']['z']['dirichlet'] = True
problem.parameters['Pr']  = Prandtl
problem.parameters['Tau'] = diffusivity_r
problem.parameters['Rp']  = Rp = density_r
problem.parameters['pm']  = f_or_d
# Mass conservation equation
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(T) - (dx(dx(T)) + dz(Tz)) + pm*w               = -(u*dx(T) + w*Tz)")
problem.add_equation("dt(S) - Tau*(dx(dx(S)) + dz(Sz)) + pm*w/Rp        = -(u*dx(S) + w*Sz)")
problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + Pr*dx(p)        = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + Pr*dz(p) - Pr*(T-S) = -(u*dx(w) + w*wz)")
# Definitions for easier derivative syntax
problem.add_equation("Tz - dz(T) = 0")
problem.add_equation("Sz - dz(S) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
# Boundary contitions
problem.add_bc("left(T) = 0")
problem.add_bc("left(S) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(T) = 0")
problem.add_bc("right(S) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
T = solver.state['T']
Tz = solver.state['Tz']
S = solver.state['S']
Sz = solver.state['Sz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
#   z_bottom, z_top
zb, zt = z_basis.interval
#   1e-3*noise adds small random fluctuations
#   (zt-z)(z-zb) shapes the fluctuations, making them
#       larger in the middle and vanish at the top and bottom
pert =  1e-3 * noise * (zt - z) * (z - zb)
T['g'] = 1 * pert
T.differentiate('z', out=Tz)
S['g'] = 1 * pert
S.differentiate('z', out=Sz)

# Initial timestep
dt = 0.125

# Integration parameters
solver.stop_sim_time = 25
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
#flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            #logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
