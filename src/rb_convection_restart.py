"""
Dedalus script for calculating the maximum growth rates in no-slip
Rayleigh Benard convection over a range of horizontal wavenumbers.
This script can be ran serially or in parallel, and produces a plot of the
highest growth rate found for each horizontal wavenumber.
To run using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
"""

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as de
from dedalus.extras import flow_tools, plot_tools
import logging
logger = logging.getLogger(__name__)

initial_time = time.time()

# Global parameters
Nx=16; Ny= 16; Nz = 16;
Prandtl = 0.025
Rayleigh = 10**5
Lx=25; Ly=25; Lz= 1;

# Create bases and domain
x_basis = de.Fourier('x', Nx, interval=(0, Lx))
y_basis = de.Fourier('y', Ny, interval=(0, Ly))
z_basis = de.Chebyshev('z', Nz, interval=(0, Lz))
domain = de.Domain([x_basis,y_basis,z_basis], grid_dtype=np.float64)

# 3D Boussinesq
problem = de.IVP(domain, variables=['p','b','u','v','w','bz','uz','vz','wz'], time='t')
problem.meta[:]['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1

problem.add_equation("dx(u)+dy(v)+wz = 0") #Fluido Incompresible
problem.add_equation("dt(b) - P*(dx(dx(b)) + dy(dy(b)) + dz(bz))              = -(u*dx(b) +  v*dy(b) + w*bz)") 
problem.add_equation("dt(u) - R*(dx(dx(u)) + dy(dy(u)) + dz(uz)) + dx(p)      = -(u*dx(u) +  v*dy(u) + w*uz)")#Eq. Navier-Stokes
problem.add_equation("dt(v) - R*(dx(dx(v)) + dy(dy(v)) + dz(vz)) + dy(p)      = -(u*dx(v) +  v*dy(v) + w*vz)")
problem.add_equation("dt(w) - R*(dx(dx(w)) + dy(dy(w)) + dz(wz)) + dz(p) - b  = -(u*dx(w) + w*dy(w) + w*wz)")

problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")

#Condiciones de contorno de z
problem.add_bc("left(b) = 1")
problem.add_bc("left(u) = 0")
problem.add_bc("left(v) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(b) = 0")
problem.add_bc("right(p) = 0", condition="(nx == 0) and (ny == 0)")
problem.add_bc("right(u) = 0")
problem.add_bc("right(v) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0) or (ny != 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.MCNAB2)
logger.info('Solver built')

# Load restart
# archive decay timescales
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_current_snapshots = sorted(os.listdir(dir_path+'/'+'snapshots'))[-1]
current_snapshots =  dir_path + '/' + 'snapshots'+ '/' + dir_current_snapshots + '/' + dir_current_snapshots +'_p0.h5'
write, dt = solver.load_state(current_snapshots, -1)

# Initial conditions
z = domain.grid(2)
b = solver.state['b']
bz = solver.state['bz']

b['g'] = F*(1-z)

# Analysis
snapshots = solver.evaluator.add_file_handler(dir_path+'/'+'snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_system(solver.state)
# Setup storage
b_list = []
t_list = []

# Initial timestep
dt = 1e-3 
# Integration parameters
solver.stop_sim_time = 1000
solver.stop_wall_time = 100 * 60.
solver.stop_iteration = np.inf

max_dt = 10

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=0.8/2,
                     max_change=1.5, min_change=0.5, max_dt=max_dt, threshold=0.05)
CFL.add_velocities(('u', 'v', 'w'))

# Flow properties
#flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
#flow.add_property("sqrt(u*u + v*v + w*w) / R", name='Re')

#Main
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt) #, trim=True)
        log_string = 'Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt)
        logger.info(log_string)
        #b_list.append(np.copy(b['g']))
        #t_list.append(solver.sim_time)
        if solver.iteration % 1 == 0:

            #de.operators.interpolate(b, x=0, y=0).evaluate()['g'][0,0])
            #print(b['g'])
            b_list.append(np.copy(b['g']))
            t_list.append(solver.sim_time)
        logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Convert storage to arrays
    b_array = np.array(b_list, dtype=np.float64)
    t_array = np.array(t_list, dtype=np.float64)
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

    if (domain.distributor.rank==0):
        N_TOTAL_CPU = domain.distributor.comm_cart.size

        # Print statistics
        print('-' * 40)
        total_time = end_time-initial_time
        main_loop_time = end_time - start_time
        startup_time = start_time-initial_time
        n_steps = solver.iteration-1
#        s.set_array(np.ravel(b['g'][:-1,:-1].T))
        print('  startup time:', startup_time)
        print('main loop time:', main_loop_time)
        print('    total time:', total_time)
        print('    iterations:', solver.iteration)
        print(' loop sec/iter:', main_loop_time/solver.iteration)
        print('    average dt:', solver.sim_time/n_steps)
        print("          N_cores, Nx, Ny, Nz, startup     main loop,   main loop/iter, main loop/iter/grid, n_cores*main loop/iter/grid")
        print('scaling:',
              ' {:d} {:d} {:d} {:d}'.format(N_TOTAL_CPU,Nx,Ny,Nz),
              ' {:8.3g} {:8.3g} {:8.3g} {:8.3g} {:8.3g}'.format(startup_time,
                                                                main_loop_time, 
                                                                main_loop_time/n_steps, 
                                                                main_loop_time/n_steps/(Nx*Ny*Nz), 
                                                                N_TOTAL_CPU*main_loop_time/n_steps/(Nx*Ny*Nz)))
        print('-' * 40)
        xmesh, ymesh = plot_tools.quad_mesh(x=domain.grid(0, scales=domain.dealias).flatten(), y=domain.grid(2, scales=domain.dealias).flatten())
        # Plot
#        plt.figure(figsize=(12, 8))
#        print(b_array[2][:][5][:])
       # np.savetxt("hola.txt",b_array)
        #print(b_array[0][:][1][:])
        plt.pcolormesh(xmesh, ymesh, b_array[80][:][1][:].T, cmap='RdBu_r')
        #plt.axis(plot_tools.pad_limits(xmesh, ymesh))
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
#        plt.title('A dispersive shock!')
        plt.show()

