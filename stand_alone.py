"""

1D Bousinessq streamfunction equation:

dt^2(dz^2(psi) - k^2 psi) - k^2 N^2 psi - nu[dz^4(psi) + k^4 psi] = 0

where k is the horizontal wavenumber, N is stratification,
    and nu is the viscosity

This script should be ran serially (because it is 1D).

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import quad_mesh, pad_limits
#from dedalus.core.operators import GeneralFunction

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Switches
plot_windows = True
run_sim = True

###############################################################################
# Run parameters
stop_n_periods = 50             # [] oscillation periods
# Displayed domain parameters
nz     = 1024                   # [] number of grid points in the z direction
zf_dis = 0.0                    # [m] the top of the displayed z domain
Lz_dis = 1.0                    # [m] the length of the z domain between forcing and sponge
z0_dis = zf_dis - Lz_dis        # [m] The bottom of the displayed domain
###############################################################################
# Physical parameters
nu          = 1.0E-6        # [m^2/s] Viscosity (momentum diffusivity)
g           = 9.81          # [m/s^2] Acceleration due to gravity
# Problem parameters
A       = 2.0e-4                # []            Amplitude of boundary forcing
N_0     = 1.0                   # [rad/s]       Reference stratification
lam_z   = Lz_dis / 4.0          # [m]           Vertical wavelength
lam_x   = lam_z                 # [m]           Horizontal wavelength
#
m       = 2*np.pi / lam_z       # [m^-1]        Vertical wavenumber
k       = 2*np.pi / lam_x       # [m^-1]        Horizontal wavenumber
theta   = np.arctan(m/k)        # [rad]         Propagation angle from vertical
omega   = N_0 * np.cos(theta)   # [rad s^-1]    Wave frequency
T       = 2*np.pi / omega       # [s]           Wave period

print('phase speed is',omega/m,'m/s')
print('group speed is',omega*m/(k**2 + m**2),'m/s') # cushman-roisin and beckers 13.10, pg 400

###############################################################################
# Boundary forcing window parameters
a_bf    = 3*lam_z               # [m] forcing area, height above display domain
b_bf    = 0.3*a_bf              # [m] full width at half max of forcing window
c_bf    = zf_dis + 0.5*a_bf     # [m] center of forcing area
tau_bf  = 1.0e0                 # [s] time constant for boundary forcing

# Sponge layer window parameters
a_sp    = 3*lam_z               # [m] sponge area, height below display domain
b_sp    = 0.3*a_sp              # [m] full width at half max of sponge window
c_sp    = z0_dis - 0.5*a_sp     # [m] center of sponge area
tau_sp  = 1.0e0                 # [s] time constant for sponge layer

###############################################################################
# Simulated domain parameters
zf     = zf_dis + a_bf          # [m] top of simulated domain
z0     = z0_dis - a_sp          # [m] bottom of simulated domain
Lz     = zf - z0                # [m] length of simulated domain
dealias= 3/2                    # [] dealiasing factor

# Bases and domain
z_basis = de.Fourier('z', nz, interval=(z0, zf), dealias=dealias)
domain  = de.Domain([z_basis], grid_dtype=np.complex128)
z = domain.grid(0)

# Define problem
problem = de.IVP(domain, variables=['psi', 'foo'])
problem.parameters['NU'] = nu
problem.parameters['N0'] = N_0

###############################################################################
# Boundary forcing parameters
problem.parameters['A']     = A
problem.parameters['k']     = k
problem.parameters['m']     = m
problem.parameters['omega'] = omega

# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['psi_f'] = "A*sin(m*z - omega*t)"

###############################################################################
# Boundary forcing window
win_bf       = domain.new_field(name = 'win_bf')
win_bf_array = np.exp(-4*np.log(2)*((z - c_bf)/b_bf)**2)     # Gaussian
win_bf['g']  = win_bf_array
problem.parameters['win_bf'] = win_bf
problem.parameters['tau_bf'] = tau_bf   # [s] time constant for boundary forcing

# Creating forcing terms
problem.substitutions['F_term_psi'] = "win_bf * (psi_f - psi)/tau_bf"

###############################################################################
# Sponge window
win_sp       = domain.new_field(name = 'win_sp')
win_sp_array = np.exp(-4*np.log(2)*((z - c_sp)/b_sp)**2)     # Gaussian
win_sp['g']  = win_sp_array
problem.parameters['win_sp'] = win_sp
problem.parameters['tau_sp'] = tau_sp   # [s] time constant for sponge layer

# Creating sponge terms
problem.substitutions['S_term_psi'] = "win_sp * psi / tau_sp"

###############################################################################
# Plotting windows
def plot_v_profile(axis, bf_array, sp_array, z, omega=None, z0_dis=None, zf_dis=None):
    # Plot both windows
    axis.plot(bf_array, z, label='Boundary forcing')
    axis.plot(sp_array, z, label='Sponge layer')
    # Add lines to denote display domain
    if z0_dis != None and zf_dis != None:
        axis.axhline(y=z0_dis, color='k', linestyle='--')
        axis.axhline(y=zf_dis, color='k', linestyle='--')
    axis.set_xlabel('Amplitude')
    axis.set_ylabel(r'$z$ (m)')
    axis.set_ylim([min(z),max(z)])
    axis.legend()
    #
    axis.set_title(r'%s' %('Windows'))

if plot_windows:
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    plot_v_profile(axes, win_bf_array, win_sp_array, z, omega, z0_dis, zf_dis)
    plt.savefig('stand_alone_windows.png')

if run_sim == False:
    raise SystemExit(0)
###############################################################################
# Define equations
problem.add_equation("dt( dz(dz(foo)) - (k**2)*foo ) " \
                     " - NU*(dz(dz(dz(dz(psi)))) + (k**4)*psi) " \
                     " = (k**2)*(N0**2)*psi " \
                     " + F_term_psi - S_term_psi ")
# LHS must be first-order in ['dt'], so I'll define a temp variable
problem.add_equation("foo - dt(psi) = 0")
###############################################################################

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
solver.stop_sim_time  = T*stop_n_periods    # [s] number of simulated seconds until the sim stops
solver.stop_wall_time = 180 * 60.0          # [s] length in minutes * 60 = length in seconds, sim stops if exceeded
solver.stop_iteration = np.inf              # [] number of iterations before the simulation stops

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################
# Initial conditions
psi = solver.state['psi']
psi['g'] = 0.0

###############################################################################
# Analysis
def add_new_file_handler(snapshot_directory='snapshots', sdt=0.25):
    return solver.evaluator.add_file_handler(snapshot_directory, sim_dt=sdt, max_writes=25, mode='overwrite')

# Add file handler for snapshots and output state of variables
snapshots = add_new_file_handler('snapshots')
snapshots.add_system(solver.state)

###############################################################################
# Logger parameters
endtime_str      = 'Sim end period: %f'
time_factor      = T
adapt_dt         = False
logger_cadence   = 100
iteration_str    = 'Iteration: %i, t/T: %e, dt/T: %e'
flow_log_message = 'Max linear criterion = {0:f}'
###############################################################################
# Store data for final plot
store_this = psi
store_this.set_scales(1)
psi_gs = [np.copy(store_this['g']).real] # Plotting functions require float64, not complex128
psi_cr = [np.copy(store_this['c']).real]
psi_ci = [np.copy(store_this['c']).imag]
t_list = [solver.sim_time]
###############################################################################
# Main loop
try:
    logger.info(endtime_str %(solver.stop_sim_time/time_factor))
    logger.info('Starting loop')
    start_time = time.time()
    dt = 0.125
    while solver.proceed:
        # Adaptive time stepping controlled from switchboard
        if (adapt_dt):
            dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % 1 == 0:
            store_this.set_scales(1)
            psi_gs.append(np.copy(store_this['g']).real)
            psi_cr.append(np.copy(store_this['c']).real)
            psi_ci.append(np.copy(store_this['c']).imag)
            t_list.append(solver.sim_time)
        if solver.iteration % logger_cadence == 0:
            logger.info(iteration_str %(solver.iteration, solver.sim_time/time_factor, dt/time_factor))
            # logger.info(flow_log_message.format(flow.max(flow_name)))
            # if np.isnan(flow.max(flow_name)):
            #     raise NameError('Code blew up it seems')
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info(endtime_str %(solver.sim_time/time_factor))
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))


# Create space-time plot
psi_g_array = np.transpose(np.array(psi_gs))
psi_c_reals = np.transpose(np.array(psi_cr))
psi_c_imags = np.transpose(np.array(psi_ci))
t_array = np.array(t_list)

# Save arrays to files
arrays = {'psi_g_array':psi_g_array,
          'psi_c_reals':psi_c_reals,
          'psi_c_imags':psi_c_imags,
          't_array':t_array}
for arr in arrays:
    file = open('arrays/'+arr, "wb")        # "wb" selects the "write binary" mode
    np.save(file, arrays[arr])
    file.close

def plot_z_vs_t(z, t_array, T, w_array, win_bf_array, win_sp_array, k, m, omega, z0_dis=None, zf_dis=None, c_map='RdBu_r'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,5]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_v_profile(axes[0], win_bf_array, win_sp_array, z, omega, z0_dis, zf_dis)
    #
    xmesh, ymesh = quad_mesh(x=t_array/T, y=z)
    im = axes[1].pcolormesh(xmesh, ymesh, w_array, cmap=c_map)
    # Find max of absolute value for colorbar for limits symmetric around zero
    cmax = max(abs(w_array.flatten()))
    if cmax==0.0:
        cmax = 0.001 # to avoid the weird jump with the first frame
    # Set upper and lower limits on colorbar
    im.set_clim(-cmax, cmax)
    # Add colorbar to im
    cbar = plt.colorbar(im)#, format=ticker.FuncFormatter(latex_exp))
    cbar.ax.ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
    # Add lines to denote display domain
    if z0_dis != None and zf_dis != None:
        axes[1].axhline(y=z0_dis, color='k', linestyle='--')
        axes[1].axhline(y=zf_dis, color='k', linestyle='--')
    #
    axes[1].set_xlabel(r'$t/T$')
    axes[1].set_title(r'$\Psi$ (m$^2$/s)')
    plt.savefig('stand_alone_wave.png')

plot_z_vs_t(z, t_array, T, psi_g_array, win_bf_array, win_sp_array, k, m, omega, z0_dis, zf_dis, c_map='RdBu_r')
