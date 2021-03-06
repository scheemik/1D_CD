"""
Author: Mikhail Schee, 2020 August
Affiliation: University of Toronto Department of Physics

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

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Switches
plot_profiles = False
run_sim = True
separate_up_down = True

###############################################################################
# Run parameters
stop_n_periods = 25             # [] number of oscillation periods to run the sim for
dt             = 0.125          # [s] time step for simulation
# Displayed domain parameters
nz     = 1024                   # [] number of grid points in the z direction
zf_dis = 0.0                    # [m] the top of the displayed z domain
Lz_dis = 1.0                    # [m] the length of the z domain between forcing and sponge
z0_dis = zf_dis - Lz_dis        # [m] The bottom of the displayed domain
###############################################################################
# Physical parameters
nu          = 1.0E-6            # [m^2/s] Viscosity (momentum diffusivity)
g           = 9.81              # [m/s^2] Acceleration due to gravity
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

# print('phase speed is',omega/m,'m/s')
# print('group speed is',omega*m/(k**2 + m**2),'m/s') # cushman-roisin and beckers 13.10, pg 400

###############################################################################
# Boundary forcing window parameters
b_bf    = 1*lam_z               # [m] full width at half max of forcing window
a_bf    = 3*b_bf                # [m] forcing area, height above display domain
c_bf    = zf_dis + 0.5*a_bf     # [m] center of forcing area
tau_bf  = 1.0e-0                 # [s] time constant for boundary forcing

# Sponge layer window parameters
b_sp    = 1*lam_z               # [m] full width at half max of sponge window
a_sp    = 3*b_sp                # [m] sponge area, height below display domain
c_sp    = z0_dis - 0.5*a_sp     # [m] center of sponge area
tau_sp  = 1.0e-0                # [s] time constant for sponge layer

###############################################################################
# Simulated domain parameters
zf     = zf_dis + a_bf          # [m] top of simulated domain
z0     = z0_dis - a_sp          # [m] bottom of simulated domain
dealias= 3/2                    # [] dealiasing factor
Lz     = z0 - zf                # [m] spacing between each grid point
dz     = Lz / nz                # [m] spacing between each grid point

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
# Background profile in N_0
def BP_n_steps(n, z, z0_dis, zf_dis, th):
    """
    n           number of steps
    z           array of z values
    z0_dis      bottom of display domain
    zf_dis      top of display domain
    th          step thickness
    """
    # create blank array the same size as z
    BP_array = z*0+1
    # divide the display range for n steps
    Lz_dis = zf_dis - z0_dis
    # find step separation
    step_sep = Lz_dis / (n+1)
    for i in range(n):
        step_c   = zf_dis - (i+1)*step_sep
        step_top = step_c + (th/2)
        step_bot = step_c - (th/2)
        for j in range(len(BP_array)):
            if z[j] < step_top and z[j] > step_bot:
                BP_array[j] = 0
    return BP_array
# Background Profile for N_0
BP          = domain.new_field(name = 'BP')
BP_array    = BP_n_steps(1, z, z0_dis, zf_dis, 1.0/m)
BP['g']     = BP_array
problem.parameters['BP'] = BP

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
problem.substitutions['delta2dt_psi'] = "(dz(dz(foo)) - (k**2)*foo )"
problem.substitutions['S_term_psi'] = "win_sp * delta2dt_psi / tau_sp"

###############################################################################
# Plotting windows
def plot_v_profile(axis, z, omega=None, z0_dis=None, zf_dis=None, bp_array=None, bf_array=None, sp_array=None):
    # Plot profiles
    if type(bp_array) is np.ndarray:
        axis.plot(bp_array, z, label='Background Profile')
    if type(bf_array) is np.ndarray:
        axis.plot(bf_array, z, label='Boundary forcing')
    if type(sp_array) is np.ndarray:
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
    axis.set_title(r'%s' %('Profiles'))

if plot_profiles:
    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    plot_v_profile(axes, z, omega, z0_dis, zf_dis, BP_array, win_bf_array, win_sp_array)
    plt.savefig('stand_alone_windows.png')

if run_sim == False:
    raise SystemExit(0)
###############################################################################
# Define equations
problem.add_equation("dt( dz(dz(foo)) - (k**2)*foo ) " \
                     " - NU*(dz(dz(dz(dz(psi)))) + (k**4)*psi) " \
                     " = (k**2)*BP*(N0**2)*psi " \
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
psi.set_scales(1)
psi_gs = [np.copy(psi['g']).real] # Plotting functions require float64, not complex128
t_list = [solver.sim_time]
###############################################################################
# Main loop
try:
    logger.info(endtime_str %(solver.stop_sim_time/time_factor))
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        if (adapt_dt):
            dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % 1 == 0:
            psi.set_scales(1)
            psi_gs.append(np.copy(psi['g']).real)
            t_list.append(solver.sim_time)
        if solver.iteration % logger_cadence == 0:
            logger.info(iteration_str %(solver.iteration, solver.sim_time/time_factor, dt/time_factor))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info(endtime_str %(solver.sim_time/time_factor))
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))

###############################################################################
# Create space-time plot
psi_g_array = np.transpose(np.array(psi_gs))
t_array = np.array(t_list)

def plot_z_vs_t(z, t_array, T, w_array, bp_array=None, bf_array=None, sp_array=None, z0_dis=None, zf_dis=None, c_map='RdBu_r', title_str='1D forced wave', filename='stand_alone_wave.png'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,5]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_v_profile(axes[0], z, omega, z0_dis, zf_dis, bp_array, bf_array, sp_array)
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
    fig.suptitle(title_str)
    plt.savefig(filename)

title_str = r'$(\tau_{sp}, $FWHM$/\lambda)$=(%.2e, %d)' %(tau_sp,b_sp/lam_z)
plot_z_vs_t(z, t_array, T, psi_g_array, bp_array=BP_array, bf_array=None, sp_array=None, z0_dis=z0_dis, zf_dis=zf_dis, c_map='RdBu_r', title_str=title_str, filename='stand_alone_wave.png')

###############################################################################
# Complex demodulaion helper functions (filters)
if separate_up_down == False:
    raise SystemExit(0)

# fourier transform in time, filter negative freq's, inverse fourier transform
def FT_in_time(t, z, data, dt):
    # FT in time of the data (axis 1 is time)
    ftd = np.fft.fft(data, axis=1)
    # find relevant frequencies
    freq = np.fft.fftfreq(len(t), dt)
    f_grid, z_grid = np.meshgrid(freq, z)
    # Filter out negative frequencies
    for i in range(f_grid.shape[0]):
        for j in range(f_grid.shape[1]):
            if f_grid[i][j] < 0.0:
                # Gets rid of negative freq's
                ftd[i][j] = 0
            else:
                # Corrects for lost amplitude
                ftd[i][j] = ftd[i][j] * 2.0
    # inverse fourier transform in time of the data
    iftd = np.fft.ifft(ftd, axis=1)
    #   a complex valued signal where iftd.real == data, or close enough
    return iftd

# fourier transform in spatial dimension (z)
#   similar to FT in time, but switch dimensions around
def FT_in_space(t, z, data, dz):
    # FT in space (z) of the data (axis 0 is z) for positive wave numbers
    fzdp = np.fft.fft(data, axis=0)
    # make a copy for the negative wave numbers
    fzdn = fzdp.copy()
    # find relevant wavenumbers
    k_zs = np.fft.fftfreq(len(z), dz)
    t_grid, k_grid = np.meshgrid(t, k_zs)
    # Filter out one half of wavenumbers to separate up and down
    for i in range(k_grid.shape[0]):
        for j in range(k_grid.shape[1]):
            if k_grid[i][j] > 0.0:
                # for down, remove values for positive wave numbers
                fzdn[i][j] = 0.0
            else:
                # for up, remove values for negative wave numbers
                fzdp[i][j] = 0.0
    # inverse fourier transform in space (z)
    ifzdp = np.fft.ifft(fzdp, axis=0)
    ifzdn = np.fft.ifft(fzdn, axis=0)
    return ifzdp, ifzdn

###############################################################################
# Separating up and down through complex demodulation (see Mercier et al. 2008)

## Step 2
ift_z_y_p, ift_z_y_n = FT_in_space(t_array, z, psi_g_array, dz)
## Step 1
up_f = FT_in_time(t_array, z, ift_z_y_p, dt)
dn_f = FT_in_time(t_array, z, ift_z_y_n, dt)
# Get up and down fields as F = |mag_f| * exp(i*phi_f)
up_field = up_f.real * np.exp(np.real(1j * up_f.imag))
dn_field = dn_f.real * np.exp(np.real(1j * dn_f.imag))

# Plot upward-propagating waves
plot_z_vs_t(z, t_array, T, up_field, bp_array=BP_array, bf_array=None, sp_array=None, z0_dis=z0_dis, zf_dis=zf_dis, c_map='RdBu_r', title_str=title_str, filename='stand_alone_up.png')
# Plow downward-propagating waves
plot_z_vs_t(z, t_array, T, dn_field, bp_array=BP_array, bf_array=None, sp_array=None, z0_dis=z0_dis, zf_dis=zf_dis, c_map='RdBu_r', title_str=title_str, filename='stand_alone_dn.png')
