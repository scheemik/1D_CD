"""

1D Bousinessq streamfunction equation:

dt^2[dz^2(psi) - k^2 psi] - k^2 N^2 psi + f_0 dz^2(psi) - nu[dz^4(psi) + k^4 psi] = 0

where k is the horizontal wavenumber, N is stratification,
    f_0 is the Coriolis parameter, and nu is the viscosity

This script should be ran serially (because it is 1D).

"""

import numpy as np
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.extras.plot_tools import quad_mesh, pad_limits
from dedalus.core.operators import GeneralFunction

import logging
logger = logging.getLogger(__name__)

###############################################################################
# Checking command line arguments
import sys
# Arguments must be passed in the correct order
arg_array = sys.argv
# argv[0] is the name of this file
run_name = str(arg_array[1])
switchboard = str(arg_array[2])

# Add functions in helper file
import helper_functions as hf

###############################################################################
# Import SwitchBoard Parameters (sbp)
#   This import assumes the switchboard is in the same directory as the core code
import switchboard as sbp
# Physical parameters
nu          = sbp.nu            # [m^2/s] Viscosity (momentum diffusivity)
f_0         = sbp.f_0           # [s^-1]        Reference Coriolis parameter
g           = sbp.g             # [m/s^2] Acceleration due to gravity
# Problem parameters
N_0         = sbp.N_0           # [rad/s]       Reference stratification
lam_x       = sbp.lam_x         # [m]           Horizontal wavelength
lam_z       = sbp.lam_z         # [m]           Vertical wavelength
k           = sbp.k             # [m^-1]        Horizontal wavenumber
m           = sbp.m             # [m^-1]        Vertical wavenumber
k_total     = sbp.k_total       # [m^-1]        Total wavenumber
theta       = sbp.theta         # [rad]         Propagation angle from vertical
omega       = sbp.omega         # [rad s^-1]    Wave frequency
T           = sbp.T             # [s]           Wave period
print('phase speed is',omega/m,'m/s')
print('group speed is',omega*m/(k**2 + m**2),'m/s') # cushman-roisin and beckers 13.10, pg 400

###############################################################################
# Bases and domain
z_basis     = sbp.z_basis
domain      = sbp.domain
# Z grid
z_da        = sbp.z_da
z           = sbp.z
nz          = sbp.nz
dz          = sbp.dz
# Getting wavenumbers
ks          = sbp.ks

ks_mod = np.copy(np.array(ks))
kfile = open('ks_mod', "w")        # "wb" selects the "write binary" mode
np.savetxt(kfile, hf.sort_k_coeffs(ks_mod, nz))
kfile.close

# Define problem
problem = de.IVP(domain, variables=['psi', 'foo', 'psi_masked'])
problem.parameters['NU'] = nu
problem.parameters['f0'] = f_0
problem.parameters['N0'] = N_0

###############################################################################
# Forcing from the boundary

# Boundary forcing parameters
problem.parameters['A']     = sbp.A
problem.parameters['k']     = k
problem.parameters['m']     = m
problem.parameters['omega'] = omega

# Temporal ramp for boundary forcing
if sbp.temporal_ramp:
    problem.parameters['T']     = T       # [s] period of oscillation
    problem.parameters['nT']    = sbp.nT  # [] number of periods for the ramp
    problem.substitutions['ramp']   = "(1/2)*(tanh(4*t/(nT*T) - 2) + 1)"
else:
    problem.substitutions['ramp']   = "1"
# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['f_psi'] = "(A*sin(m*z - omega*t))*ramp"

###############################################################################
# Background Profile for N_0
BP = domain.new_field(name = 'BP')
BP_array = hf.BP_n_steps(sbp.n_steps, z, sbp.z0_dis, sbp.zf_dis, sbp.step_th)
BP['g'] = BP_array
problem.parameters['BP'] = BP

###############################################################################
# Masking to keep just display domain
DD_mask = domain.new_field(name = 'DD_mask')
DD_array = hf.make_DD_mask(z, sbp.z0_dis, sbp.zf_dis)
DD_mask['g'] = DD_array
problem.parameters['DD_mask'] = DD_mask

###############################################################################
# Boundary forcing window
win_bf = domain.new_field(name = 'win_bf')
win_bf['g'] = sbp.win_bf_array
problem.parameters['win_bf'] = win_bf
problem.parameters['tau_bf'] = sbp.tau_bf # [s] time constant for boundary forcing

# Creating forcing terms
problem.substitutions['F_term_psi'] = "win_bf * (f_psi - psi)/tau_bf"

###############################################################################
# Sponge window
win_sp      = domain.new_field(name = 'win_sp')
win_sp['g'] = sbp.win_sp_array
problem.parameters['win_sp'] = win_sp
problem.parameters['tau_sp'] = sbp.tau_sp # [s] time constant for sponge layer

# Creating sponge terms
#problem.substitutions['S_term_psi'] = "win_sp * psi / tau_sp"
problem.substitutions['nabla2dt_psi'] = "(dz(dz(foo)) - (k**2)*foo )"
problem.substitutions['S_term_psi'] = "win_sp * nabla2dt_psi / tau_sp"

###############################################################################
# Plotting windows
if sbp.plot_windows:
    hf.plot_v_profiles(BP_array, sbp.win_bf_array, sbp.win_sp_array, z, omega, sbp.z0_dis, sbp.zf_dis, title_str=run_name)

###############################################################################
# Define equations
problem.add_equation("dt( dz(dz(foo)) - (k**2)*foo ) + f0*(dz(dz(psi))) " \
                     " - NU*(dz(dz(dz(dz(psi)))) + (k**4)*psi) " \
                     " = (k**2)*((N0*BP)**2)*psi " \
                     " + F_term_psi - S_term_psi ")
# LHS must be first-order in ['dt'], so I'll define a temp variable
problem.add_equation("foo - dt(psi) = 0")
# Create copy of psi which is masked to the display domain
problem.add_equation("psi_masked = DD_mask*psi")
###############################################################################

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
solver.stop_sim_time  = sbp.sim_time_stop
solver.stop_wall_time = sbp.stop_wall_time
solver.stop_iteration = sbp.stop_iteration

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################

# Initial conditions
psi = solver.state['psi']
psi_masked = solver.state['psi_masked']

psi['g']        = sbp.psi_initial
psi_masked['g'] = sbp.psi_initial * DD_array


###############################################################################
# Analysis
def add_new_file_handler(snapshot_directory='snapshots/new', sdt=sbp.snap_dt):
    return solver.evaluator.add_file_handler(snapshot_directory, sim_dt=sdt, max_writes=sbp.snap_max_writes, mode=sbp.fh_mode)

# Add file handler for snapshots and output state of variables
snapshots = add_new_file_handler('snapshots')
snapshots.add_system(solver.state)

###############################################################################
# CFL
# CFL = flow_tools.CFL(solver, initial_dt=sbp.dt, cadence=sbp.CFL_cadence,
#                      safety=sbp.CFL_safety, max_change=sbp.CFL_max_change,
#                      min_change=sbp.CFL_min_change, max_dt=sbp.CFL_max_dt,
#                      threshold=sbp.CFL_threshold)
# CFL.add_velocities(('w'))
###############################################################################
# Flow properties
# flow_name       = sbp.flow_name
# flow            = flow_tools.GlobalFlowProperty(solver, cadence=sbp.flow_cadence)
# flow.add_property(sbp.flow_property, name=flow_name)
###############################################################################
# Logger parameters
endtime_str     = sbp.endtime_str
time_factor     = T
adapt_dt        = sbp.adapt_dt
logger_cadence  = sbp.logger_cadence
iteration_str   = sbp.iteration_str
flow_log_message= sbp.flow_log_message
###############################################################################
# Store data for final plot
store_this = psi #psi_masked
store_this.set_scales(1)
psi_gs = [np.copy(store_this['g']).real] # Plotting functions require float64, not complex128
psi_cr = [np.copy(store_this['c']).real]
psi_ci = [np.copy(store_this['c']).imag]
t_list = [solver.sim_time]
###############################################################################
# Filtering in wavenumber space

Filter_ks = domain.new_field()
new_f_psi = domain.new_field()

Filter_ks['c'][ks==0] = 0

def filter_psi():
    psi['c'][1] = 0j
    #de.operators.GeneralFunction(domain, layout='c', func=Filter_ks*psi, out=new_f_psi)
# filter_psi()
# print('new_f_psi:')
# print(new_f_psi['g'].shape)
# new_psi_g = [np.copy(new_f_psi['g']).real]
###############################################################################
# Main loop
try:
    logger.info(endtime_str %(solver.stop_sim_time/time_factor))
    logger.info('Starting loop')
    start_time = time.time()
    dt = sbp.dt
    while solver.proceed:
        # Adaptive time stepping controlled from switchboard
        if (adapt_dt):
            dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % 1 == 0:
            store_this.set_scales(1)
            filter_psi()
            this_psi_g = np.copy(store_this['g']).real
            this_psi_c = np.copy(store_this['c']).real
            psi_gs.append(this_psi_g)

            # new_psi_g.append(np.copy(new_f_psi['g']).real)

            psi_cr.append(hf.sort_k_coeffs(np.copy(store_this['c']).real, nz))
            psi_ci.append(hf.sort_k_coeffs(np.copy(store_this['c']).imag, nz))
            t_list.append(solver.sim_time)
        if solver.iteration % logger_cadence == 0:
            #print(this_psi_c)
            #print('max in filtered psi',max(abs(this_psi_c)))
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
print('psi_g_array:')
print(psi_g_array.shape)
# new_psi_array = np.transpose(np.array(new_psi_g))
# print('new_psi_array:')
# print(new_psi_array.shape)
psi_c_reals = np.transpose(np.array(psi_cr))
psi_c_imags = np.transpose(np.array(psi_ci))
t_array = np.array(t_list)

def FT_in_space(t, z, data, dz):
    # FT in space (z) of the data (axis 1 is z) for positive k_z
    A = np.fft.fft(data, axis=1)
    # make a copy for the negative k_z
    B = np.copy(A)
    # find relevant wave numbers
    k_zs = np.fft.fftfreq(len(z), dz)
    kz_grid, t_grid = np.meshgrid(k_zs, t, indexing='ij')
    # Create mask to keep positive k_zs, relies on integer arithmetic
    kz_p_mask = np.ceil((np.sign(kz_grid) + 1.0)/2)
    #  -1 becomes 0, negative kzs are masked out
    #   0 becomes 1, no change to kzs of 0 - masked out
    #   1 becomes 1, no change to positive kzs
    A = A * kz_p_mask
    # Create mask to keep negative k_xs - relies on integer arithmetic
    kz_n_mask = np.abs(np.ceil((np.sign(kz_grid) - 1.0)/2))
    #  -1 becomes 1, no change to negative kzs
    #   0 becomes 1, no change to kzs of 0
    #   1 becomes 0, positive kzs are masked out
    B = B * kz_n_mask
    # inverse fourier transform in space (x)
    A_zp = np.fft.ifft(A, axis=1)
    B_zn = np.fft.ifft(B, axis=1)
    return A_zp.real, B_zn.real

#psi_A, psi_B = FT_in_space(t_array, z, psi_g_array, dz)

# Save arrays to files
arrays = {'psi_g_array':psi_g_array,
          'psi_c_reals':psi_c_reals,
          'psi_c_imags':psi_c_imags,
          't_array':t_array,
          'BP_array':BP_array}
for arr in arrays:
    file = open('arrays/'+arr, "wb")        # "wb" selects the "write binary" mode
    np.save(file, arrays[arr])
    file.close
