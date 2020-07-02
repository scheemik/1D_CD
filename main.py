"""
# 1D Bousinessq equations:
#
# dz(w) + iku = 0
# dt(b) - ka(dzz-k^2)b = -N^2*w -(wdz + iku)b
# dt(u) - nu(dzz-k^2)u + ikp = -(wdz + iku)u
# dt(w) - nu(dzz-k^2)w + dz(p) - b = -(wdz + iku)w
#
# where k is the horizontal wavenumber

1D Bousinessq streamfunction equation:

dtt(dzz(psi) - k^2 psi) - k^2 N^2 psi + f_0 dzz(psi) - nu(dzz(psi) + k^4 psi)

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
#kappa       = sbp.kappa         # [m^2/s] Thermal diffusivity
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

###############################################################################
# Bases and domain
z_basis     = sbp.z_basis
domain      = sbp.domain
# Z grid
z_da        = sbp.z_da
z           = sbp.z
# Getting wavenumbers
ks          = sbp.ks

# Define problem
# problem = de.IVP(domain, variables=['b', 'p', 'u', 'w'])
# problem.parameters['NU'] = nu
# problem.parameters['KA'] = kappa
# problem.parameters['N0'] = N_0
problem = de.IVP(domain, variables=['psi', 'foo', 'psi_masked'])
problem.parameters['NU'] = nu
problem.parameters['f0'] = f_0
problem.parameters['N0'] = N_0

###############################################################################
# Forcing from the boundary

# Boundary forcing parameters
# A = sbp.A
problem.parameters['A']     = sbp.A
problem.parameters['k']     = k
problem.parameters['m']     = m
problem.parameters['omega'] = omega

# # Polarization relation from boundary forcing file
# PolRel = {'u': -A*(g*omega*m)/(N_0**2*k),
#           'w':  A*(g*omega)/(N_0**2),
#           'b':  A*g}
#         # 'p': -A*(g*m)/(k**2+m**2)}
# # Creating forcing amplitudes
# for fld in ['u', 'w', 'b']:#, 'p']:
#     BF = domain.new_field()
#     BF['g'] = PolRel[fld]
#     problem.parameters['BF' + fld] = BF  # pass function in as a parameter.
#     del BF

# Temporal ramp for boundary forcing
if sbp.temporal_ramp:
    problem.parameters['T']     = T       # [s] period of oscillation
    problem.parameters['nT']    = sbp.nT  # [] number of periods for the ramp
    problem.substitutions['ramp']   = "(1/2)*(tanh(4*t/(nT*T) - 2) + 1)"
else:
    problem.substitutions['ramp']   = "1"
# # Substitutions for boundary forcing (see C-R & B eq 13.7)
# problem.substitutions['fu'] = "BFu*sin(m*z - omega*t)*ramp"
# problem.substitutions['fw'] = "BFw*sin(m*z - omega*t)*ramp"
# problem.substitutions['fb'] = "BFb*cos(m*z - omega*t)*ramp"
# # problem.substitutions['fp'] = "BFp*sin(m*z - omega*t)*ramp"
problem.substitutions['f_psi'] = "A*sin(m*z - omega*t)*ramp"

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
problem.parameters['tau_bf'] = sbp.tau_bf # [s] time constant for sponge layer

# Creating forcing terms
# for fld in ['u', 'w', 'b']:#, 'p']:
#     # terms will be = win_bf * (f(psi) - psi)
#     problem.substitutions['F_term_' + fld] = "win_bf * (f"+fld+" - "+fld+")/tau_bf"
# problem.substitutions['bf_term'] = " win_bf * (a*sin(-m*z - omega*t) - w)*ramp"
#   Terms will be = win_bf * (f(psi) - psi)
problem.substitutions['F_term_psi'] = "win_bf * (f_psi - psi)/tau_bf"

###############################################################################
# Sponge window
win_sp      = domain.new_field(name = 'win_sp')
win_sp['g'] = sbp.win_sp_array
problem.parameters['win_sp'] = win_sp
problem.parameters['tau_sp'] = sbp.tau_sp # [s] time constant for sponge layer

# Creating sponge terms
# for fld in ['u', 'w', 'b']:#, 'p']:
#     problem.substitutions['S_term_' + fld] = "win_sp * "+fld+" / tau_sp"
# problem.substitutions['sp_term'] = "-win_sp * w / tau"
problem.substitutions['S_term_psi'] = "win_sp * psi / tau_sp"

###############################################################################
# Plotting windows
if sbp.plot_windows:
    hf.plot_v_profiles(BP_array, sbp.win_bf_array, sbp.win_sp_array, z, omega, sbp.z0_dis, sbp.zf_dis, title_str=run_name)

###############################################################################
# Define equations
#   Non-linear terms and NCC need to be on RHS
#   BP, F_terms, S_terms are NCC, Advection is nonlinear

# problem.add_equation("dz(w) - k*u = 0")
# problem.add_equation("dt(b) - KA*(dz(dz(b)) - (k**2)*b) " \
#                      " = -((N0*BP)**2)*w - (w*dz(b) + k*u*b) " \
#                      " + F_term_b - S_term_b ")
# problem.add_equation("dt(u) - NU*(dz(dz(u)) - (k**2)*u) + k*p " \
#                      " = - (w*dz(u) + k*u*u) " \
#                      " + F_term_u - S_term_u ")
# problem.add_equation("dt(w) - NU*(dz(dz(w)) - (k**2)*w) + dz(p) - b " \
#                      " = - (w*dz(w) + k*u*w) " \
#                      " + F_term_w - S_term_w ")

problem.add_equation("dt( dz(dz(foo)) - (k**2)*foo ) + f0*(dz(dz(psi))) " \
                     " - NU*(dz(dz(dz(dz(psi)))) + (k**4)*psi) " \
                     " = (k**2)*((N0*BP)**2)*psi " \
                     " + F_term_psi - S_term_psi ")
# LHS must be first-order in ['dt'], so I'll define a temp variable
problem.add_equation("foo - dt(psi) = 0")
# Create copy of psi which is masked to the display domain
problem.add_equation("psi_masked = DD_mask*psi")

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')
solver.stop_sim_time  = sbp.sim_time_stop
solver.stop_wall_time = sbp.stop_wall_time
solver.stop_iteration = sbp.stop_iteration

# Above code modified from here: https://groups.google.com/forum/#!searchin/dedalus-users/%22wave$20equation%22%7Csort:date/dedalus-users/TJEOwHEDghU/g2x00YGaAwAJ

###############################################################################

# Initial conditions
# b = solver.state['b']
# u = solver.state['u']
# w = solver.state['w']
psi = solver.state['psi']
psi_masked = solver.state['psi_masked']

# b['g'] = 0.0
# u['g'] = 0.0
# w['g'] = 0.0
psi['g'] = 0.0
psi_masked['g'] = 0.0

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
# w.set_scales(1)
# w_list = [np.copy(w['g'])]
# t_list = [solver.sim_time]
store_this = psi_masked
store_this.set_scales(1)
psi_gs = [np.copy(store_this['g']).real] # Plotting functions require float64, not complex128
psi_cr = [np.copy(store_this['c']).real]
psi_ci = [np.copy(store_this['c']).imag]
# psi.set_scales(1)
# psi_gs = [np.copy(psi['g']).real] # Plotting functions require float64, not complex128
# psi_cr = [np.copy(psi['c']).real]
# psi_ci = [np.copy(psi['c']).imag]
t_list = [solver.sim_time]
###############################################################################
# Main loop
try:
    logger.info(endtime_str %(solver.stop_sim_time/time_factor))
    logger.info('Starting loop')
    start_time = time.time()
    dt = sbp.dt
    while solver.proceed:
        # Adaptive time stepping controlled from switchboard
        # if (adapt_dt):
        #     dt = CFL.compute_dt()
        solver.step(dt)
        if solver.iteration % 1 == 0:
            # w.set_scales(1)
            # w_list.append(np.copy(w['g']))
            # t_list.append(solver.sim_time)
            store_this.set_scales(1)
            psi_gs.append(np.copy(store_this['g']).real)
            psi_cr.append(np.copy(store_this['c']).real)
            psi_ci.append(np.copy(store_this['c']).imag)
            # si.set_scales(1)
            # psi_gs.append(np.copy(psi['g']).real)
            # psi_cr.append(np.copy(psi['c']).real)
            # psi_ci.append(np.copy(psi['c']).imag)
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
# w_array = np.transpose(np.array(w_list))
# t_array = np.array(t_list)
psi_g_array = np.transpose(np.array(psi_gs))
psi_c_reals = np.transpose(np.array(psi_cr))
psi_c_imags = np.transpose(np.array(psi_ci))
t_array = np.array(t_list)

# Save arrays to files
arrays = {'psi_g_array':psi_g_array,
          'psi_c_reals':psi_c_reals,
          'psi_c_imags':psi_c_imags,
          't_array':t_array,
          'BP_array':BP_array}
for arr in arrays:
    file = open('arrays/'+arr, "wb")
    np.save(file, arrays[arr])
    file.close
