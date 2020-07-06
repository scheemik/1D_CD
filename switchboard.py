# Switchboard for Dedalus experiment
"""
Description:
This is the switchboard for the Dedalus experiment. This file contains parameter values called by multiple scripts. This insures they all get the same values
"""

import numpy as np
from dedalus import public as de
# import sys
# sys.path.append("../") # Adds higher directory to python modules path
# import helper_functions as hf

###############################################################################
# Main parameters, the ones I'll change a lot. Many more below

# Run parameters
stop_n_periods = 40             # [] oscillation periods

# Displayed domain parameters
nz     = 1024                   # [] number of grid points in the z direction
zf_dis = 0.0                    # [m] the top of the displayed z domain
Lz_dis = 1.0                    # [m] the length of the z domain between forcing and sponge
#
z0_dis = zf_dis - Lz_dis        # [m] The bottom of the displayed domain

# Problem parameters
A       = 2.0e-4                # []            Amplitude of boundary forcing
N_0     = 1.0                   # [rad/s]       Reference stratification
f_0     = 0.000                 # [s^-1]        Reference Coriolis parameter
set_case= 1                     # Picks combination of variables to set in switch below
if set_case == 1:
    lam_z   = Lz_dis / 4.0          # [m]           Vertical wavelength
    lam_x   = lam_z                 # [m]           Horizontal wavelength
    #
    m       = 2*np.pi / lam_z       # [m^-1]        Vertical wavenumber
    k       = 2*np.pi / lam_x       # [m^-1]        Horizontal wavenumber
    k_total = np.sqrt(k**2 + m**2)  # [m^-1]        Total wavenumber
    theta   = np.arctan(m/k)        # [rad]         Propagation angle from vertical
    omega   = N_0 * np.cos(theta)   # [rad s^-1]    Wave frequency
elif set_case == 2:
    lam_z   = Lz_dis / 8.0          # [m]           Vertical wavelength
    omega   = 0.7071                # [rad s^-1]    Wave frequency
    #
    m       = 2*np.pi / lam_z       # [m^-1]        Vertical wavenumber
    theta   = np.arccos(omega/N_0)  # [rad]         Propagation angle from vertical
    k       = m/np.tan(theta)       # [m^-1]        Horizontal wavenumber
    k_total = np.sqrt(k**2 + m**2)  # [m^-1]        Total wavenumber
    lam_x   = 2*np.pi / k           # [m]           Horizontal wavelength
elif set_case == 3:
    lam_z   = Lz_dis / 8.0          # [m]           Vertical wavelength
    theta   = 0.7854 # 45deg        # [rad]         Propagation angle from vertical
    #
    m       = 2*np.pi / lam_z       # [m^-1]        Vertical wavenumber
    k       = m/np.tan(theta)       # [m^-1]        Horizontal wavenumber
    k_total = np.sqrt(k**2 + m**2)  # [m^-1]        Total wavenumber
    lam_x   = 2*np.pi / k           # [m]           Horizontal wavelength
    omega   = N_0 * np.cos(theta)   # [rad s^-1]    Wave frequency

T       = 2*np.pi / omega       # [s]           Wave period

###############################################################################
# ON / OFF Switches

# Determine whether adaptive time stepping is on or off
adapt_dt                = False

# Simulation starting conditions
temporal_ramp           = True  # If True, slowly ramps the amplitude up to avoid shock wave
nT                      = 3.0   # number of oscillation periods long the ramp lasts
steady_state_start      = False  # If True, automatically turns off temporal ramp

# Terms in equations of motion
viscous_term            = True
pressure_term           = True
advection_term          = False
buoyancy_term           = True
diffusivity_term        = True
rotation_term           = False
# Diffusion / dissipation of reflections
use_sponge              = False
use_rayleigh_friction   = False
boundary_forcing_region = False  # If False, waves will be forced over entire domain

# Measurements
take_ef_comp  = False # Energy flux terms recorded separately
# Records snapshots of total vertical energy flux
take_ef_snaps = False # Total energy flux recorded

###############################################################################
# Boundary forcing window 1
a_bf    = 1.0                   # [] amplitude ("height") of the forcing window
b_bf    = lam_z                 # [m] full width at half max of forcing window
buff_bf = 1.5*b_bf              # [m] distance from top boundary to center of forcing
tau_bf  = 1.0e0                 # [s] time constant for boundary forcing

# Sponge layer window 1
a_sp    = 1.0                   # [] amplitude ("height") of the sponge window
b_sp    = lam_z                 # [m] full width at half max of sponge window
buff_sp = 1.5*b_sp              # [m] distance from bottom boundary to center of sponge
tau_sp  = 1.0e-1                # [s] time constant for sponge layer

###############################################################################
###############################################################################
# Simulated domain parameters
zf     = zf_dis + 2*buff_bf     # [m] top of simulated domain
z0     = z0_dis - 2*buff_sp     # [m] bottom of simulated domain
Lz     = zf - z0                # [m] length of simulated domain
dealias= 3/2                    # [] dealiasing factor

# Bases and domain
z_basis = de.Fourier('z', nz, interval=(z0, zf), dealias=dealias)
domain = de.Domain([z_basis], grid_dtype=np.complex128)#float64)
# Z grid
z_da = domain.grid(0, scales=domain.dealias)
z = domain.grid(0)
# Getting wavenumbers
ks = z_basis.wavenumbers

# Background profile in N_0
n_steps = 0
step_th = 1.0/m

# Boundary forcing window 2
c_bf    = zf - buff_bf          # [m] location of center of boundary forcing window
if boundary_forcing_region == True:
    win_bf_array = a_bf*np.exp(-4*np.log(2)*((z - c_bf)/b_bf)**2)
else:
    win_bf_array = z*0.0 + 1.0  # Forcing over entire domain
    use_sponge = False          # Having sp layer and full domain forcing causes problems

# Sponge layer window 2
c_sp    = z0 + buff_sp          # [m] location of center of sponge window window
if use_sponge == True:
    win_sp_array = a_sp*np.exp(-4*np.log(2)*((z - c_sp)/b_sp)**2)
else:
    win_sp_array = z * 0.0      # No sponge

###############################################################################
# Run parameters
dt              = 0.125         # [s] initial time step size
snap_dt         = 32*dt          # [s] time step size for snapshots
snap_max_writes = 100           # [] max number of writes per snapshot file
fh_mode         = 'overwrite'   # file handling mode, either 'overwrite' or 'append'
# Stopping conditions for the simulation
sim_time_stop  =T*stop_n_periods# [s] number of simulated seconds until the sim stops
stop_wall_time = 180 * 60.0     # [s] length in minutes * 60 = length in seconds, sim stops if exceeded
stop_iteration = np.inf         # [] number of iterations before the simulation stops

# temporal ramp - probably should remove, already defined above
# temporal_ramp = True
# nT = 3.0

###############################################################################
# Physical parameters
nu          = 1.0E-6        # [m^2/s] Viscosity (momentum diffusivity)
kappa       = 1.4E-7        # [m^2/s] Thermal diffusivity
Prandtl     = nu / kappa    # [] Prandtl number, about 7.0 for water at 20 C
Rayleigh    = 1e6
g           = 9.81          # [m/s^2] Acceleration due to gravity

###############################################################################
# Plotting parameters

plot_spacetime = True
plot_wavespace = True
plot_windows   = True
# If true, plot will include full simulated domain, if false, just the display domain
plot_full_domain = True

# Dark mode on or off (ideally would make plots that have white text and alpha background)
dark_mode = False
cmap = 'RdBu_r'
# import colorcet as cc
# cmap = cc.CET_D4

# Presentation mode on or off (increases size of fonts and contrast of colors)
presenting = False

# Vertical profile and Wave field animation
# If True, plots b, p, u, and w. If false, plots profile and w
plot_all_variables = False
# If True, the sponge layer plot will be plotted to the right of the animation
plot_sponge        = False
# If True, the Rayleigh friction plot will replace background profile
plot_rf            = False
plot_twin          = False

# Auxiliary snapshot plots
plot_ef_total = True
plot_ef_comps = False

# Miscellaneous
# Fudge factor to make plots look nicer
buffer = 0.04
# Extra buffer for a constant vertical profile
extra_buffer = 0.5
# Display ratio of vertical profile plot
vp_dis_ratio = 2.0 # Profile plot gets skinnier as this goes up
# The number of ticks on the top color bar
n_clrbar_ticks = 3
# Overall font size of plots
font_size   = 12
scale       = 2.5
dpi         = 100

# Animation parameters
fps = 20

###############################################################################
# Snapshot parameters
snapshots_dir   = 'snapshots'
snap_dt         = 0.25
snap_max_writes = 25

# Background profile snapshot parameters
take_bp_snaps   = True
# Sponge layer snapshot parameters
take_sl_snaps   = True
# Rayleigh friction snapshot parameters
take_rf_snaps   = True

# Define all vertical profile snapshots in an array of dictionaries
#   Meant for profiles that are constant in time
take_vp_snaps = True
vp_snap_dir = 'vp_snapshots'
vp_snap_dicts = [
           {'take_vp_snaps':   take_bp_snaps,
            'vp_task':         "N0*BP",
            'vp_task_name':    'bp'},

           {'take_vp_snaps':   take_sl_snaps,
            'vp_task':         "SL",
            'vp_task_name':    'sl'},

           {'take_vp_snaps':   take_rf_snaps,
            'vp_task':         "RF",
            'vp_task_name':    'rf'}
            ]

# Auxiliary snapshot directory
aux_snap_dir = 'aux_snapshots'

###############################################################################
# Initial conditions
if steady_state_start:
    psi_initial = A * np.sin(m*z)
else:
    psi_initial = 0.0

###############################################################################
# CFL parameters
CFL_cadence     = 10
CFL_safety      = 1
CFL_max_change  = 1.5
CFL_min_change  = 0.5
CFL_max_dt      = 0.125
CFL_threshold   = 0.05
###############################################################################
# Flow properties
flow_cadence    = 10
flow_property   = "(k*u + m*w)/omega"
flow_name       = 'Lin_Criterion'
###############################################################################
# Logger parameters
endtime_str     = 'Sim end period: %f'
logger_cadence  = 100
iteration_str   = 'Iteration: %i, t/T: %e, dt/T: %e'
flow_log_message= 'Max linear criterion = {0:f}'
###############################################################################

###############################################################################
################    Shouldn't need to edit below here    #####################
###############################################################################
#
# ###############################################################################
# # Imports for preparing physics modules
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# from shutil import copy2, rmtree
# import os
# import sys
# p_module_dir = './_modules_physics/'
#
# ###############################################################################
# # Boundary forcing
#
# # Need to add the path before every import
# sys.path.insert(0, p_module_dir)
# import boundary_forcing as bf
# # See boundary forcing file for the meaning of these variables
# N_0     = bf.N_0        # [rad s^-1]
# k       = bf.k          # [m^-1]
# omega   = bf.omega      # [rad s^-1]
# theta   = bf.theta      # [rad]
# k_x     = bf.k_x        # [m^-1]
# k_z     = bf.k_z        # [m^-1]
# lam_x   = bf.lam_x      # [m]
# lam_z   = bf.lam_z      # [m]
# T       = bf.T          # [s]
# A       = bf.A          # []
# nT      = bf.nT         # []
# PolRel  = bf.PolRel     # Dictionary of coefficients for variables
# # Dedalus specific string substitutions
# bf_slope= bf.bf_slope
# bfl_edge= bf.bfl_edge
# bfr_edge= bf.bfr_edge
# window  = bf.window
# ramp    = bf.ramp
# fu      = bf.fu
# fw      = bf.fw
# fb      = bf.fb
# fp      = bf.fp
#
# # Calculate stop_sim_time if use_stop_sim_time=False
# if use_stop_sim_time == False:
#     stop_sim_time = stop_n_periods * T
# # Set restart simulation parameters
# restart_add_time = stop_sim_time
# restart_file  = 'restart.h5'
#
# ###############################################################################
# # Equations of Motion and Boundary Conditions
#
# # Need to add the path before every import
# sys.path.insert(0, p_module_dir)
# import eqs_and_bcs as eq
# # Equations of motion
# eq1_mc      = eq.eq1_mc
# eq2_es      = eq.eq2_es
# eq3_hm      = eq.eq3_hm
# eq4_vm      = eq.eq4_vm
# eq5_bz      = eq.eq5_bz
# eq6_uz      = eq.eq6_uz
# eq7_wz      = eq.eq7_wz
# # Boundary contitions
# bc1         = eq.bc1_u_bot
# bc2         = eq.bc2_u_top
# bc3         = eq.bc3_w_bot
# bc3_cond    = eq.bc3_w_cond
# bc4         = eq.bc4_w_top
# bc5         = eq.bc5_b_bot
# bc6         = eq.bc6_b_top
# bc7         = eq.bc7_p_bot
# bc7_cond    = eq.bc7_p_cond
#
#
# ###############################################################################
# # Sponge Layer Profile
#
# # Need to add the path before every import
# sys.path.insert(0, p_module_dir)
# import sponge_layer as sl
# if take_sl_snaps==False:
#     plot_sponge = False
# if use_sponge==True:
#     # The sponge layer profile generator function
#     build_sl_array = sl.build_sl_array
#     # Redefine the vertical domain length if need be
#     L_z = L_z + sl.sl_thickness
#     z_sim_f = sl.z_sl_bot
#     if dis_eq_sim==True:
#         L_z_dis = L_z
# else:
#     build_sl_array = sl.build_no_sl_array
#
# ###############################################################################
# # Rayleigh Friction Profile
#
# # Need to add the path before every import
# sys.path.insert(0, p_module_dir)
# import rayleigh_friction as rf
# if take_rf_snaps==False:
#     plot_rf = False
# if use_rayleigh_friction==True:
#     # The sponge layer profile generator function
#     build_rf_array = rf.build_rf_array
#     # Redefine the vertical domain length if need be
#     if use_sponge==False:
#         L_z = L_z + rf.rf_thickness
#         z_sim_f = rf.z_rf_bot
#         if dis_eq_sim==True:
#             L_z_dis = L_z
# else:
#     build_rf_array = rf.build_no_rf_array
#
# ###############################################################################
# # Energy Flux Measurements
#
# # Need to add the path before every import
# sys.path.insert(0, p_module_dir)
# import energy_flux as ef
# if take_ef_snaps==False and take_ef_comp==False:
#     plot_ef = False
# ef_snap_dicts = ef.ef_snap_dicts
#
# ###############################################################################
# # Cleaning up the _modules-physics directory tree
# for some_dir in os.scandir(p_module_dir):
#     # Iterate through subdirectories in _modules-physics
#     if some_dir.is_dir():
#         dir=some_dir.name
#         # If the directory isn't __pycache__, then delete it
#         if dir!='__pycache__':
#             dir_path = p_module_dir + dir
#             rmtree(dir_path, ignore_errors=True)
