"""

This script plots:
    z vs. t with Psi in a heatmap
    k vs. t with Psi in a heatmap in both real and imaginary parts

"""

import numpy as np
import matplotlib.pyplot as plt

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

###############################################################################
# Get depth and wavenumber axes
z           = sbp.z
ks          = sbp.ks


# Save arrays to files
arrays = {'psi_g_array':[],
          'psi_c_reals':[],
          'psi_c_imags':[],
          't_array':[],
          'BP_array':[]}
for arr in arrays:
    file = open('arrays/'+arr, "rb")
    arrays[arr] = np.load(file)
    file.close

if sbp.plot_spacetime:
    hf.plot_z_vs_t(z, arrays['t_array'], T, arrays['psi_g_array'], arrays['BP_array'], k, m, omega, sbp.z0_dis, sbp.zf_dis, plot_full_domain=sbp.plot_full_domain, nT=sbp.nT, title_str=run_name)

if sbp.plot_wavespace:
    hf.plot_k_vs_t(ks, arrays['t_array'], T, arrays['psi_c_reals'], arrays['psi_c_imags'], k, m, omega, title_str=run_name)

if sbp.plot_amplitude:
    hf.plot_A_vs_t(arrays['t_array'], T, arrays['psi_g_array'], sbp.A, k, m, omega, nT=sbp.nT, title_str=run_name)
