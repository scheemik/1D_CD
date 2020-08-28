"""
Performs post-processing actions. Run with $ python3 post_process.py snapshots/*.h5

Usage:
    post_process.py <files>... [--output=<dir>]

Options:
    <files>         # h5 snapshot files

"""

import h5py
import numpy as np
import matplotlib
#matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dedalus.extras.plot_tools import quad_mesh
#plt.ioff()

# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
h5_files = args['<files>']

import switchboard as sbp
import helper_functions as hf

# Parameters
tasks = ['psi']
T = sbp.T
skip_nT = 0
k = sbp.k
m = sbp.m
omega = sbp.omega
z0_dis = sbp.z0_dis
zf_dis = sbp.zf_dis
dt = sbp.dt
dz = sbp.dz
plot_f_d=sbp.plot_full_domain
nT=sbp.nT
n_steps = sbp.n_steps
step_th = sbp.step_th

###############################################################################
# Helper functions

def get_h5_data(tasks, h5_files):
    for task in tasks:
        for filename in h5_files:
            with h5py.File(filename, mode='r') as f:
                # The [()] syntax returns all data from an h5 object
                psi = f['tasks'][task]
                # Need to transpose into the correct orientation
                #   Also need to convert to np.array for plotting function
                psi_array = np.transpose(np.array(psi[()]))
                # Just plotting the real part for now
                psi_real = psi_array.real
                t = np.array(f['scales']['sim_time'])
                z = np.array(f['scales']['z']['1.0'])
    return t, z, psi_real

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

t, z, data = get_h5_data(tasks, h5_files)

t_then_z = False
if t_then_z == True:
    ## Step 1
    ift_t_y = FT_in_time(t, z, data, dt)
    ### Step 2
    ift_z_y_p, ift_z_y_n = FT_in_space(t, z, ift_t_y, dz)
    # Get up and down fields as F = |mag_f| * exp(i*phi_f)
    up_field = ift_z_y_p.real * np.exp(np.real(1j * ift_z_y_p.imag))
    dn_field = ift_z_y_n.real * np.exp(np.real(1j * ift_z_y_n.imag))
else:
    ## Step 1
    ift_z_y_p, ift_z_y_n = FT_in_space(t, z, data, dz)
    ## Step 2
    up_f = FT_in_time(t, z, ift_z_y_p, dt)
    dn_f = FT_in_time(t, z, ift_z_y_n, dt)
    # Get up and down fields as F = |mag_f| * exp(i*phi_f)
    up_field = up_f.real * np.exp(np.real(1j * up_f.imag))
    dn_field = dn_f.real * np.exp(np.real(1j * dn_f.imag))

BP_array = hf.BP_n_steps(sbp.n_steps, sbp.z, z0_dis, zf_dis, sbp.step_th)

if sbp.plot_spacetime:
    hf.plot_z_vs_t(z, t, T, data, BP_array, k, m, omega, z0_dis, zf_dis, plot_full_domain=plot_f_d, nT=nT)

if sbp.plot_spacetime:
    hf.plot_z_vs_t(z, t, T, up_field, BP_array, k, m, omega, z0_dis, zf_dis, plot_full_domain=plot_f_d, nT=nT, filename='f_1D_up_field.png')

if sbp.plot_spacetime:
    hf.plot_z_vs_t(z, t, T, dn_field, BP_array, k, m, omega, z0_dis, zf_dis, plot_full_domain=plot_f_d, nT=nT, filename='f_1D_dn_field.png')
