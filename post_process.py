"""
Performs post-processing actions.

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
omega = sbp.omega
skip_nT = 0

###############################################################################
# Helper functions

###############################################################################
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

t_array, z_array, data = get_h5_data(tasks, h5_files)

BP_array = hf.BP_n_steps(sbp.n_steps, sbp.z, sbp.z0_dis, sbp.zf_dis, sbp.step_th)

if sbp.plot_spacetime:
    hf.plot_z_vs_t(z_array, t_array, T, data, BP_array, sbp.k, sbp.m, sbp.omega, sbp.z0_dis, sbp.zf_dis, plot_full_domain=sbp.plot_full_domain, nT=sbp.nT)
