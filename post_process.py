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
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()

# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
h5_files = args['<files>']

import switchboard as sbp
import helper_functions as hf


# Merge snapshots into one file (might make a really big file)
from dedalus.tools import post
post.merge_sets("snapshots/analysis.h5", h5_files, cleanup=True)


# Parameters
tasks = ['psi']
T = sbp.T
omega = sbp.omega
skip_nT = 0

# Parameters for background profile
# n_steps = sbp.n_steps
# z0_dis  = sbp.z0_dis
# zf_dis  = sbp.zf_dis
# step_th = sbp.step_th

###############################################################################
# Helper functions

###############################################################################
# for task in tasks:


# dsets will be an array containing all the data
#   it will have a size of: tasks x timesteps x 2 x nx x nz (5D)
# dsets = []
# for task in tasks:
#     task_tseries = []
#     for filename in h5_files:
#         with h5py.File(filename, mode='r') as f:
#             dset = f['tasks'][task]
#             # Check dimensionality of data
#             if len(dset.shape) != 2:
#                 raise ValueError("This only works for 2D datasets")
#             # The [()] syntax returns all data from an h5 object
#             task_grid = np.array(dset[()])
#             z_scale = f['scales']['z']['1.0']
#             z_axis = np.array(z_scale[()])
#             t_scale = f['scales']['sim_time']
#             t_axis = np.array(t_scale[()])
#             for i in range(len(t_axis)):
#                 # Skip any times before specified number of T's
#                 if(t_axis[i] > skip_nT*T):
#                     time_slice = [t_axis[i], np.transpose(task_grid[i])]
#                     task_tseries.append(time_slice)
#     dsets.append(task_tseries)

# BP_array = hf.BP_n_steps(sbp.n_steps, sbp.z, sbp.z0_dis, sbp.zf_dis, sbp.step_th)
#
# print('dset is',dsets[0][0][1])

# if sbp.plot_spacetime:
#     hf.plot_z_vs_t(z, arrays['t_array'], T, arrays['psi_g_array'], arrays['BP_array'], k, m, omega, sbp.z0_dis, sbp.zf_dis, plot_full_domain=sbp.plot_full_domain, nT=sbp.nT, title_str=run_name)
