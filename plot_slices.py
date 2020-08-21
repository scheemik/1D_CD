"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py NAME SWITCHBOARD <files>... [--output=<dir>]

Options:
    NAME            # Name to put in plot title
    SWITCHBOARD     # Name of switchboard file
    --output=<dir>  # Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
name = args['NAME']
switchboard = args['SWITCHBOARD']
h5_files = args['<files>']

import switchboard as sbp
import helper_functions as hf

import pathlib
output_path = pathlib.Path(args['--output']).absolute()

# Check if output path exists, if not, create it
import os
if not os.path.exists(output_path):
    if rank==0:
        os.makedirs(output_path)

# Labels
hori_label = r'$\Psi$ (m$^2$/s)'
vert_label = r'$z$ (m)'

# Parameters
dpi = sbp.dpi
tasks = ['psi']
rows = 1
cols = 2
n_x_ticks = 3
n_z_ticks = 7
round_to_decimal = 1
title_size = 'medium'
suptitle_size = 'large'
T = sbp.T
omega = sbp.omega

# number of oscillation periods to skip at the start
skip_nT = 0

# Parameters for background profile
n_steps = sbp.n_steps
z0_dis  = sbp.z0_dis
zf_dis  = sbp.zf_dis
step_th = sbp.step_th

###############################################################################
# Helper functions

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

###############################################################################

# dsets will be an array containing all the data
#   it will have a size of: tasks x timesteps x 2 x nx x nz (5D)
dsets = []
for task in tasks:
    task_tseries = []
    for filename in h5_files:
        with h5py.File(filename, mode='r') as f:
            dset = f['tasks'][task]
            # Check dimensionality of data
            if len(dset.shape) != 2:
                raise ValueError("This only works for 2D datasets")
            # The [()] syntax returns all data from an h5 object
            task_grid = np.array(dset[()])
            z_scale = f['scales']['z']['1.0']
            z_axis = np.array(z_scale[()])
            t_scale = f['scales']['sim_time']
            t_axis = np.array(t_scale[()])
            for i in range(len(t_axis)):
                # Skip any times before specified number of T's
                if(t_axis[i] > skip_nT*T):
                    time_slice = [t_axis[i], np.transpose(task_grid[i])]
                    task_tseries.append(time_slice)
    dsets.append(task_tseries)

# Find length of time series
t_len = len(dsets[0])

# Build array for background profile
BP_array = hf.BP_n_steps(n_steps, z_axis, z0_dis, zf_dis, step_th)

# Iterate across time, plotting and saving a frame for each timestep
for i in range(t_len):
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,4]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, ax = plt.subplots(nrows=rows, ncols=cols, gridspec_kw=plot_ratios, sharey=True)
    # Plot background profile
    hf.plot_BP(ax[0], BP_array, z_axis, omega, z0_dis, zf_dis)
    # Plot the task
    j = 0
    hf.plot_task(ax[1], i, j, z_axis, dsets)
    # format axis labels and ticks
    hf.format_labels_and_ticks(ax[1], hori_label)
    # Add title to task plot
    #ax[1].set_title(tasks[j], fontsize=title_size)
    # add display bounds
    hf.add_dis_bounds(ax[0], z0_dis, zf_dis)
    hf.add_dis_bounds(ax[1], z0_dis, zf_dis)
    # Add title for overall figure
    t = dsets[0][i][0]
    #title_str = '{:}, $t/T=${:2.2f}'
    title_str = '$t/T=${:2.2f}'
    #fig.suptitle(title_str.format(name, t/T), fontsize=suptitle_size)
    fig.suptitle(title_str.format(t/T), fontsize=suptitle_size)
    # Save figure as image in designated output directory
    save_fig_as_frame(fig, i, output_path, dpi)
    plt.close(fig)
