# Plotting the
"""
Description:
This contains helper functions for the Dedalus code so the same version of functions can be called by multiple scripts
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import colorcet as cc
# Add functions in helper file
import helper_functions as hf

###############################################################################

x_min = 0
x_max = np.pi/2
nx = 1000
y_min = 0
y_max = 5
ny = 1000

# points of interest
theta_1 = np.pi/4 # 45 degrees
mL_1    = 1.0
theta_2 = np.pi/4 # 45 degrees
mL_2    = 3.5

x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)

theta, mL = np.meshgrid(x, y)
# Sutherland and Yewchuk 2004 eq 2.4
def T_for_1_layer(theta, mL):
    return 1 / (1 + (np.sinh(mL)/np.sin(2*theta))**2)
Tcoeff = T_for_1_layer(theta, mL)

fig, axes = plt.subplots(nrows=1, ncols=1)
im = axes.pcolormesh(theta, mL, Tcoeff, cmap=cc.cm.linear_worb_100_25_c53)
axes.plot(theta_1, mL_1, 'ko', markersize=10, markerfacecolor='none', markeredgewidth=1.0)
axes.plot(theta_2, mL_2, 'ko', markersize=10, markerfacecolor='none', markeredgewidth=1.0)
cbar = plt.colorbar(im)#, format=ticker.FuncFormatter(latex_exp))
hf.format_labels_and_ticks(axes, r'$\theta_1$', n_ticks=4, tick_formatter=hf.rad_to_degs)
# axes.set_xlabel(r'$\theta_1$')
axes.set_ylabel(r'$m_0L$')
axes.set_title(r'$|\mathcal{T}|/|\mathcal{I}|$ for 1 layer')

plt.savefig('Tcoeff_Ghaemsaidi_fig3a.png')

print('For mL=1.0 and theta=45, T=',str(T_for_1_layer(theta_1, mL_1)))
print('For mL=3.5 and theta=45, T=',str(T_for_1_layer(theta_2, mL_2)))
