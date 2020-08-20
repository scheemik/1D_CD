# Helper functions for Dedalus experiment
"""
Description:
This contains helper functions for the Dedalus code so the same version of functions can be called by multiple scripts
"""

import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import ticker
from dedalus.extras.plot_tools import quad_mesh, pad_limits

###############################################################################
# Takes an exponential number and returns a string formatted nicely for latex
#   Expects numbers in the format 7.0E+2
def latex_exp(num, pos=None):
    if (isinstance(num, int)):
        # integer type, don't reformat
        return num
    else:
        float_str = "{:.1E}".format(num)
        if "E" in float_str:
            base, exponent = float_str.split("E")
            exp = int(exponent)
            b   = float(base)
            str1 = '$'
            if (exp == -1):
                str1 = str1 + str(b/10.0)
            elif (exp == 0):
                str1 = str1 + str(base)
            elif (exp == 1):
                str1 = str1 + str(b*10.0)
            elif (exp == 2):
                str1 = str1 + str(b*100.0)
            else:
                str1 = str1 + str(base) + r'\cdot10^{' + str(exp) + '}'
            str1 = str1 + '$'
            return r"{0}".format(str1)
        else:
            return float_str

# takes in a value in radians and outputs the value in degrees
def rad_to_degs(num, pos=None):
    deg = int(round(math.degrees(num)))
    deg_str = str(deg) + r'$^\circ$'
    return deg_str

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

# Mask to just keep display domain
def make_DD_mask(z, z0_dis, zf_dis):
    """
    z           array of z values
    z0_dis      bottom of display domain
    zf_dis      top of display domain
    """
    # create blank array the same size as z
    DD_array = z*0
    # Set display domain range to have value 1
    for i in range(len(DD_array)):
        if z[i] < zf_dis and z[i] > z0_dis:
            DD_array[i] = 1
    return DD_array

def add_dis_bounds(ax, z0_dis=None, zf_dis=None):
    line_color = my_clrs['black']
    if z0_dis != None:
        ax.axhline(y=z0_dis, color=line_color, linestyle='--')
        ax.axhline(y=zf_dis, color=line_color, linestyle='--')

# Plot background profile
def plot_BP(ax, BP, z, omega=None, z0_dis=None, zf_dis=None):
    ax.plot(BP, z, color=my_clrs['N_0'], label=r'$N_0$')
    ax.set_xlabel(r'$N_0$ (s$^{-1}$)')
    ax.set_ylabel(r'$z$ (m)')
    ax.set_title(r'Background Profile')
    ax.set_ylim([min(z),max(z)])
    if omega != None:
        ax.axvline(x=omega, color=my_clrs['omega'], linestyle='--', label=r'$\omega$')
        ax.legend()

def plot_v_profiles(BP_array, bf_array, sp_array, z, omega=None, z0_dis=None, zf_dis=None, title_str='Forced 1D Wave'):
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,4]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_BP(axes[0], BP_array, z, omega)
    add_dis_bounds(axes[0], z0_dis, zf_dis)
    #
    axes[1].plot(bf_array, z, color=my_clrs['F_bf'], label='Boundary forcing')
    axes[1].plot(sp_array, z, color=my_clrs['F_sp'], label='Sponge layer')
    axes[1].plot(make_DD_mask(z, z0_dis, zf_dis), z, color=my_clrs['black'], label='Display Domain Mask')
    add_dis_bounds(axes[1], z0_dis, zf_dis)
    axes[1].set_xlabel('Amplitude')
    #axes[1].set_ylabel(r'$z$')
    axes[1].set_title(r'Windows')
    axes[1].legend()
    #
    fig.suptitle(r'%s' %(title_str))
    plt.savefig('f_1D_windows.png')

###############################################################################

def plot_z_vs_t(z, t_array, T, w_array, BP_array, k, m, omega, z0_dis=None, zf_dis=None, plot_full_domain=True, nT=0.0, c_map='RdBu_r', title_str='Forced 1D Wave'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1],
                   'width_ratios': [1,5]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=1, ncols=2, gridspec_kw=plot_ratios, sharey=True)
    #
    plot_BP(axes[0], BP_array, z, omega)
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
    #
    if plot_full_domain:
        add_dis_bounds(axes[0], z0_dis, zf_dis)
    else:
        axes[0].set_ylim([z0_dis,zf_dis])
        axes[1].set_ylim([z0_dis,zf_dis])
        axes[1].set_xlim([nT,t_array[-1]/T])
    #
    axes[1].set_xlabel(r'$t/T$')
    axes[1].set_title(r'$\Psi$ (m$^2$/s)')
    param_formated_str = latex_exp(k)+', '+latex_exp(m)+', '+latex_exp(omega)
    fig.suptitle(r'%s, $(k,m,\omega)$=(%s)' %(title_str, param_formated_str))
    #plt.show()
    plt.savefig('f_1D_wave.png')

def plot_A_vs_t(t_array, T, data_array, A, k, m, omega, nT=0.0, title_str='Forced 1D Wave'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1,1],
                   'width_ratios': [1]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=2, ncols=1, gridspec_kw=plot_ratios, sharex=True)
    #
    max_amps = t_array * 0.0
    data_T = np.transpose(data_array)
    # print('shape of max_amps:',max_amps.shape)
    # print('shape of data_T:',data_T.shape)
    for i in range(len(max_amps)):
        max_amps[i] = max(data_T[i])
    ramp_array = A*(1/2)*(np.tanh(4*t_array/(nT*T) - 2) + 1)
    #
    axes[0].plot(t_array/T, max_amps, color=my_clrs['b'], label=r'$A$')
    axes[1].plot(t_array/T, ramp_array, color=my_clrs['b'], label=r'$ramp$')
    axes[1].axvline(x=nT, color=my_clrs['black'], linestyle='--')
    #
    axes[0].set_xlim(t_array[0]/T, t_array[-1]/T)
    axes[1].set_xlabel(r'$t/T$')
    axes[0].set_ylabel(r'Amplitude')
    axes[1].set_ylabel(r'Ramp')
    param_formated_str = latex_exp(A)+', '+latex_exp(k)+', '+latex_exp(m)+', '+latex_exp(omega)
    fig.suptitle(r'%s, $(A,k,m,\omega)$=(%s)' %(title_str, param_formated_str))
    #plt.show()
    plt.savefig('f_1D_A_vs_t.png')

def sort_k_coeffs(arr, nz):
    sorted_arr = np.zeros(nz-1)
    sorted_arr[0:(nz//2-1)] = arr[(nz//2):(nz-1)]
    sorted_arr[(nz//2-1):(nz-1)] = arr[0:(nz//2)]
    return sorted_arr

def plot_k_vs_t(ks, t_array, T, real_array, imag_array, k, m, omega, c_map='RdBu_r', title_str='Forced 1D Wave'):
    # Set aspect ratio of overall figure
    w, h = mpl.figure.figaspect(0.5)
    # This dictionary makes each subplot have the desired ratios
    # The length of heights will be nrows and likewise len(widths)=ncols
    plot_ratios = {'height_ratios': [1,1],
                   'width_ratios': [1]}
    # Set ratios by passing dictionary as 'gridspec_kw', and share y axis
    fig, axes = plt.subplots(figsize=(w,h), nrows=2, ncols=1, gridspec_kw=plot_ratios, sharey=True)
    #
    xmesh, ymesh = quad_mesh(x=t_array/T, y=sort_k_coeffs(ks,1024))
    im_r = axes[0].pcolormesh(xmesh, ymesh, real_array, cmap=c_map)
    im_i = axes[1].pcolormesh(xmesh, ymesh, imag_array, cmap=c_map)
    # # Find max of absolute value for colorbar for limits symmetric around zero
    # cmax = max(abs(data_array.flatten()))
    # if cmax==0.0:
    #     cmax = 0.001 # to avoid the weird jump with the first frame
    # # Set upper and lower limits on colorbar
    # im.set_clim(-cmax, cmax)
    # Add colorbar to im
    cbar_r = plt.colorbar(im_r, ax=axes[0])#, format=ticker.FuncFormatter(latex_exp))
    cbar_r.ax.ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
    cbar_i = plt.colorbar(im_i, ax=axes[1])#, format=ticker.FuncFormatter(latex_exp))
    cbar_i.ax.ticklabel_format(style='sci', scilimits=(-2,2), useMathText=True)
    axes[1].set_xlabel(r'$t/T$')
    axes[0].set_ylabel(r'$k$ (m$^{-1}$)')
    axes[1].set_ylabel(r'$k$ (m$^{-1}$)')
    axes[0].set_title(r'real')
    axes[1].set_title(r'imag')
    wiggle_room = 3
    axes[0].set_ylim([wiggle_room*k, -wiggle_room*k])
    axes[1].set_ylim([wiggle_room*k, -wiggle_room*k])
    param_formated_str = latex_exp(k)+', '+latex_exp(m)+', '+latex_exp(omega)
    fig.suptitle(r'%s, $(k,m,\omega)$=(%s)' %(title_str, param_formated_str))
    plt.savefig('f_1D_wave_spectra.png')

###############################################################################

# Make a plot for one time slice
def plot_task(ax, time_i, task_j, z_ax, dsets):
    # plot line of psi vs. z
    im = ax.plot(dsets[task_j][time_i][1], z_ax, color=my_clrs['b'])
    # Find max of absolute value for data to make symmetric around zero
    xmax = max(abs(max(dsets[task_j][time_i][1].flatten())), abs(min(dsets[task_j][time_i][1].flatten())))
    if xmax==0.0:
        xmax = 0.001 # to avoid the weird jump with the first frame
    # format range of plot extent
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(z_ax[0], z_ax[-1])

def find_tick_locations(x0, xf, n_ticks=3):
    locs = [0]*n_ticks
    locs[0]  = x0
    locs[-1] = xf
    if n_ticks <= 3 and x0 < 0 and xf>0:
        locs[1] = 0.0
    else:
        spacing = (xf-x0)/(n_ticks-1)
        for i in range(1,n_ticks):
            locs[i] = x0 + spacing*i
    return locs

def format_labels_and_ticks(ax, hori_label, n_ticks=3, tick_formatter=latex_exp):
    # add labels
    ax.set_xlabel(hori_label)
    # fix horizontal ticks
    x0, xf = ax.get_xlim()
    tick_locs = find_tick_locations(x0, xf, n_ticks)
    ax.xaxis.set_ticks(tick_locs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(tick_formatter))

def format_labels_and_ticks_v(ax, vert_label):
    # add labels
    ax.set_ylabel(vert_label)
    # fix vertical ticks
    y0, yf = ax.get_ylim()
    ax.yaxis.set_ticks([y0, 0.0, yf])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))

###############################################################################
# Plotting colors from style guide

CUSTOM_COLORS ={'lightcornflowerblue2': '#a4c2f4',
                'lightred3': '#f2c1c1'}

TAB_COLORS =   {'tab:blue': '#1f77b4',
                'tab:orange': '#ff7f0e',
                'tab:green': '#2ca02c',
                'tab:red': '#d62728',
                'tab:purple': '#ffffff',
                'tab:brown': '#ffffff',
                'tab:pink': '#ffffff',
                'tab:gray': '#ffffff',
                'tab:olive': '#ffffff',
                'tab:cyan': '#ffffff'
                }

CSS4_COLORS =  {'aliceblue': '#F0F8FF',
                'antiquewhite': '#FAEBD7',
                'aqua': '#00FFFF',
                'aquamarine': '#7FFFD4',
                'azure': '#F0FFFF',
                'beige': '#F5F5DC',
                'bisque': '#FFE4C4',
                'black': '#000000',
                'blanchedalmond': '#FFEBCD',
                'blue': '#0000FF',
                'blueviolet': '#8A2BE2',
                'brown': '#A52A2A',
                'burlywood': '#DEB887',
                'cadetblue': '#5F9EA0',
                'chartreuse': '#7FFF00',
                'chocolate': '#D2691E',
                'coral': '#FF7F50',
                'cornflowerblue': '#6495ED',
                'cornsilk': '#FFF8DC',
                'crimson': '#DC143C',
                'cyan': '#00FFFF',
                'darkblue': '#00008B',
                'darkcyan': '#008B8B',
                'darkgoldenrod': '#B8860B',
                'darkgray': '#A9A9A9',
                'darkgreen': '#006400',
                'darkgrey': '#A9A9A9',
                'darkkhaki': '#BDB76B',
                'darkmagenta': '#8B008B',
                'darkolivegreen': '#556B2F',
                'darkorange': '#FF8C00',
                'darkorchid': '#9932CC',
                'darkred': '#8B0000',
                'darksalmon': '#E9967A',
                'darkseagreen': '#8FBC8F',
                'darkslateblue': '#483D8B',
                'darkslategray': '#2F4F4F',
                'darkslategrey': '#2F4F4F',
                'darkturquoise': '#00CED1',
                'darkviolet': '#9400D3',
                'deeppink': '#FF1493',
                'deepskyblue': '#00BFFF',
                'dimgray': '#696969',
                'dimgrey': '#696969',
                'dodgerblue': '#1E90FF',
                'firebrick': '#B22222',
                'floralwhite': '#FFFAF0',
                'forestgreen': '#228B22',
                'fuchsia': '#FF00FF',
                'gainsboro': '#DCDCDC',
                'ghostwhite': '#F8F8FF',
                'gold': '#FFD700',
                'goldenrod': '#DAA520',
                'gray': '#808080',
                'green': '#008000',
                'greenyellow': '#ADFF2F',
                'grey': '#808080',
                'honeydew': '#F0FFF0',
                'hotpink': '#FF69B4',
                'indianred': '#CD5C5C',
                'indigo': '#4B0082',
                'ivory': '#FFFFF0',
                'khaki': '#F0E68C',
                'lavender': '#E6E6FA',
                'lavenderblush': '#FFF0F5',
                'lawngreen': '#7CFC00',
                'lemonchiffon': '#FFFACD',
                'lightblue': '#ADD8E6',
                'lightcoral': '#F08080',
                'lightcyan': '#E0FFFF',
                'lightgoldenrodyellow': '#FAFAD2',
                'lightgray': '#D3D3D3',
                'lightgreen': '#90EE90',
                'lightgrey': '#D3D3D3',
                'lightpink': '#FFB6C1',
                'lightsalmon': '#FFA07A',
                'lightseagreen': '#20B2AA',
                'lightskyblue': '#87CEFA',
                'lightslategray': '#778899',
                'lightslategrey': '#778899',
                'lightsteelblue': '#B0C4DE',
                'lightyellow': '#FFFFE0',
                'lime': '#00FF00',
                'limegreen': '#32CD32',
                'linen': '#FAF0E6',
                'magenta': '#FF00FF',
                'maroon': '#800000',
                'mediumaquamarine': '#66CDAA',
                'mediumblue': '#0000CD',
                'mediumorchid': '#BA55D3',
                'mediumpurple': '#9370DB',
                'mediumseagreen': '#3CB371',
                'mediumslateblue': '#7B68EE',
                'mediumspringgreen': '#00FA9A',
                'mediumturquoise': '#48D1CC',
                'mediumvioletred': '#C71585',
                'midnightblue': '#191970',
                'mintcream': '#F5FFFA',
                'mistyrose': '#FFE4E1',
                'moccasin': '#FFE4B5',
                'navajowhite': '#FFDEAD',
                'navy': '#000080',
                'oldlace': '#FDF5E6',
                'olive': '#808000',
                'olivedrab': '#6B8E23',
                'orange': '#FFA500',
                'orangered': '#FF4500',
                'orchid': '#DA70D6',
                'palegoldenrod': '#EEE8AA',
                'palegreen': '#98FB98',
                'paleturquoise': '#AFEEEE',
                'palevioletred': '#DB7093',
                'papayawhip': '#FFEFD5',
                'peachpuff': '#FFDAB9',
                'peru': '#CD853F',
                'pink': '#FFC0CB',
                'plum': '#DDA0DD',
                'powderblue': '#B0E0E6',
                'purple': '#800080',
                'rebeccapurple': '#663399',
                'red': '#FF0000',
                'rosybrown': '#BC8F8F',
                'royalblue': '#4169E1',
                'saddlebrown': '#8B4513',
                'salmon': '#FA8072',
                'sandybrown': '#F4A460',
                'seagreen': '#2E8B57',
                'seashell': '#FFF5EE',
                'sienna': '#A0522D',
                'silver': '#C0C0C0',
                'skyblue': '#87CEEB',
                'slateblue': '#6A5ACD',
                'slategray': '#708090',
                'slategrey': '#708090',
                'snow': '#FFFAFA',
                'springgreen': '#00FF7F',
                'steelblue': '#4682B4',
                'tan': '#D2B48C',
                'teal': '#008080',
                'thistle': '#D8BFD8',
                'tomato': '#FF6347',
                'turquoise': '#40E0D0',
                'violet': '#EE82EE',
                'wheat': '#F5DEB3',
                'white': '#FFFFFF',
                'whitesmoke': '#F5F5F5',
                'yellow': '#FFFF00',
                'yellowgreen': '#9ACD32'}


my_clrs       =  {'b': TAB_COLORS['tab:blue'],
                  'w': (1, 0, 0),               # - r
                  'u': (0, 0, 1),               # - b
                  'v'  : (0, 0.5, 0),           # - g
                  'p': CSS4_COLORS['plum'],
                  'diffusion': CSS4_COLORS['peru'],
                  'viscosity': CSS4_COLORS['peru'],
                  'N_0': TAB_COLORS['tab:blue'],
                  'rho': CSS4_COLORS['slateblue'],
                  'advection': CSS4_COLORS['indianred'],
                  'coriolis': CSS4_COLORS['teal'],
                  'omega': CSS4_COLORS['slategray'],
                  'F_bf': '#008080',            # - teal
                  'F_sp': '#CD853F',            # - peru
                  'temperature': '#B22222',     # - firebrick
                  'salinity': '#4682B4',        # - steelblue
                  'incident': '#8A2BE2',        # - blueviolet
                  'transmission': '#4169E1',    # - royalblue
                  'reflection': '#FF6347',      # - tomato
                  'linear': CSS4_COLORS['forestgreen'],
                  'nonlinear': CSS4_COLORS['indianred'],
                  'arctic': CSS4_COLORS['cornflowerblue'],
                  'cold-fresh': CUSTOM_COLORS['lightcornflowerblue2'],
                  'warm-salty': CUSTOM_COLORS['lightred3'],
                  'black': (0, 0, 0),
                  'white': (1, 1, 1)}
