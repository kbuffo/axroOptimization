import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import fmin_slsqp

def correct_tao_to_wavefront_1d(init_fig, target_wavefront, ifs, smin=0., smax=1.0, bounds=None, timeit=True):
    """
    Computes the voltages for a TAO that induce a figure change that best matches the target wavefront in 1 dimension using matrix inversion and least squares minimiaztion.
    
    INPUTS: (init_fig, target_fig, and ifs must be in the same data units!)
    init_fig: array of shape (j,) where j is the number of data points in the initial figure of the mirror
    target_fig: array of shape (j,) where j is the number of data points in the target wavefront of the mirror
    ifs: array of shape (N, j) where N is the number of cells of the mirror, and j is the number of data points in each IF. If you don't want to include a cell in the correction (because it's malfuncting), you can set ifs[i] = 0 for those cell(s) to not be included.
    smax: Sets the upperbound that independent variable can be set to during the
        minimization using sequential least squares programming.
        smax is related to the voltage that the IFs you are using were taken at. smax is the
        max scaled value relative to what the voltages the IFs were taken at.
        i.e., theoretical IFs were taken at 1 V -> smax = 10.0
                measured IFs were taken at 10 V -> smax = 1.0
    timeit: Times how long the correction takes to compute and prints it.
    
    RETURNS:
    fig_change: the array the represents the best figure change applied to the mirror (shape is (j,))
    corr_wavefront: the corrected wavefront that best matches target_wavefront (shape is (j,))
    corr_volts: the optimal voltages to apply to the mirror to produce fig_change (shape is (N,))
    """
    
    start_time = time.time()
    # Transpose ifs
    ifs = np.copy(ifs).T
    # get theoretical figure change to achieve
    theo_fig_change = target_wavefront - init_fig
    # initial voltages
    v0 = np.zeros(ifs.shape[1])
    # call the optimizer
    default_acc = 1e-6
    default_epsilon = 1.4901161193847656e-08
    default_iter = 100
    if bounds is None:
        bounds = [(smin, smax) for i in range(ifs.shape[1])]
    corr_volts = fmin_slsqp(RMSmeritFunction, v0, bounds=bounds, args=(theo_fig_change, ifs),\
                           iprint=4,fprime=RMSmeritFunction_deriv,iter=1000,\
                           acc=1e-10, disp=True, epsilon=default_epsilon, full_output=0)
    
    fig_change = np.dot(ifs, corr_volts)
    corr_wavefront = init_fig + fig_change
    time_elapsed = time.time() - start_time
    if timeit:
        print('Time elapsed: {:.2f} s = {:.2f} min'.format(time_elapsed, time_elapsed/60.))
    return fig_change, corr_wavefront, corr_volts

def RMSmeritFunction(voltages, theo_fig_change, ifs):
    rms = np.mean((np.dot(ifs,voltages)-theo_fig_change)**2)
    return rms

def RMSmeritFunction_deriv(voltages, theo_fig_change, ifs):
    deriv = np.dot(2*(np.dot(ifs,voltages)-theo_fig_change),ifs)/np.size(theo_fig_change)
    return deriv

def plot_1d_correction(target_wavefront, corr_wavefront, voltages, dx, meas_wavefront=None, figsize=(13, 5), fontsize=12, figure_units='nm', overall_title='', plot_titles=['Wavefront Correction With Matrix Inversion', 'Difference', 'Optimal Voltages'], colors=['red', 'blue', 'purple', 'black'], meas_wavefront_colors=['green', 'orange'], labels=['Target Wavefront', 'Theoretically Corrected Wavefront'], meas_labels=['Measured Wavefront', 'Target - Measured', 'Predicted - Measured']):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    azimuth_vals = np.linspace(-(target_wavefront.shape[0]/2)*dx, (target_wavefront.shape[0]/2)*dx, target_wavefront.shape[0])
    cell_nums = [i+1 for i in range(voltages.shape[0])]
    
    ax[0].plot(azimuth_vals, target_wavefront, color=colors[0], label=labels[0])
    ax[0].plot(azimuth_vals, corr_wavefront, linestyle='dashed', color=colors[1], label=labels[1])
    if meas_wavefront is not None:
        ax[0].plot(azimuth_vals, meas_wavefront, linestyle='dashed', color=meas_wavefront_colors[0], label=meas_labels[0])
    ax[0].legend(fontsize=fontsize-2)
    if meas_wavefront is not None:
        ax[1].plot(azimuth_vals, target_wavefront-meas_wavefront, color=colors[2], label=meas_labels[1]+', RMS: {:.2f} {}'.format(np.std(target_wavefront-meas_wavefront), figure_units))
        ax[1].plot(azimuth_vals, corr_wavefront-meas_wavefront, color=meas_wavefront_colors[1], label=meas_labels[2]+', RMS: {:.2f} {}'.format(np.std(corr_wavefront-meas_wavefront), figure_units))
    else:
        ax[1].plot(azimuth_vals, target_wavefront-corr_wavefront, color=colors[2], label='RMS: {:.2f} {}'.format(np.std(target_wavefront-corr_wavefront), figure_units))
    ax[1].legend(fontsize=fontsize-2)
    ax[2].plot(cell_nums, voltages, marker='.', color=colors[3])
    ax[2].set_xticks(cell_nums)
    ax[2].set_xticklabels(cell_nums, rotation=90)

    ylabels = ['Figure ({})'.format(figure_units)] * 2 + ['Voltages (V)']
    xlabels = ['Azimuthal Dimension (mm)'] * 2 + ['Cell Number']
    for i in range(3):
        ax[i].set_ylabel(ylabels[i], fontsize=fontsize)
        ax[i].set_xlabel(xlabels[i], fontsize=fontsize)
        ax[i].set_title(plot_titles[i], fontsize=fontsize)
    fig.suptitle(overall_title, fontsize=fontsize)
    fig.tight_layout()
    return fig