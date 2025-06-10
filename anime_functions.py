import numpy as np
from scipy import ndimage as nd
from scipy.interpolate import griddata
from operator import itemgetter
from itertools import chain
import copy
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import matplotlib.colors as mcolors
plt.rcParams['savefig.facecolor']='white'

import utilities.figure_plotting as fp
import imaging.man as man
import imaging.analysis as alsis
import imaging.stitch as stitch
import imaging.fitting as fit
from imaging.zernikemod import ptov_r
import axroOptimization.if_functions as iff
#import axroOptimization.evaluateMirrors as eva
#import axroOptimization.solver as solver
try: import construct_connections as cc
except: import axroHFDFCpy.construct_connections as cc

def displayCells(cell_lists, 
                 dx, 
                 imbounds=None, 
                 vbounds=None, 
                 colormap='jet',
                 figsize=None, 
                 title_fntsz=14, 
                 ax_fntsz=12, 
                 tickSize=None,
                 tickLabelSize=None,
                 plot_titles=None,
                 global_title='', 
                 cbar_title='Figure (microns)',
                 x_title='Azimuthal Dimension (mm)',
                 y_title='Axial Dimension (mm)',
                 frame_time=500, 
                 repeat_bool=False, 
                 dispR=False,
                 cell_nos=True, 
                 stats=True,
                 rms=None,
                 PV=True,
                 PVr=False,
                 PVq=False,
                 lambda_PV=False,
                 staticStatsTextbox=False,
                 stats_fntsz=None,
                 dispMaxInds=None, 
                 dispBoxCoords=None,
                 linecolors=['white', 'fuchsia'], 
                 details=False, 
                 includeCables=False,
                 merits=None, 
                 merits_decimalplaces=2,
                 showImageNumber=True, 
                 date='', 
                 stats_textbox_coords=[0.03, 0.7],
                 stats_ha='right',
                 stats_va='bottom',
                 show_maxInd_textbox=True,
                 N_rows=1,
                 plots_wspace=0.3,
                 extent=None,
                 frame_num_label=None, 
                 stats_units=['um'], 
                 cbarLabelPerPlot=False):
    """
    Wrapper function that runs displayIFs but takes in a list of lists of cell objects.
    dispMaxInds and dispBoxCoords should be specified with a list of the plot numbers you
    want to display those attributes for.
    Ex. dispMaxInds = [1, 2] will only show maxInds on the first and second plots whose
    IFs are specified by cell_lists.
    """
    ifs_list, maxInds_list, boxCoords_list = [], [], []
    if not dispMaxInds: dispMaxInds = []
    else: dispMaxInds = np.array(dispMaxInds) - 1
    if not dispBoxCoords: dispBoxCoords = []
    else: dispBoxCoords = np.array(dispBoxCoords) - 1
    details_list = cell_lists[0]
    if not details: details_list = None

    for i, cell_list in enumerate(cell_lists):
        cell_nos_array, ifs, maxInds, boxCoords = cc.get_cell_arrays(cell_list)
        if i not in dispMaxInds: maxInds = None
        if i not in dispBoxCoords: boxCoords = None
        ifs_list.append(ifs)
        maxInds_list.append(maxInds)
        boxCoords_list.append(boxCoords)

    if not cell_nos: cell_nos_array = None

    ani, fps = displayIFs(ifs_list, dx, imbounds=imbounds, vbounds=vbounds,
                        colormap=colormap,
                        figsize=figsize, title_fntsz=title_fntsz, ax_fntsz=ax_fntsz,
                        plot_titles=plot_titles,
                        global_title=global_title, cbar_title=cbar_title,
                        x_title=x_title,
                        y_title=y_title,
                        frame_time=frame_time, repeat_bool=repeat_bool, dispR=dispR,
                        cell_nos=cell_nos_array, stats=stats, rms=rms, PV=PV, PVr=PVr, PVq=PVq,
                        lambda_PV=lambda_PV, dispMaxInds=maxInds_list,
                        dispBoxCoords=boxCoords_list,
                        linecolors=linecolors, details=details_list, includeCables=includeCables,
                        merits=merits, showImageNumber=showImageNumber, date=date,
                        stats_textbox_coords=stats_textbox_coords, show_maxInd_textbox=show_maxInd_textbox,
                        N_rows=N_rows, frame_num_label=frame_num_label, stats_units=stats_units, tickSize=tickSize, 
                        tickLabelSize=tickLabelSize, merits_decimalplaces=merits_decimalplaces, plots_wspace=plots_wspace, 
                        staticStatsTextbox=staticStatsTextbox, stats_fntsz=stats_fntsz, stats_ha=stats_ha, 
                        stats_va=stats_va, extent=extent, cbarLabelPerPlot=cbarLabelPerPlot)
    return ani, fps

def displayIFs(ifs,
                dx,
                imbounds=None,
                vbounds=None,
                colormap='jet',
                figsize=None,
                title_fntsz=14,
                ax_fntsz=12,
                tickSize=None,
                tickLabelSize=12,
                plot_titles=None,
                global_title='',
                cbar_title='Figure (microns)',
                x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)',
                frame_time=500,
                repeat_bool=False,
                dispR=False,
                cell_nos=None,
                stats=False,
                rms=None,
                PV=True,
                PVr=False,
                PVq=False,
                lambda_PV=False,
                PV_labels=None,
                staticStatsTextbox=True,
                stats_fntsz=12,
                dispMaxInds=None,
                dispBoxCoords=None,
                linecolors=['white', 'fuchsia'],
                details=None,
                includeCables=False,
                merits=None,
                merits_decimalplaces=2,
                showImageNumber=True,
                image_aspect='auto',
                date='',
                stats_textbox_coords=[0.03, 0.97],
                stats_ha='right',
                stats_va='bottom',
                show_maxInd_textbox=True,
                N_rows=1,
                stats_units=['um'],
                plots_wspace=0.3,
                extent=None,
                frame_num_label=None,
                cbarLabelPerPlot=True):

    """
    Displays IFs stacks, side by side.

    ifs: list of 3D arrays. The length of the list determines how many plots will
    be made.

    dx: pixel spacing for images.

    imbounds: List to specify the range of images to animate: [upperBound, lowerBound]
    If cell_nos is given, the entries of imbounds specify the cell numbers to
    animate. If cell_nos is not given, the entries of imbounds specify the images to animate.

    vbounds: list or list of lists to specify the color range of the IFs: [lowerBound, upperBound]
    If one list is given, the bounds are applied to all plots. If a list of lists are given, unique
    bounds are generated for each plot. Default is to generate separate vbounds for each plot.

    figsize: specify the size of the figure as a tuple: i.e. (width, height) -> (8, 8).

    plot_titles: list of plot titles for each plot generated. len(plot_titles) must equal len(ifs).

    global_title: Title for figure.

    frame_time: time in ms between each frame.

    repeat_bool: if True loops animation when finished.

    cell_nos: 1D array of cell nos that correspond to the IFs stacks provided.

    stats: If true, display the rms and P-to-V of each image.
    
    rms: None or an array of shape N x M. N is the number of frames to render,
         M is the number of subplots, and the data entries represent the root mean square.
         If None and stats=True, rms values are automatically calculated.

    dispMaxInds: A list whose elments are 3D arrays that are the maxInds for the
    corresponding IF stack given.

    details: A list of cell objects for the plots, assuming all plots are showing the same cell each frame.
    If included, details allows grid coords, region, pin cell #, board, dac, channel,
    and cables (optionally) to be displayed on every frame.

    includeCables: If set to True, details will display the cable connections in addition to
    the other information included in the cell objects lists.

    merits: An array of shape N x M x 2 matrix. N is the number of frames to render,
            M is the number of subplots, and the first entry along the last axis is the E68 value, and
            the last is the HPD. NaN values on the array will not be shown.
    PVr: False, or 'zernike', or 'legendre', or an array of shape N x M. N is the number of frames to render,
            M is the number of subplots, and the data entries represent the robust peak-to-valley. If 'zernike', the PVr 
            is calculated using a 35 zernike term fit: zernikemod.ptov_r(d, N=35). This is ideal for rotationally symmetric 
            arrays. If 'legendre', the PVr is calculated using 10th a order legendre fit. This is used for arrays that are not
            rotationally symmetric.
    PVq: False or float Q between 0. to 1. If not False, the peak to value is calculated of the arrays after disregarding the
        highest and lowest pixels and only considering the remaining Q percent of pixels.
    
    lambda_PV: False or float. Converts the PV value to be in terms of lambda, the wavelength of the interferometer used. If this
        desired, lambda_PV should be set to the wavelength of the interferometer in units matching the units of the data.

    extent: [left, right, bottom, top] in data coordinates
    """
    ifs_ls = ifs.copy()
    N_plots = len(ifs_ls)
    ifs_ls = [iff.conv_2D_to_3D(if_map) if if_map.ndim==2 else if_map for if_map in ifs_ls] # convert any 2D arrays into 3D arrays
    if '_' in date: date = date.replace('_', '')
    if not figsize:
        figsize = (7*len(ifs_ls), 6)
    fig = plt.figure(figsize=figsize)
    if extent is None:
        # extent = fp.mk_extent(ifs_ls[0][0], dx)
        extent = [fp.mk_extent(ifs_ls[i][0], dx) for i in range(len(ifs_ls))]
    if not plot_titles:
        plot_titles = [''] * len(ifs_ls)
    axs, caxs = init_subplots(N_plots, fig, plot_titles, # initalize subplots
                            x_title, y_title, title_fntsz, ax_fntsz, N_rows=N_rows, tickSize=tickSize, tickLabelSize=tickLabelSize)
    if isinstance(imbounds, list) and len(imbounds) == 1:
        imbounds = imbounds * 2
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, cell_nos, ifs_ls) # get imbounds
    vbounds_ls = get_vbounds(ifs_ls, vbounds, [lbnd, ubnd]) # get vbounds
    idx_txt, cell_nos = get_index_label(ifs_ls, cell_nos, frame_num_label) # "Image #:" or "Cell #:"
    if type(cbar_title) != list:
        cbar_title = [cbar_title] * len(ifs_ls)
    if (not cbarLabelPerPlot) and (not all(title == cbar_title[0] for title in cbar_title)):
        cbarLabelPerPlot = True
    rms_ls, ptov_ls = [], []
    if stats: # get rms and P-to-V of IFs
        rms_ls, ptov_ls = get_stats(ifs_ls, PV=PV, PVr=PVr, PVq=PVq, lambda_PV=lambda_PV, rms=rms)
    if not isinstance(stats_units, list):
        stats_units = [stats_units]
    if len(stats_units) < N_plots:
        stats_units = stats_units * ((N_plots - len(stats_units)) + 1)

    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        # generate features for a frame
        feature_ls = make_frame(axs, ifs_ls, dx, i, extent, colormap, vbounds_ls,
                                global_title, idx_txt, cell_nos, title_fntsz,
                                dispMaxInds, ax_fntsz, stats, rms_ls, ptov_ls, dispBoxCoords,
                                linecolors, details, includeCables, merits, dispR, showImageNumber, date,
                                stats_textbox_coords, show_maxInd_textbox, stats_units=stats_units, PV=PV, PVr=PVr, 
                                PVq=PVq, lambda_PV=lambda_PV, PV_labels=PV_labels, stats_fontsize=stats_fntsz, 
                                staticStatsTextbox=staticStatsTextbox, stats_ha=stats_ha,
                                stats_va=stats_va, merits_decimalplaces=merits_decimalplaces, image_aspect=image_aspect)
        frames.append(feature_ls)

    for i in range(len(ifs_ls)): # attach static colorbar
        cbar = fig.colorbar(frames[0][i], cax=caxs[i])
        cbar.ax.tick_params(axis='both', which='both', width=tickSize, labelsize=tickLabelSize)
        if (cbarLabelPerPlot) or (not cbarLabelPerPlot and i == len(ifs_ls)-1):
            cbar.set_label(cbar_title[i], fontsize=ax_fntsz)
    # include the date
    if date != '' and showImageNumber == True:
        date_text = plt.gcf().text(0.78, 0.98, 'Date: {}'.format(date), fontsize=title_fntsz,
                                    ha='right', va='top')
    if details is not None:
        fig.subplots_adjust(top=0.85, bottom=0.3, hspace=0.5, wspace=0.3)
    else:
        fig.subplots_adjust(top=0.85, hspace=0.5, wspace=plots_wspace)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    # fps = int(1 / (frame_time/1000))
    fps = 1 / (frame_time/1000)
    if dispSingle:
        print('dispSingle is True.')
        ani = fig
    return ani, fps # return animation and frames per second for saving to GIF


def init_subplots(N_plots, fig, title_ls, x_title, y_title, title_fontsize, ax_fontsize, N_rows=1, 
                  tickSize=None, tickLabelSize=None, xtickLabelRotation=None, ytickLabelRotation=None):
    """
    Initializes a row of subplots based on the number of IF stacks provided.
    Generates the axes and sets their features.
    """
    ax_ls, cax_ls = [], []
    # if not given a list of titles, convert single string to list
    if type(x_title) != list: x_title = [x_title]*N_plots
    if type(y_title) != list: y_title = [y_title]*N_plots
    gs = gridspec.GridSpec(N_rows, N_plots)
    for i in range(N_plots):
        ax = fig.add_subplot(gs[i])
        ax.set_title(title_ls[i], fontsize=title_fontsize)
        ax.set_xlabel(x_title[i], fontsize=ax_fontsize)
        ax.set_ylabel(y_title[i], fontsize=ax_fontsize)
        #if tickLabelSize is not None:
        ax.tick_params(axis='x', which='major', width=tickSize, labelsize=tickLabelSize, 
                       labelrotation=xtickLabelRotation)
        ax.tick_params(axis='y', which='major', width=tickSize, labelsize=tickLabelSize, 
                       labelrotation=ytickLabelRotation)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.10)
        ax_ls.append(ax)
        cax_ls.append(cax)
    return ax_ls, cax_ls
    pass


def get_vbounds(ifs_ls, vbounds, imbounds):
    """
    Formats the user provided vbounds into a list of vbounds.
    """
    vbounds_ls = []
    vbound_single_image = False
    if np.abs(imbounds[0]-imbounds[1]) == 1: vbound_single_image = True
    if vbounds is None and not vbound_single_image:
        # generate different vbounds for each plot (based on the stack of IFs for each plot)
        for ifs in ifs_ls:
            vbounds_ls.append([np.nanmin(ifs), np.nanmax(ifs)])
    elif vbounds is None and vbound_single_image:
        # generate different vbounds for each plot (based on a single IF for each plot)
        for ifs in ifs_ls:
            vbounds_ls.append([np.nanmin(ifs[imbounds[0]]), np.nanmax(ifs[imbounds[0]])])
    elif type(vbounds[0]) is list and len(vbounds) == len(ifs_ls):
        # use different user-supplied vbounds for each plot
        vbounds_ls = vbounds
    elif type(vbounds) is list and len(vbounds) == 2:
        # use the same user-supplied vbounds for each plot
        vbounds_ls = [vbounds]*len(ifs_ls)
    # print(vbounds_ls)
    return vbounds_ls


def get_index_label(ifs_ls, cell_nos, frame_num_label):
    """
    Returns a label that denotes whether we are displaying images or explicitly
    indexed cells.
    """
    if frame_num_label is not None:
        idx_label = frame_num_label
        if cell_nos is None:
            cell_nos = np.arange(0, len(ifs_ls[0]))
    elif type(cell_nos) != type(None):
        idx_label = 'Cell #:'
    else:
        idx_label = 'Image #:'
        cell_nos = np.arange(0, len(ifs_ls[0]))
    return idx_label, cell_nos


def get_imbounds(imbounds, cell_nos, ifs_ls):
    """
    Formats the user provided imbounds to get the appropriate images to display
    from the IF stacks.
    """
    if isinstance(imbounds, list) and (len(imbounds) == 1):
        imbounds += imbounds
    displaySingle = False
    if imbounds is not None: # the user provided imbounds
        if imbounds[0] == imbounds[1]:
            displaySingle = True # show only a single frame
        if type(cell_nos) != type(None):
            # match the imbounds to the cell numbers
            try:
                lowerBound = int(np.where(cell_nos == imbounds[0])[0])
                upperBound = int(np.where(cell_nos == imbounds[1])[0] + 1)
            except: print('One or more of your imbounds does not match the given cell numbers.')
        else: # explicitly use the imbounds as indexes when cell_nos are not present
            lowerBound, upperBound = imbounds[0], imbounds[1]+1
    else: # show all images supplied
        lowerBound, upperBound = 0, ifs_ls[0].shape[0]
        if ifs_ls[0].shape[0] == 1: displaySingle = True
    return upperBound, lowerBound, displaySingle


def get_stats(ifs_ls, PV=True, PVr=False, PVq=False, lambda_PV=False, rms=None):
    """
    Returns a list of rms and P-to-V values for a list of IF stacks.
    """
    rms_ls, ptov_ls = [], []
    allowed_PVr_types = ['zernike', 'legendre']
    PVr_type = None
    if type(PVr) == str:
        PVr_type = PVr
        PVr = True
    if (PV and np.any(PVr)) or (PV and PVq) or (np.any(PVr) and PVq):
        print('Error: Only one of the following stats may be specified while the others must be False: [PV, PVr, PVq]')
        return None
    for i, ifs in enumerate(ifs_ls): # iterate through each subplot
        if np.any(rms): # user provided rms vals
            rms_vals = rms[:, i].flatten()
        else: # calculate rms vals
            rms_vals = np.array([alsis.rms(ifs[i]) for i in range(ifs.shape[0])])
        if PV: # use direct PV
            ptov_vals = np.array([alsis.ptov(ifs[i]) for i in range(ifs.shape[0])])
        if np.any(PVr): # use robust PV
            if PVr_type: # user specified PVr type to calculate
                if PVr_type == 'zernike': # calculate zernike PVr
                    ptov_vals = np.array([ptov_r(ifs[i], N=35) for i in range(ifs.shape[0])])
                elif PVr_type == 'legendre': # calculate legendre PVr
                    ptov_vals = np.array([fit.ptov_r_leg(ifs[i], xo=10, yo=10) for i in range(ifs.shape[0])])
                else:
                    print("Error: PVr type '{}' not recognized. Avialable PVr types are : {}".format(PVr_type, allowed_PVr_types))
            else: # user provided PVr values
                ptov_vals = PVr[:, i].flatten()
        if PVq: # use PVq
            ptov_vals = np.array([alsis.ptov_q(ifs[i], q=PVq) for i in range(ifs.shape[0])])
        rms_ls.append(rms_vals)
        ptov_ls.append(ptov_vals)
    if lambda_PV:
        ptov_ls = [lambda_PV/ptov_vals for ptov_vals in ptov_ls]
    return rms_ls, ptov_ls
        
    # if PV: # use direct PV
    #     for ifs in ifs_ls:
    #         rms_vals = np.array([alsis.rms(ifs[i]) for i in range(ifs.shape[0])])
    #         ptov_vals = np.array([alsis.ptov(ifs[i]) for i in range(ifs.shape[0])])
    #         rms_ls.append(rms_vals)
    #         ptov_ls.append(ptov_vals)
    # if np.any(PVr): # use Zernike fit PV
    #     for i, ifs in enumerate(ifs_ls): # i indexes subplot number
    #         rms_vals = np.array([alsis.rms(ifs[i]) for i in range(ifs.shape[0])])
    #         if type(PVr) == bool: # calculate PVr
    #             ptov_vals = np.array([ptov_r(ifs[i], N=35) for i in range(ifs.shape[0])])
    #         else: # user supplied PVr
    #             ptov_vals = PVr[:, i].flatten()
    #         rms_ls.append(rms_vals)
    #         ptov_ls.append(ptov_vals)
    # if PVq: # Use Q PV
    #     for ifs in ifs_ls:
    #         rms_vals = np.array([alsis.rms(ifs[i]) for i in range(ifs.shape[0])])
    #         ptov_vals = np.array([alsis.ptov_q(ifs[i], q=PVq) for i in range(ifs.shape[0])])
    #         rms_ls.append(rms_vals)
    #         ptov_ls.append(ptov_vals)
    # if lambda_PV:
    #     ptov_ls = [lambda_PV/ptov_vals for ptov_vals in ptov_ls]
    # return rms_ls, ptov_ls


def make_frame(axs, ifs_ls, dx, frame_num, extent, colormap, vbounds_ls,
                global_title, idx_txt, cell_nos, title_fntsz, dispMaxInds, ax_fntsz,
                stats, rms_ls, ptov_ls, dispBoxCoords, linecolors, details, includeCables,
                merits, dispR, showImageNumber, date, stats_textbox_coords, show_maxInd_textbox,
                stats_units, stats_fontsize=None, staticStatsTextbox=True, stats_ha='right', 
                stats_va='bottom', merits_decimalplaces=2, PV=True, PVr=False, PVq=False, 
                lambda_PV=False, PV_labels=None, image_aspect='auto'):
    """
    Generates all the features that will be animated in the figure.
    """
    feature_ls = [] # list that holds all the features of a frame

    for i, ifs in enumerate(ifs_ls): # plot the data
        image = axs[i].imshow(ifs[frame_num], extent=extent[i], aspect=image_aspect,
                            cmap=colormap, vmin=vbounds_ls[i][0], vmax=vbounds_ls[i][1])
        feature_ls.append(image)
    cell_no = cell_nos[frame_num] # make the global title
    if showImageNumber:
        if idx_txt == 'Cell #:' or idx_txt == 'Image #:':
            txtstring = global_title + '\n' + idx_txt + ' {}\n'.format(int(cell_no))
        else:
            txtstring = global_title + '\n' + idx_txt + ' {:.4f}\n'.format(cell_no)
    else:
        txtstring = global_title
        if date != '': txtstring += '\nDate: ' + date
    title_plt_text = plt.gcf().text(0.5, 0.94, txtstring, fontsize=title_fntsz,
                            ha='center', va='center')

    # initialize lines and text entries as blank
    # structure of vlines and hlines is:
    # [ [ifs1_vline_prim,ifs1_vline_sec], [ifs2_vline_prim, ifs2_vline_sec], ...]
    # structure maxvals is:
    # [ ifs1_maxval, ifs2_maxval, ...]
    vlines = [[ax.text(0,0,''), ax.text(0,0,'')] for ax in axs]
    hlines = [[ax.text(0,0,''), ax.text(0,0,'')] for ax in axs]
    maxvals = [ax.text(0,0,'') for ax in axs]
    if dispMaxInds is not None: # draw the lines and text boxes for maxInds
        vlines, hlines, maxvals = illustrate_maxInds(dispMaxInds, frame_num, axs, vlines,
                                                    hlines, maxvals, dx, ifs_ls,
                                                    ax_fntsz, linecolors, show_maxInd_textbox)
    # unpack nested lists into regular lists
    vlines, hlines = list(chain(*vlines)), list(chain(*hlines))

    # initialize the rms and ptov text boxes as blank
    stats_textboxes = [ax.text(0,0,'') for ax in axs]
    if stats: # draw the stats text boxes
        stats_textboxes = illustrate_stats(axs, frame_num, rms_ls, ptov_ls,
                                            ax_fntsz, stats_textbox_coords, stats_units,
                                            staticTextbox=staticStatsTextbox, ha=stats_ha,
                                            va=stats_va, stats_fontsize=stats_fontsize, PV=PV, 
                                            PVr=PVr, PVq=PVq, lambda_PV=lambda_PV, PV_labels=PV_labels)

    # initialize the box coord rectangles as blank
    boxCoords_rectangles = [ax.text(0,0,'') for ax in axs]
    if dispBoxCoords is not None:
        boxCoords_rectangles = illustrate_boxCoords(axs, frame_num, dispBoxCoords,
                                                    ifs_ls, dx, linecolors)

    # initalize the details text boxes as blank
    details_boxes = [ax.text(0,0, '') for ax in axs]
    if details is not None:
        details_boxes = illustrate_details(frame_num, details, includeCables, ax_fntsz)

    # initalize the merit text boxes as blank
    merit_boxes = [ax.text(0,0, '') for ax in axs]
    if merits is not None:
        merit_boxes = illustrate_merits(axs, frame_num, merits, ax_fntsz, 
                                        merits_decimalplaces=merits_decimalplaces, 
                                        merits_fontsize=stats_fontsize)

    # initialize the dispR textboxes as blank
    dispR_boxes = [ax.text(0,0, '') for ax in axs]
    if dispR is True:
        dispR_boxes = illustrate_dispR(axs, frame_num, ax_fntsz)

    # combine all features into single list
    feature_ls += [title_plt_text] + vlines + hlines + maxvals + stats_textboxes \
                + boxCoords_rectangles + details_boxes + merit_boxes + dispR_boxes
    return feature_ls


def illustrate_maxInds(dispMaxInds, frame_num, axs, vlines, hlines, maxvals, dx,
                        ifs_ls, ax_fntsz, linecolors, show_maxInd_textbox):
    """
    Creates the coordinate tracking lines for IFs and the associated textbox.
    """
    for i, maxInds in enumerate(dispMaxInds):
        if maxInds is None: continue
        ifs = ifs_ls[i]
        primary_if_txt, secondary_if_txt = '', ''
        if maxInds[frame_num][0][0] >= 0 and maxInds[frame_num][0][1] >= 0:
            # check if primary maxInd coordinates are on grid before drawing
            hlines[i][0] = axs[i].axhline(y=(ifs.shape[1]/2-maxInds[frame_num][0][0])*dx,
                                                    xmin=0, xmax=1, color=linecolors[0])
            vlines[i][0] = axs[i].axvline(x=(maxInds[frame_num][0][1]-ifs.shape[2]/2)*dx,
                                            ymin=0, ymax=1, color=linecolors[0])
            primary_if_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds[frame_num][0][0], maxInds[frame_num][0][1], maxInds[frame_num][0][2])
        if maxInds[frame_num][1][0] >= 0 and maxInds[frame_num][1][1] >= 0:
            # check if secondary maxInd coordinates are on grid before drawing
            hlines[i][1] = axs[i].axhline(y=(ifs.shape[1]/2-maxInds[frame_num][1][0])*dx,
                                            xmin=0, xmax=1, color=linecolors[1])
            vlines[i][1] = axs[i].axvline(x=(maxInds[frame_num][1][1]-ifs.shape[2]/2)*dx,
                                            ymin=0, ymax=1, color=linecolors[1])
            secondary_if_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds[frame_num][1][0], maxInds[frame_num][1][1], maxInds[frame_num][1][2])
        # construct the maxInd textbox for each axis
        maxInd_txt = primary_if_txt + secondary_if_txt
        x_txt_pos, y_txt_pos = 0.03, 0.97
        if (0 <= maxInds[frame_num][0][0] < ifs.shape[1]*0.15) or (0 <= maxInds[frame_num][1][0] < ifs.shape[1]*0.15):
            # move IF text box if it will block IF
            y_txt_pos = 0.22
        if show_maxInd_textbox:
            maxvals[i] = axs[i].text(x_txt_pos, y_txt_pos, maxInd_txt, color='black', fontsize=ax_fntsz-4,
                            transform=axs[i].transAxes, va='top', bbox=dict(facecolor='white', alpha=0.65))
    return vlines, hlines, maxvals


def illustrate_stats(axs, frame_num, rms_ls, ptov_ls, ax_fntsz, stats_textbox_coords,
                    stats_units, staticTextbox=False, ha='right', va='bottom', 
                    stats_fontsize=None, PV=True, PVr=False, PVq=False, lambda_PV=False, PV_labels=None):
    """
    Creates the textbox that will display the RMS and P-to-V values.
    """
    stats_textbox_ls = []
    if stats_fontsize is None:
        stats_fontsize = ax_fntsz
    if PV_labels:
        pv_label = PV_labels
    elif PV:
        pv_label = ['PV']*len(rms_ls)
    elif type(PVr)==str or type(PVr)==np.ndarray:
        pv_label = ['PVr']*len(rms_ls)
    elif PVq:
        pv_label = ['PVq']*len(rms_ls)
    for i in range(len(rms_ls)):
        rms_val = rms_ls[i][frame_num]
        pv_val = ptov_ls[i][frame_num]
        if lambda_PV: # write PV in terms of lambda
            if pv_val <= 1:
                pv_val = 1. / pv_val
                stats_txt = "RMS: {:.2f} {}\n{}: {:.2f}{}".format(rms_val, stats_units[i], pv_label[i], pv_val, r'$\lambda$')
            else:
                pv_val = pv_val
                stats_txt = "RMS: {:.2f} {}\n{}: {}{:.2f}".format(rms_val, stats_units[i], pv_label[i], r'$\lambda\:/\:$', pv_val)
        else: # write PV in terms of data units
            stats_txt = "RMS: {:.2f} {}\n{}: {:.2f} {}".format(rms_val, stats_units[i], pv_label[i], pv_val, stats_units[i])
        y_txt_pos, x_txt_pos = stats_textbox_coords[0], stats_textbox_coords[1]
        if (len(stats_textbox_coords) != 3) and (frame_num > len(rms_ls[i])/2) and (not staticTextbox):
            # move text box if it will block IF
            x_txt_pos = 0.03
            ha = 'left'
        stats_textbox = axs[i].text(x_txt_pos, y_txt_pos, stats_txt, fontsize=stats_fontsize,
                                    transform=axs[i].transAxes, va=va, ha=ha,
                                    bbox=dict(facecolor='white', alpha=0.65))
        stats_textbox_ls.append(stats_textbox)
    return stats_textbox_ls

def illustrate_boxCoords(axs, frame_num, dispBoxCoords, ifs_ls, dx, linecolors):
    """
    Creates the rectangles that will enclose each IF.
    """
    rectangles = []
    for i, boxCoords in enumerate(dispBoxCoords):
        if boxCoords is None:
            rectangles.append(axs[i].text(0, 0, ''))
            continue
        ifs = ifs_ls[i]
        # anchor is (x_anchor, y_anchor) bottom left of box
        # primary IF
        y1_anchor = (ifs.shape[1]/2 - boxCoords[frame_num][0][2][0])*dx
        x1_anchor = (boxCoords[frame_num][0][2][1]-ifs.shape[2]/2)*dx
        height1 = (boxCoords[frame_num][0][2][0]-boxCoords[frame_num][0][0][0])*dx
        width1 = (boxCoords[frame_num][0][1][1]-boxCoords[frame_num][0][0][1])*dx
        rec1 = axs[i].add_patch(Rectangle((x1_anchor, y1_anchor), width1, height1,
                                edgecolor=linecolors[0], facecolor='none'))
        rectangles.append(rec1)
        # check if secondary IF exists
        if -1 not in boxCoords[frame_num][1]:
            y2_anchor = (ifs.shape[1]/2 - boxCoords[frame_num][1][2][0])*dx
            x2_anchor = (boxCoords[frame_num][1][2][1]-ifs.shape[2]/2)*dx
            height2 = (boxCoords[frame_num][1][2][0]-boxCoords[frame_num][1][0][0])*dx
            width2 = (boxCoords[frame_num][1][1][1]-boxCoords[frame_num][1][0][1])*dx
            rec2 = axs[i].add_patch(Rectangle((x2_anchor, y2_anchor), width2, height2,
                                    edgecolor=linecolors[1], facecolor='none'))
            rectangles.append(rec2)
    return rectangles

def illustrate_details(frame_num, details, includeCables, ax_fntsz):
    """
    Creates the textboxes that will display the cell details.
    """
    cell = details[frame_num]
    loc_header = '--------------------------------------------Location--------------------------------------------\n'
    loc_txt = 'Grid Coords: {} || Region: {} || Pin Cell #: {} || Shorted Cell #: {}\n'.format(cell.grid_coord, cell.region, cell.pin_cell_no, cell.short_cell_no)
    loc_textstring = loc_header + loc_txt
    contr_header = '---------------------------------------------Control----------------------------------------------\n'
    contr_txt = 'BRD: {} || DAC: {} || CH: {}'.format(cell.board_num, cell.dac_num, cell.channel)
    contr_textstring = contr_header + contr_txt
    if includeCables:
        cable_header = '\n---------------------------------------------Cables---------------------------------------------\n'
        cable_txt1 = 'P0XV Cable, Pin: {} || PV Cable, Pin: {}\n'.format([cell.p0xv_cable, cell.p0xv_pin], [cell.pv_cable, cell.pv_pin])
        cable_txt2 = 'P0XA Cable, Pin: {} || PA Cable, Pin: {}\n'.format([cell.p0xa_cable, cell.p0xa_pin], [cell.pa_cable, cell.pa_pin])
        cable_txt3 = 'J Port, Pin: {} || AOUT Pin: {}  '.format([cell.j_port, cell.j_pin], cell.aout_pin)
        cable_textstring = cable_header + cable_txt1  + cable_txt2 + cable_txt3
    else: cable_textstring = ''
    details_textstring = loc_textstring  + contr_textstring + cable_textstring
    details_textbox = plt.gcf().text(0.5, 0.1, details_textstring, fontsize=ax_fntsz-4,
                                    ha='center', va='center', bbox=dict(facecolor='white', alpha=0.65))
    return [details_textbox]


def illustrate_merits(axs, frame_num, merits, ax_fntsz, merits_decimalplaces=2, merits_fontsize=None):
    """
    Creates the textboxes that will display the HPD and E68 merit values.
    """
    if merits_fontsize is None:
        merits_fontsize = ax_fntsz
    merit_textboxes = []
    # print('merits given:\n', merits)
    # print('checking:', merits[frame_num])

    for i in range(len(axs)):
        if np.any(np.isnan(merits[frame_num][i])):
            continue
        else:
            #hpd = np.round(merits[frame_num][i][1], decimals=merits_decimalplaces)
            #e68 = np.round(merits[frame_num][i][0], decimals=merits_decimalplaces)
            hpd = merits[frame_num][i][1]
            e68 = merits[frame_num][i][0]
            #print(hpd, e68)
        merit_txt = "PSF HPD: {:.{}f} arcsec\nPSF E68: {:.{}f} arcsec"\
                    .format(hpd, merits_decimalplaces, e68, merits_decimalplaces)
        #print(merit_txt)
        x_txt_pos, y_txt_pos = 0.03, 0.03
        merit_textbox = axs[i].text(x_txt_pos, y_txt_pos, merit_txt, fontsize=merits_fontsize,
                                    transform=axs[i].transAxes, va='bottom',
                                    bbox=dict(facecolor='white', alpha=0.65))
        merit_textboxes.append(merit_textbox)
    return merit_textboxes

def illustrate_dispR(axs, frame_num, ax_fntsz):
    """
    Creates the textboxes that will display the large and small radii of C1S04.
    """
    textboxes = []
    for ax in axs:
        large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=12, color='red',
                                        ha='center', va='center', transform=ax.transAxes,
                                        bbox=dict(facecolor='white', alpha=0.65))
        small_R_text = ax.text(0.5, 0.85, 'Smaller R', fontsize=12, color='red',
                                ha='center', va='center', transform=ax.transAxes,
                                bbox=dict(facecolor='white', alpha=0.65))
        textboxes.append(large_R_text)
        textboxes.append(small_R_text)
    return textboxes

def displayMultipleVoltMaps(voltMaps_ls, 
                            map_nos=None, 
                            voltMaplabel='Map:',
                            showMap_no=True,
                            suppress_thresh=None,
                            include_suppressThresh_txt=False,
                            cell_num_array=cc.cell_order_array,
                            showCellNumLabels=True,
                            imbounds=None, 
                            vbounds=None, 
                            colormap='inferno',
                            figsize=None, 
                            title_fntsz=14, 
                            ax_fntsz=12,
                            cellNumLabel_fontsize=11,
                            tickSize=None,
                            tickLabelSize=None,
                            xtickLabelRotation=None, 
                            ytickLabelRotation=None,
                            plot_titles=None,
                            global_title='Voltage Maps', 
                            cbar_titles='Voltage (V)',
                            x_title='Column',
                            y_title='Row',
                            frame_time=1000, 
                            repeat_bool=False,
                            includeMeanVolt=False,
                            plots_wspace=0.3,
                            cell_border_color='black',
                            N_rows=1):
    voltMaps_ls = [np.copy(voltMaps) if voltMaps.ndim==3 else np.copy(iff.conv_2D_to_3D(voltMaps)) for voltMaps in voltMaps_ls]
    if map_nos is None:
        map_nos = np.arange(voltMaps_ls[0].shape[0])
    if not figsize: 
        figsize = (7*len(voltMaps_ls), 6)
    if suppress_thresh:
        voltMaps_ls = [np.where(voltMaps<suppress_thresh, np.nan, voltMaps) for voltMaps in voltMaps_ls]
    if type(cbar_titles) != list:
        cbar_titles = [cbar_titles] * len(voltMaps_ls)
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, map_nos, voltMaps_ls) # get imbounds
    vbounds_ls = get_vbounds(voltMaps_ls, vbounds, [lbnd, ubnd]) # get vbounds
    print('vbounds:', vbounds_ls)
    fig = plt.figure(figsize=figsize)
    axs, caxs = init_subplots(len(voltMaps_ls), fig, plot_titles, x_title, y_title, title_fntsz, ax_fntsz, 
                              N_rows=N_rows, tickSize=tickSize, tickLabelSize=tickLabelSize, xtickLabelRotation=xtickLabelRotation, 
                              ytickLabelRotation=ytickLabelRotation)
    for ax in axs:
        configure_voltMap_subplot(ax, voltMaps_ls[0], suppress_thresh, ax_fntsz, cell_border_color=cell_border_color)
    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        for j in range(len(voltMaps_ls)):
            voltMap_feature_ls = make_voltMap_frame(axs[j], voltMaps_ls[j], i, vbounds_ls[j], colormap, ax_fntsz, includeMeanVolt, 
                                                    showCellNumLabels=showCellNumLabels, cellNumLabel_fontsize=cellNumLabel_fontsize, 
                                                    supText_fontsize=None, cell_num_array=cell_num_array, showMap_no=showMap_no)
            frames.append(voltMap_feature_ls)
    # add the suppressing text
    if suppress_thresh and include_suppressThresh_txt:
        suppress_txt = plt.gcf().text(0.825, 0.85, 'Suppressing voltages below: {} V'\
                                        .format(suppress_thresh),
                                fontsize=ax_fntsz-2, ha='right', va='bottom')
    # attach static colorbar
    for i in range(len(voltMaps_ls)):
        cbar = fig.colorbar(frames[i][0], cax=caxs[i])
        cbar.ax.tick_params(axis='both', which='both', width=tickSize, labelsize=tickLabelSize)
        cbar.set_label(cbar_titles[i], fontsize=ax_fntsz)
    fig.suptitle(global_title, fontsize=title_fntsz)
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=plots_wspace)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    # fps = int(1 / (frame_time/1000))
    fps = 1 / (frame_time/1000)
    if dispSingle:
        ani = fig
        print('dispSingle is True')
    return ani, fps # return animation and frames per second for saving to GIF

def displayVoltMaps(voltMaps_input, date='', map_nos=None, voltMaplabel='Map:',
                    suppress_thresh=None, cell_num_array=cc.cell_order_array,
                    imbounds=None, vbounds=None, colormap='jet',
                    figsize=None, title_fntsz=14, ax_fntsz=12,
                    global_title='Voltage Maps', cbar_title='Voltage (V)',
                    x_title='Column',
                    y_title='Row',
                    frame_time=1000, repeat_bool=False,
                    includeMeanVolt=False,
                    title_y_pos=0.94, showMap_no=True,
                    cell_border_color='white', ax_plot_title=''):
    voltMaps = np.copy(voltMaps_input)
    # if a single 2D voltmap is given, convert it to a 3D array
    if voltMaps.ndim < 3: voltMaps = iff.conv_2D_to_3D(voltMaps)
    if map_nos is None:
        map_nos = np.arange(voltMaps.shape[0])
    # if not figsize and not suppress_thresh: figsize=(6.5,6)
    # elif not figsize and suppress_thresh: figsize=(6, 6)
    if not figsize: figsize = (6, 6)
    if '_' in date: date = date.replace('_', '')
    if suppress_thresh: voltMaps = np.where(voltMaps<suppress_thresh, np.nan, voltMaps)
    if not vbounds: vbounds = [np.nanmin(voltMaps), np.nanmax(voltMaps)]
    fig = plt.figure(figsize=figsize)
    extent = None
    axs, caxs = init_subplots(1, fig, [ax_plot_title], # initalize subplot
                            x_title, y_title, title_fntsz, ax_fntsz)
    ax, cax = axs[0], caxs[0]
    if isinstance(imbounds, list) and len(imbounds) == 1:
        imbounds = imbounds * 2
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, map_nos, [voltMaps]) # get imbounds
    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        if ax_plot_title == '':
            # make the animated title text
            if date != '': date_text = '\nDate: {}'.format(date)
            else: date_text = ''
            if showMap_no:
                mapNo_text = '\n' + voltMaplabel + ' {}'.format(map_nos[i])
            else: mapNo_text = ''
            combined_text = global_title + date_text + mapNo_text
            title_plt_text = plt.gcf().text(0.5, title_y_pos, combined_text,
                                            fontsize=title_fntsz,
                                            ha='center', va='center')
        else:
            title_plt_text = plt.gcf().text(0, 0, '')
        voltMap_feature_ls = make_voltMap_frame(ax, voltMaps, i, vbounds,
                                                colormap, ax_fntsz, includeMeanVolt, 
                                                cell_num_array=cell_num_array)
        frames.append(voltMap_feature_ls + [title_plt_text])
    # add the suppressing text
    if suppress_thresh:
        suppress_txt = plt.gcf().text(0.825, 0.85, 'Suppressing voltages below: {} V'\
                                        .format(suppress_thresh),
                                fontsize=ax_fntsz-2, ha='right', va='bottom')
    # attach static colorbar
    cbar = fig.colorbar(frames[0][0], cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz, labelpad=10)
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.8)
    # adjust tick labels
    ax.set_xticks([i for i in range(voltMaps.shape[2])])
    ax.set_yticks([i for i in range(voltMaps.shape[1])])
    ax.set_xticklabels([i+1 for i in range(voltMaps.shape[2])])
    ax.set_yticklabels([i+1 for i in range(voltMaps.shape[1])])
    add_cell_labels(ax, ax_fntsz, voltMaps, border_color=cell_border_color, cell_labels=False)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    # fps = int(1 / (frame_time/1000))
    fps = 1 / (frame_time/1000)
    if dispSingle:
        ani = fig
        print('dispSingle is True')
    return ani, fps # return animation and frames per second for saving to GIF


def displayPolling(voltMaps_input, date='', voltTimes_input=np.array([]), suppress_thresh=None, 
                   display_suppressThresh=True,
                    imbounds=None, vbounds=None, colormap='turbo', cell_num_array=cc.cell_order_array,
                    figsize=None, 
                    title_fntsz=14, 
                    ax_fntsz=12,
                    tickSize=None,
                    tickLabelSize=None,
                    supText_fntsz=None,
                    showCellNumLabels=True, 
                    cellNumLabelSize=None,
                    xtickLabelRotation=None,
                    ytickLabelRotation=None,
                    global_title='', cbar_title='Voltage (V)',
                    x_title='Column',
                    y_title='Row',
                    frame_time=1000, repeat_bool=False, 
                    includeMeanVolt=True,
                    cell_border_color='white', 
                    title_hspace=None):
    voltMaps, voltTimes = np.copy(voltMaps_input), np.copy(voltTimes_input)
    # if a single 2D voltmap is given, convert it to a 3D array
    if voltMaps.ndim < 3: voltMaps = iff.conv_2D_to_3D(voltMaps)
    map_nos = np.arange(voltMaps.shape[0])
    # if not figsize and not suppress_thresh: figsize=(6.5,6)
    # elif not figsize and suppress_thresh: figsize=(6, 6)
    # elif not figsize and suppress_thresh: figsize=(6, 6)
    if not figsize: figsize = (6, 6)
    if '_' in date: date = date.replace('_', '')
    if suppress_thresh: voltMaps = np.where(voltMaps<suppress_thresh, np.nan, voltMaps)
    if not vbounds: vbounds = [np.nanmin(voltMaps), np.nanmax(voltMaps)]
    fig = plt.figure(figsize=figsize)
    extent = None
    axs, caxs = init_subplots(1, fig, [''], # initalize subplot
                            x_title, y_title, title_fntsz, ax_fntsz, tickSize=tickSize, 
                            tickLabelSize=tickLabelSize, xtickLabelRotation=xtickLabelRotation, 
                            ytickLabelRotation=ytickLabelRotation)
    ax, cax = axs[0], caxs[0]
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, map_nos, [voltMaps]) # get imbounds
    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        # make the animated title text
        if date != '': 
            date_text = '\nDate: {}'.format(date)
        else: 
            date_text = ''
        mapNo_text = '\nMap #: {}'.format(map_nos[i])
        if voltTimes.size == 0: 
            volt_text = ''
        else: 
            volt_text = 't = {:.2f} {}'.format(voltTimes[i], 'min')
            mapNo_text = '\n'
        combined_text = global_title + date_text + mapNo_text + volt_text
        title_plt_text = plt.gcf().text(0.5, 0.94, combined_text, fontsize=title_fntsz,
                                ha='center', va='center')
        voltMap_feature_ls = make_voltMap_frame(ax, voltMaps, i, vbounds, colormap, ax_fntsz, 
                                                includeMeanVolt, cell_num_array=cell_num_array, 
                                                showCellNumLabels=showCellNumLabels,
                                                cellNumLabel_fontsize=cellNumLabelSize)
        # add all features to current frame
        frames.append(voltMap_feature_ls+[title_plt_text])
    # add the suppressing text
    if suppress_thresh and display_suppressThresh:
        suppress_txt = ax.text(0.5, -0.05, 'Not displaying voltages below: {} V'\
                                        .format(suppress_thresh), fontsize=supText_fntsz, 
                               ha='center', va='top', transform=fig.transFigure)
    # attach static colorbar
    cbar = fig.colorbar(frames[0][0], cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz, labelpad=10)
    cbar.ax.tick_params(labelsize=tickLabelSize, width=tickSize)
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.8)
    # adjust tick labels
    ax.set_xticks([i for i in range(voltMaps.shape[2])])
    ax.set_yticks([i for i in range(voltMaps.shape[1])])
    ax.set_xticklabels([i+1 for i in range(voltMaps.shape[2])])
    ax.set_yticklabels([i+1 for i in range(voltMaps.shape[1])])
    add_cell_labels(ax, ax_fntsz, voltMaps, border_color=cell_border_color, cell_labels=False)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    if title_hspace is not None:
        fig.tight_layout(rect=(0, 0, 1, title_hspace))
    # fps = int(1 / (frame_time/1000))
    fps = 1 / (frame_time/1000)
    if dispSingle: ani = fig
    return ani, fps # return animation and frames per second for saving to GIF

def add_cell_labels(ax, fontsize, voltMaps, cell_labels=True, cell_borders=True,
                    border_color='white', cell_order_array=cc.cell_order_array):
    if cell_labels:
        for i in cell_order_array.flatten():
            args = np.argwhere(cell_order_array==i)[0]
            row, col = args[0], args[1]
            color = 'white'
            if np.any(np.isnan(voltMaps[:, row, col])): color='black'
            ax.text(col, row, int(i), color=color, ha='center', va='center',
                    fontsize=fontsize-6)
    if cell_borders:
        row_nums = np.array([i for i in range(voltMaps.shape[1])])
        col_nums = np.array([i for i in range(voltMaps.shape[2])])
        hline_locs = (row_nums[1:]+row_nums[:-1]) / 2
        vline_locs = (col_nums[1:]+col_nums[:-1]) / 2
        for loc in hline_locs: ax.axhline(loc, color=border_color)
        for loc in vline_locs: ax.axvline(loc, color=border_color)
        # print(hline_locs)

def displayIFs_wVoltMaps(input_cells,
                        ifs,
                        dx,
                        date='',
                        imbounds=None,
                        voltMap_vbounds=None,
                        if_vbounds=None,
                        cell_num_array=cc.cell_order_array,
                        voltMap_colormap='inferno',
                        if_colormap='jet',
                        suppress_thresh=None,
                        voltMap_type='high',
                        voltMapSubplotPlacement=0,
                        showVoltMapCellNos=True,
                        figsize=None,
                        title_fntsz=14,
                        ax_fntsz=12,
                        met_tickLabelSize=None,
                        met_tickSize=None,
                        met_xtickLabelRotation=None,
                        met_ytickLabelRotation=None,
                        volt_tickLabelSize=None,
                        volt_tickSize=None,
                        volt_xtickLabelRotation=None,
                        volt_ytickLabelRotation=None,
                        voltMapCellNo_fontsize=None,
                        plot_titles=None,
                        global_title='',
                        cbar_titles=['Voltage (V)', 'Figure (microns)'],
                        x_titles=None,
                        y_titles=None,
                        frame_time=500,
                        repeat_bool=False,
                        dispR=False,
                        cell_nos=None,
                        stats=False,
                        PV=True, 
                        PVr=False, 
                        PVq=False,
                        lambda_PV=False,
                        PV_labels=None,
                        rms=None,
                        dispMaxInds=None,
                        dispBoxCoords=None,
                        linecolors=['white', 'fuchsia'],
                        details=None,
                        includeCables=False,
                        merits=None,
                        merits_decimalplaces=2,
                        showImageNumber=True,
                        image_aspect='equal',
                        includeMeanVolt=False,
                        stats_textbox_coords=[0.03, 0.7],
                        staticStatsTextbox=False,
                        stats_fontsize=None,
                        stats_ha='right',
                        stats_va='bottom',
                        show_maxInd_textbox=True,
                        cell_border_color='white',
                        stats_units=['um'],
                        plots_wspace=0.3,
                        plots_topspace=0.85,
                        frame_num_label=None,
                        extent=None,
                        cbarLabelPerPlot=False):
    cells = copy.deepcopy(input_cells)
    # get a 3D array of voltMaps
    if len(cells) > 1:
        if voltMap_type == 'high': voltMaps = np.copy(np.stack([cell.high_voltMap for cell in cells], axis=0))
        elif voltMap_type == 'gnd': voltMaps = np.copy(np.stack([cell.gnd_voltMap for cell in cells], axis=0))
    else:
        if voltMap_type == 'high': voltMaps = iff.conv_2D_to_3D(cells[0].high_voltMap)
        elif voltMap_type == 'gnd': voltMaps = iff.conv_2D_to_3D(cells[0].gnd_voltMap)
    if suppress_thresh: voltMaps = np.where(voltMaps<suppress_thresh, np.nan, voltMaps)
    if not voltMap_vbounds: voltMap_vbounds = [np.nanmin(voltMaps), np.nanmax(voltMaps)]

    ifs_ls = ifs.copy()
    N_plots = len(ifs_ls) + 1
    ifs_ls = [iff.conv_2D_to_3D(if_map) if if_map.ndim==2 else if_map for if_map in ifs_ls] # convert any 2D arrays into 3D arrays
    if '_' in date: date = date.replace('_', '')

    if not figsize:
        figsize = (7*N_plots, 6)
    fig = plt.figure(figsize=figsize)
    if extent is None:
        extent = [fp.mk_extent(ifs_ls[i][0], dx) for i in range(len(ifs_ls))]
    if not plot_titles:
        plot_titles = [''] * N_plots
    if not x_titles: x_titles = ['Column'] + ['Azimuthal Dimension (mm)'] * len(ifs_ls)
    if not y_titles: y_titles = ['Row'] + ['Axial Dimension (mm)'] * len(ifs_ls)
    if not isinstance(stats_units, list):
        stats_units = [stats_units]
    if len(stats_units) < N_plots:
        stats_units = stats_units * ((N_plots - len(stats_units)) + 1)
    axs, caxs = init_subplots(N_plots, fig, plot_titles, # initalize subplots
                            x_titles, y_titles, title_fntsz, ax_fntsz, tickSize=met_tickSize, 
                            tickLabelSize=met_tickLabelSize, xtickLabelRotation=met_xtickLabelRotation, 
                            ytickLabelRotation=met_ytickLabelRotation)
    # adjust the tick params of the voltMap subplot separately
    axs[voltMapSubplotPlacement].tick_params(axis='x', which='major', width=volt_tickSize, labelsize=volt_tickLabelSize, 
                                             labelrotation=volt_xtickLabelRotation)
    axs[voltMapSubplotPlacement].tick_params(axis='y', which='major', width=volt_tickSize, labelsize=volt_tickLabelSize, 
                                            labelrotation=volt_ytickLabelRotation)

    # configure the ticks and static text for the voltMap subplot
    configure_voltMap_subplot(axs[voltMapSubplotPlacement], # axs[0]
                                voltMaps, suppress_thresh, ax_fntsz,
                                cell_border_color=cell_border_color)
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, cell_nos, ifs_ls) # get imbounds
    vbounds_ls = get_vbounds(ifs_ls, if_vbounds, [lbnd, ubnd]) # get vbounds for IFs if needed
    idx_txt, cell_nos = get_index_label(ifs_ls, cell_nos, frame_num_label) # "Image #:" or "Cell #:"
    if len(cbar_titles) < N_plots: # get the correct number of cbar titles
        cbar_titles += ['Figure (microns)']*(N_plots-len(cbar_titles))
    rms_ls, ptov_ls = [], []
    if stats: # get rms and P-to-V of IFs
        rms_ls, ptov_ls = get_stats(ifs_ls, PV=PV, PVr=PVr, PVq=PVq, lambda_PV=lambda_PV, rms=rms)

    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        # generate features for a frame
        if_feature_ls = make_frame(axs[:voltMapSubplotPlacement]+axs[voltMapSubplotPlacement+1:], # axs[1:]
                                ifs_ls, dx, i, extent, if_colormap, vbounds_ls,
                                global_title, idx_txt, cell_nos, title_fntsz,
                                dispMaxInds, ax_fntsz, stats, rms_ls, ptov_ls, dispBoxCoords,
                                linecolors, details, includeCables, merits, dispR,
                                showImageNumber, date, stats_textbox_coords, show_maxInd_textbox,
                                stats_units, PV=PV, PVr=PVr, PVq=PVq, lambda_PV=lambda_PV, PV_labels=PV_labels,
                                staticStatsTextbox=staticStatsTextbox, stats_fontsize=stats_fontsize,
                                stats_ha=stats_ha, stats_va=stats_va, merits_decimalplaces=merits_decimalplaces, image_aspect=image_aspect)

        voltMap_feature_ls = make_voltMap_frame(axs[voltMapSubplotPlacement], # axs[0]
                                                voltMaps, i, voltMap_vbounds,
                                                voltMap_colormap, ax_fntsz, includeMeanVolt,
                                                showCellNumLabels=showVoltMapCellNos, cellNumLabel_fontsize=voltMapCellNo_fontsize,
                                                cell_num_array=cell_num_array)
        total_feature_ls = if_feature_ls + voltMap_feature_ls # the first element in the voltmap feature list is the image
        frames.append(total_feature_ls)

    figMap_cbar_titles = cbar_titles[:voltMapSubplotPlacement]+\
                        cbar_titles[voltMapSubplotPlacement+1:]
    figMap_caxs = caxs[:voltMapSubplotPlacement]+\
                caxs[voltMapSubplotPlacement+1:]
    for i in range(N_plots-1): # attach static colorbar to IFs
        cbar = fig.colorbar(frames[0][i], cax=figMap_caxs[i])
        cbar.set_label(figMap_cbar_titles[i], fontsize=ax_fntsz)
        cbar.ax.tick_params(width=met_tickSize, labelsize=met_tickLabelSize)

    cbar = fig.colorbar(frames[0][len(if_feature_ls)], cax=caxs[voltMapSubplotPlacement])
    cbar.set_label(cbar_titles[voltMapSubplotPlacement], fontsize=ax_fntsz)
    cbar.ax.tick_params(width=volt_tickSize, labelsize=volt_tickLabelSize)
    # include the date
    if date != '':
        date_text = plt.gcf().text(0.80, 0.98, 'Date: {}'.format(date), fontsize=title_fntsz,
                                ha='right', va='top')
    if details is not None:
        fig.subplots_adjust(top=0.85, bottom=0.3, hspace=0.5, wspace=0.3)
    else:
        fig.subplots_adjust(top=plots_topspace, hspace=0.5, wspace=plots_wspace)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    # fps = int(1 / (frame_time/1000))
    fps = 1 / (frame_time/1000)
    if dispSingle: ani = fig
    return ani, fps # return animation and frames per second for saving to GIF

def configure_voltMap_subplot(ax, voltMaps, suppress_thresh, ax_fntsz, cell_border_color='white'):
    if suppress_thresh:
        suppress_txt = plt.gcf().text(0.5, 0.5, 'Suppressing voltages below: {} V'\
                                        .format(suppress_thresh),
                                fontsize=ax_fntsz-2, ha='right', va='bottom')
    ax.set_xticks([i for i in range(voltMaps.shape[2])])
    ax.set_yticks([i for i in range(voltMaps.shape[1])])
    ax.set_xticklabels([i+1 for i in range(voltMaps.shape[2])])
    ax.set_yticklabels([i+1 for i in range(voltMaps.shape[1])])
    add_cell_labels(ax, ax_fntsz, voltMaps, border_color=cell_border_color, cell_labels=False)

def make_voltMap_frame(ax, voltMaps, i, vbounds, colormap, ax_fntsz, includeMeanVolt,
                        showCellNumLabels=True, cellNumLabel_fontsize=None, supText_fontsize=None, 
                        cell_num_array=cc.cell_order_array, showMap_no=False):
    extent = None
    if supText_fontsize is None:
        supText_fontsize = ax_fntsz - 2
    # show the voltMap
    image = ax.imshow(voltMaps[i], extent=extent, aspect='auto',
                        cmap=colormap, vmin=vbounds[0], vmax=vbounds[1])
    if vbounds[0] is None:
        vbounds[0] = np.nanmin(voltMaps[i])
    if vbounds[1] is None:
        vbounds[1] = np.nanmax(voltMaps[i])
    # include the mean value for the voltMap?
    if includeMeanVolt: 
        mean_text = 'Mean: {:.2f} V'.format(np.nanmean(voltMaps[i]))
    else: 
        mean_text = ''
    mean_plt_text = ax.text(0., 1., mean_text, fontsize=supText_fontsize, 
                            ha='left', va='bottom', transform=ax.transAxes)
    cell_labels = [] # change the color of cells labeles based on which cell was set to 10 V
    if cellNumLabel_fontsize is None:
        cellNumLabel_fontsize = ax_fntsz-6
    flat_cell_num_array = np.sort(cell_num_array[~np.isnan(cell_num_array)].flatten())
    for j in flat_cell_num_array:
        args = np.argwhere(cell_num_array==j)[0]
        #print(type(args))
        #print('cell num: {}, args: {}'.format(j, args))
        row, col = args[0], args[1]
        color = 'white'
        if np.any(np.isnan(voltMaps[i, row, col])):
            color='black'
        if voltMaps[i][row][col] > 0.8*(max(vbounds)-min(vbounds)) + min(vbounds):#(np.nanmax(voltMaps[i])-np.nanmin(voltMaps[i])) + np.nanmin(voltMaps[i]):
            color='black'
        if showCellNumLabels:
            cell_label = ax.text(col, row, int(j), color=color, ha='center', va='center', fontsize=cellNumLabel_fontsize)
        else:
            cell_label = ax.text(col, row, '', color=color, ha='center', va='center', fontsize=cellNumLabel_fontsize)
        cell_labels.append(cell_label)
    return [image, mean_plt_text] + cell_labels

def plot_1D_averaged_ifs(ifs, dx, cell_nos=None, mean_direction=1, x_label='Azimuthal Dimension (mm)', y_label='Mean Axial Figure (um)', title='1D Averaged IFs', figsize=(8, 4), fontsize=12, 
                         linecolors=None, offset=-7.5, N_plots=1):
    mean_ifs = np.nanmean(ifs, axis=mean_direction)
    xvals = np.linspace(-mean_ifs.shape[1]/2*dx, mean_ifs.shape[1]/2*dx, mean_ifs.shape[1])
    if linecolors is None:
        tab_colors = list(mcolors.TABLEAU_COLORS)
        linecolors = [tab_colors[i%len(tab_colors)] for i in range(mean_ifs.shape[0])]
    lines_per_plot = int(round(mean_ifs.shape[0]/N_plots))
    fig, ax = plt.subplots(N_plots, 1, figsize=figsize)
    if N_plots == 1:
        ax = [ax, None]
    j = 0
    for i in range(mean_ifs.shape[0]):
        #print('i: {}, j: {}'.format(i, j))
        ax[j].set_xlabel(x_label, fontsize=fontsize)
        ax[j].set_ylabel(y_label, fontsize=fontsize)
        ax[j].plot(xvals, mean_ifs[i], color=linecolors[i])
        if cell_nos is not None:
            min_arg = np.argmin(mean_ifs[i])
            ax[j].text(xvals[min_arg], mean_ifs[i][min_arg]+offset, str(int(cell_nos[i])), color=linecolors[i], va='center', ha='center', fontsize=fontsize)
        if i > 0 and i%lines_per_plot == 0:
            j += 1
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()
    return fig


##############################################################################
############################### EXPERIMENTAL FUNCTIONS #######################
##############################################################################

def displayIFs_grid(ifs, dx, imbounds=None, vbounds=None, colormap='jet',
                    figsize=(11,10), title_fontsize=14, ax_fontsize=12,
                    plot_titles=[''], row_titles=[''], global_title='',
                    x_label='Azimuthal Dimension (mm)', y_label='Axial Dimension (mm)',
                    cbar_title='Figure (microns)', cell_nos=[None], stats=False, maxInds=[None],
                    dispRadii=False, banded_rows=True, frame_time=500, repeat_bool=False):
    """
    ifs: list of ifs or list of lists of ifs, where each grouped list of ifs
    specifies the row number.
    imbounds: list with [frame min, frame max] to specify what range of frames to render.
    dx: list of dx value(s) to use for each row. A list with one entry will be used
    for all subplots.
    vbounds: list of lists specifying the figure range to display per row:
    i.e. vbounds=[[vmin1,vmax1], [None, None], [None,vmax3]]
    plot_titles: list of plot titles that correspond with the order that ifs are given.
    row_titles: list of titles for each row of plots.
    global_title: overall title for figure.
    cell_nos: list of 1D arrays that contain the cell numbers for each if of a given row.
    The cell numbers for each row must be an explicit entry in the cell_nos list.
    stats: if true, display the RMS and PV of all plots on each frame
    maxInds: list of 3D arrays containing the maxInds for each plot. The maxInds
    for each plot must be explicitly given in the list.
    share_row_cbar: if true, only one color bar per row is generated.
    dispRadii: if true, displays the large radii and small radii for C1S04.
    banded_rows: alternates facecolor for each row.
    frame_time: time in ms to display each frame.
    repeat_bool: if true, animation will repeat after completion.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(global_title+'\n', fontsize=title_fontsize+2)
    ############################################################################
    # format the data inputs for sequential plotting and rendering
    ############################################################################
    if type(ifs[0]) is not list: ifs = [ifs]
    N_rows = len(ifs) # total number of rows
    N_plots = 0 # total number of plots
    for plot_ls in ifs:
        N_plots += len(plot_ls) # count the total number of plots to make
    if dx[0] is not list: dx = dx*N_rows # expand dx if not specified
    # expand plot_titles and row titles if not specified
    if len(plot_titles) != N_plots: plot_titles += ['']*(N_plots-len(plot_titles))
    if len(row_titles) != N_rows: row_titles += ['']*(N_rows-len(row_titles))
    # determine the frame_num_label based on whether cell_nos were supplied for
    # a given row
    if len(cell_nos) != N_rows: cell_nos += [None]*(N_rows-len(cell_nos))
    frame_num_label = [] # labels whether we are index by image, or by cell number
    for i in range(N_rows):
        if type(cell_nos[i]) is None: frame_num_label.append('\nImage #:')
        else: frame_num_label.append('\nCell #:')
    # expand maxInds if not specified
    if len(maxInds) != N_plots: maxInds += [None]*(N_plots-len(maxInds))
    # expand vbounds if not specified
    if vbounds is None:
        print('You literally gave None for vbounds.')
        vbounds = []
        for i in range(N_rows):
            row_ifs = ifs[i] # list of ifs for a row
            # find the min and max values per row and append to vbounds
            rowmin = min([np.nanmin(ifs) for ifs in row_ifs])
            rowmax = max([np.nanmax(ifs) for ifs in row_ifs])
            vbounds.append([rowmin, rowmax])
    else:
        for i, vbound_row in enumerate(vbounds):
            # print('vbounds', vbounds)
            # print("row: {}, vbound_row: {}".format(i, vbound_row))
            # print('This row has {} sets of ifs.'.format(len(ifs[i])))
            if vbound_row[0] is None: vbound_row[0] = min([np.nanmin(if_stack) for if_stack in ifs[i]])
            if vbound_row[1] is None: vbound_row[1] = max([np.nanmax(if_stack) for if_stack in ifs[i]])
    ############################################################################
    # Iteratively generate subfigures rows and subplot columns
    ############################################################################
    gs = fig.add_gridspec(N_rows, 1)
    plot_num = 0 # current plot number
    axs = [] # list of axes that will be created
    for i in range(N_rows):
        # create and format subfigure
        subfig = fig.add_subfigure(gs[i, 0])
        subfig.suptitle(row_titles[i]+frame_num_label[i], fontsize=title_fontsize)
        if banded_rows and i % 2 == 0: subfig.set_facecolor('0.75')
        N_cols = len(ifs[i])
        for j in range(N_cols):
            # create each subplot
            ax = subfig.add_subplot(1, N_cols, j+1)
            print('row is', i, 'column is', j, 'total columns is', N_cols)
            im = format_subplot(ax, plot_titles[plot_num], title_fontsize,
                                ax_fontsize, x_label, y_label,
                                ifs[i][j][0], dx[i], 'equal', colormap, vbounds[i],
                                N_cols)
            # create each colorbar
            format_colorbar(im, ax, subfig, cbar_title, N_cols, ax_fontsize, ax_fontsize)
            # if j+1 == N_cols:
            plot_num += 1
        subfig.subplots_adjust(bottom=0.15, top=0.85, hspace=0.8, wspace=0.8)

    return fig

def format_subplot(ax, plot_title, title_fontsize, ax_fontsize, x_label, y_label,
                    data, dx, aspect, colormap, vbounds, N_cols):
    """
    Return the image from formating a subplot within a figure.
    """
    ax.set_title(plot_title, fontsize=title_fontsize-(N_cols-1))
    ax.set_xlabel(x_label, fontsize=ax_fontsize-(N_cols-1))
    ax.set_ylabel(y_label, fontsize=ax_fontsize-(N_cols-1))
    ax.tick_params(axis='both', which='major', labelsize=10-((N_cols-1)*0.5))
    ax.tick_params(axis='both', which='minor', labelsize=8)
    extent = fp.mk_extent(data, dx)
    im = ax.imshow(data, extent=extent, aspect=aspect, cmap=colormap,
                    vmin=vbounds[0], vmax=vbounds[1])

    return im

def format_colorbar(im, ax, fig, cbar_title, N_cols, cbar_fontsize, tick_fontsize,
                    location='right', orientation='vertical', fraction=0.15,
                    shrink=1.0, pad=0.0):
    """
    Takes in an imshow image and associated axes and adds a color bar to it.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_title, fontsize=cbar_fontsize-(N_cols-1))
    cbar.ax.tick_params(labelsize=tick_fontsize-(N_cols+1))
