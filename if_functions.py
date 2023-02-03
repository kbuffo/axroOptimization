import numpy as np
from scipy import ndimage as nd
from scipy.interpolate import griddata
from operator import itemgetter
from itertools import chain
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
plt.rcParams['savefig.facecolor']='white'

import utilities.figure_plotting as fp
import imaging.man as man
import imaging.analysis as alsis
import imaging.stitch as stitch
import axroOptimization.evaluateMirrors as eva
import axroOptimization.solver as solver

def ptov(d):
    """Calculate peak to valley for an image"""
    return np.nanmax(d) - np.nanmin(d)

##############################################################################
######################### IF FORMATTING FUNCTIONS ############################
##############################################################################

def format_theoFig_IFs(fn, savename=None):
    """
    Format theoretical figure IFs produced by Vlad.
    Takes in a .fits file with unformatted IFs, and formats them.
    This includes:
    1. Transposing the data so that it's shape is (IF #, ypixels, xpixels)
    2. Multiplying by -1 so that the IF figure distortions are positive values
    3. Convert data from meters to microns
    4. Regrids the image data so that (ypixels, xpixels) -> (200, 200)
    5. Reflects the images across x and y axes so that first IF appears bottom
        left on plot.
    """
    # load data from fits file
    unform_ifs = pyfits.getdata(fn)
    scaled_ifs = 1.e6*(-np.transpose(unform_ifs, axes=(2,0,1))) # steps 1-3
    # step 4:
    scaled_ifs = np.array([man.newGridSize(scaled_ifs[i], (200, 200)) for i in range(len(scaled_ifs))])
    form_ifs = np.flip(scaled_ifs, axis=(1,2)) # step 5
    if savename:
        pyfits.writeto(savename, form_ifs, overwrite=True)
    return form_ifs

def format_theoSlp_IFs(fn, savename=None):
    """
    Format theoretical slope IFs produced by Vlad.
    Takes in a .fits file with unformatted IFs, and formats them.
    This includes:
    1. Transposing the data so that it's shape is (IF #, ypixels, xpixels)
    2. Convert data from radians to arcsec.
    3. Regrids the image data so that (ypixels, xpixels) -> (200, 200)
    4. Reflects the images across x and y axes so that first IF appears bottom
        left on plot.
    """
    # load data from fits file
    unform_ifs = pyfits.getdata(fn)
    # steps 1-2:
    scaled_ifs = 206264.806247*(np.transpose(unform_ifs, axes=(2,0,1)))
    # step 3:
    scaled_ifs = np.array([man.newGridSize(scaled_ifs[i], (200, 200)) for i in range(len(scaled_ifs))])
    form_ifs = np.flip(scaled_ifs, axis=(1,2)) # step 4
    if savename:
        pyfits.writeto(savename, form_ifs, overwrite=True)
    return form_ifs

def validateIFs(ifs_input, dx, triBounds=[10., 25.], triPlacement='all',
                pvThresh=0.3, printout=False):
    """
    Filter a set of measured IFs and return a set of true measured IFs
    and false IFs.
    triBounds: [ylength, xlength] of triangle in mm.
    placement: ['tl', 'tr', 'bl', 'br'] for triangle placement.
    pvThresh: IF peak figure threshold value (in microns) for filtering IFs.
    printout: print the results of filtering each image.
    """
    ifs = np.copy(ifs_input)
    corner_ifs = zeroCorners(ifs, dx, bounds=triBounds, placement=triPlacement)
    true_ifs = []
    false_ifs = []
    t_counter = -1
    f_counter = -1
    for i in range(corner_ifs.shape[0]):
        pv = ptov(corner_ifs[i])
        maxval = np.nanmax(corner_ifs[i])
        if pv >= pvThresh and maxval >= pvThresh/2.:
            true_ifs.append(ifs[i])
            status = True
            t_counter += 1
        else:
            false_ifs.append(ifs[i])
            status = False
            f_counter += 1
        if printout:
            print('|| Image no: {} || PV: {:.4f} || max: {:.4f} || Threshold: {:.4f} || Status: {} || True/False Counter {} ||'.format(i,pv,maxval,pvThresh, status,[t_counter, f_counter]))
            print('==================================================================================================================')
    true_ifs = np.array(true_ifs)
    false_ifs = np.array(false_ifs)
    return true_ifs, false_ifs

def match_ifs(ifs1, ifs2, maxInds1, maxInds2):
    """
    Returns:
        * the images in ifs2 that match the images in ifs1. The images in ifs2 are normalized to match the peak of ifs1. ifs2 must be of size >= the size of ifs1.
        * the maxInds in ifs2 that match the maxInds in ifs1. The maxInds in ifs2 are normalized to match the peak of ifs1.
        * the cell numbers of ifs2 that match with ifs1

    Typical use case: ifs1 = measured ifs, ifs2 = theoretical ifs
    will return: the set of theoretical ifs that match the measured ifs
    """
    ifs2_match = np.zeros(ifs1.shape)
    maxInds2_match = np.zeros(maxInds1.shape)
    cell_nos = np.zeros(len(ifs1))

    for i in range(ifs1.shape[0]):
        ifs2_matched_idx = find_nearest_coord_idx(maxInds1[i], maxInds2)
        ifs2_matched_if = np.copy(ifs2[ifs2_matched_idx])
        maxInds2_matched_maxInd = np.copy(maxInds2[ifs2_matched_idx])
        ratio = maxInds1[i][0][2] / maxInds2[ifs2_matched_idx][0][2]
        ifs2_matched_if *= ratio # normalize the if to match the peak with if1
        maxInds2_matched_maxInd[0][2] *= ratio # normalize the maxInd figure value
        ifs2_match[i] = ifs2_matched_if
        maxInds2_match[i] = maxInds2_matched_maxInd
        cell_nos[i] = ifs2_matched_idx+1

    duplicate_cell_nos = find_repeat_elements(cell_nos)
    if duplicate_cell_nos.size == 0:
        print('A unique cell number was found for each IF.')
    else:
        print('The following cell numbers were matched more than once:', duplicate_cell_nos)
    return ifs2_match, maxInds2_match, cell_nos

def find_nearest_coord_idx(maxCoord, maxInds):
    """
    Returns the index from maxInds whose elements are closest to maxCoord.
    maxInds is a n x 2 x 3 array with row elements of [y_value, x_value, figure_value]
    maxCoord is a single maxInd (size: 1 x 2 x 3)
    """
    return np.sqrt((maxCoord[0][0]-maxInds[:, 0, 0])**2 + (maxCoord[0][1]-maxInds[:, 0, 1])**2).argmin()

def find_repeat_elements(array):
    """
    Return the repeated elements of an array.
    """
    unique_array, repeat_counts = np.unique(array, return_counts=True)
    duplicates = unique_array[repeat_counts>1]
    return duplicates

def get_imageNum(cell_nos, cell_no):
    """
    Finds the index of a cell number from an array of cell numbers.
    """
    return np.argwhere(cell_nos==cell_no)[0][0]

##############################################################################
##################### ARRAY MASKING FUNCTIONS ################################
##############################################################################

def frameIFs(ifs_input, dx, triBounds=[20.,22.], edgeTrim=6., triPlacement='all',
            edgePlacement='all', triVal=0., edgeVal=0.):
    """
    Combines zeroEdges and zeroCorners to create a "frame" of NaNs around an image.
    """
    ifs = np.copy(ifs_input)
    edges_ifs = zeroEdges(ifs, dx, amount=edgeTrim, placement=edgePlacement, setval=edgeVal)
    frame_ifs = zeroCorners(edges_ifs, dx, bounds=triBounds, placement=triPlacement, setval=triVal)
    return frame_ifs

def zeroEdges(ifs_input, dx, amount=5., placement='all', setval=0.):
    """
    Applies edges to a set images used to filter out noise in IFs.
    amount: length from edges to set to setval
    placement: ['t', 'b', 'l', 'r'] or 'all' for where to place edges
    setval: value to set edges at in array
    """
    ifs = np.copy(ifs_input)
    pixTrim = int(round(amount/dx))
    if 't' in placement or 'all' in placement:
        ifs[:, :pixTrim, :] = setval
    if 'b' in placement or 'all' in placement:
        ifs[:, -pixTrim:, :] = setval
    if 'l' in placement or 'all' in placement:
        ifs[:, :, :pixTrim] = setval
    if 'r' in placement or 'all' in placement:
        ifs[:, :, -pixTrim:] = setval
    return ifs

def zeroCorners(ifs_input, dx, bounds=[10., 25.], placement='all', setval=0.):
    """
    Applies triangles to a set images used to filter out noise in IFs.
    bounds = [ylength, xlength] of triangle in mm.
    placement: ['tl', 'tr', 'bl', 'br'] for triangle placement
    setval: value to set triangles at in array
    """
    ifs = np.copy(ifs_input)
    yDisp = int(round(bounds[0]/dx))
    xDisp = int(round(bounds[1]/dx))
    slope = xDisp/yDisp
    corners = np.ones(ifs.shape)
    for iy in range(yDisp):
        if iy == 0:
            ix = xDisp
        else:
            ix = xDisp - int(round(iy*slope))
        if ix < 0:
            break
        if 'tl' in placement or 'all' in placement:
            corners[:, iy, :ix] = 0
        if 'tr' in placement or 'all' in placement:
            corners[:, iy, -ix:] = 0
        if 'bl' in placement or 'all' in placement:
            corners[:, -iy, :ix] = 0
        if 'br' in placement or 'all' in placement:
            corners[:, -iy, -ix:] = 0
    ifs[corners==0] = setval
    return ifs


def isoIFs(ifs, dx, boxCoords, setval=np.nan):
    iso_ifs = np.full((ifs.shape), np.nan)
    for i in range(ifs.shape[0]):
        # Primary IF
        y_lbnd = int(round(boxCoords[i][0][0][0])) # top left
        y_ubnd = int(round(boxCoords[i][0][2][0])) # bottom left
        x_lbnd = int(round(boxCoords[i][0][0][1])) # top left
        x_ubnd = int(round(boxCoords[i][0][1][1])) # top right
        for j in range(y_ubnd-y_lbnd):
            iso_ifs[i][y_lbnd+j][x_lbnd:x_ubnd+1] = ifs[i][y_lbnd+j][x_lbnd:x_ubnd+1]
        # Secondary IF
        if -1 not in boxCoords[i][1]:
            y_lbnd = int(round(boxCoords[i][1][0][0])) # top left
            y_ubnd = int(round(boxCoords[i][1][2][0])) # bottom left
            x_lbnd = int(round(boxCoords[i][1][0][1])) # top left
            x_ubnd = int(round(boxCoords[i][1][1][1])) # top right
            for j in range(y_ubnd-y_lbnd):
                iso_ifs[i][y_lbnd+j][x_lbnd:x_ubnd+1] = ifs[i][y_lbnd+j][x_lbnd:x_ubnd+1]
    return iso_ifs

##############################################################################
############################### ANIMATION FUNCTIONS ##########################
##############################################################################

def displayIFs(ifs, dx, imbounds=None, vbounds=None, colormap='jet',
                figsize=None, title_fntsz=14, ax_fntsz=12,
                plot_titles = None,
                global_title='', cbar_title='Figure (microns)',
                x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)',
                frame_time=500, repeat_bool=False, dispR=False,
                cell_nos=None, stats=False, dispMaxInds=None, dispBoxCoords=None,
                linecolors=['white', 'fuchsia'], details=None, includeCables=False):

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

    dispMaxInds: A list whose elments are 3D arrays that are the maxInds for the
    corresponding IF stack given.

    details: A list of cell objects for the plots, assuming all plots are showing the same cell each frame.
    If includedm details allows grid coords, region, pin cell #, board, dac, channel,
    and cables (optionally) to be displayed on every frame.

    includeCables: If set to True, details will display the cable connections in addition to
    the other information included in the cell objects lists.
    """
    ifs_ls = ifs.copy()
    N_plots = len(ifs_ls)
    if not figsize:
        figsize = (7*len(ifs_ls), 6)
    fig = plt.figure(figsize=figsize)
    extent = fp.mk_extent(ifs_ls[0][0], dx)
    if not plot_titles:
        plot_titles = [''] * len(ifs_ls)
    axs, caxs = init_subplots(N_plots, fig, plot_titles, # initalize subplots
                            x_title, y_title, title_fntsz, ax_fntsz)
    vbounds_ls = get_vbounds(ifs_ls, vbounds) # get vbounds
    ubnd, lbnd, dispSingle = get_imbounds(imbounds, cell_nos, ifs_ls) # get imbounds
    idx_txt, cell_nos = get_index_label(ifs_ls, cell_nos) # "Image #:" or "Cell #:"
    if type(cbar_title) != list:
        cbar_title = [cbar_title] * len(ifs_ls)
    rms_ls, ptov_ls = [], []
    if stats: # get rms and P-to-V of IFs
        rms_ls, ptov_ls = get_stats(ifs_ls)

    frames = []
    for i in range(lbnd, ubnd): # create frames for animation
        # generate features for a frame
        feature_ls = make_frame(axs, ifs_ls, dx, i, extent, colormap, vbounds_ls,
                                global_title, idx_txt, cell_nos, title_fntsz,
                                dispMaxInds, ax_fntsz, stats, rms_ls, ptov_ls, dispBoxCoords,
                                linecolors, details, includeCables)
        frames.append(feature_ls)

    for i in range(len(ifs_ls)): # attach static colorbar
        cbar = fig.colorbar(frames[0][i], cax=caxs[i])
        if i == len(ifs_ls)-1:
            cbar.set_label(cbar_title[i], fontsize=ax_fntsz)
    if details is not None:
        fig.subplots_adjust(top=0.85, bottom=0.3, hspace=0.5, wspace=0.3)
    else:
        fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.3)
    # create animation
    ani = animation.ArtistAnimation(fig, frames, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    fps = int(1 / (frame_time/1000))
    if dispSingle: ani = fig
    return ani, fps # return animation and frames per second for saving to GIF


def init_subplots(N_plots, fig, title_ls, x_title, y_title, title_fontsize, ax_fontsize):
    """
    Initializes a row of subplots based on the number of IF stacks provided.
    Generates the axes and sets their features.
    """
    ax_ls, cax_ls = [], []
    gs = gridspec.GridSpec(1, N_plots)
    for i in range(N_plots):
        ax = fig.add_subplot(gs[i])
        ax.set_title(title_ls[i], fontsize=title_fontsize)
        ax.set_xlabel(x_title, fontsize=ax_fontsize)
        ax.set_ylabel(y_title, fontsize=ax_fontsize)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.10)
        ax_ls.append(ax)
        cax_ls.append(cax)
    return ax_ls, cax_ls
    pass


def get_vbounds(ifs_ls, vbounds):
    """
    Formats the user provided vbounds into a list of vbounds.
    """
    vbounds_ls = []
    if vbounds is None:
        # generate different vbounds for each plot
        for ifs in ifs_ls:
            vbounds_ls.append([np.nanmin(ifs), np.nanmax(ifs)])
    elif type(vbounds[0]) is list and len(vbounds) == len(ifs_ls):
        # use different user-supplied vbounds for each plot
        vbounds_ls = vbounds
    elif type(vbounds) is list and len(vbounds) == 2:
        # use the same user-supplied vbounds for each plot
        vbounds_ls = [vbounds]*len(ifs_ls)
    return vbounds_ls


def get_index_label(ifs_ls, cell_nos):
    """
    Returns a label that denotes whether we are displaying images or explicitly
    indexed cells.
    """
    if type(cell_nos) != type(None):
        idx_label = 'Cell'
    else:
        idx_label = 'Image'
        cell_nos = np.arange(0, len(ifs_ls[0]))
    return idx_label, cell_nos


def get_imbounds(imbounds, cell_nos, ifs_ls):
    """
    Formats the user provided imbounds to get the appropriate images to display
    from the IF stacks.
    """
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
    return upperBound, lowerBound, displaySingle


def get_stats(ifs_ls):
    """
    Returns a list of rms and P-to-V values for a list of IF stacks.
    """
    rms_ls, ptov_ls = [], []
    for ifs in ifs_ls:
        rms_vals = np.array([alsis.rms(ifs[i]) for i in range(ifs.shape[0])])
        ptov_vals = np.array([alsis.ptov(ifs[i]) for i in range(ifs.shape[0])])
        rms_ls.append(rms_vals)
        ptov_ls.append(ptov_vals)
    return rms_ls, ptov_ls


def make_frame(axs, ifs_ls, dx, frame_num, extent, colormap, vbounds_ls,
                global_title, idx_txt, cell_nos, title_fntsz, dispMaxInds, ax_fntsz,
                stats, rms_ls, ptov_ls, dispBoxCoords, linecolors, details, includeCables):
    """
    Generates all the features that will be animated in the figure.
    """
    feature_ls = [] # list that holds all the features of a frame

    for i, ifs in enumerate(ifs_ls): # plot the data
        image = axs[i].imshow(ifs[frame_num], extent=extent, aspect='auto',
                            cmap=colormap, vmin=vbounds_ls[i][0], vmax=vbounds_ls[i][1])
        feature_ls.append(image)

    cell_no = cell_nos[frame_num] # make the global title
    txtstring = global_title + '\n' + idx_txt + ' #: {}'.format(int(cell_no))
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
                                                    ax_fntsz, linecolors)
    # unpack nested lists into regular lists
    vlines, hlines = list(chain(*vlines)), list(chain(*hlines))

    # initialize the rms and ptov text boxes as blank
    stats_textboxes = [ax.text(0,0,'') for ax in axs]
    if stats: # draw the stats text boxes
        stats_textboxes = illustrate_stats(axs, frame_num, rms_ls, ptov_ls,
                                            ax_fntsz)

    # initialize the box coord rectangles as blank
    boxCoords_rectangles = [ax.text(0,0,'') for ax in axs]
    if dispBoxCoords is not None:
        boxCoords_rectangles = illustrate_boxCoords(axs, frame_num, dispBoxCoords,
                                                    ifs_ls, dx, linecolors)

    # initalize the details text boxes as blank
    details_boxes = [ax.text(0,0, '') for ax in axs]
    if details is not None:
        details_boxes = illustrate_details(frame_num, details, includeCables, ax_fntsz)

    # combine all features into single list
    feature_ls += [title_plt_text] + vlines + hlines + maxvals + stats_textboxes + boxCoords_rectangles + details_boxes
    return feature_ls


def illustrate_maxInds(dispMaxInds, frame_num, axs, vlines, hlines, maxvals, dx,
                        ifs_ls, ax_fntsz, linecolors):
    """
    Creates the coordinate tracking lines for IFs and the associated textbox.
    """
    for i, maxInds in enumerate(dispMaxInds):
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
        maxvals[i] = axs[i].text(x_txt_pos, y_txt_pos, maxInd_txt, color='black', fontsize=ax_fntsz-4,
                            transform=axs[i].transAxes, va='top', bbox=dict(facecolor='white', alpha=0.65))
    return vlines, hlines, maxvals


def illustrate_stats(axs, frame_num, rms_ls, ptov_ls, ax_fntsz):
    """
    Creates the textbox that will display the RMS and P-to-V values.
    """
    stats_textbox_ls = []
    for i in range(len(rms_ls)):
        stats_txt = "RMS: {:.2f} um\nPV: {:.2f} um".format(rms_ls[i][frame_num], ptov_ls[i][frame_num])
        x_txt_pos, y_txt_pos = 0.7, 0.03
        # print('frame #:', frame_num, 'threshold:', len(rms_ls[i])/2)
        if frame_num > len(rms_ls[i])/2: # move text box if it will block IF
            x_txt_pos = 0.03
        stats_textbox = axs[i].text(x_txt_pos, y_txt_pos, stats_txt, fontsize=ax_fntsz,
                                    transform=axs[i].transAxes, va='bottom',
                                    bbox=dict(facecolor='white', alpha=0.65))
        stats_textbox_ls.append(stats_textbox)
    return stats_textbox_ls

def illustrate_boxCoords(axs, frame_num, dispBoxCoords, ifs_ls, dx, linecolors):
    """
    Creates the rectangles that will enclose each IF.
    """
    rectangles = []
    for i, boxCoords in enumerate(dispBoxCoords):
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


##############################################################################
############################### PlOT FUNCTIONS ###############################
##############################################################################

def stats_hist(stats, bins, x_labels, plot_titles=None, global_title=None,
                figsize=(8,5), colors=None, N_xticks=20, disp_Nbins=False,
                showMean=None, ylog=False):
    """
    Returns a row of histograms displaying the distributions of different stats.
    stats: list of 1D arrays that contain a set of stats. Each element of the list gets its own plot.
    x_labels: list of titles for x-axis of each plot.
    plot_titles: list of titles for each plot.
    global_title: title for overall figure.
    color: list of colors of histogram bars.
    N_xticks: number of xtick markers to generate
    """

    fig, axs = plt.subplots(1, len(stats), figsize=figsize)
    if len(stats) == 1: axs = np.array(axs)
    Nbins_text = ''
    if disp_Nbins: Nbins_text = '\nN Bins: {}'.format(bins)
    if global_title: fig.suptitle(global_title)
    if not plot_titles: plot_titles = [''] * len(stats)
    if not colors: colors = ['limegreen'] * len(stats)
    for i, ax in enumerate(axs.flat):
        ax.set_title(plot_titles[i]+Nbins_text)
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel('# of Cells')
        ax.hist(stats[i], bins=bins, range=None, color=colors[i],
                label='mean: {:.2f} {}\nmedian: {:.2f} {}'.format(np.nanmean(stats[i]), showMean, np.nanmedian(stats[i]), showMean))
        xticks = np.linspace(np.nanmin(stats[i]), np.nanmax(stats[i]), num=N_xticks)
        xtick_labels = xticks.round(2)
        ax.set_xticks(ticks=xticks, labels=xtick_labels, rotation=45, fontsize=10-(2*(int(N_xticks/10)-1)))
        if ylog: ax.set_yscale('log')
        if showMean is not None: ax.legend()
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.tight_layout()
    return fig


def cell_yield_scatter(maxInds, image_shape, dx, vbounds=None, colormap='jet',
                    figsize=(8,8), title_fntsz=14, ax_fntsz=12,
                    global_title="C1S04 Spatial Distribution of IFs' Maximum Figure Change",
                    plot_titles=None, cbar_titles=None, x_title='Azimuthal Dimension (mm)',
                    y_title='Axial Dimension (mm)', xlabels=np.arange(-45, 50, 5), ylabels=np.arange(-50, 55, 5)):

        """
        Returns a row of scatter plots showing all the coordinates of maxInds for a given IF stack, along
        with their corresponding figure change.
        maxInds: list of 3D numpy arrays. Each element in the list gets a separate plot.
        image_shape: the shape of the associated IF images (i.e. (200, 200)).
        dx: pixel spacing for IF imagesvbounds
        vbounds: a list like [lowerBound, upperBound] to set a single colorbar scale
        for all plots.
        x_labels, y_labels: an array of values that will be used as the labels for the axes' tick marks.
        """
        N_plots = len(maxInds)
        if not vbounds: vbounds = [None, None]
        if not plot_titles: plot_titles = [''] * N_plots
        if not cbar_titles: cbar_titles = ['Figure (microns)'] * N_plots
        fig, axs = plt.subplots(1, N_plots, figsize=figsize)
        if N_plots == 1: axs = np.array(axs)
        fig.suptitle(global_title, fontsize=title_fntsz)
        xticks = xlabels/dx + image_shape[1]/2 # calculate tick mark locations based on labels
        yticks = ylabels/dx + image_shape[0]/2
        for i, ax in enumerate(axs.flat):
            # need to flip the y coordinates since scatter has origin at bottom
            # maxInds has origin at top
            yvals = image_shape[0] - maxInds[i][:, 0, 0]
            xvals = maxInds[i][:, 0, 1] # get the array of x coordinates
            maxvals = maxInds[i][:, 0, 2] # get the array of figure vals
            scatter_plot = ax.scatter(xvals, yvals, c=maxvals, vmin=vbounds[0],
                                    vmax=vbounds[1], cmap=colormap)
            # ax.axhline(y=52.65, color='green')
            # ax.axvline(x=55.5, color='blue')
            ax.set_xticks(xticks) # adjust axis parameters
            ax.set_yticks(yticks)
            ax.set_xticklabels(xlabels, rotation=45)
            ax.set_yticklabels(ylabels)
            ax.set_xlabel(x_title, fontsize=ax_fntsz)
            ax.set_ylabel(y_title, fontsize=ax_fntsz)
            ax.set_title(plot_titles[i], fontsize=title_fntsz)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.10)
            cbar = plt.colorbar(scatter_plot, cax=cax)
            cbar.set_label(cbar_titles[i], fontsize=ax_fntsz)
            ax.grid(True)
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.35)
        return fig

##############################################################################
############################### STITCHING FUNCTIONS ##########################
##############################################################################
def conv_2D_to_3D(array):
    """
    Convert a 2D array into a 3D array.
    """
    return array.reshape(1, array.shape[0], array.shape[1])

def calc_maxInds_distance(maxInds1, maxInds2):
    """
    Returns a 1D array with the distances between 2 sets of maxInds.
    """
    # sqrt((y1-y2)^2 + (x1-x2)^2)
    distances = np.sqrt((maxInds1[:,0,0] - maxInds2[:,0,0])**2
                            + (maxInds1[:,0,1] - maxInds2[:,0,1])**2)
    return distances


def get_box_coords(maxInds, extent, image_shape):
    # 1st dimension: image number
    # 2nd dimension: primary or secondary IF
    # 3rd dimension: coordinates (from 1st row to last is
    # top left, top right, bottom left, bottom right)
    # 4th dimension: 1st column is y_ind, 2nd column is x_ind for a coordinate
    box_coords = np.zeros((maxInds.shape[0], 2, 4, 2))
    for i in range(maxInds.shape[0]):
        # get box coords for an image
        box_coord = calc_box_coords(maxInds[i], extent, image_shape)
        box_coords[i] = box_coord
    return box_coords

def calc_box_coords(maxInd, extent, image_shape):
    box_coords = np.full((2,4,2), -1.)
    y_image_shape, x_image_shape = float(image_shape[0]), float(image_shape[1])
    # PRIMARY IF
    y1, x1 = maxInd[0][0], maxInd[0][1]
    # y for top left and top right
    if y1 - extent > 0:
        box_coords[0][0][0] = y1 - extent # y for top left
        box_coords[0][1][0] = y1 - extent # y for top right
    else: box_coords[0][0][0], box_coords[0][1][0] = 0., 0.
    # y for bottom left and bottom right
    if y1 + extent < y_image_shape:
        box_coords[0][2][0] = y1 + extent # y for bottom left
        box_coords[0][3][0] = y1 + extent # y for bottom right
    else: box_coords[0][2][0], box_coords[0][3][0] = y_image_shape, y_image_shape
    # x for top left and bottom left
    if x1 - extent > 0:
        box_coords[0][0][1] = x1 - extent # x for top left
        box_coords[0][2][1] = x1 - extent # x for bottom left
    else: box_coords[0][0][1], box_coords[0][2][1] = 0., 0.
    # x for top right and bottom right
    if x1 + extent < x_image_shape:
        box_coords[0][1][1] = x1 + extent # x for top right
        box_coords[0][3][1] = x1 + extent # x for bottom right
    else: box_coords[0][1][1], box_coords[0][3][1] = x_image_shape, x_image_shape

    # SECONDARY IF
    if -1 not in maxInd[1]:
        y2, x2 = maxInd[1][0], maxInd[1][1]
        # y for top left and top right
        if y2 - extent > 0:
            box_coords[1][0][0] = y2 - extent # y for top left
            box_coords[1][1][0] = y2 - extent # y for top right
        else: box_coords[1][0][0], box_coords[1][1][0] = 0., 0.
        # y for bottom left and bottom right
        if y2 + extent < y_image_shape:
            box_coords[1][2][0] = y2 + extent # y for bottom left
            box_coords[1][3][0] = y2 + extent # y for bottom right
        else: box_coords[1][2][0], box_coords[1][3][0] = y_image_shape, y_image_shape
        # x for top left and bottom left
        if x2 - extent > 0:
            box_coords[1][0][1] = x2 - extent # x for top left
            box_coords[1][2][1] = x2 - extent # x for bottom left
        else: box_coords[1][0][1], box_coords[1][2][1] = 0., 0.
        # x for top right and bottom right
        if x2 + extent < x_image_shape:
            box_coords[1][1][1] = x2 + extent # x for top right
            box_coords[1][3][1] = x2 + extent # x for bottom right
        else: box_coords[1][1][1], box_coords[1][3][1] = x_image_shape, x_image_shape
    return box_coords

def get_maxInds_from_boxCoords(boxCoords, ifs):
    maxInds = np.full((boxCoords.shape[0], 2, 3), -1.)
    for i in range(boxCoords.shape[0]):
        # primary IF
        y1 = boxCoords[i][0][0][0] # top left
        y2 = boxCoords[i][0][2][0] # bottom right
        x1 = boxCoords[i][0][0][1] # top left
        x2 = boxCoords[i][0][1][1] # top right
        maxInds[i][0][0] = (y2-y1)/2 + y1# calculate maxInds
        maxInds[i][0][1] = (x2-x1)/2 + x1
        maxInds[i][0][2] = ifs[i][int(round((y2-y1)/2+y1))][int(round((x2-x1)/2+x1))]
        # secondary IF
        if -1 not in boxCoords[i][1]:
            y1 = boxCoords[i][1][0][0] # top left
            y2 = boxCoords[i][1][2][0] # bottom right
            x1 = boxCoords[i][1][0][1] # top left
            x2 = boxCoords[i][1][1][1] # top right
            maxInds[i][1][0] = (y2-y1)/2 + y1# calculate maxInds
            maxInds[i][1][1] = (x2-x1)/2 + x1
            maxInds[i][1][2] = ifs[i][int(round((y2-y1)/2+y1))][int(round((x2-x1)/2+x1))]
    return maxInds

def stitch_ifs(ifs1, ifs2, maxInds1, maxInds2, boxCoords1, boxCoords2, degree):
    """
    Computes a transformation to bring maxInds2 and ifs2 in line with ifs1 and maxInds1.
    The transformation is then applied to ifs2 and maxInds2 and then returned.

    The type of transformation is determined by degree, which can equal the following:
    2: translation only (dx, dy)
    3: translation and rotation (dx, dy, theta)
    4: translation, rotation, and magnification (dx, dy, theta, mag)
    5: translation, rotation, and separate magnifications (dx, dy, theta, x_mag, y_mag)
    """
    yf1, xf1 = maxInds1[:,0,0], maxInds1[:,0,1] # get x and y coordinates from maxInds
    yf2, xf2 = maxInds2[:,0,0], maxInds2[:,0,1]
    # calculate transformations
    tx, ty, theta, mag, x_mag, y_mag = None, None, None, None, None, None
    if degree == 2:
        print('Still a work in progress')
        pass
    if degree == 3:
        tx, ty, theta = stitch.matchFiducials(xf1, yf1, xf2, yf2)
    elif degree == 4:
        tx, ty, theta, mag = stitch.matchFiducials_wMag(xf1, yf1, xf2, yf2)
    elif degree == 5:
        tx, ty, theta, x_mag, y_mag = stitch.matchFiducials_wSeparateMag(xf1, yf1, xf2, yf2)
    else:
        print('degree must be 2, 3, 4, or 5.')
        pass
    if tx < 0: xf1 -= tx
    if ty < 0: yf1 -= ty
    print('\nCalculated transformation:\n dx: {}\n dy: {}\n theta: {}\n mag: {}\n x_mag: {}\n y_mag: {}\n'.format(tx, ty, theta, mag, x_mag, y_mag))

    # stitching process
    stitch_ifs = np.zeros(ifs2.shape)
    stitch_boxCoords = np.full(boxCoords2.shape, -1.)
    for i in range(ifs2.shape[0]):
        img1, img2 = ifs1[i], ifs2[i]
        y1_boxCoords = boxCoords2[i,0,:,0] # primary IF
        x1_boxCoords = boxCoords2[i,0,:,1]
        y2_boxCoords = boxCoords2[i,1,:,0] # secondary IF
        x2_boxCoords = boxCoords2[i,1,:,1]
        #Get x,y,z points from reference image
        x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,np.shape(img1)[1]],\
                               ylim=[0,np.shape(img1)[0]])
        #Get x,y,z points from stitched image
        x2,y2,z2 = man.unpackimage(img2,xlim=[0,np.shape(img2)[1]],\
                               ylim=[0,np.shape(img2)[0]])

        #Apply transformations to x,y coords of stitched image
        if degree == 2:
            pass
        elif degree == 3:
            x2_t, y2_t = stitch.transformCoords(x2, y2, tx, ty, theta)
            x1_t_boxCoords, y1_t_boxCoords = stitch.transformCoords(x1_boxCoords, y1_boxCoords, tx, ty, theta)
            if -1 not in y2_boxCoords:
                x2_t_boxCoords, y2_t_boxCoords = stitch.transformCoords(x2_boxCoords, y2_boxCoords, tx, ty, theta)
        elif degree == 4:
            x2_t, y2_t = stitch.transformCoords_wMag(x2, y2, tx, ty, theta, mag)
            x1_t_boxCoords, y1_t_boxCoords = stitch.transformCoords_wMag(x1_boxCoords, y1_boxCoords, tx, ty, theta, mag)
            if -1 not in y2_boxCoords:
                x2_t_boxCoords, y2_t_boxCoords = stitch.transformCoords_wMag(x2_boxCoords, y2_boxCoords, tx, ty, theta, mag)
        elif degree == 5:
            x2_t, y2_t = stitch.transformCoords_wSeparateMag(x2, y2, tx, ty, theta, x_mag, y_mag)
            x1_t_boxCoords, y1_t_boxCoords = stitch.transformCoords_wSeparateMag(x1_boxCoords, y1_boxCoords, tx, ty, theta, x_mag, y_mag)
            if -1 not in y2_boxCoords:
                x2_t_boxCoords, y2_t_boxCoords = stitch.transformCoords_wSeparateMag(x2_boxCoords, y2_boxCoords, tx, ty, theta, x_mag, y_mag)

        # Interpolate stitched image onto expanded image grid
        newimg = griddata((x2_t,y2_t),z2,(x1,y1),method='linear')
        newimg = newimg.reshape(np.shape(img1))
        print('Image {}: Interpolation ok'.format(i+1))
        # Images should now be in the same reference frame
        # Time to apply tip/tilt/piston to minimize RMS
        newimg = stitch.matchPistonTipTilt(img1,newimg)
        stitch_ifs[i] = newimg
        stitch_boxCoords[i,0,:,0] = y1_t_boxCoords
        stitch_boxCoords[i,0,:,1] = x1_t_boxCoords
        if -1 not in y2_boxCoords:
            stitch_boxCoords[i,1,:,0] = y2_t_boxCoords
            stitch_boxCoords[i,1,:,1] = x2_t_boxCoords
    stitch_maxInds = get_maxInds_from_boxCoords(stitch_boxCoords, stitch_ifs)

    return stitch_ifs, stitch_maxInds, stitch_boxCoords, (tx, ty, theta, mag, x_mag, y_mag)

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

##############################################################################
############################### OLD FUNCTIONS ################################
##############################################################################
# def find_nearest(maxInds, point):
#     """
#     Finds nearest neighbor between a meas IF maximum and the set
#     of theo IFs by minimizing the distance between the two.
#     """
#     y, x = point[0], point[1]
#     yvals, xvals = [i[2] for i in maxInds], [i[3] for i in maxInds]
#     idx = np.sqrt((y-yvals)**2 + (x-xvals)**2).argmin()
#     return idx

# def orientOrigin(unsrt_ifs, unsrt_maxInds):
#     """
#     Sort a list of IF maximum indicies such that the origin is formatted
#     to be at lower left when displaying IFs using imshow (in accordance
#     with chosen IF order format). The default origin is top left in Numpy.
#     """
#     maxInds = np.copy(unsrt_maxInds[:, 0, :])
#     maxInds[:, 1] = np.floor(maxInds[:, 1]/10)*10 # make the x ind for all Ifs in the same column by taking the floor
#     # we want to apply the floor to the tenths place of the x ind, hence we divide by 10 first, then multiply by ten after taking the floor
#     maxInds[:, 0] = -1*maxInds[:, 0] # invert the y inds so that we can orient the bottom as origin
#     idx_ls = [[i, maxInds[i][0], maxInds[i][1]] for i in range(len(maxInds))] # list whose elements are [idx_num, y_ind, x_ind]
#     srt_idx_ls = sorted(idx_ls, key=lambda k: [k[2], k[1]]) # sort the maxInds based on column first, then row
#
#     srt_ifs = np.zeros(unsrt_ifs.shape)
#     srt_maxInds = np.zeros(unsrt_maxInds.shape)
#     for i in range(len(srt_idx_ls)):
#         idx = srt_idx_ls[i][0]
#         srt_ifs[i] = unsrt_ifs[idx]
#         srt_maxInds[i] = unsrt_maxInds[idx]
#
#     return srt_ifs, srt_maxInds
#
# def getMaxInds(ifs):
#     """
#     Takes in a set of IFs and returns a list of lists containing the following:
#     [cell #, maximum val, ypix of max, xpix of max]
#     """
#     maxInds = []
#     for i in range(ifs.shape[0]):
#         maxInd = np.unravel_index(np.argmax(ifs[i], axis=None), ifs[i].shape)
#         maxval, maxInd_y, maxInd_x = ifs[i][maxInd], maxInd[0], maxInd[1]
#         maxInds.append([i, maxval, maxInd_y, maxInd_x])
#     srt_maxInds = orientOrigin(maxInds)
#     return srt_maxInds
#
# def alignMaxInds(maxInds, cell_gap):
#     """
#     Aligns a set of IF maximums to be in grouped columns.
#     By specifying cell_gap in pixels, we can align the indicies
#     of IF maximums to be in adjacent columns such that they are ordered
#     properly.
#     """
#     aligned_maxInds = []
#     x0 = maxInds[0][3]
#     counter = 0
#     for i in maxInds:
#         xval = i[3]
#         # print('=======================')
#         # print('Counter =', counter)
#         # print('x0 =', x0)
#         # print('xval =', xval)
#         if abs(xval-x0)>0 and abs(xval-x0)<cell_gap:
#             # print('xval is stray...aligning')
#             aligned_maxInds.append([i[0],i[1],i[2],x0])
#         else:
#             # print('xval is inline')
#             aligned_maxInds.append(i)
#             x0 = xval
#         counter+=1
#     aligned_maxInds = orientOrigin(aligned_maxInds)
#     return aligned_maxInds
#
# def orderIFs(input_ifs, dx, triBounds=[10., 25.], edgeTrim=5.0, cell_gap=5.0,
#                 triPlacement='all', edgePlacement='all'):
#     """
#     Orders a set of IFs into the standard IF formatted order.
#     triBounds: specifies [ymm, xmm] to place triangles to zero out corners of
#     images.
#     edgeTrim: in mm, how much to zero out images from the edges
#     triPlacement: ['tl','tr','br','bl'] or 'all' to specify where triangles are
#     placed.
#     edgePlacement: ['t', 'b', 'l', 'r'] or 'all' to specify which edges to trim
#     cell_gap: specify the spacing between adjacent cells in mm.
#     """
#     cell_gap = int(round(cell_gap/dx))
#     ifs = np.copy(input_ifs)
#     if triBounds: # zero out corners
#         ifs = zeroCorners(input_ifs, dx, bounds=triBounds, placement=triPlacement)
#     if edgeTrim: # zero out edges
#         ifs = zeroEdges(ifs, dx, amount=edgeTrim, placement=edgePlacement)
#     ifs[ifs<0] = 0 # zero out negativ values
#     maxInds = getMaxInds(ifs) # find and sort indicies of max vals for each image
#     maxInds = alignMaxInds(maxInds, cell_gap) # align the indicies of max vals
#     srt_ifs = np.zeros(ifs.shape)
#     for i in range(len(maxInds)): # reorder input ifs based on maxInds
#         srt_ifs[i] = input_ifs[maxInds[i][0]]
#         maxInds[i][0] = i
#     return srt_ifs, maxInds # return sorted ifs and max indicies
#
# def matchIFs(input_ifs1, input_ifs2, dx, cell_gap=[5, 3.048],
#                 triBounds=[None, None], edgeTrim=[None, None],
#                 triPlacement=['all', 'all'], edgePlacement=['all', 'all'],
#                 matchScale=False):
#     """
#     Matches a set of measured IFs to corresponding theoretical IFs.
#
#     input_ifs1: set of measured IFs of size (L x M x N)
#     input_ifs2: set of theoretical IFs of size (K x M x N), where
#                 K >= L
#     cell_gap: [measured, theoretical] gap between cells in mm to fit to gridd
#                 of maxInds
#     triBounds: [[measured], [theoretical]] bounds to zero out corners
#     edgeTrim: [[measured], [theoretical]] bounds to zero out edges
#     triPlacement: [[measured], [theoretical]] placements for triangles
#     edgePlacement: [[measured], [theoretical]] placements for edges
#     matchScale:  scales the figure data of the theoretical IFs to the
#                 the same as measured IFs
#     returns: (ordered set of measured IFs, matching set of theoretical IFs)
#     """
#
#     # measured ifs
#     ifs1, idx1 = orderIFs(input_ifs1, dx, cell_gap=cell_gap[0],
#                             triBounds=triBounds[0], edgeTrim=edgeTrim[0],
#                             triPlacement=triPlacement[0], edgePlacement=edgePlacement[0])
#     # theoretical ifs
#     ifs2, idx2 = orderIFs(input_ifs2, dx, cell_gap=cell_gap[1],
#                             triBounds=triBounds[1], edgeTrim=edgeTrim[1],
#                             triPlacement=triPlacement[1], edgePlacement=edgePlacement[1])
#
#     matching_ifs = np.zeros(input_ifs1.shape)
#     matching_idx = []
#     cell_nos = []
#     for i in range(len(idx1)):
#         max_point = [idx1[i][2], idx1[i][3]]
#         matched_idx = find_nearest(idx2, max_point)
#         # print('matching index:', matched_idx)
#         matching_ifs[i] = ifs2[matched_idx]
#         cell_nos.append(matched_idx)
#         matching_idx.append(list(idx2[matched_idx][:]))
#         # print(matching_idx, '\n')
#         # print(matching_idx[i])
#         idx2[matched_idx][-2:] = [np.inf, np.inf]
#         # print(idx2[matched_idx])
#     if matchScale:
#         for i in range(matching_ifs.shape[0]):
#             matching_ifs[i] *= idx1[i][1]/matching_idx[i][1]
#     # print(matching_idx)
#     return ifs1, matching_ifs, idx1, matching_idx, cell_nos

# def displayIFs(ifs, dx, imbounds=None, vbounds=None, colormap='jet',
#                 figsize=(8,5), title_fntsz=14, ax_fntsz=12,
#                 title='Influence Functions',
#                 x_title='Azimuthal Dimension (mm)',
#                 y_title='Axial Dimension (mm)',
#                 cbar_title='Figure (microns)',
#                 frame_time=500, repeat_bool=False, dispR=False,
#                 cell_nos=None, stats=False, dispMaxInds=None):
#     """
#     Displays set of IFs in a single animation on one figure.
#     """
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax.set_ylabel(y_title, fontsize = ax_fntsz)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.10)
#     extent = fp.mk_extent(ifs[0], dx)
#     if not vbounds:
#         vbounds = [np.nanmin(ifs), np.nanmax(ifs)]
#     if isinstance(cell_nos, type(None)) == False:
#         idx_txt = 'Cell'
#     else:
#         idx_txt = 'Image'
#     if imbounds:
#         if isinstance(cell_nos, type(None)) == False:
#             lbnd = cell_nos.index(imbounds[0])
#             ubnd = cell_nos.index(imbounds[1])+1
#         else:
#             lbnd = imbounds[0]
#             ubnd = imbounds[1]+1
#     else:
#         lbnd = 0
#         ubnd = ifs.shape[0]
#     if stats:
#         rmsVals = [alsis.rms(ifs[i]) for i in range(ifs.shape[0])]
#         ptovVals = [alsis.ptov(ifs[i]) for i in range(ifs.shape[0])]
#     ims = []
#     for i in range(lbnd, ubnd):
#         im = ax.imshow(ifs[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds[0], vmax=vbounds[1])
#         if isinstance(cell_nos, type(None)) == False:
#             cell_no = cell_nos[i]
#         else: cell_no = i
#
#         txtstring = title + '\n' + idx_txt + ' #: {}'.format(cell_no)
#         title_plt_text = ax.text(0.5, 1.075, txtstring, fontsize=title_fntsz,
#                                 ha='center', va='center', transform=ax.transAxes)
#
#         vline_primary, hline_primary = ax.text(0,0, ''), ax.text(0,0, '')
#         vline_secondary, hline_secondary = ax.text(0,0, ''), ax.text(0,0, '')
#         stats_plt_txt = ax.text(0,0, '')
#         maxval = ax.text(0,0, '')
#
#         if type(dispMaxInds).__module__ == np.__name__:
#             hline_secondary = ax.axhline(y=(ifs.shape[1]/2-dispMaxInds[i][1][0])*dx, xmin=0, xmax=1, color='fuchsia')
#             vline_secondary = ax.axvline(x=(dispMaxInds[i][1][1]-ifs.shape[2]/2)*dx, ymin=0, ymax=1, color='fuchsia')
#             hline_primary = ax.axhline(y=(ifs.shape[1]/2-dispMaxInds[i][0][0])*dx, xmin=0, xmax=1, color='white')
#             vline_primary = ax.axvline(x=(dispMaxInds[i][0][1]-ifs.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             # maxval_txt = "IF Max Figure: {:.3f} um\ny: {:.0f}, x: {:.0f}".format(dispMaxInds[i][2], dispMaxInds[i][0], dispMaxInds[i][1])
#             primary_if_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(dispMaxInds[i][0][0], dispMaxInds[i][0][1], dispMaxInds[i][0][2])
#             secondary_if_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(dispMaxInds[i][1][0], dispMaxInds[i][1][1], dispMaxInds[i][1][2])
#             maxInd_txt = primary_if_txt + secondary_if_txt
#             maxval = ax.text(0.03, 0.90, maxInd_txt, color='black', fontsize=ax_fntsz-4,
#                                 transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.85))
#
#         if stats:
#             stats_txtstring = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals[i+lbnd], ptovVals[i+lbnd])
#             stats_plt_txt = ax.text(0.03, 0.05, stats_txtstring, fontsize=ax_fntsz,
#                                     transform=ax.transAxes)
#
#         ims.append([im, title_plt_text, stats_plt_txt, hline_primary, vline_primary, hline_secondary, vline_secondary, maxval])
#         # else:
#         #     ims.append([im, title_plt_text])
#     cbar = fig.colorbar(ims[0][0], cax=cax)
#     cbar.set_label(cbar_title, fontsize=ax_fntsz)
#     if dispR:
#         large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=12, color='red',
#                                 ha='center', va='center', transform=ax.transAxes)
#         small_R_text = ax.text(0.5, 0.9255, 'Smaller R', fontsize=12, color='red',
#                                 ha='center', va='center', transform=ax.transAxes)
#     ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
#                                     repeat=repeat_bool)
#     fps = int(1 / (frame_time/1000))
#     if stats:
#         return ani, fps, [rmsVals, ptovVals]
#     else:
#         return ani, fps
#
# def displayIFs_diff(ifs1, ifs2, ifs3, dx, imbounds=None, vbounds=None, colormap='jet',
#                 figsize=(18,6), title_fntsz=14, ax_fntsz=12,
#                 first_title='', second_title='', third_title='',
#                 global_title='', cbar_title='Figure (microns)',
#                 x_title='Azimuthal Dimension (mm)',
#                 y_title='Axial Dimension (mm)',
#                 frame_time=500, repeat_bool=False, dispR=False,
#                 cell_nos=None, stats=False, dispMaxInds=None):
#
#     """
#     Displays 3 sets of IFs adjacent to one another, in a single animation,
#     on one figure.
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(1,3)
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1])
#     ax3 = fig.add_subplot(gs[2])
#     ax1.set_title(first_title, fontsize=title_fntsz)
#     ax2.set_title(second_title, fontsize=title_fntsz)
#     ax3.set_title(third_title, fontsize=title_fntsz)
#     ax1.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax2.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax3.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax1.set_ylabel(y_title, fontsize = ax_fntsz)
#     div1 = make_axes_locatable(ax1)
#     div2 = make_axes_locatable(ax2)
#     div3 = make_axes_locatable(ax3)
#     cax1 = div1.append_axes("right", size="5%", pad=0.10)
#     cax2 = div2.append_axes("right", size="5%", pad=0.10)
#     cax3 = div3.append_axes("right", size="5%", pad=0.10)
#     extent = fp.mk_extent(ifs1[0], dx)
#     dispSingle = False
#     if not vbounds:
#         vbounds = [np.nanmin([ifs1, ifs2, ifs3]), np.nanmax([ifs1, ifs2, ifs3])]
#         vbounds1 = [np.nanmin(ifs1), np.nanmax(ifs1)]
#         vbounds2 = [np.nanmin(ifs2), np.nanmax(ifs2)]
#         vbounds3 = [np.nanmin(ifs3), np.nanmax(ifs3)]
#     elif vbounds == 'share':
#         vbounds = [np.nanmin([ifs1, ifs2, ifs3]), np.nanmax([ifs1, ifs2, ifs3])]
#         vbounds1, vbounds2, vbounds3 = vbounds, vbounds, vbounds
#     else:
#         vbounds1, vbounds2, vbounds3 = vbounds, vbounds, vbounds
#     if cell_nos is not None:
#         idx_txt = 'Cell'
#     else:
#         idx_txt = 'Image'
#     if imbounds:
#         if imbounds[0] == imbounds[1]:
#             dispSingle = True
#         if cell_nos is not None:
#             # lbnd = cell_nos.index(imbounds[0])
#             # ubnd = cell_nos.index(imbounds[1])+1
#             lbnd = int(np.where(cell_nos == imbounds[0])[0])
#             ubnd = int(np.where(cell_nos == imbounds[1])[0] + 1)
#             print(lbnd, ubnd)
#         else:
#             lbnd = imbounds[0]
#             ubnd = imbounds[1]+1
#     else:
#         lbnd = 0
#         ubnd = ifs1.shape[0]
#     if type(cbar_title) != list:
#         cbar_title = [cbar_title] * 3
#     if stats:
#         rmsVals1 = [alsis.rms(ifs1[i]) for i in range(ifs1.shape[0])]
#         rmsVals2 = [alsis.rms(ifs2[i]) for i in range(ifs2.shape[0])]
#         rmsVals3 = [alsis.rms(ifs3[i]) for i in range(ifs3.shape[0])]
#         ptovVals1 = [alsis.ptov(ifs1[i]) for i in range(ifs1.shape[0])]
#         ptovVals2 = [alsis.ptov(ifs2[i]) for i in range(ifs2.shape[0])]
#         ptovVals3 = [alsis.ptov(ifs3[i]) for i in range(ifs3.shape[0])]
#     if dispMaxInds:
#         maxInds1 = dispMaxInds[0]
#         maxInds2 = dispMaxInds[1]
#
#     ims = []
#     for i in range(lbnd, ubnd):
#         im1 = ax1.imshow(ifs1[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds1[0], vmax=vbounds1[1])
#         im2 = ax2.imshow(ifs2[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds2[0], vmax=vbounds2[1])
#         im3 = ax3.imshow(ifs3[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds3[0], vmax=vbounds3[1])
#         if cell_nos is not None:
#             cell_no = cell_nos[i]
#         else: cell_no = i
#
#         txtstring = global_title + '\n' + idx_txt + ' #: {}'.format(cell_no)
#         title_plt_text = plt.gcf().text(0.5, 0.94, txtstring, fontsize=title_fntsz,
#                                 ha='center', va='center')
#
#         vline1_primary, hline1_primary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline1_secondary, hline1_secondary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline2_primary, hline2_primary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline2_secondary, hline2_secondary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         # vline3_primary, hline3_primary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         # vline3_secondary, hline3_secondary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         maxval1, maxval2, maxval3 = ax1.text(0,0, ''), ax1.text(0,0, ''), ax1.text(0,0, '')
#         stats_plt_txt1 = ax1.text(0,0, '')
#         stats_plt_txt2 = ax1.text(0,0, '')
#         stats_plt_txt3 = ax1.text(0,0, '')
#
#         if dispMaxInds:
#             # print('vline1 x coord: {:.2f}'.format((maxInds1[i][1]*dx)-ifs1.shape[1]/2))
#             hline1_primary = ax1.axhline(y=(ifs1.shape[1]/2-maxInds1[i][0][0])*dx, xmin=0, xmax=1, color='white')
#             vline1_primary = ax1.axvline(x=(maxInds1[i][0][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             hline1_secondary = ax1.axhline(y=(ifs1.shape[1]/2-maxInds1[i][1][0])*dx, xmin=0, xmax=1, color='fuchsia')
#             vline1_secondary = ax1.axvline(x=(maxInds1[i][1][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='fuchsia')
#             hline2_primary = ax2.axhline(y=(ifs1.shape[1]/2-maxInds2[i][0][0])*dx, xmin=0, xmax=1, color='white')
#             vline2_primary = ax2.axvline(x=(maxInds2[i][0][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             hline2_secondary = ax2.axhline(y=(ifs1.shape[1]/2-maxInds2[i][1][0])*dx, xmin=0, xmax=1, color='fuchsia')
#             vline2_secondary = ax2.axvline(x=(maxInds2[i][1][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='fuchsia')
#
#             primary_if1_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds1[i][0][0], maxInds1[i][0][1], maxInds1[i][0][2])
#             secondary_if1_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds1[i][1][0], maxInds1[i][1][1], maxInds1[i][1][2])
#             maxInd1_txt = primary_if1_txt + secondary_if1_txt
#             maxval1 = ax1.text(0.03, 0.90, maxInd1_txt, color='black', fontsize=ax_fntsz-4,
#                                 transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.85))
#             primary_if2_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds2[i][0][0], maxInds2[i][0][1], maxInds2[i][0][2])
#             secondary_if2_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds2[i][1][0], maxInds2[i][1][1], maxInds2[i][1][2])
#             maxInd2_txt = primary_if2_txt + secondary_if2_txt
#             maxval2 = ax2.text(0.03, 0.90, maxInd1_txt, color='black', fontsize=ax_fntsz-4,
#                                 transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.85))
#             #
#             # maxval1_txt = "IF Max Figure: {:.3f} um".format(maxInds1[i][2])
#             # maxval1 = ax1.text(0.03, 0.95, maxval1_txt, fontsize=ax_fntsz-4,
#             #                     transform=ax1.transAxes)
#             # vline2 = ax2.axvline(x=(maxInds2[i][1]-ifs2.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             # hline2 = ax2.axhline(y=(ifs2.shape[1]/2-maxInds2[i][0])*dx, xmin=0, xmax=1, color='white')
#             # maxval2_txt = "IF Max Figure: {:.3f} um".format(maxInds2[i][2])
#             # maxval2 = ax2.text(0.03, 0.95, maxval2_txt, fontsize=ax_fntsz-4,
#             #                     transform=ax2.transAxes)
#
#         if stats:
#             stats_txt1 = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals1[i], ptovVals1[i])
#             stats_txt2 = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals2[i], ptovVals2[i])
#             stats_txt3 = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals3[i], ptovVals3[i])
#             stats_plt_txt1 = ax1.text(0.03, 0.05, stats_txt1, fontsize=ax_fntsz,
#                                     transform=ax1.transAxes)
#             stats_plt_txt2 = ax2.text(0.03, 0.05, stats_txt2, fontsize=ax_fntsz,
#                                     transform=ax2.transAxes)
#             stats_plt_txt3 = ax3.text(0.03, 0.05, stats_txt3, fontsize=ax_fntsz,
#                                     transform=ax3.transAxes)
#         ims.append([im1, im2, im3, stats_plt_txt1, stats_plt_txt2, stats_plt_txt3, title_plt_text,
#                         vline1_primary, hline1_primary, vline1_secondary, hline1_secondary,
#                          vline2_primary, hline2_primary, vline2_secondary, hline2_secondary,
#                          maxval1, maxval2])
#
#     cbar1 = fig.colorbar(ims[0][0], cax=cax1)
#     cbar2 = fig.colorbar(ims[0][1], cax=cax2)
#     cbar3 = fig.colorbar(ims[0][2], cax=cax3)
#     cbar3.set_label(cbar_title[2], fontsize=ax_fntsz)
#     fig.subplots_adjust(top=0.80, hspace=0.5, wspace=0.35)
#     # plt.suptitle(global_title,fontsize=title_fntsz)
#     ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
#                                     repeat=repeat_bool)
#     fps = int(1 / (frame_time/1000))
#     if dispSingle:
#         return fig, None, [[rmsVals1, ptovVals1], [rmsVals2, ptovVals2], [rmsVals3, ptovVals3]]
#     elif stats:
#         return ani, fps, [[rmsVals1, ptovVals1], [rmsVals2, ptovVals2], [rmsVals3, ptovVals3]]
#     else:
#         return ani, fps
#
#
# def displayIFs_compare(ifs1, ifs2, dx, imbounds=None, vbounds=None, colormap='jet',
#                 figsize=(10,5), title_fntsz=14, ax_fntsz=12,
#                 first_title='', second_title='',
#                 global_title='', cbar_title='Figure (microns)',
#                 x_title='Azimuthal Dimension (mm)',
#                 y_title='Axial Dimension (mm)',
#                 frame_time=500, repeat_bool=False, dispR=False,
#                 cell_nos=None, stats=False, dispMaxInds=None):
#
#     """
#     Displays 2 sets of IFs adjacent to one another, in a single animation,
#     on one figure.
#     """
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(1,2)
#     ax1 = fig.add_subplot(gs[0])
#     ax2 = fig.add_subplot(gs[1])
#     ax1.set_title(first_title, fontsize=title_fntsz)
#     ax2.set_title(second_title, fontsize=title_fntsz)
#     ax1.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax2.set_xlabel(x_title, fontsize = ax_fntsz)
#     ax1.set_ylabel(y_title, fontsize = ax_fntsz)
#     div1 = make_axes_locatable(ax1)
#     div2 = make_axes_locatable(ax2)
#     cax1 = div1.append_axes("right", size="5%", pad=0.10)
#     cax2 = div2.append_axes("right", size="5%", pad=0.10)
#     extent = fp.mk_extent(ifs1[0], dx)
#     dispSingle = False
#     if not vbounds:
#         vbounds = [np.nanmin([ifs1, ifs2]), np.nanmax([ifs1, ifs2])]
#         vbounds1 = [np.nanmin(ifs1), np.nanmax(ifs1)]
#         vbounds2 = [np.nanmin(ifs2), np.nanmax(ifs2)]
#     elif vbounds == 'share':
#         vbounds = [np.nanmin([ifs1, ifs2]), np.nanmax([ifs1, ifs2])]
#         vbounds1, vbounds2 = vbounds, vbounds
#     else:
#         vbounds1, vbounds2 = vbounds, vbounds
#     if cell_nos:
#         idx_txt = 'Cell'
#     else:
#         idx_txt = 'Image'
#     if imbounds:
#         if imbounds[0] == imbounds[1]:
#             dispSingle = True
#         if cell_nos:
#             lbnd = cell_nos.index(imbounds[0])
#             ubnd = cell_nos.index(imbounds[1])+1
#         else:
#             lbnd = imbounds[0]
#             ubnd = imbounds[1]+1
#     else:
#         lbnd = 0
#         ubnd = ifs1.shape[0]
#     if type(cbar_title) != list:
#         cbar_title = [cbar_title] * 2
#     if stats:
#         rmsVals1 = [alsis.rms(ifs1[i]) for i in range(ifs1.shape[0])]
#         rmsVals2 = [alsis.rms(ifs2[i]) for i in range(ifs2.shape[0])]
#         ptovVals1 = [alsis.ptov(ifs1[i]) for i in range(ifs1.shape[0])]
#         ptovVals2 = [alsis.ptov(ifs2[i]) for i in range(ifs2.shape[0])]
#     if dispMaxInds:
#         maxInds1 = dispMaxInds[0]
#         maxInds2 = dispMaxInds[1]
#
#     ims = []
#     for i in range(lbnd, ubnd):
#         im1 = ax1.imshow(ifs1[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds1[0], vmax=vbounds1[1])
#         im2 = ax2.imshow(ifs2[i], extent=extent, aspect='auto', cmap=colormap,
#                         vmin=vbounds2[0], vmax=vbounds2[1])
#         if cell_nos:
#             cell_no = cell_nos[i]
#         else: cell_no = i
#
#         txtstring = global_title + '\n' + idx_txt + ' #: {}'.format(cell_no)
#         title_plt_text = plt.gcf().text(0.5, 0.94, txtstring, fontsize=title_fntsz,
#                                 ha='center', va='center')
#
#         vline1_primary, hline1_primary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline1_secondary, hline1_secondary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline2_primary, hline2_primary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         vline2_secondary, hline2_secondary = ax1.text(0,0, ''), ax1.text(0,0, '')
#         maxval1, maxval2 = ax1.text(0,0, ''), ax1.text(0,0, '')
#         stats_plt_txt1 = ax1.text(0,0, '')
#         stats_plt_txt2 = ax1.text(0,0, '')
#
#         if dispMaxInds:
#             # print('vline1 x coord: {:.2f}'.format((maxInds1[i][1]*dx)-ifs1.shape[1]/2))
#             hline1_primary = ax1.axhline(y=(ifs1.shape[1]/2-maxInds1[i][0][0])*dx, xmin=0, xmax=1, color='white')
#             vline1_primary = ax1.axvline(x=(maxInds1[i][0][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             hline1_secondary = ax1.axhline(y=(ifs1.shape[1]/2-maxInds1[i][1][0])*dx, xmin=0, xmax=1, color='fuchsia')
#             vline1_secondary = ax1.axvline(x=(maxInds1[i][1][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='fuchsia')
#             hline2_primary = ax2.axhline(y=(ifs1.shape[1]/2-maxInds2[i][0][0])*dx, xmin=0, xmax=1, color='white')
#             vline2_primary = ax2.axvline(x=(maxInds2[i][0][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='white')
#             hline2_secondary = ax2.axhline(y=(ifs1.shape[1]/2-maxInds2[i][1][0])*dx, xmin=0, xmax=1, color='fuchsia')
#             vline2_secondary = ax2.axvline(x=(maxInds2[i][1][1]-ifs1.shape[2]/2)*dx, ymin=0, ymax=1, color='fuchsia')
#
#             primary_if1_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds1[i][0][0], maxInds1[i][0][1], maxInds1[i][0][2])
#             secondary_if1_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds1[i][1][0], maxInds1[i][1][1], maxInds1[i][1][2])
#             maxInd1_txt = primary_if1_txt + secondary_if1_txt
#             maxval1 = ax1.text(0.03, 0.90, maxInd1_txt, color='black', fontsize=ax_fntsz-4,
#                                 transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.85))
#             primary_if2_txt = "1st IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds2[i][0][0], maxInds2[i][0][1], maxInds2[i][0][2])
#             secondary_if2_txt = "\n2nd IF || y: {:.0f} || x: {:.0f} || fig: {:.2f} um ||".format(maxInds2[i][1][0], maxInds2[i][1][1], maxInds2[i][1][2])
#             maxInd2_txt = primary_if2_txt + secondary_if2_txt
#             maxval2 = ax2.text(0.03, 0.90, maxInd1_txt, color='black', fontsize=ax_fntsz-4,
#                                 transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.85))
#
#         if stats:
#             stats_txt1 = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals1[i], ptovVals1[i])
#             stats_txt2 = "RMS: {:.2f} um\nPV: {:.2f} um".format(rmsVals2[i], ptovVals2[i])
#             stats_plt_txt1 = ax1.text(0.03, 0.05, stats_txt1, fontsize=ax_fntsz,
#                                     transform=ax1.transAxes)
#             stats_plt_txt2 = ax2.text(0.03, 0.05, stats_txt2, fontsize=ax_fntsz,
#                                     transform=ax2.transAxes)
#         ims.append([im1, im2, stats_plt_txt1, stats_plt_txt2, title_plt_text,
#                     vline1_primary, hline1_primary, vline1_secondary, hline1_secondary,
#                     vline2_primary, hline2_primary, vline2_secondary, hline2_secondary,
#                     maxval1, maxval2])
#
#     cbar1 = fig.colorbar(ims[0][0], cax=cax1)
#     cbar2 = fig.colorbar(ims[0][1], cax=cax2)
#     cbar1.set_label(cbar_title[0], fontsize=ax_fntsz)
#     cbar2.set_label(cbar_title[1], fontsize=ax_fntsz)
#     fig.subplots_adjust(top=0.80, hspace=0.5, wspace=0.35)
#     # plt.suptitle(global_title,fontsize=title_fntsz)
#     ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
#                                     repeat=repeat_bool)
#     fps = int(1 / (frame_time/1000))
#     if dispSingle:
#         return fig, None, [[rmsVals1, ptovVals1], [rmsVals2, ptovVals2]]
#     elif stats:
#         return ani, fps, [[rmsVals1, ptovVals1], [rmsVals2, ptovVals2]]
#     else:
#         return ani, fps


# def cell_yield_scatter(maxInds, img_shp, dx, vbounds=None, colormap='jet',
#                     figsize=(8,8), title_fntsz=14, ax_fntsz=12,
#                     title="C1S04 Spatial Distribution of\nMeasured IFs' Maximum Figure Change",
#                     cbar_title='Figure (microns)',
#                     x_title='Azimuthal Dimension (mm)',
#                     y_title='Axial Dimension (mm)'):
#         maxvals = np.array([i[2] for i in maxInds])
#         xvals = np.array([i[1] for i in maxInds])
#         yvals = np.array([i[0] for i in maxInds])
#         fig, ax = plt.subplots(figsize=figsize)
#         scatter_plot = ax.scatter(xvals, yvals, c=maxvals, cmap=colormap)
#         xlabels, ylabels = np.arange(-45, 50, 5), np.arange(-50, 55, 5)
#         xticks = xlabels/dx + img_shp[1]/2
#         yticks = ylabels/dx + img_shp[0]/2
#         ax.set_xticks(xticks)
#         ax.set_yticks(yticks)
#         ax.set_xticklabels(xlabels, rotation=45)
#         ax.set_yticklabels(ylabels)
#         ax.set_xlabel(x_title, fontsize=ax_fntsz)
#         ax.set_ylabel(y_title, fontsize=ax_fntsz)
#         ax.set_title(title, fontsize=title_fntsz)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.10)
#         cbar = plt.colorbar(scatter_plot, cax=cax)
#         cbar.set_label(cbar_title, fontsize=ax_fntsz)
#         # cbar.ax.tick_params(labelsize=tick_fntsz)
#         ax.grid(True)
#         return fig
#
# def compare_cell_yield_scatter(maxInds1, maxInds2, img_shp, dx, vbounds=None, colormap='jet',
#                     figsize=(12,6), title_fntsz=14, ax_fntsz=12,
#                     global_title="C1S04 Spatial Distribution of IFs' Maximum Figure Change",
#                     first_title='', second_title='',
#                     cbar_title='Figure (microns)',
#                     x_title='Azimuthal Dimension (mm)',
#                     y_title='Axial Dimension (mm)'):
#         maxvals1 = np.array([i[2] for i in maxInds1])
#         xvals1 = np.array([i[1] for i in maxInds1])
#         yvals1 = np.array([i[0] for i in maxInds1])
#         maxvals2 = np.array([i[2] for i in maxInds2])
#         xvals2 = np.array([i[1] for i in maxInds2])
#         yvals2 = np.array([i[0] for i in maxInds2])
#
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
#         scatter_plot1 = ax1.scatter(xvals1, yvals1, c=maxvals1, cmap=colormap)
#         scatter_plot2 = ax2.scatter(xvals2, yvals2, c=maxvals2, cmap=colormap)
#         xlabels, ylabels = np.arange(-45, 50, 5), np.arange(-50, 55, 5)
#         xticks = xlabels/dx + img_shp[1]/2
#         yticks = ylabels/dx + img_shp[0]/2
#         for ax, scatter_plot in zip([ax1, ax2], [scatter_plot1, scatter_plot2]):
#             ax.set_xticks(xticks)
#             ax.set_yticks(yticks)
#             ax.set_xticklabels(xlabels, rotation=45)
#             ax.set_yticklabels(ylabels)
#             ax.set_xlabel(x_title, fontsize=ax_fntsz)
#             ax.set_ylabel(y_title, fontsize=ax_fntsz)
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.10)
#             cbar = plt.colorbar(scatter_plot, cax=cax)
#             cbar.set_label(cbar_title, fontsize=ax_fntsz)
#             ax.grid(True)
#         ax1.set_title(first_title, fontsize=ax_fntsz)
#         ax2.set_title(second_title, fontsize=ax_fntsz)
#         fig.suptitle(global_title, fontsize=title_fntsz)
#         fig.tight_layout(rect=[0, 0, 1, 0.93])
#         fig.subplots_adjust(wspace=0.35)
#         return fig
#
# def get_ticks(xvals, yvals):
#     cell_gap = 5.
#     u_xvals = np.unique(xvals)
#     u_yvals = np.unique(yvals)
#     xticks = [u_xvals[0]]
#     yticks = [u_yvals[0]]
#     for i in range(len(u_xvals)+1):
#         if i == len(u_xvals): break
#         if np.abs(u_xvals[i]-xticks[-1]) >= cell_gap:
#             xticks.append(u_xvals[i])
#         else: continue
#     for i in range(len(u_yvals)+1):
#         if i == len(u_yvals): break
#         if np.abs(u_yvals[i]-yticks[-1]) >= cell_gap:
#             yticks.append(u_yvals[i])
#         else: continue
#     return np.array(xticks), np.array(yticks)
#
# def get_ticklabels(xticks, yticks, img_shp, dx):
#     x0, y0 = img_shp[1]/2, img_shp[0]/2
#     xlabels = np.round((xticks-x0)*dx, decimals=0)
#     ylabels = np.round((yticks-y0)*dx, decimals=0)
#     xlabels = [int(x) for x in xlabels]
#     ylabels = [int(y) for y in ylabels]
#     return xlabels, ylabels

# def isoIFs(input_ifs, dx, maxInds, setval=np.nan, extent=15):
#     ifs = np.copy(input_ifs)
#     for i in range(ifs.shape[0]):
#         y_ind = int(round(maxInds[i][0][0]))
#         x_ind = int(round(maxInds[i][0][1]))
#         # print('i:', i, 'y_ind:', y_ind, 'x_ind', x_ind)
#         # ifs[i, :y_ind-extent, :] = setval
#         # ifs[i, y_ind+extent:, :] = setval
#         # ifs[i, :, :x_ind-extent] = setval
#         # ifs[i, :, x_ind+extent:] = setval
#         if y_ind-extent > extent:
#             ifs[i, :y_ind-extent, :] = setval
#         if ifs.shape[1]-y_ind-extent > extent:
#             ifs[i, y_ind+extent:, :] = setval
#         if x_ind-extent > extent:
#             ifs[i, :, :x_ind-extent] = setval
#         if ifs.shape[2]-x_ind-extent > extent:
#             ifs[i, :, x_ind+extent:] = setval
#     ifs = zeroEdges(ifs, dx, amount=3., placement='all', setval=setval)
#     return ifs

# def stitch_ifs(ifs1, ifs2, maxInds1, maxInds2, degree=5, extent=15):
#     """
#     Computes a transformation to bring maxInds2 and ifs2 in line with ifs1 and maxInds1.
#     The transformation is then applied to ifs2 and maxInds2 and then returned.
#
#     The type of transformation is determined by degree, which can equal the following:
#     2: translation only (dx, dy)
#     3: translation and rotation (dx, dy, theta)
#     4: translation, rotation, and magnification (dx, dy, theta, mag)
#     5: translation, rotation, and separate magnifications (dx, dy, theta, x_mag, y_mag)
#     """
#     yf1, xf1 = maxInds1[:,0,0], maxInds1[:,0,1] # get x and y coordinates from maxInds
#     yf2, xf2 = maxInds2[:,0,0], maxInds2[:,0,1]
#     # calculate transformations
#     tx, ty, theta, mag, x_mag, y_mag = None, None, None, None, None, None
#     if degree == 2:
#         print('Still a work in progress')
#         pass
#     if degree == 3:
#         tx, ty, theta = stitch.matchFiducials(xf1, yf1, xf2, yf2)
#     elif degree == 4:
#         tx, ty, theta, mag = stitch.matchFiducials_wMag(xf1, yf1, xf2, yf2)
#     elif degree == 5:
#         tx, ty, theta, x_mag, y_mag = stitch.matchFiducials_wSeparateMag(xf1, yf1, xf2, yf2)
#     else:
#         print('degree must be 2, 3, 4, or 5.')
#         pass
#     # if tx: tx = round(tx)
#     # if ty: ty = round(ty)
#     # if theta: theta = round(theta)
#     # if mag: mag = round(mag)
#     # if x_mag: x_mag = round(x_mag)
#     # if y_mag: y_mag = round(y_mag)
#     print('\nCalculated transformation:\n dx: {}\n dy: {}\n theta: {}\n mag: {}\n x_mag: {}\n y_mag: {}\n'.format(tx, ty, theta, mag, x_mag, y_mag))
#     if tx < 0: xf1 -= tx
#     if ty < 0: yf1 -= ty
#
#     # stitching process
#     stitch_ifs, stitch_maxInds = np.zeros(ifs2.shape), np.zeros(maxInds2.shape)
#     for i in range(ifs2.shape[0]):
#         img1, img2 = ifs1[i], ifs2[i]
#         #Get x,y,z points from reference image
#         x1,y1,z1 = man.unpackimage(img1,remove=False,xlim=[0,np.shape(img1)[1]],\
#                                ylim=[0,np.shape(img1)[0]])
#         #Get x,y,z points from stitched image
#         x2,y2,z2 = man.unpackimage(img2,xlim=[0,np.shape(img2)[1]],\
#                                ylim=[0,np.shape(img2)[0]])
#         #Apply transformations to x,y coords of stitched image
#         if degree == 2:
#             pass
#         elif degree == 3:
#             x2_t, y2_t = stitch.transformCoords(x2, y2, tx, ty, theta)
#             stitch_maxInds[i,0,0], stitch_maxInds[i,0,1] = stitch.transformCoords(maxInds2[i,0,0], maxInds2[i,0,1], tx, ty, theta)
#             if -1 not in maxInds2[i][1]:
#                 stitch_maxInds[i,1,0], stitch_maxInds[i,1,1] = stitch.transformCoords(maxInds2[i,1,0], maxInds2[i,1,1], tx, ty, theta)
#         elif degree == 4:
#             x2_t, y2_t = stitch.transformCoords_wMag(x2, y2, tx, ty, theta, mag)
#             stitch_maxInds[i,0,0], stitch_maxInds[i,0,1] = stitch.transformCoords_wMag(maxInds2[i,0,0], maxInds2[i,0,1], tx, ty, theta, mag)
#             if -1 not in maxInds2[i][1]:
#                 stitch_maxInds[i,1,0], stitch_maxInds[i,1,1] = stitch.transformCoords_wMag(maxInds2[i,1,0], maxInds2[i,1,1], tx, ty, theta, mag)
#         elif degree == 5:
#             x2_t, y2_t = stitch.transformCoords_wSeparateMag(x2, y2, tx, ty, theta, x_mag, y_mag)
#             stitch_maxInds[i,0,0], stitch_maxInds[i,0,1] = stitch.transformCoords_wSeparateMag(maxInds2[i,0,0], maxInds2[i,0,1], tx, ty, theta, x_mag, y_mag)
#             if -1 not in maxInds2[i][1]:
#                 stitch_maxInds[i,1,0], stitch_maxInds[i,1,1] = stitch.transformCoords_wSeparateMag(maxInds2[i,1,0], maxInds2[i,1,1], tx, ty, theta, x_mag, y_mag)
#
#         # Interpolate stitched image onto expanded image grid
#         newimg = griddata((x2_t,y2_t),z2,(x1,y1),method='linear')
#         print('Image {}: Interpolation ok'.format(i+1))
#         newimg = newimg.reshape(np.shape(img1))
#
#         # Images should now be in the same reference frame
#         # Time to apply tip/tilt/piston to minimize RMS
#         newimg = stitch.matchPistonTipTilt(img1,newimg)
#         stitch_ifs[i] = newimg
#         stitch_maxInds[i,0,2] = stitch_ifs[i][int(round(stitch_maxInds[i,0,0]))][int(round(stitch_maxInds[i,0,1]))]
#         if -1 not in maxInds2[i][1]:
#             stitch_maxInds[i,1,2] = stitch_ifs[i][int(round(stitch_maxInds[i,1,0]))][int(round(stitch_maxInds[i,1,1]))]
#         else:
#             stitch_maxInds[i,1,:] = -1
#     return stitch_ifs, stitch_maxInds, (tx, ty, theta, mag, x_mag, y_mag)
