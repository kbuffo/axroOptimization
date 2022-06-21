import numpy as np
from scipy import ndimage as nd
from operator import itemgetter
from astropy.io import fits as pyfits
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
plt.rcParams['savefig.facecolor']='white'

import utilities.figure_plotting as fp
import imaging.man as man
import axroOptimization.evaluateMirrors as eva
import axroOptimization.solver as solver

def ptov(d):
    """Calculate peak to valley for an image"""
    return np.nanmax(d) - np.nanmin(d)

def find_nearest(maxInds, point):
    """
    Finds nearest neighbor between a meas IF maximum and the set
    of theo IFs by minimizing the distance between the two.
    """
    y, x = point[0], point[1]
    yvals, xvals = [i[2] for i in maxInds], [i[3] for i in maxInds]
    idx = np.sqrt((y-yvals)**2 + (x-xvals)**2).argmin()
    return idx

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

def orientOrigin(maxInds):
    """
    Sort a list of IF maximum indicies such that the origin is formatted
    to be at lower left when displaying IFs using imshow (in accordance
    with chosen IF order format). The default origin is top left in Numpy.
    """
    for i in maxInds:
        i[2] *= -1
    srt_maxInds = sorted(maxInds, key=lambda k: [k[3], k[2]])
    for i in srt_maxInds:
        i[2] *= -1
    return srt_maxInds

def getMaxInds(ifs):
    """
    Takes in a set of IFs and returns a list of lists containing the following:
    [cell #, maximum val, ypix of max, xpix of max]
    """
    maxInds = []
    for i in range(ifs.shape[0]):
        maxInd = np.unravel_index(np.argmax(ifs[i], axis=None), ifs[i].shape)
        maxval, maxInd_y, maxInd_x = ifs[i][maxInd], maxInd[0], maxInd[1]
        maxInds.append([i, maxval, maxInd_y, maxInd_x])
    srt_maxInds = orientOrigin(maxInds)
    return srt_maxInds

def alignMaxInds(maxInds):
    """
    Aligns a set of IF maximums to be in grouped columns.
    By specifying cell_gap in pixels, we can align the indicies
    of IF maximums to be in adjacent columns such that they are ordered
    properly.
    """
    aligned_maxInds = []
    x0 = maxInds[0][3]
    cell_gap = 6
    counter = 0
    for i in maxInds:
        xval = i[3]
        # print('=======================')
        # print('Counter =', counter)
        # print('x0 =', x0)
        # print('xval =', xval)
        if abs(xval-x0)>0 and abs(xval-x0)<cell_gap:
            # print('xval is stray...aligning')
            aligned_maxInds.append([i[0],i[1],i[2],x0])
        else:
            # print('xval is inline')
            aligned_maxInds.append(i)
            x0 = xval
        counter+=1
    aligned_maxInds = orientOrigin(aligned_maxInds)
    return aligned_maxInds

def orderIFs(input_ifs, dx, triBounds=[10., 25.], edgeTrim=5.0,
                triPlacement='all', edgePlacement='all'):
    """
    Orders a set of IFs into the standard IF formatted order.
    triBounds: specifies [ymm, xmm] to place triangles to zero out corners of
    images.
    edgeTrim: in mm, how much to zero out images from the edges
    triPlacement: ['tl','tr','br','bl'] or 'all' to specify where triangles are
    placed.
    edgePlacement: ['t', 'b', 'l', 'r'] or 'all' to specify which edges to trim
    """
    ifs = np.copy(input_ifs)
    if triBounds: # zero out corners
        ifs = zeroCorners(input_ifs, dx, bounds=triBounds, placement=triPlacement)
    if edgeTrim: # zero out edges
        ifs = zeroEdges(ifs, dx, amount=edgeTrim, placement=edgePlacement)
    ifs[ifs<0] = 0 # zero out negativ values
    maxInds = getMaxInds(ifs) # find and sort indicies of max vals for each image
    maxInds = alignMaxInds(maxInds) # align the indicies of max vals
    srt_ifs = np.zeros(ifs.shape)
    for i in range(len(maxInds)): # reorder input ifs based on maxInds
        srt_ifs[i] = input_ifs[maxInds[i][0]]
        maxInds[i][0] = i
    return srt_ifs, maxInds # return sorted ifs and max indicies

def matchIFs(input_ifs1, input_ifs2, dx, triBounds=[None, None],
                edgeTrim=[None, None], triPlacement=['all', 'all'],
                edgePlacement=['all', 'all'], matchScale=False):
    """
    Matches a set of measured IFs to corresponding theoretical IFs.

    input_ifs1: set of measured IFs of size (L x M x N)
    input_ifs2: set of theoretical IFs of size (K x M x N), where
                K >= L
    triBounds: [[measured], [theoretical]] bounds to zero out corners
    edgeTrim: [[measured], [theoretical]] bounds to zero out edges
    triPlacement: [[measured], [theoretical]] placements for triangles
    edgePlacement: [[measured], [theoretical]] placements for edges
    matchScale:  scales the figure data of the theoretical IFs to the
                the same as measured IFs
    returns: (ordered set of measured IFs, matching set of theoretical IFs)
    """

    # measured ifs
    ifs1, idx1 = orderIFs(input_ifs1, dx, triBounds=triBounds[0],
                            edgeTrim=edgeTrim[0], triPlacement=triPlacement[0],
                            edgePlacement=edgePlacement[0])
    # theoretical ifs
    ifs2, idx2 = orderIFs(input_ifs2, dx, triBounds=triBounds[1],
                            edgeTrim=edgeTrim[1], triPlacement=triPlacement[1],
                            edgePlacement=edgePlacement[1])
    matching_ifs = np.zeros(input_ifs1.shape)
    matching_idx = []
    for i in range(len(idx1)):
        max_point = [idx1[i][2], idx1[i][3]]
        matched_idx = find_nearest(idx2, max_point)
        matching_ifs[i] = ifs2[matched_idx]
        matching_idx.append(idx2[matched_idx])
    if matchScale:
        for i in range(matching_ifs.shape[0]):
            matching_ifs[i] *= idx1[i][1]/matching_idx[i][1]
    return ifs1, matching_ifs

def displayIFs(ifs, dx, imbounds=None, vbounds=None, colormap='jet',
                figsize=(8,5), title_fntsz=14, ax_fntsz=12,
                title='Influence Functions',
                x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)',
                cbar_title='Figure (microns)',
                frame_time=500, repeat_bool=False, dispR=False):
    """
    Displays set of IFs in a single animation on one figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_title, fontsize = ax_fntsz)
    ax.set_ylabel(y_title, fontsize = ax_fntsz)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    extent = fp.mk_extent(ifs[0], dx)
    if not vbounds:
        vbounds = [np.nanmin(ifs), np.nanmax(ifs)]
    if imbounds:
        lbnd = imbounds[0]
        ubnd = imbounds[1]
    else:
        lbnd = 0
        ubnd = ifs.shape[0]
    ims = []
    for i in range(ifs[lbnd:ubnd+1].shape[0]):
        im = ax.imshow(ifs[i+lbnd], extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds[0], vmax=vbounds[1])
        cell_no = i + lbnd
        txtstring = title + '\nCell #: {}'.format(cell_no)
        title_plt_text = ax.text(0.5, 1.075, txtstring, fontsize=title_fntsz,
                                ha='center', va='center', transform=ax.transAxes)
        ims.append([im, title_plt_text])
    cbar = fig.colorbar(ims[0][0], cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz)
    if dispR:
        large_R_text = ax.text(0.5, 0.075, 'Larger R', fontsize=12, color='red',
                                ha='center', va='center', transform=ax.transAxes)
        small_R_text = ax.text(0.5, 0.9255, 'Smaller R', fontsize=12, color='red',
                                ha='center', va='center', transform=ax.transAxes)
    ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    fps = int(1 / (frame_time/1000))
    return ani, fps

def displayIFs_diff(ifs1, ifs2, ifs3, dx, imbounds=None, vbounds=None, colormap='jet',
                figsize=(18,6), title_fntsz=14, ax_fntsz=12,
                first_title='', second_title='', third_title='',
                global_title='', cbar_title='Figure (microns)',
                x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)',
                frame_time=500, repeat_bool=False, dispR=False):

    """
    Displays 3 sets of IFs adjacent to one another, in a single animation,
    on one figure.
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax1.set_title(first_title, fontsize=title_fntsz)
    ax2.set_title(second_title, fontsize=title_fntsz)
    ax3.set_title(third_title, fontsize=title_fntsz)
    ax1.set_xlabel(x_title, fontsize = ax_fntsz)
    ax2.set_xlabel(x_title, fontsize = ax_fntsz)
    ax3.set_xlabel(x_title, fontsize = ax_fntsz)
    ax1.set_ylabel(y_title, fontsize = ax_fntsz)
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    div3 = make_axes_locatable(ax3)
    cax1 = div1.append_axes("right", size="5%", pad=0.10)
    cax2 = div2.append_axes("right", size="5%", pad=0.10)
    cax3 = div3.append_axes("right", size="5%", pad=0.10)
    extent = fp.mk_extent(ifs1[0], dx)
    if not vbounds:
        vbounds = [np.nanmin([ifs1, ifs2, ifs3]), np.nanmax([ifs1, ifs2, ifs3])]
        vbounds1 = [np.nanmin(ifs1), np.nanmax(ifs1)]
        vbounds2 = [np.nanmin(ifs2), np.nanmax(ifs2)]
        vbounds3 = [np.nanmin(ifs3), np.nanmax(ifs3)]
    else:
        vbounds1, vbounds2, vbounds3 = vbounds, vbounds, vbounds
    if imbounds:
        lbnd = imbounds[0]
        ubnd = imbounds[1]
    else:
        lbnd = 0
        ubnd = ifs1.shape[0]
    if type(cbar_title) != list:
        cbar_title = [cbar_title] * 3
    ims = []
    for i in range(ifs1[lbnd:ubnd+1].shape[0]):
        im1 = ax1.imshow(ifs1[i+lbnd], extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds1[0], vmax=vbounds1[1])
        im2 = ax2.imshow(ifs2[i+lbnd], extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds2[0], vmax=vbounds2[1])
        im3 = ax3.imshow(ifs3[i+lbnd], extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds3[0], vmax=vbounds3[1])
        cell_no = i + lbnd
        txtstring = '\nCell #: {}'.format(cell_no)
        title_plt_text = plt.gcf().text(0.5, 0.95, txtstring, fontsize=title_fntsz,
                                ha='center', va='center')
        ims.append([im1, im2, im3, title_plt_text])
    cbar1 = fig.colorbar(ims[0][0], cax=cax1)
    cbar2 = fig.colorbar(ims[0][1], cax=cax2)
    cbar3 = fig.colorbar(ims[0][2], cax=cax3)
    cbar3.set_label(cbar_title[2], fontsize=ax_fntsz)
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.35)
    plt.suptitle(global_title,fontsize=title_fntsz)
    ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    fps = int(1 / (frame_time/1000))
    return ani, fps
