import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
plt.rcParams['savefig.facecolor']='white'
import utilities.figure_plotting as fp
import imaging.man as man
import axroOptimization.evaluateMirrors as eva
import axroOptimization.solver as solver

def formatIFs(fn, savename=None):
    """
    Takes in a .fits file with unformatted IFs, and formats them.
    This includes:
    1. Transposing the data so that it's shape is (IF #, ypixels, xpixels)
    2. Multiplying by -1 so that the IF figure distortions are positive values
    3. Convert data from meters to microns
    4. Regrids the image data so that (ypixels, xpixels) -> (200, 200)
    """
    # load data from fits file
    unform_ifs = pyfits.getdata(fn)
    scaled_ifs = 1.e6*(-np.transpose(unform_ifs))
    form_ifs = [man.newGridSize(scaled_ifs[i], (200, 200)) for i in range(len(scaled_ifs))]
    form_ifs = np.array(form_ifs)
    if savename:
        pyfits.writeto(savename, form_ifs, overwrite=True)
    return form_ifs

def meanMax(arrays):
    maxvals = []
    for i in range(len(arrays)):
        maxval = np.nanmax(arrays[i])
        maxvals.append(maxval)
    maxvals = np.array(maxvals)
    mean_max = np.mean(maxvals)
    return maxvals, mean_max

def filterIF(ifs, maxvals, mean_max, tol, printout):
    true_ifs_index = []
    false_ifs_index = []
    t_counter = 0
    f_counter = 0
    for i in range(len(ifs)):
        if maxvals[i] > (1.-tol)*mean_max:
            true_ifs_index.append(i)
            status = 'True'
            t_counter+=1
        else:
            false_ifs_index.append(i)
            status = 'False'
            f_counter+=1
        if printout:
            print('|| Cell no: {} || maxval: {:.4f} || value to beat: {:.4f} || Status: {} || True/False Counter {} ||'.format(i+1,maxvals[i], (1.-tol)*mean_max, status, [t_counter, f_counter]))
            print('=====================================================================================================')
    true_ifs_index = np.array(true_ifs_index)
    false_ifs_index = np.array(false_ifs_index)
    return true_ifs_index, false_ifs_index


def validateIFs(ifs, shade_ifs, mirror_len, tol=0.05, printout=False):
    maxvals, mean_max = meanMax(shade_ifs)
    true_ifs_index, false_ifs_index = filterIF(ifs, maxvals, mean_max, tol, printout)
    true_ifs = []
    false_ifs = []
    for i in true_ifs_index:
        true_ifs.append(ifs[i])
    for i in false_ifs_index:
        false_ifs.append(ifs[i])
    true_ifs = np.array(true_ifs)
    false_ifs = np.array(false_ifs)
    return true_ifs, false_ifs

def addToStack(img1, img2, place='first'):
    """
    Appends an image or a set of images to another image or set of
    images. img1 and img2 must be of shapes (L x M x N) and (K x M x N)
    """
    if img1.ndim == 2 and img2.ndim == 2:
        if place == 'first':
            imstack = np.vstack((img1, img2))
        elif place == 'last':
            imstack = np.vstack((img2, img1))
    if img1.ndim == 2 and img2.ndim == 3:
        img1 = np.reshape(img1, (1, img1.shape[1], img1.shape[2]))
        print('img1 shape', img1.shape)
        if place == 'first':
            imstack = np.vstack((img1, img2))
        elif place == 'last':
            imstack = np.vstack((img2, img1))
    if img1.ndim == 3 and img2.ndim == 2:
        if place == 'first':
            imstack = np.vstack((img1[:], img2))
        elif place == 'last':
            imstack = np.vstack((img2, img1[:]))
    if img1.ndim == 3 and img2.ndim == 3:
        if place == 'first':
            imstack = np.vstack((img1[:], img2[:]))
        elif place == 'last':
            imstack = np.vstack((img2[:], img1[:]))
    return imstack

def displayIFs(ifs, dx, imbounds=None, vbounds=None, colormap='jet',
                figsize=(8,5), title_fntsz=14, ax_fntsz=12,
                title='Influence Functions',
                x_title='Azimuthal Dimension (mm)',
                y_title='Axial Dimension (mm)',
                cbar_title='Figure (microns)',
                frame_time=500, repeat_bool=False):

    """
    Displays set of IFs in a single animation on one figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_title, fontsize = ax_fntsz)
    ax.set_ylabel(y_title, fontsize = ax_fntsz)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    extent = fp.mk_extent(ifs[0], dx)
    # a = np.zeros((1, ifs1.shape[1], ifs1.shape[2])) + 0.1
    # ifs = np.vstack((a, ifs1))
    if not vbounds:
        vbounds = [np.nanmin(ifs), np.nanmax(ifs)]
    if imbounds:
        lbnd = imbounds[0]
        ubnd = imbounds[1]
    else:
        lbnd = 0
        ubnd = ifs.shape[0]
    ims = []
    for i in range(ifs[lbnd:ubnd].shape[0]):
        im = ax.imshow(ifs[i+lbnd], extent=extent, aspect='auto', cmap=colormap,
                        vmin=vbounds[0], vmax=vbounds[1])
        cell_no = i + lbnd + 1
        txtstring = title + '\nCell #: {}'.format(cell_no)
        title_plt_text = ax.text(0.5, 1.075, txtstring, fontsize=title_fntsz,
                                ha='center', va='center', transform=ax.transAxes)
        ims.append([im, title_plt_text])

    cbar = fig.colorbar(ims[0][0], cax=cax)
    cbar.set_label(cbar_title, fontsize=ax_fntsz)
    ani = animation.ArtistAnimation(fig, ims, interval=frame_time, blit=False,
                                    repeat=repeat_bool)
    fps = int(1 / (frame_time/1000))
    return ani, fps
