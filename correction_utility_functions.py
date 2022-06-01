from numpy import *
import matplotlib.pyplot as plt
import os
import glob
import pickle
import astropy.io.fits as pyfits
from astropy.modeling import models
from matplotlib import gridspec
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdb

import utilities.imaging.man as man
import utilities.imaging.stitch as stitch
import utilities.metrology as met
import utilities.fourier as fourier
import utilities.imaging.fitting as fit

import axroOptimization.evaluateMirrors as eva

home_directory = os.getcwd()

def printer():
    print('Hello correlation utilitiy functions!')

##########################################################################################################
# Utility functions.

def pv(img):
    return nanmax(img) - nanmin(img)

def convertToAxialSlopes(img,dx):
    return gradient(img,dx)[0]*3600*180/pi

def stripWithShade(dist,shade):
    output = copy(dist)
    output = man.newGridSize(dist,shape(shade))
    output[shade == 0] = NaN
    return man.stripnans(output)

##########################################################################################################
# CTF specific functions

def generateAxialSineModel(amp,period,ylen,xlen,dy,phase = 0.0):
    '''
    Constructs a 1D sine model, oriented in the axial direction, from the
    format of an example image.
    '''
    x,y = meshgrid(linspace(0,1,xlen),linspace(0,1,ylen))
    g = models.Sine1D(amp,ylen*dy/period,phase)
    return g(y)

def generate2DLegendreModel(xo,yo,xlen,ylen,coeffs = None):
    '''

    '''
    x,y = meshgrid(linspace(-1,1,xlen),linspace(-1,1,ylen))
    g = models.Legendre2D(xo,yo)
    if (coeffs is not None) & (len(g.__dict__['_parameters']) == len(coeffs)):
        try:
            g.__dict__['_parameters'] = coeffs
        except:
            pdb.set_trace()
    return g(x,y)

##########################################################################################################
# Iteration assessment functions.

def readCylWFSRaw(fn):
    """
    Load in data from WFS measurement of cylindrical mirror.
    Assumes that data was processed using processHAS, and loaded into
    a .fits file.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = pyfits.getdata(fn)
    d = man.stripnans(d)

    # Negate to make bump positive.
    d = -d

    return d

def reshapeMeasToCorrection(raw_correction,shape_match,mask_fraction):
    # Loading the as-measured correction and processing it appropriately to be stripped of
    # exterior NaNs, bump positive, and have best fit cylinder removed (like dist_map and the ifs).
    # This raw correction has its own distinct shape of order 120 by 100.

    # Creating a perimeter shademask consistent with the size of the measured change.
    meas_shade = eva.slv.createShadePerimeter(shape(raw_correction),axialFraction = mask_fraction,azFraction = mask_fraction)

    # Now making the measured relative change directly comparable to the area of the
    # distortion map we are trying to correct by putting the shade mask in place, and
    # then interpolating to the size of dist_map.
    rel_change = copy(raw_correction)
    rel_change[meas_shade == 0] = NaN
    rel_change = man.newGridSize(rel_change,shape_match)
    return rel_change

def getIterMeasResults(directory,desired_shape,mask_fraction = 30./101.6,name_search = 'DistortionToCorrect'):
    os.chdir(directory)
    fig_files = glob.glob('*' + name_search + '*')
    figs = [reshapeMeasToCorrection(readCylWFSRaw(fn),desired_shape,mask_fraction) for fn in fig_files]
    os.chdir(home_directory)
    return figs

def getIterTheoResults(directory,desired_shape,mask_fraction = 30./101.6,name_search = 'DistortionToCorrect'):
    os.chdir(directory)
    fig_files = glob.glob('*' + name_search + '*')
    figs = [reshapeMeasToCorrection(pyfits.getdata(fn),desired_shape,mask_fraction) for fn in fig_files]
    os.chdir(home_directory)
    return figs

def getIterVoltages(directory,name_search = 'OptVolts'):
    os.chdir(directory)
    volt_files = glob.glob('*' + name_search + '*')
    volts = [loadtxt(fn) for fn in volt_files]
    os.chdir(home_directory)
    return volts

def evalAxSlopes(img,dx):
    return gradient(img,dx)[0]*3600*180/pi

def evalSlopeImprovement(dist,cor,fig,dx_eff):
    dist_std = nanstd(evalAxSlopes(dist*10**-4,dx_eff))
    theo_cor_std = nanstd(evalAxSlopes((dist + cor)*10**-3,dx_eff))
    meas_cor_std = nanstd(evalAxSlopes((dist + fig)*10**-3,dx_eff))
    res_std = nanstd(evalAxSlopes((cor - fig)*10**-3,dx_eff))
    return dist_std,theo_cor_std,meas_cor_std,res_std

##########################################################################################################
# Plotting functions.

def mirror_subplot(data_img,ax,title,cbar_label,extent = None,vmin = None,vmax = None,draw_cbar = True,merit = None,
                     merit1_label = 'PSF E68', merit2_label = 'PSF HPD',merit1_unit = 'asec.',merit2_unit = 'asec.'):
    '''
    The default figure plot style I want to use. Needs a specified input
    data set, plotting axis and title. Options include an extent, vmin/vmax args,
    and adding a merit function to the plot.
    '''
    im = ax.imshow(data_img,extent = extent,vmin = vmin,vmax = vmax)
    ax.set_xlabel('Azimuthal Dimension (mm)',fontsize = 16)
    ax.set_ylabel('Axial Dimension (mm)',fontsize = 16)
    ax.set_title(title,fontsize = 16)
    if draw_cbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = plt.colorbar(im, cax = cax)
        cbar.set_label(cbar_label,fontsize = 16)

    if merit is not None:
        ax.text(0.05,0.05,merit1_label + ': ' + "{:4.3f}".format(merit[0]) + ' ' + merit1_unit,ha = 'left',transform = ax.transAxes)
        ax.text(0.05,0.10,merit2_label + ': ' + "{:3.3f}".format(merit[1]) + ' ' + merit2_unit,ha = 'left',transform = ax.transAxes)

def plot_correction_inline(input_dist,fc,cor,dx,first_title = '',second_title = '',third_title = '',
                             cbar_label = '',global_title = '',save_file = None,vbounds = None,dist_merit = None,\
                             fc_merit = None,cor_merit = None,
                             merit1_label = 'PSF E68', merit2_label = 'PSF HPD',merit1_unit = 'asec.',merit2_unit = 'asec.'):
    '''
    '''
    fig = plt.figure(figsize = (18,5))
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_dist = man.stripnans(input_dist - nanmean(input_dist))
    plot_fc = man.newGridSize(man.stripnans(fc - nanmean(fc)),shape(plot_dist))
    plot_cor = man.newGridSize(man.stripnans(cor - nanmean(cor)),shape(plot_dist))

    extent = [-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx,-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx]

    if vbounds is None:
        vmin,vmax = nanmin([plot_dist,plot_fc,plot_cor]),nanmax([plot_dist,plot_fc,plot_cor])
    else:
        [vmin,vmax] = vbounds

    mirror_subplot(plot_dist,ax1,first_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = dist_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)
    mirror_subplot(plot_fc,ax2,second_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = fc_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)
    mirror_subplot(plot_cor,ax3,third_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = cor_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)

    fig.subplots_adjust(top = 0.74,hspace = 0.4,wspace = 0.4)

    plt.suptitle(global_title,fontsize = 20)

    if save_file != None:
        plt.savefig(save_file)
        plt.close()
    return plot_dist,plot_fc,plot_cor

def plot_measured_correction_sixfig(input_dist,theo_corr,meas_corr0,meas_corr1,dx,first_title = '',second_title = '',third_title = '',
                                    fourth_title = '',fifth_title = '',sixth_title = '', cbar_label = '',global_title = '',save_file = None,
                                    dist_merit = None, meas_corr_merit0 = None,meas_corr_merit1 = None,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (12,16))
    gs = gridspec.GridSpec(3,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])

    plot_dist = man.stripnans(input_dist - nanmean(input_dist))
    plot_theo_corr = man.newGridSize(man.stripnans(theo_corr - nanmean(theo_corr)),shape(plot_dist))
    plot_meas_corr0 = man.newGridSize(man.stripnans(meas_corr0 - nanmean(meas_corr0)),shape(plot_dist))
    plot_meas_corr1 = man.newGridSize(man.stripnans(meas_corr1 - nanmean(meas_corr1)),shape(plot_dist))

    extent = [-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx,-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx]
    if vbounds == None:
        vmin = nanmin([plot_dist,plot_theo_corr,plot_meas_corr0,plot_dist + plot_meas_corr0,plot_meas_corr1,plot_dist + plot_meas_corr1]),
        vmax = nanmax([plot_dist,plot_theo_corr,plot_meas_corr,plot_dist + plot_meas_corr,plot_meas_corr1,plot_dist + plot_meas_corr1])
    else:
        [vmin,vmax] = vbounds

    mirror_subplot(plot_dist,ax1,first_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = dist_merit)
    mirror_subplot(plot_theo_corr,ax2,second_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = None)
    mirror_subplot(plot_meas_corr0,ax3,third_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = None)
    mirror_subplot(plot_meas_corr0 + plot_dist,ax4,fourth_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = meas_corr_merit0)
    mirror_subplot(plot_meas_corr1,ax5,fifth_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = None)
    mirror_subplot(plot_meas_corr1 + plot_dist,ax6,sixth_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax, merit = meas_corr_merit1)

    fig.subplots_adjust(hspace = 0.4,wspace = 0.3)

    plt.suptitle(global_title,fontsize = 20)

    if save_file != None:
        plt.savefig(save_file)
        plt.close()

    return fig,(ax1,ax2,ax3,ax4,ax5,ax6)

def plot_compare_theo_meas_corr(plot_dist,plot_theo_corr,plot_meas_corr,plot_compare_theo_meas,dx,first_title = '',second_title = '',third_title = '',fourth_title = '',
                             cbar_label = '',global_title = '',save_file = None,vbounds = [-1.,1.],dist_merit = None,\
                             theo_corr_merit = None,meas_corr_merit = None,slope = False):
    '''
    '''
    fig = plt.figure(figsize = (12,12))
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    extent = [-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx,-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx]

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_corr,plot_dist + plot_corr]),nanmax([plot_dist,plot_corr,plot_dist + plot_corr])
    else:
        [vmin,vmax] = vbounds

    mirror_subplot(plot_dist,ax1,first_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax,merit = dist_merit)
    mirror_subplot(plot_theo_corr + plot_dist,ax2,second_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax,merit = theo_corr_merit)
    mirror_subplot(plot_meas_corr + plot_dist,ax3,third_title,cbar_label,extent = extent,vmin = vmin,vmax = vmax,merit = meas_corr_merit)

    if slope is False:
        mirror_subplot(plot_compare_theo_meas,ax4,fourth_title,cbar_label,extent = extent,vmin = -0.100,vmax = 0.100,merit = [nanstd(plot_compare_theo_meas*10**3),pv(plot_compare_theo_meas)*10**3],
                        merit1_label = 'RMS', merit2_label = 'PV',merit1_unit = 'nm',merit2_unit = 'nm')
    else:
        mirror_subplot(plot_compare_theo_meas,ax4,fourth_title,cbar_label,extent = extent,vmin = -2,vmax = 2,merit = [nanstd(plot_compare_theo_meas),pv(plot_compare_theo_meas)],
                       merit1_label = 'RMS', merit2_label = 'PV',merit1_unit = 'asec.',merit2_unit = 'asec.')

    fig.subplots_adjust(top = 0.85,hspace = 0.4,wspace = 0.4)

    plt.suptitle(global_title,fontsize = 20)

    if save_file != None:
        plt.savefig(save_file)
    return plot_dist,plot_theo_corr,plot_meas_corr

def plot_fig_slope_sidebyside(input_data,input_slopes,dx,individual_title = '',global_title = '',
                              save_file = None,vbounds_fig = None,vbounds_slope = None,
                              fig_merit = None,slope_merit = None,slope_unit = 'asec.',plot_to_use = None,draw_cbar = True):
    '''
    '''
    if plot_to_use is None:
        fig = plt.figure(figsize = (12,5))
        gs = gridspec.GridSpec(1,2)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
    else:
        fig,(ax1,ax2) = plot_to_use

    plot_figdata,plot_slopedata = input_data,input_slopes

    extent = [-shape(plot_figdata)[0]/2*dx,shape(plot_figdata)[0]/2*dx,-shape(plot_figdata)[0]/2*dx,shape(plot_figdata)[0]/2*dx]

    if vbounds_fig is None:
        vbounds_fig = [nanmin([plot_figdata]),nanmax([plot_figdata])]
    if vbounds_slope is None:
        vbounds_slope = [nanmin([plot_slopedata]),nanmax([plot_slopedata])]

    mirror_subplot(plot_figdata,ax1,individual_title + 'Figure Space',cbar_label = 'Figure (microns)',extent = extent,vmin = vbounds_fig[0],vmax = vbounds_fig[1], merit = fig_merit,
                   merit1_label = 'RMS', merit2_label = 'PV',merit1_unit = 'um',merit2_unit = 'um',draw_cbar = draw_cbar)
    mirror_subplot(plot_slopedata,ax2,individual_title + 'Axial Slope Space',cbar_label = 'Slope (arcseconds)',extent = extent,vmin = vbounds_slope[0],vmax = vbounds_slope[1], merit = slope_merit,
                   merit1_label = 'RMS', merit2_label = 'PV',merit1_unit = slope_unit,merit2_unit = slope_unit,draw_cbar = draw_cbar)

    fig.subplots_adjust(top = 0.9,hspace = 0.4,wspace = 0.4)
    plt.suptitle(global_title,fontsize = 20)

    if save_file != None:
        plt.savefig(save_file)
        plt.close()
    return fig,(ax1,ax2)

def mirror_subplot_vlad(data_img,ax,title,cbar_label,extent = None,vmin = None,vmax = None,draw_cbar = True,merit = None,
                     merit1_label = 'PSF E68', merit2_label = 'PSF HPD',merit1_unit = 'asec.',merit2_unit = 'asec.'):
    '''
    The default figure plot style I want to use. Needs a specified input
    data set, plotting axis and title. Options include an extent, vmin/vmax args,
    and adding a merit function to the plot.
    '''
    im = ax.imshow(data_img,extent = extent,vmin = vmin,vmax = vmax)
    ax.set_xlabel('Azimuthal Dimension (mm)',fontsize = 12)
    ax.set_ylabel('Axial Dimension (mm)',fontsize = 12)
    ax.set_title(title,fontsize = 16)
    if draw_cbar is True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = plt.colorbar(im, cax = cax,format='%.0e')
        cbar.set_label(cbar_label,fontsize = 12)

    if merit is not None:
        ax.text(0.05,0.05,merit1_label + ': ' + "{:3.2e}".format(merit[0]) + ' ' + merit1_unit,ha = 'left',transform = ax.transAxes)
        ax.text(0.05,0.10,merit2_label + ': ' + "{:3.2e}".format(merit[1]) + ' ' + merit2_unit,ha = 'left',transform = ax.transAxes)

def plot_correction_inline_vlad(input_dist,fc,cor,dx,first_title = '',second_title = '',third_title = '',
                             cbar_label = '',global_title = '',save_file = None,dist_merit = None,vbounds = None,\
                             fc_merit = None,cor_merit = None,
                             merit1_label = 'PSF E68', merit2_label = 'PSF HPD',merit1_unit = 'asec.',merit2_unit = 'asec.'):
    '''
    '''
    fig = plt.figure(figsize = (18,5))
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_dist = man.stripnans(input_dist - nanmean(input_dist))
    plot_fc = man.newGridSize(man.stripnans(fc - nanmean(fc)),shape(plot_dist))
    plot_cor = man.newGridSize(man.stripnans(cor - nanmean(cor)),shape(plot_dist))

    extent = [-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx,-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx]

    #if vbounds is None:
    #    vmin,vmax = nanmin([plot_dist,plot_fc,plot_cor]),nanmax([plot_dist,plot_fc,plot_cor])
    #else:
    #    [vmin,vmax] = vbounds

    mirror_subplot_vlad(plot_dist,ax1,first_title,cbar_label,extent = extent, merit = dist_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)
    mirror_subplot_vlad(plot_fc,ax2,second_title,cbar_label,extent = extent, merit = fc_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)
    mirror_subplot_vlad(plot_cor,ax3,third_title,cbar_label,extent = extent, merit = cor_merit,merit1_label = merit1_label, merit2_label = merit2_label,merit1_unit = merit1_unit,merit2_unit = merit2_unit)

    fig.subplots_adjust(top = 0.74,hspace = 0.3,wspace = 0.5)

    plt.suptitle(global_title,fontsize = 20)

    if save_file != None:
        plt.savefig(save_file)
        plt.close()
    return plot_dist,plot_fc,plot_cor

#####################################################################################
#####################################################################################




def plot_slumped_data_map(slump_data,shademask):
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(slump_data,extent = [-50,50,-50,50])
    ax1.set_xlabel('Azimuthal Dimension (mm)',fontsize = 16)
    ax1.set_ylabel('Axial Dimension (mm)',fontsize = 16)
    ax1.set_title('Dimple Removed, 10th Order\nLegendre Fit To Slumped Data',fontsize = 16)

    inner_region = stripWithShade(slump_data,shademask)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im, cax = cax1)
    cbar.set_label('Figure (microns)')

    ax1.add_patch(patches.Rectangle((-35.05,-35.05),70.1,70.1,fill = False))
    ax1.text(0.05,0.05,'PV: ' + "{:3.1f}".format(pv(slump_data)) + ' um',ha = 'left',transform = ax1.transAxes,fontsize = 16)
    ax1.text(0.05,0.08,'RMS: ' + "{:3.1f}".format(nanstd(slump_data)) + ' um',ha = 'left',transform = ax1.transAxes,fontsize = 16)
    ax1.text(-32,-32,'PV: ' + "{:3.1f}".format(pv(inner_region)) + ' um',ha = 'left',fontsize = 16)
    ax1.text(-32,-29,'RMS: ' + "{:3.1f}".format(nanstd(inner_region)) + ' um',ha = 'left',fontsize = 16)

def plot_bffc_map(bffc):
    fig = plt.figure(figsize = (10,10))
    ax1 = fig.add_subplot(111)
    im = ax1.imshow(bffc,extent = [-30.1,30.1,-30.1,30.1])
    ax1.set_xlabel('Azimuthal Dimension (mm)',fontsize = 16)
    ax1.set_ylabel('Axial Dimension (mm)',fontsize = 16)
    ax1.set_title('Theoretical Best Fit Figure Change\nTo Correct Slumped Mirror Data',fontsize = 16)

    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im, cax = cax1)
    cbar.set_label('Figure (microns)')

    ax1.add_patch(patches.Rectangle((-35.05,-35.05),70.1,70.1,fill = False))
    ax1.text(0.05,0.05,'PV: ' + "{:3.1f}".format(pv(bffc)) + ' um',ha = 'left',transform = ax1.transAxes,fontsize = 16)
    ax1.text(0.05,0.08,'RMS: ' + "{:3.1f}".format(nanstd(bffc)) + ' um',ha = 'left',transform = ax1.transAxes,fontsize = 16)

def plot_computed_correction(input_dist,comp_corr,dx,shade,first_title = '',second_title = '',sum_title = '', \
                            cbar_label = '',global_title = '',save_file = None,est_perf = False, \
                            dist_merit = None,corr_merit = None,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (12,10))
    gs = gridspec.GridSpec(4,4)
    ax1 = fig.add_subplot(gs[0:2,0:2])
    ax2 = fig.add_subplot(gs[0:2,2:4])
    ax3 = fig.add_subplot(gs[2:4,1:3])
    fig.subplots_adjust(top = 0.9,hspace = 1.0,wspace = 1.0)

    plot_corr = man.stripnans(comp_corr)
    corr_shade = ~isnan(comp_corr)
    plot_dist = stripWithShade(input_dist,corr_shade)

    extent = [-shape(plot_corr)[0]/2*dx,shape(plot_corr)[0]/2*dx,-shape(plot_corr)[0]/2*dx,shape(plot_corr)[0]/2*dx]

    if shape(plot_dist) != shape(plot_corr):
        print("Something's fucked here, mate")
        pdb.set_trace()

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_corr,plot_dist + plot_corr]),nanmax([plot_dist,plot_corr,plot_dist + plot_corr])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_dist + plot_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(sum_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    fig.subplots_adjust(top = 0.83,hspace = 0.5,wspace = 1.5)

    plt.suptitle(global_title,fontsize = 20)

    if est_perf == True:
        print('Computing performance for plotting... Be patient!')
        dist_merit = eva.computeMeritFunctions(plot_dist,[dx])
        corr_merit = eva.computeMeritFunctions(plot_dist + plot_corr,[dx])

        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax3.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(corr_merit[0]) + ' asec.',ha = 'left',transform = ax3.transAxes)
        ax3.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(corr_merit[1]) + ' asec.',ha = 'left',transform = ax3.transAxes)

    elif logical_and(dist_merit is not None,corr_merit is not None):
        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax3.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(corr_merit[0]) + ' asec.',ha = 'left',transform = ax3.transAxes)
        ax3.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(corr_merit[1]) + ' asec.',ha = 'left',transform = ax3.transAxes)

    if save_file != None:
        plt.savefig(save_file)
    return fig,(ax1,ax2,ax3)

def plot_computed_correction_inline(input_dist,fc,cor,dx,shade,first_title = '',second_title = '',sum_title = '', \
                            cbar_label = '',global_title = '',save_file = None,est_perf = False, \
                            dist_merit = None,corr_merit = None,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (18,5))
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    #fig.subplots_adjust(top = 0.9,hspace = 1.0,wspace = 1.0)

    plot_corr = man.stripnans(comp_corr)
    corr_shade = ~isnan(comp_corr)
    plot_dist = stripWithShade(input_dist,corr_shade)

    extent = [-shape(plot_corr)[0]/2*dx,shape(plot_corr)[0]/2*dx,-shape(plot_corr)[0]/2*dx,shape(plot_corr)[0]/2*dx]

    if shape(plot_dist) != shape(plot_corr):
        print("Something's fucked here, mate")
        pdb.set_trace()

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_corr,plot_dist + plot_corr]),nanmax([plot_dist,plot_corr,plot_dist + plot_corr])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_dist + plot_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(sum_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    fig.subplots_adjust(top = 0.7,hspace = 0.05,wspace = 0.6)

    plt.suptitle(global_title,fontsize = 20)

    if est_perf == True:
        print('Computing performance for plotting... Be patient!')
        dist_merit = eva.computeMeritFunctions(plot_dist,[dx])
        corr_merit = eva.computeMeritFunctions(plot_dist + plot_corr,[dx])

        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax3.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(corr_merit[0]) + ' asec.',ha = 'left',transform = ax3.transAxes)
        ax3.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(corr_merit[1]) + ' asec.',ha = 'left',transform = ax3.transAxes)

    elif logical_and(dist_merit is not None,corr_merit is not None):
        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax3.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(corr_merit[0]) + ' asec.',ha = 'left',transform = ax3.transAxes)
        ax3.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(corr_merit[1]) + ' asec.',ha = 'left',transform = ax3.transAxes)

    if save_file != None:
        plt.savefig(save_file)
    return fig,(ax1,ax2,ax3)

def plot_measured_correction(input_dist,theo_corr,meas_corr,dx,first_title = '',second_title = '',third_title = '',sum_title = '',
                             cbar_label = '',global_title = '',save_file = None,est_perf = False,dist_merit = None, meas_corr_merit = None,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (12,10))
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    #fig.subplots_adjust(top = 0.9,hspace = 0.1,wspace = 0.1)

    plot_dist = man.stripnans(input_dist - nanmean(input_dist))
    plot_theo_corr = man.newGridSize(man.stripnans(theo_corr - nanmean(theo_corr)),shape(plot_dist))
    plot_meas_corr = man.newGridSize(man.stripnans(meas_corr - nanmean(meas_corr)),shape(plot_dist))

    extent = [-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx,-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx]
    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_theo_corr,plot_meas_corr,plot_dist + plot_meas_corr]),nanmax([plot_dist,plot_theo_corr,plot_meas_corr,plot_dist + plot_meas_corr])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_theo_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_meas_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(third_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    ax4.imshow(plot_meas_corr + plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax4.set_xlabel('Azimuthal Dimension (mm)')
    ax4.set_ylabel('Axial Dimension (mm)')
    ax4.set_title(sum_title)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.10)
    cbar4 = plt.colorbar(im, cax = cax4)
    cbar4.set_label(cbar_label)

    fig.subplots_adjust(top = 0.85,hspace = 0.4,wspace = 0.4)

    plt.suptitle(global_title,fontsize = 20)

    if est_perf == True:
        print('Computing performance for plotting... Be patient!')
        dist_merit = eva.computeMeritFunctions(plot_dist,[dx])
        corr_merit = eva.computeMeritFunctions(plot_dist + plot_meas_corr,[dx])

        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax4.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(meas_corr_merit[0]) + ' asec.',ha = 'left',transform = ax4.transAxes)
        ax4.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(meas_corr_merit[1]) + ' asec.',ha = 'left',transform = ax4.transAxes)
    if save_file != None:
        plt.savefig(save_file)
    return fig,(ax1,ax2,ax3,ax4)

def plot_measured_correction_for_iteration(fig,input_dist,theo_corr,meas_corr,dx,first_title = '',second_title = '',third_title = '',sum_title = '',
                             cbar_label = '',global_title = '',save_file = None,est_perf = False,dist_merit = None, meas_corr_merit = None, vbounds = [-1.,1.]):
    '''
    '''
    gs = gridspec.GridSpec(2,2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    plot_dist = man.stripnans(input_dist - nanmean(input_dist))
    plot_theo_corr = man.newGridSize(man.stripnans(theo_corr - nanmean(theo_corr)),shape(plot_dist))
    plot_meas_corr = man.newGridSize(man.stripnans(meas_corr - nanmean(meas_corr)),shape(plot_dist))

    extent = [-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx,-shape(plot_theo_corr)[0]/2*dx,shape(plot_theo_corr)[0]/2*dx]

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_corr,plot_dist + plot_corr]),nanmax([plot_dist,plot_corr,plot_dist + plot_corr])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_theo_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_meas_corr,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(third_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    ax4.imshow(plot_meas_corr + plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax4.set_xlabel('Azimuthal Dimension (mm)')
    ax4.set_ylabel('Axial Dimension (mm)')
    ax4.set_title(sum_title)
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.10)
    cbar4 = plt.colorbar(im, cax = cax4)
    cbar4.set_label(cbar_label)

    fig.subplots_adjust(top = 0.9,hspace = 0.4,wspace = 0.4)

    plt.suptitle(global_title,fontsize = 20)

    if est_perf == True:
        ax1.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax4.text(0.05,0.05,'PSF RMS: ' + "{:4.1f}".format(meas_corr_merit[0]) + ' asec.',ha = 'left',transform = ax4.transAxes)
        ax4.text(0.05,0.10,'PSF HPD: ' + "{:3.1f}".format(meas_corr_merit[1]) + ' asec.',ha = 'left',transform = ax4.transAxes)
    if save_file != None:
        plt.savefig(save_file)
    return plot_dist,plot_theo_corr,plot_meas_corr


def plot_computed_correction_trifig_inline(input_dist,fc,cor,dx,first_title = '',second_title = '',third_title = '', \
                            cbar_label = '',global_title = '',save_file = None,est_perf = False, \
                            dist_merit = None,fc_merit = None,cor_merit = None,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (18,5))
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_dist,plot_fc,plot_cor = input_dist,fc,cor

    extent = [-shape(plot_dist)[1]/2*dx,shape(plot_dist)[1]/2*dx,-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx]

    if (shape(plot_dist) != shape(plot_cor)) | (shape(plot_dist) != shape(plot_fc)):
        print("Something's fucked here, mate")
        pdb.set_trace()

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_fc,plot_cor]),nanmax([plot_dist,plot_fc,plot_cor])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_fc,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_cor,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(third_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    fig.subplots_adjust(top = 0.7,hspace = 0.05,wspace = 0.6)

    plt.suptitle(global_title,fontsize = 20)

    #if est_perf == True:
    #    if logical_and(dist_merit is None,cor_merit is None):
    #        dist_merit = eva.computeMeritFunctions(plot_dist,[dx])
    #        cor_merit = eva.computeMeritFunctions(plot_cor,[dx])
    if dist_merit is not None:
        ax1.text(0.05,0.05,'PSF RMS: ' + "{:5.2f}".format(dist_merit[0]) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax1.text(0.05,0.10,'PSF HPD: ' + "{:4.2f}".format(dist_merit[1]) + ' asec.',ha = 'left',transform = ax1.transAxes)
    if fc_merit is not None:
        ax2.text(0.05,0.05,'PSF RMS: ' + "{:5.2f}".format(fc_merit[0]) + ' asec.',ha = 'left',transform = ax2.transAxes)
        ax2.text(0.05,0.10,'PSF HPD: ' + "{:4.2f}".format(fc_merit[1]) + ' asec.',ha = 'left',transform = ax2.transAxes)
    if cor_merit is not None:
        ax3.text(0.05,0.05,'PSF RMS: ' + "{:5.2f}".format(cor_merit[0]) + ' asec.',ha = 'left',transform = ax3.transAxes)
        ax3.text(0.05,0.10,'PSF HPD: ' + "{:4.2f}".format(cor_merit[1]) + ' asec.',ha = 'left',transform = ax3.transAxes)

    if save_file != None:
        plt.savefig(save_file)
    return fig,(ax1,ax2,ax3)

def plot_computed_corrections_trifig_inline_slopes(input_dist,fc,cor,dx,first_title = '',second_title = '',third_title = '', \
                            cbar_label = '',global_title = '',save_file = None,est_perf = False,vbounds = None):
    '''
    '''
    fig = plt.figure(figsize = (18,5))
    gs = gridspec.GridSpec(1,3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    plot_dist,plot_fc,plot_cor = input_dist,fc,cor

    extent = [-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx,-shape(plot_dist)[0]/2*dx,shape(plot_dist)[0]/2*dx]

    if (shape(plot_dist) != shape(plot_cor)) | (shape(plot_dist) != shape(plot_fc)):
        print("Something's fucked here, mate")
        pdb.set_trace()

    if vbounds == None:
        vmin,vmax = nanmin([plot_dist,plot_fc,plot_cor]),nanmax([plot_dist,plot_fc,plot_cor])
    else:
        [vmin,vmax] = vbounds

    im = ax1.imshow(plot_dist,extent = extent,vmin = vmin,vmax = vmax)
    ax1.set_xlabel('Azimuthal Dimension (mm)')
    ax1.set_ylabel('Axial Dimension (mm)')
    ax1.set_title(first_title)
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.10)
    cbar1 = plt.colorbar(im, cax = cax1)
    cbar1.set_label(cbar_label)

    ax2.imshow(plot_fc,extent = extent,vmin = vmin,vmax = vmax)
    ax2.set_xlabel('Azimuthal Dimension (mm)')
    ax2.set_ylabel('Axial Dimension (mm)')
    ax2.set_title(second_title)
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.10)
    cbar2 = plt.colorbar(im, cax = cax2)
    cbar2.set_label(cbar_label)

    ax3.imshow(plot_cor,extent = extent,vmin = vmin,vmax = vmax)
    ax3.set_xlabel('Azimuthal Dimension (mm)')
    ax3.set_ylabel('Axial Dimension (mm)')
    ax3.set_title(third_title)
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.10)
    cbar3 = plt.colorbar(im, cax = cax3)
    cbar3.set_label(cbar_label)

    fig.subplots_adjust(top = 0.7,hspace = 0.05,wspace = 0.6)

    plt.suptitle(global_title,fontsize = 20)

    if est_perf == True:
        ax1.text(0.05,0.05,'RMS Ax. Slope: ' + "{:5.2f}".format(nanstd(plot_dist)) + ' asec.',ha = 'left',transform = ax1.transAxes)
        ax2.text(0.05,0.05,'RMS Ax. Slope: ' + "{:5.2f}".format(nanstd(plot_fc)) + ' asec.',ha = 'left',transform = ax2.transAxes)
        ax3.text(0.05,0.05,'RMS Ax. Slope: ' + "{:5.2f}".format(nanstd(plot_cor)) + ' asec.',ha = 'left',transform = ax3.transAxes)

    if save_file != None:
        plt.savefig(save_file)
    return fig,(ax1,ax2,ax3)
