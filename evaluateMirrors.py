import numpy as np
import matplotlib.pyplot as plt
import axroOptimization.scattering as scat
import utilities.imaging.man as man
import axroOptimization.solver as slv
import axroOptimization.conicsolve as conic
import axroOptimization.correction_utility_functions as cuf
import axroHFDFCpy.construct_connections as cc
import time

import pdb

def printer():
    print('Hello evaluate mirrors!')

def correctXrayTestMirror(d,ifs,shade=None,dx=None,azweight=.015,smax=5.,\
                          bounds=None,regrid_figure_change = False,avg_slope_remove = True,\
                          matlab_opt = False):
    """
    Get distortion on same grid as IFs and run correction.
    Rebin result onto original distortion grid and apply.
    dx should be on IF grid size
    """
    #Rebin to IF grid
    if d.shape != ifs[0].shape:
        d2 = man.newGridSize(d,np.shape(ifs[0]))
    else:
        d2 = d

    #Handle shademask
    if shade is None:
        shade = np.ones(np.shape(d2))

    #Run correction
    volt = slv.correctDistortion(d2,ifs,shade,dx=dx,azweight=azweight,
                                smax=smax,bounds=bounds,avg_slope_remove = avg_slope_remove,
                                matlab_opt = matlab_opt)
    # Compute the correction on the same scale as the original
    # data. This correction will need to be added to the original
    # data to yield the final corrected figure.
    ifs2 = ifs.transpose(1,2,0)
    cor2 = np.dot(ifs2,volt)

    if regrid_figure_change == True:
        cor3 = man.newGridSize(cor2,np.shape(d),method='linear')
        #Handle shademask
        cor2[shade==0] = np.nan
        cornan = man.newGridSize(cor2,np.shape(d),method='linear')
        cor3[np.isnan(cornan)] = np.nan
        fc = cor3
    else:
        #Handle shademask
        # cor2[shade==0] = np.nan
        fc = cor2

    return fc,volt

def computeMeritFunctions(d,dx,x0=np.linspace(-1.,1.,10001),\
                          R0 = 220.,Z0 = 8400.,wave = 1.24e-6,\
                          renorm=True):
    """
    RMS axial slope
    Axial sag
    d in microns
    dx value in a list: [dx]
    wavelength in mm.
    """
    #Remove NaNs
    d = man.stripnans(d)

    #Compute PSF
    primfoc = conic.primfocus(R0,Z0) # distance to primary focus
    # print('Distance to primary focus: {:.3f} mm'.format(primfoc))
    dx2 = x0[1]-x0[0]
    resa = scat.primary2DPSF(d,dx[0],R0 = R0,Z0 = Z0,x0=x0,wave = wave)

    #Make sure over 95% of flux falls in detector
    integral = np.sum(resa)*dx2

    if integral < .95:
        print('Possible sampling problem. Integral: {:.5f}'.format(np.sum(resa)*dx2))
        # print(str(np.sum(resa)*dx2))
    if integral > 1.5:
        print('Possible aliasing problem. Integral: {:.5f}'.format(np.sum(resa)*dx2))
        # print('Possible aliasing problem')
        # print(str(np.sum(resa)*dx2))

    #Normalize the integral to account for some flux
    #scattered beyond the detector
    if renorm is True:
        resa = resa/integral

    cdf = np.cumsum(resa)*dx2

    #Compute PSF merit functions
    # Computing the rms rigorously, but not usefully.
    #rmsPSF = np.sqrt(np.sum(resa*x0**2)*dx2-(np.sum(resa*x0)*dx2)**2)
    # Computing the rms by assuming a Gaussian profile.
    # E68
    rmsPSF = x0[np.argmin(np.abs(cdf-.84))]-\
             x0[np.argmin(np.abs(cdf-.16))]
    # HPD
    hpdPSF = x0[np.argmin(np.abs(cdf-.75))]-\
             x0[np.argmin(np.abs(cdf-.25))]

    e68 = rmsPSF/primfoc*180/np.pi*60**2 # arcsec
    hpd = hpdPSF/primfoc*180/np.pi*60**2

    return e68, hpd, [x0,resa]

def correctHFDFC3(d,ifs,shade=None,dx=None,azweight=.015,smax=5.,\
                          bounds=None):
    """
    Get distortion on same grid as IFs and run correction.
    Rebin result onto original distortion grid and apply.
    dx should be on IF grid size.
    Needs update on doc string!
    """
    #Rebin to IF grid
    d2 = man.newGridSize(d,np.shape(ifs[0]))

    #Handle shademask
    if shade is None:
        shade = np.ones(np.shape(d2))

    #Run correction
    volt = slv.correctDistortion(d2,ifs,shade,\
                                            dx=dx,azweight=azweight,\
                                            smax=smax,bounds=bounds)

    #Add correction to original data
    ifs2 = ifs.transpose(1,2,0)
    cor2 = np.dot(ifs2,volt)
    #Handle shademask
    cor2[shade==0] = np.nan
    cor3 = man.newGridSize(cor2,np.shape(d),method='linear')

    return cor3,volt

def correct_distortions(dist_maps, ifs, dx, shademask=None, smax=1.0,
                        computeInitialPerformances=True, timeit=True):
    """
    Compute the optimal figure/slope change for a given input distortion(s) and calculate
    the inital and final performances of the X-ray mirror along with voltages needed for correction.

    INPUTS:
    dist_maps: A single 2D array or 3D array of unshaded distortion maps in figure space.
    ifs: 3D array of influence functions
    dx: pixel spacing value (mm/pixel)
    shademask: The shademask that will be used to strip the edges of the figures.
    smax: Sets the upperbound that independent variable can be set to during the
        minimization using sequential least squares programming.
        smax is related to the voltage that the IFs you are using were taken at. smax is the
        max scaled value relative to what the voltages the IFs were taken at.
        i.e., theoretical IFs were taken at 1 V -> smax = 10.0
                measured IFs were taken at 10 V -> smax = 1.0
    OUTPUTS:
    dist_fig_maps: A 3D array of the shaded distortion maps in figure space
    dist_slp_maps: A 3D array of the shaded distortion maps in slope space
    fig_change_maps: A 3D array of the shaded optimal figure change maps
    slope_change_maps: A 3D array of the shaded optimal slope change maps
    corrected_fig_maps: A 3D array of the corrected distortion maps in figure space
    corrected_slp_maps: A 3D array of the corrected distortion maps in slope space
    initial_performances: A list of lists that give the E68 and HPD for the shaded distortion maps
        initial_performances[i][0] and initial_performances[i][1] give the E68 and HPD for
        dist_fig_maps[i]
    final_performances: A list of lists that give the E68 and HPD for the corrected maps.
        final_performances[i][0] and final_performances[i][1] give the E68 and HPD for
        corrected_fig_maps[i]
    volts_array: A N x M array that contains the voltages needed to correct the shaded distortion maps.
        N indexes the distortion map and M indexes the cell number for that distortion map
        volts_array[i, :] is the array of volts needed to produce fig_change_maps[i] given dist_fig_maps[i]
    """
    func_start_time = time.time()
    dist_fig_maps = [] # the shaded distortion maps in figure space
    dist_slp_maps = [] # the shaded distortion maps in slope space
    # initial_performances = [] # list of performances of distortion maps
    # final_performances = [] # list of performances of corrected maps
    fig_change_maps = [] # shaded maps that show optimal figure change
    slope_change_maps = [] # shaded maps that show optimal slope change
    corrected_fig_maps = [] # shaded maps that show the corrected mirror in figure space
    corrected_slp_maps = [] # shaded maps that show the corrected mirror in slope space
    if dist_maps.ndim == 2: # check if only a single distortion map was provided
        dist_maps = dist_maps.reshape(1, dist_maps.shape[0], dist_maps.shape[1])
    if ifs.ndim == 2: # check if only a single IF was provided
        ifs = ifs.reshape(1, ifs.shape[0], ifs.shape[1])
    volts_array = np.full((dist_maps.shape[0], ifs.shape[0]), np.nan) # list of lists that show the volts associated with the optimal figure/slope change
    merits = np.full((dist_maps.shape[0], 3, 2), np.nan) # the array of E68 and HPD values

    for i in range(dist_maps.shape[0]):
        iter_start_time = time.time()
        print('-'*15+'Map: {}'.format(i)+'-'*15)
        # strip the shade from the distortion map
        if shademask is not None:
            dist_shd_map = cuf.stripWithShade(dist_maps[i], shademask)
        else:
            dist_shd_map = dist_maps[i]
        # compute the inital performance of the distortion map
        if computeInitialPerformances:
            initial_performance = computeMeritFunctions(dist_shd_map, [dx])
            print('Initial E68: {:.3f} arcsec, HPD: {:.3f} arcsec'.format(initial_performance[0], initial_performance[1]))
        # compute the optimal figure change (use the unshaded dist_map in figure space)
        fig_change_map, volts = correctXrayTestMirror(dist_maps[i], ifs, shademask, [dx], azweight=0,
                                                        smax=smax, matlab_opt=False)
        # null the mean figure value in slope change map
        fig_change_map -= np.nanmean(fig_change_map)
        # compute the corrected map
        # print('original dist map shape:', dist_maps[i].shape)
        # print('fig_change_map shape:', fig_change_map.shape)
        corrected_fig_map = dist_maps[i] + fig_change_map
        if shademask is not None:
            corrected_fig_shd_map = cuf.stripWithShade(corrected_fig_map, shademask)
        else:
            corrected_fig_shd_map = corrected_fig_map
        final_performance = computeMeritFunctions(corrected_fig_shd_map, [dx])
        print('Final E68: {:.3f} arcsec, HPD: {:.3f} arcsec'.format(final_performance[0], final_performance[1]))

        # strip the shade from figure maps and add them to lists:
        if shademask is not None:
            fig_shd_change_map = cuf.stripWithShade(fig_change_map, shademask)
        else:
            fig_shd_change_map = fig_change_map

        dist_fig_maps.append(dist_shd_map) # previously shaded during initial_performance
        fig_change_maps.append(fig_shd_change_map)
        corrected_fig_maps.append(corrected_fig_shd_map)

        # convert the shaded figure maps to shaded slope maps:
        dist_slp_maps.append(cuf.convertToAxialSlopes(dist_shd_map*1e-3, dx))
        slope_change_maps.append(cuf.convertToAxialSlopes(fig_shd_change_map*1e-3, dx))
        corrected_slp_maps.append(cuf.convertToAxialSlopes(corrected_fig_shd_map*1e-3, dx))

        # add the other outputs to their respective lists/array:
        if computeInitialPerformances:
            merits[i][0][0] = initial_performance[0]
            merits[i][0][1] = initial_performance[1]
        merits[i][-1][0] = final_performance[0]
        merits[i][-1][1] = final_performance[1]
        # initial_performances.append([initial_performance[0], initial_performance[1]]) # [E68, HPD]
        # final_performances.append([final_performance[0], final_performance[1]])
        volts_array[i, :] = np.array(volts)
        if timeit:
            print('Map {} time elapsed: {:.3f} min'.format(i, (time.time()-iter_start_time)/60.))

    # convert figure maps to 3D arrays
    if len(dist_fig_maps) > 1:
        dist_fig_maps = np.stack(dist_fig_maps, axis=0)
        dist_slp_maps = np.stack(dist_slp_maps, axis=0)
        fig_change_maps = np.stack(fig_change_maps, axis=0)
        slope_change_maps = np.stack(slope_change_maps, axis=0)
        corrected_fig_maps = np.stack(corrected_fig_maps, axis=0)
        corrected_slp_maps = np.stack(corrected_slp_maps, axis=0)
    else:
        dist_fig_maps = np.reshape(dist_fig_maps, (1, dist_fig_maps[0].shape[0], dist_fig_maps[0].shape[1]))
        dist_slp_maps = np.reshape(dist_slp_maps, (1, dist_slp_maps[0].shape[0], dist_slp_maps[0].shape[1]))
        fig_change_maps = np.reshape(fig_change_maps, (1, fig_change_maps[0].shape[0], fig_change_maps[0].shape[1]))
        slope_change_maps = np.reshape(slope_change_maps, (1, slope_change_maps[0].shape[0], slope_change_maps[0].shape[1]))
        corrected_fig_maps = np.reshape(corrected_fig_maps, (1, corrected_fig_maps[0].shape[0], corrected_fig_maps[0].shape[1]))
        corrected_slp_maps = np.reshape(corrected_slp_maps, (1, corrected_slp_maps[0].shape[0], corrected_slp_maps[0].shape[1]))

    if timeit:
        min_elapsed = (time.time()-func_start_time)/60.
        print('Function time elapsed: {:.3f} min = {:.3f} hours'.format(min_elapsed, min_elapsed/60.))

    return dist_fig_maps, dist_slp_maps, fig_change_maps, slope_change_maps, \
            corrected_fig_maps, corrected_slp_maps, merits, volts_array

def convert_volts_array_to_voltMaps(volts_array, cells, max_voltage, smax):
    voltMaps = np.full((volts_array.shape[0], cc.cell_order_array.shape[0], cc.cell_order_array.shape[1]),
                        np.nan)
    for i in range(volts_array.shape[0]):
        for j in range(len(cells)):
            args = np.argwhere(cc.cell_order_array==cells[j].no)[0]
            y_arg, x_arg = args[0], args[1]
            voltMaps[i][args[0]][args[1]] = volts_array[i][j] * (max_voltage/smax)

    return voltMaps

def convert_voltMaps_to_volts_array(voltMaps, cells, max_voltage, smax):
    if voltMaps.ndim == 2:
        voltMaps = voltMaps.reshape(1, voltMaps.shape[0], voltMaps.shape[1])
    volts_arrays = np.full((voltMaps.shape[0], len(cells)), np.nan)
    for i in range(voltMaps.shape[0]):
        voltMap = voltMaps[i]
        for j in range(len(cells)):
            cell_no = cells[j].no
            args = np.argwhere(cc.cell_order_array==cell_no)[0]
            y_arg, x_arg = args[0], args[1]
            voltage = voltMap[y_arg][x_arg]
            volts_arrays[i][j] = voltage * (smax/max_voltage)

    return volts_arrays


def applyVoltsArray_to_IFstack(ifs, volts_array, dx=None, shademask=None, convertToAxialSlopes=False):
    """
    Computes a figure change using a set of IFs and and an array of voltage
    prescriptions.

    ifs: An array of shape N x M x L, where N indexes the IF number, and each IF's
    data is of shape (M, L). The IF data should be in microns.
    volts_array: A one dimensional array of size N, where each element of the
    array is indexed to match the IF number.
    dx: pixel spacing (mm/pixel)

    if shademask is specified, the resulting figure/slope change map will have shade
    stripped from it.

    if convertToAxialSlopes is True, the resulting figure change map will be converted
    to axial slope space

    Returns the figure or slope change when applying the volts_array to the ifs.
    """
    if ifs.shape[0] != volts_array.shape[0]:
        print('Error: The number of IFs provided does not match the number of voltage prescriptions.')
        return None
    else:
        scaled_IFs = np.copy(ifs)
        for i in range(ifs.shape[0]):
            scaled_IFs[i] = scaled_IFs[i] * volts_array[i]
        change_map = np.sum(scaled_IFs, axis=0)
        if shademask is not None:
            change_map = cuf.stripWithShade(change_map, shademask)
        if convertToAxialSlopes:
            change_map = cuf.convertToAxialSlopes(change_map*1e-3, dx)
    return change_map

#def correctForCTF(d,ifs,shade=None,dx=None,azweight=.015,smax=1.0,\
#                          bounds=None,avg_slope_remove = True):
#    """
#    Get distortion on same grid as IFs and run correction.
#    Rebin result onto original distortion grid and apply.
#    dx should be on IF grid size.
#    Needs update on doc string!
#    """
#    #Rebin to IF grid
#    d2 = man.newGridSize(d,np.shape(ifs[0]))
#
#    #Handle shademask
#    if shade is None:
#        shade = np.ones(np.shape(d2))
#
#    #Run correction
#    volt = slv.correctDistortion(d2,ifs,shade,dx=dx,azweight=azweight,\
#                                smax=smax,bounds=bounds,avg_slope_remove = avg_slope_remove)
#    # Compute the correction on the same scale as the original
#    # data. This correction will need to be added to the original
#    # data to yield the final corrected figure.
#    ifs2 = ifs.transpose(1,2,0)
#    cor2 = np.dot(ifs2,volt)
#    #Handle shademask
#    cor2[shade==0] = np.nan
#
#    return cor2,volt
