import numpy as np
import pdb,os
from scipy.optimize import fmin_slsqp,least_squares
import astropy.io.fits as pyfits
import scipy.interpolate as interp
import utilities.transformations as tr
import utilities.fourier as fourier

def nyquistOptimizer(distp,ifsp,shadep,smin=0.,smax=5.,bounds=None,matlab_opt = False):
    """
    Cleaner implementation of optimizer. ifs and distortion should
    already be in whatever form (amplitude or slope) desired.
    IFs should have had prepareIFs already run on them.
    Units should be identical between the two.
    """
    #Load in data
    if type(distp)==str:
        distp = pyfits.getdata(distp)
    if type(ifsp)==str:
        ifs = pyfits.getdata(ifsp)
    if type(shadep)==str:
        shadep = pyfits.getdata(shadep)

    # remove the shade perimeter from the prepared distortion and IFs
    shd_distp, shd_ifsp, div = stripShade_from_prepared_dist_and_ifs(distp, ifsp, shadep)
    #Handle bounds
    if bounds is None:
        bounds = []
        for i in range(np.shape(ifsp)[1]):
            bounds.append((smin,smax))

    #np.savetxt('ifs.txt',ifs)
    #np.savetxt('dist.txt',distortion)

    if matlab_opt is True:
        optv = matlab_lsqlin_optimization(shd_ifsp,shd_distp,bounds)
    else: #Call optimizer algorithm
        optv = fmin_slsqp(nyquistMeritFunction,np.zeros(np.shape(shd_ifsp)[1]),\
                          bounds=bounds,args=(shd_distp,shd_ifsp,div),\
                          iprint=2,fprime=None,iter=1000,\
                          acc=1.e-10, disp=False)

    return optv

def stripShade_from_prepared_dist_and_ifs(distp, ifsp, shadep):
    """
    Removes the shademask from a prepared distortion and IFs.
    div is either 1 or 2, and allows us to construct the original
    distortion and IF arrays based on if we included azimuthal weighting.
    """
    #Remove shadepmask
    # Covering the case with no azimuthal weighting.
    if len(distp) == len(shadep):
        ifsp = ifsp[shadep==1]
        distp = distp[shadep==1]
        div = 1
    # covering the case with azimuthal weighting
    elif len(distp) == 2*len(shadep):
        ifsp = np.vstack((ifsp[:len(shadep)][shadep==1],ifsp[len(shadep):][shadep==1]))
        distp = np.concatenate((distp[:len(shadep)][shadep==1],distp[len(shadep):][shadep==1]))
        div = 2
    else:
        print('distp not an expected length relative to the shadep -- investigation needed.')
        pdb.set_trace()

    #Remove nans
    ind = ~np.isnan(distp)
    ifs = ifsp[ind]
    distortion = distp[ind]

    return distortion, ifs, div

def nyquistMeritFunction(voltages, distortion, ifuncs, div, nyquist_freq=0.1):
    """
    voltages is 1D array of shape (N,) weights for the N number of influence functions.
    distortion is 1D array of shape (2*j*k,) where (j, k) is the shape of distortion
    image after the shade has been stripped.
    ifuncs is 2D array of shape (2*j*k, N), where (j, k) is the shape of a single
    IF after the shade has been stripped, and N is the number of IFs
    shade is 2D array shade mask
    Simply compute sum(ifuncs*voltages-distortion)**2)
    """
    # print()
    # print('merit voltages shape: {}'.format(voltages.shape))
    # print('merit distortion shape: {}'.format(distortion.shape))
    # print('merit ifuncs shape: {}'.format(ifuncs.shape))
    # apply voltages to IFs and compute the corrected map
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = corrected_map[:int(len(corrected_map)/div)]
    # convert the axial and azimuthal slope arrays to 2D
    array_len_2d = int(np.sqrt(len(shd_distp_axial)))
    shd_distp_axial_2d = shd_distp_axial.reshape((array_len_2d, array_len_2d))
    # get the PSDs of the 2D axial slope map
    axial_f, axial_c = fourier.meanPSD(shd_distp_axial_2d, axis=0)
    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
    if div == 2: # If the azweight is specified, compute the azimuthal slope rms
        shd_distp_azimuth = corrected_map[int(len(corrected_map)/div):]
        shd_distp_azimuth_2d = shd_distp_azimuth.reshape((array_len_2d, array_len_2d))
        azimuth_f, azimuth_c = fourier.meanPSD(shd_distp_azimuth_2d, axis=0)
        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    # print('rms: {:.3f}'.fromat(rms))
    return rms

def nyquistMeritDerivative(voltages, distortion, ifuncs, div):
    pass
