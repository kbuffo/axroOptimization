import numpy as np
import pdb,os
from scipy.optimize import fmin_slsqp,least_squares
from scipy.fftpack import fft2, ifft2
#from axroOptimization.solver import ampMeritFunction, ampMeritDerivative
#from axroOptimization.correction_utility_functions import stripWithShade
#from axroOptimization.evaluateMirrors import applyVoltsArray_to_IFstack, computeMeritFunctions
import astropy.io.fits as pyfits
import scipy.interpolate as interp
import utilities.transformations as tr
import utilities.fourier as fourier

def nyquistOptimizer(distp, ifsp, shadep, smin=0., smax=5., bounds=None, matlab_opt=False, 
                     dx=None, azweight=0, correctionShape=None, v0=None, meritFunc='nyquistMeritFunction', 
                     nyquistFreq=0.1):
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

    shd_distp, shd_ifsp = stripShade_from_prepared_dist_and_ifs(distp, ifsp, shadep, azweight)
    #Handle bounds
    if bounds is None:
        bounds = []
        for i in range(np.shape(ifsp)[1]):
            bounds.append((smin,smax))
    
    default_acc = 1e-6
    default_epsilon = 1.4901161193847656e-08
    default_iter = 100

    if v0 is None:
        v0 = np.zeros(np.shape(shd_ifsp)[1])
    meritFunction, meritFunction_deriv = get_meritFunction(meritFunc)

    if matlab_opt is True:
        optv = matlab_lsqlin_optimization(shd_ifsp,shd_distp,bounds)
    else: #Call optimizer algorithm
         optv = fmin_slsqp(meritFunction, v0,\
                           bounds=bounds, args=(shd_distp,shd_ifsp, azweight, correctionShape, nyquistFreq),\
                           iprint=4,fprime=meritFunction_deriv,iter=1000,\
                           acc=1e-10, disp=True, epsilon=default_epsilon, full_output=0)
    return optv

def stripShade_from_prepared_dist_and_ifs(distp, ifsp, shadep, azweight):
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
    # covering the case with azimuthal weighting
    elif len(distp) == 2*len(shadep):
        ifsp = np.vstack((ifsp[:len(shadep)][shadep==1],ifsp[len(shadep):][shadep==1]))
        distp = np.concatenate((distp[:len(shadep)][shadep==1],distp[len(shadep):][shadep==1]))
    else:
        print('distp not an expected length relative to the shadep -- investigation needed.')
        pdb.set_trace()

    #Remove nans
    ind = ~np.isnan(distp)
    ifs = ifsp[ind]
    distortion = distp[ind]

    return distortion, ifs

def get_meritFunction(meritFunc):
    available_meritFuncs = ['RMS_directMeritFunction', 'nyquistMeritFunction', 'nyquistMeritFunction_reflectRows']
    if meritFunc == 'RMS_directMeritFunction':
        meritFunction = RMS_directMeritFunction
        meritDeriv = RMS_directMeritFunction_deriv
    elif meritFunc == 'nyquistMeritFunction':
        meritFunction = nyquistMeritFunction
        meritDeriv = None
    elif meritFunc == 'nyquistMeritFunction_reflectRows':
        meritFunction = nyquistMeritFunction_reflectRows
        meritDeriv = None
    else:
        print("Error: merit function '{}' is not defined. \
        Merit function options are {}".format(meritFunc, available_meritFuncs))
        meritFunction = None
        meritDeriv = None
    return meritFunction, meritDeriv

############################## DIRECT RMS MERIT FUNCTIONS ##############################

def RMS_directMeritFunction(voltages, distortion, ifuncs, azweight, correctionShape, nyquistFreq=0.1):
    """Simple merit function calculator.
    voltages is 1D array of shape (N,) weights for the N number of influence functions.
    distortion is 1D array of shape (2*j*k,) where (j, k) is the shape of distortion
    image after the shade has been stripped.
    ifuncs is 2D array of shape (2*j*k, N), where (j, k) is the shape of a single
    IF after the shade has been stripped, and N is the number of IFs
    shade is 2D array shade mask
    Simply compute sum(ifuncs*voltages-distortion)**2)
    """
    rms = np.mean((np.dot(ifuncs,voltages)-distortion)**2)
    return rms

def RMS_directMeritFunction_deriv(voltages, distortion, ifuncs, azweight, correctionShape, nyquistFreq=0.1):
    """Compute derivatives with respect to voltages of
    simple RMS()**2 merit function
    """
    deriv = np.dot(2*(np.dot(ifuncs,voltages)-distortion),ifuncs)/\
           np.size(distortion)
    return deriv

############################## FLATTENED ARRAY PSD NYQUIST MERIT FUNCTIONS ##############################

def nyquistMeritFunction(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.05):
    """
    BEST MERIT FUNCTION I'VE BEEN ABLE TO MAKE SO FAR. TAKES 5-6 MINUTES TO SOLVE A SINGLE CORRECTION.
    """
    if azweight > 0:
        div = 2
    else:
        div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    shd_distp_axial_flatten = shd_distp_axial.flatten(order='C')
    # get the PSD of the flattened axial slope map
    axial_f, axial_c = fourier.realPSD(shd_distp_axial_flatten, win=np.hanning)
    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
    if azweight > 0: # If the azweight is specified, compute the azimuthal slope rms
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        shd_distp_azimuth_flatten = shd_distp_azimuth.flatten(order='C')
        azimuth_f, azimuth_c = fourier.realPSD(shd_distp_azimuth_flatten, win=np.hanning)
        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    #print('axial rms: {}, azimuth rms: {:.3f}, rms: {:.3f}'.format(axial_rms, azimuth_rms, rms))
    return rms

def nyquistMeritFunction_reflectRows(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.05, 
                                     axis='columns'):
    div = 2
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    shd_distp_axial_flatten = flatten_and_reverse_every_other(shd_distp_axial, axis='columns')
    # get the PSD of the flattened axial slope map
    axial_f, axial_c = fourier.realPSD(shd_distp_axial_flatten, win=np.hanning)
    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
    if azweight > 0: # If the azweight is specified, compute the azimuthal slope rms
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        shd_distp_azimuth_flatten = flatten_and_reverse_every_other(shd_distp_azimuth, axis='rows')
        azimuth_f, azimuth_c = fourier.realPSD(shd_distp_azimuth_flatten, win=np.hanning)
        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    #print('axial rms: {}, azimuth rms: {:.3f}, rms: {:.3f}'.format(axial_rms, azimuth_rms, rms))
    return rms

def flatten_and_reverse_every_other(arr, axis='rows'):
    flattened_array = arr.flatten()

    if axis == 'rows':
        # Reverse values of every other row
        reversed_rows = [row[::-1] if i % 2 != 0 else row for i, row in enumerate(arr)]
        # Flatten the reversed rows and append to the flattened array
        reversed_flattened_array = np.array(reversed_rows).flatten()

    elif axis == 'columns':
        # Reverse values of every other column
        reversed_columns = [col[::-1] if i % 2 != 0 else col for i, col in enumerate(arr.T)]
        # Flatten the reversed columns and append to the flattened array
        reversed_flattened_array = np.array(reversed_columns).flatten()

    else:
        raise ValueError("Invalid 'axis' argument. Use 'rows' or 'columns'.")

    return reversed_flattened_array


def nyquistMeritFunctionColwise(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
    """
    TAKES 7-8 MINUTES TO SOLVE A SINGLE CORRECTION.
    voltages is 1D array of shape (N,) weights for the N number of influence functions.
    distortion is 1D array of shape (2*j*k,) where (j, k) is the shape of distortion
    image after the shade has been stripped.
    ifuncs is 2D array of shape (2*j*k, N), where (j, k) is the shape of a single
    IF after the shade has been stripped, and N is the number of IFs
    shade is 2D array shade mask
    Simply compute sum(ifuncs*voltages-distortion)**2)
    """
    if azweight > 0:
        div = 2
    else:
        div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    shd_distp_axial_flatten = shd_distp_axial.flatten(order='F')
    # get the PSDs of the 2D axial slope map
    axial_f, axial_c = fourier.realPSD(shd_distp_axial_flatten, win=np.hanning)
    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
    if azweight > 0: # If the azweight is specified, compute the azimuthal slope rms
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        shd_distp_azimuth_flatten = shd_distp_azimuth.flatten(order='F') # order='F' works best for both axial and azimuth
        azimuth_f, azimuth_c = fourier.realPSD(shd_distp_azimuth_flatten, win=np.hanning)
        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    # print('rms: {:.3f}'.fromat(rms))
    return rms

def residual_PV(voltages, distortion, ifuncs, azweight, correctionShape):
    """
    Find the peak-to-valley of the axial slope map residual and the (optionally) the azimuthal slope map residual
    """
    if azweight > 0:
        div = 2
    else:
        div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    axial_pv = np.nanmax(shd_distp_axial) - np.nanmin(shd_distp_axial)
    if azweight > 0:
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        azimuth_pv = np.nanmax(shd_distp_azimuth) - np.nanmin(shd_distp_azimuth)
    else:
        azimuth_pv = 0
    pv = np.sqrt(axial_pv**2+azimuth_pv**2)
    return pv

def nyquistMeritFunction_BiDir_PSD(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1, include_PV=False):
    """
    Take a flattened PSD both row-wise and column-wise and add in quadrature. If you include using residual_PV(),
    the time to solve 6 distortions is ~8 hours
    """
    rowwise_rms = nyquistMeritFunction(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=nyquist_freq)
    colwise_rms = nyquistMeritFunctionColwise(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=nyquist_freq)
    final_rms = np.sqrt(rowwise_rms**2+colwise_rms**2)
    if include_PV:
        pv = residual_PV(voltages, distortion, ifuncs, azweight, correctionShape)
    else:
        pv = 1
    return pv * final_rms

############################## MEAN SLOPE PSD NYQUIST MERIT FUNCTIONS ##############################

def nyquistMeritFunction_meanSlope(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1, axial_mean_axis=1, 
                                   azimuth_mean_axis=0):
    """
    Take the mean of the axial and (optionally) the azimuthal slope values and take a single PSD of each.
    axial_mean_axis is the direction to take the mean of the axial slope values: 0 for row-wise, 1 for column-wise
    azimuth_mean_axis is the direction to take the mean of the azimuthal slope values (if azweight is not 0): 0 for row-wise, 1 for column-wise
    """
    if azweight > 0:
        div = 2
    else:
        div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    mean_shd_distp_axial = np.nanmean(shd_distp_axial, axis=axial_mean_axis)
    axial_f, axial_c = fourier.realPSD(mean_shd_distp_axial, win=np.hanning)
    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[1], nyquist_freq, 1e-5)
    if azweight > 0:
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        mean_shd_distp_azimuth = np.nanmean(shd_distp_azimuth, axis=azimuth_mean_axis)
        azimuth_f, azimuth_c = fourier.realPSD(mean_shd_distp_azimuth, win=np.hanning)
        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[1], nyquist_freq, 1e-5)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    return rms

############################## 2D PSD NYQUIST MERIT FUNCTIONS ##############################

def nyquistMeritFunction_2DPSD(voltages, distortion, ifuncs, azweight, correctionShape, 
                               nyquist_freq=0.1):
    #if azweight > 0:
    #    div = 2
    #else:
    #    div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    div = 2
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    axial_fx, axial_fy, axial_c = fourier.real2DPSD(shd_distp_axial, win=np.hanning, rms_norm=True)
    #axial_rms, _, _, _ = fourier.computeFreqBand2D(axial_fx, axial_fy, axial_c, 
    #                                               axial_fx[0], axial_fx[-1], 
    #                                               axial_fy[0], nyquist_freq)
    axial_rms = fourier.computeFreqBand2D_meritFunc(axial_fx, axial_fy, axial_c, 
                                                    axial_fx[0], nyquist_freq, 
                                                    axial_fy[0], axial_fx[-1])
    if azweight > 0:
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        azimuth_fx, azimuth_fy, azimuth_c = fourier.real2DPSD(shd_distp_azimuth, win=np.hanning, rms_norm=True)
        #azimuth_rms, _, _, _ = fourier.computeFreqBand2D(azimuth_fx, azimuth_fy, azimuth_c, 
        #                                                 azimuth_fx[0], nyquist_freq, 
        #                                                 azimuth_fy[0], azimuth_fy[-1])
        azimuth_rms = fourier.computeFreqBand2D_meritFunc(azimuth_fx, azimuth_fy, azimuth_c, 
                                                          azimuth_fx[0], azimuth_fy[-1], 
                                                          azimuth_fy[0], nyquist_freq)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
    print('axial_rms: {}, azimuthal_rms: {}, rms: {}'.format(axial_rms, azimuth_rms, rms))
    return rms


############################## OTHER NYQUIST MERIT FUNCTIONS ##############################

def nyquistMeritFunctionLowPass(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
    corrected_map = np.dot(ifuncs,voltages)-distortion
    div = 2
    dx = 0.508
    axial_corr_dist = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    axial_corr_dist_lp = fourier.lowpass(axial_corr_dist, dx, nyquist_freq)
    axial_rms = np.mean(axial_corr_dist_lp**2)
    if azweight > 0:
        azimuth_corr_dist = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        azimuth_corr_dist_lp = fourier.lowpass(azimuth_corr_dist, dx, nyquist_freq)
        azimuth_rms = np.mean(azimuth_corr_dist_lp**2)
    else:
        azimuth_rms = 0
    rms = np.sqrt(axial_rms**2 + azimuth_rms**2)
    print('axial rms: {}, azimuth rms: {}, rms: {}'.format(axial_rms, azimuth_rms, rms))
    return rms

#def nyquistMeritFunctionLowPass_deriv2(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
#    res = np.dot(2*(np.dot(ifuncs,voltages)-distortion),ifuncs)/\
#           np.size(distortion)
#    #print('deriv:\n{}'.format(res))
#    return res

#def nyquistMeritFunctionLowPass_deriv(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
#    corrected_map = np.dot(ifuncs,voltages)-distortion
#    div = 2
#    lowpass_corr_dist = fourier.lowpass(corrected
#    deriv = np.sqrt(axial_deriv**2 + azimuth_deriv**2)
#    print('deriv:\n{}'.format(deriv))
#    return deriv


#def nyquistMeritFunctionLowPass_deriv(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
#    corrected_map = np.dot(ifuncs,voltages)-distortion
#    div = 2
#    #axial_ifs_2d = np.stack([ifuncs[:int(ifuncs.shape[0]/div), i].reshape(correctionShape) for i in range(ifuncs.shape[1])], axis=0)
#    shd_axial_corr_2d = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
#    #axial_ifs_1d = np.stack([axial_ifs_2d[i].flatten() for i in range(axial_ifs_2d.shape[0])], axis=1)
#    shd_axial_corr_1d = shd_axial_corr_2d.flatten()
#    #axial_deriv = np.dot(2*shd_axial_corr_1d, axial_ifs_1d)/shd_axial_corr_1d.size
#    #print('axial_ifs_2d: {}, shd_axial_corr_2d: {}, axial_ifs_1d: {}, shd_axial_corr_1d: {}'.format(axial_ifs_2d.shape, shd_axial_corr_2d.shape, 
#    #                                                                                                axial_ifs_1d.shape, shd_axial_corr_1d.shape))
#    if azweight > 0:
#        #azimuth_ifs_2d = np.stack([ifuncs[int(ifuncs.shape[0]/div):, i].reshape(correctionShape) for i in range(ifuncs.shape[1])], axis=0)
#        shd_azimuth_corr_2d = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
#        #azimuth_ifs_1d = np.stack([azimuth_ifs_2d[i].flatten() for i in range(azimuth_ifs_2d.shape[0])], axis=1)
#        shd_azimuth_corr_1d = shd_azimuth_corr_2d.flatten()
#        #azimuth_deriv = np.dot(2*shd_azimuth_corr_1d, azimuth_ifs_1d)/shd_azimuth_corr_1d.size
#    else:
#        azimuth_deriv = np.zeros(ifuncs.shape[1])
#    dist_1d = np.concatenate((shd_axial_corr_1d, shd_azimuth_corr_1d), axis=0)
    
#    deriv = nyquistMeritFunctionLowPass_deriv2(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1)#np.sqrt(axial_deriv**2 + azimuth_deriv**2)
#    return deriv

def nyquistMeritFunctionLong(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
    """
    TAKES ABOUT 20 HRS TO COMPUTE THE CORRECTIONS FOR THE 6 HFDFC3 DISTORTIONS AND DOES NOT DO BETTER
    THAN nyquistMeritFunction()
    voltages is 1D array of shape (N,) weights for the N number of influence functions.
    distortion is 1D array of shape (2*j*k,) where (j, k) is the shape of distortion
    image after the shade has been stripped.
    ifuncs is 2D array of shape (2*j*k, N), where (j, k) is the shape of a single
    IF after the shade has been stripped, and N is the number of IFs
    shade is 2D array shade mask
    Simply compute sum(ifuncs*voltages-distortion)**2)
    """
    if azweight > 0:
        div = 2
    else:
        div = 1
    corrected_map = np.dot(ifuncs,voltages)-distortion
    # get the axial slope values
    shd_distp_axial = np.reshape(corrected_map[:int(len(corrected_map)/div)], correctionShape)
    # get the PSDs of the 2D axial slope map
    axial_rms_vals = []
    for axial_col in shd_distp_axial.transpose():
        axial_f, axial_c = fourier.realPSD(axial_col, win=np.hanning)
        axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
        axial_rms_vals.append(axial_rms)
    weighted_axial_rms = np.sqrt(np.sum(np.array(axial_rms_vals)**2))
    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
    if azweight > 0: # If the azweight is specified, compute the azimuthal slope rms
        shd_distp_azimuth = np.reshape(corrected_map[int(len(corrected_map)/div):], correctionShape)
        azimuth_rms_vals = []
        for azimuth_col in shd_distp_azimuth.transpose():
            azimuth_f, azimuth_c = fourier.realPSD(azimuth_col, win=np.hanning)
            azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
            azimuth_rms_vals.append(azimuth_rms)
        weighted_azimuth_rms = np.sqrt(np.sum(np.array(azimuth_rms_vals)**2))
    else:
        weighted_azimuth_rms = 0
    rms = np.sqrt(weighted_axial_rms**2+weighted_azimuth_rms**2)
    # print('rms: {:.3f}'.fromat(rms))
    return rms


#def nyquistMeritFunction2(voltages, distortion, ifuncs, azweight, correctionShape, nyquist_freq=0.1):
#    """
#    voltages is 1D array of shape (N,) weights for the N number of influence functions.
#    distortion is 1D array of shape (2*j*k,) where (j, k) is the shape of distortion
#    image after the shade has been stripped.
#    ifuncs is 2D array of shape (2*j*k, N), where (j, k) is the shape of a single
#    IF after the shade has been stripped, and N is the number of IFs
#    shade is 2D array shade mask
#    Simply compute sum(ifuncs*voltages-distortion)**2)
#    """
#    # print()
#    # print('merit voltages shape: {}'.format(voltages.shape))
#    # print('merit distortion shape: {}'.format(distortion.shape))
#    # print('merit ifuncs shape: {}'.format(ifuncs.shape))
#    # apply voltages to IFs and compute the corrected map
#    if azweight > 0:
#        div = 2
#    else:
#        div = 1
#    corrected_map = np.dot(ifuncs,voltages)-distortion
#    # get the axial slope values
#    shd_distp_axial = corrected_map[:int(len(corrected_map)/div)]
#    # convert the axial and azimuthal slope arrays to 2D
#    array_len_2d = int(np.sqrt(len(shd_distp_axial)))
#    shd_distp_axial_2d = shd_distp_axial.reshape((array_len_2d, array_len_2d))
#    # get the PSDs of the 2D axial slope map
#    axial_f, axial_c = fourier.meanPSD(shd_distp_axial_2d, axis=0)
#    axial_rms = fourier.computeFreqBand(axial_f, axial_c, axial_f[0], nyquist_freq, 1e-5)
#    # repeat this process for the  azimuthal slope values (if any as indicated azweight)
#    if azweight > 0: # If the azweight is specified, compute the azimuthal slope rms
#        shd_distp_azimuth = corrected_map[int(len(corrected_map)/div):]
#        shd_distp_azimuth_2d = shd_distp_azimuth.reshape((array_len_2d, array_len_2d))
#        azimuth_f, azimuth_c = fourier.meanPSD(shd_distp_azimuth_2d, axis=0)
#        azimuth_rms = fourier.computeFreqBand(azimuth_f, azimuth_c, azimuth_f[0], nyquist_freq, 1e-5)
#    else:
#        azimuth_rms = 0
#    rms = np.sqrt(axial_rms**2+azimuth_rms**2)
#    # print('rms: {:.3f}'.fromat(rms))
#    return rms


#def stripShade_from_original_dist_and_ifs(orig_dist, orig_ifs, shademask):
#    if shademask is not None:
#        shd_orig_dist = stripWithShade(orig_dist, shademask)
#        shd_orig_ifs = np.stack([stripWithShade(orig_ifs[i], shademask) for i in range(orig_ifs.shape[0])], axis=0)
#    else:
#        shd_orig_dist = np.copy(orig_dist)
#        shd_orig_ifs = np.copy(orig_ifs)
#    return shd_orig_dist, shd_orig_ifs

#def hpdMeritFunction(voltages, distortion, ifuncs, dx):
#    fig_change_map = applyVoltsArray_to_IFstack(ifuncs, voltages)
#    residual_map = distortion - fig_change_map
#    dx_unpack = dx[0]
#    performance = computeMeritFunctions(residual_map, [dx_unpack])
#    hpd = performance[1]
#    print('voltages:', voltages)
#    print('residual hpd: {:.3f} arcsec'.format(hpd))
#    return hpd

# def nyquistMeritFunction3(voltages, distortion, ifuncs, div, nyquist_freq=0.1):
#     """
#     1.	Compute real psd on the flattened slope array and integrate that up to Nyquist freq
#     """
#     corrected_map = np.dot(ifuncs,voltages)-distortion
#     f, c = fourier.realPSD(corrected_map)
#     rms = fourier.computeFreqBand(f, c, f[0], nyquist_freq, 1e-5)
#     return rms

# def nyquistMeritFunction3(voltages, distortion, ifuncs, div, nyquist_freq=0.1, mirror_len=101.6):
#     """
#     2.	Compute the Fourier Transform using fft2
#     """
#     # apply voltages to IFs and compute the corrected map
#     corrected_map = np.dot(ifuncs,voltages)-distortion
#     # get the axial slope values
#     shd_distp_axial = corrected_map[:int(len(corrected_map)/div)]
#     # convert the axial and azimuthal slope arrays to 2D
#     array_len_2d = int(np.sqrt(len(shd_distp_axial)))
#     shd_distp_axial_2d = shd_distp_axial.reshape((array_len_2d, array_len_2d))
#     # compute the 2D FFT
#     axial_f = fft2(shd_distp_axial_2d)
#     cutoff = int(nyquist_freq/(mirror_len/shd_distp_axial_2d.shape[0]))
#     # set high frequencies to 0
#     axial_f[:cutoff, :] = 0
#     axial_f[:, :cutoff] = 0
#     axial_f[-cutoff:, :] = 0
#     axial_f[:, -cutoff:] = 0
#     # Perform Inverse Fourier Transform
#     axial_filter_map = ifft2(axial_f).real
#     # Calculate RMS of the filtered data
#     axial_rms = np.sqrt(np.mean(axial_filter_map**2))
#     if div == 2:
#         shd_distp_azimuth = corrected_map[int(len(corrected_map)/div):]
#         shd_distp_azimuth_2d = shd_distp_azimuth.reshape((array_len_2d, array_len_2d))
#         azimuth_f = fft2(shd_distp_azimuth_2d)
#         azimuth_f[:cutoff, :] = 0
#         azimuth_f[:, :cutoff] = 0
#         azimuth_f[-cutoff:, :] = 0
#         azimuth_f[:, -cutoff:] = 0
#         azimuth_filter_map = ifft2(azimuth_f).real
#         azimuth_rms = np.sqrt(np.mean(azimuth_filter_map**2))
#     else: 
#         azimuth_rms = 0
#     rms = np.sqrt(axial_rms**2+azimuth_rms**2)
#     return rms

# def nyquistMeritDerivative(voltages, distortion, ifuncs, div):
#     pass
