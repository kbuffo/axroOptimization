import numpy as np
import pdb,os
from scipy.optimize import fmin_slsqp,least_squares
import astropy.io.fits as pyfits
import scipy.interpolate as interp
import utilities.transformations as tr
import utilities.fourier as fourier

def optimizer(distortion, ifs, shade, smin=0., smax=5., bounds=None):
    """
    Cleaner implementation of optimizer. ifs and distortion should
    already be in whatever form (amplitude or slope) desired.
    IFs should have had prepareIFs already run on them.
    Units should be identical between the two.
    """

    #Remove shademask
    # Covering the case with no azimuthal weighting.
    if len(distortion) == len(shade):
        ifs = ifs[shade==1]
        distortion = distortion[shade==1]
    elif len(distortion) == 2*len(shade):
        ifs = np.vstack((ifs[:len(shade)][shade==1],ifs[len(shade):][shade==1]))
        distortion = np.concatenate((distortion[:len(shade)][shade==1],distortion[len(shade):][shade==1]))
    else:
        print('Distortion not an expected length relative to the shade -- investigation needed.')
        pdb.set_trace()

    #Remove nans
    ind = ~np.isnan(distortion)
    ifs = ifs[ind]
    distortion = distortion[ind]

    print('final ifs shape: {}'.format(ifs.shape))
    print('final distortion shape: {}'.format(distortion.shape))

    #Handle bounds
    if bounds is None:
        bounds = []
        for i in range(np.shape(ifs)[1]):
            bounds.append((smin,smax))

    #np.savetxt('ifs.txt',ifs)
    #np.savetxt('dist.txt',distortion)

    if matlab_opt is True:
        optv = matlab_lsqlin_optimization(ifs,distortion,bounds)

    #Call optimizer algorithm
    else:
        optv = fmin_slsqp(ampMeritFunction,np.zeros(np.shape(ifs)[1]),\
                          bounds=bounds,args=(distortion,ifs),\
                          iprint=2,fprime=ampMeritDerivative,iter=1000,\
                          acc=1.e-10, disp=False)
    return optv
