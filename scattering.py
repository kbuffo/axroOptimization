import numpy as np
import matplotlib.pyplot as plt
import pdb
from axroOptimization.conicsolve import primrad,primfocus,woltparam
import axroOptimization.scatter_v3 as scatter
import utilities.imaging.man as man
from numpy import matlib as npm
# import axroOptimization.scatter as scatter

def printer():
    print('Hello scattering!')

def computeHEW(x0,psf):
    """
    Compute the HEW from a PSF via the CDF method
    """
    dx2 = np.diff(x0)[0]
    cdf = np.cumsum(psf)*dx2
    hpdPSF = x0[np.argmin(np.abs(cdf-.75))]-\
             x0[np.argmin(np.abs(cdf-.25))]
    return hpdPSF

def primary2DPSF(img,dx,R0=220.,Z0=8400.,x0=np.linspace(-2.,2.,1001),\
                 wave=1.24e-6):
    """
    Create height vector and radius img based on
    radial distortion input data.
    Then pass to a F2PY scattering function to
    compute PSF over observation points.
    wavelength in mm.
    """
    #Remove NaNs if they exist
    #Create height vector
    graze = woltparam(R0,Z0)[0]
    foc = primfocus(R0,Z0)
    # z = np.arange(np.shape(img)[0])*dx*np.cos(graze)+Z0
    z = np.arange(np.shape(img)[0])*dx*np.cos(graze)+Z0 # Jeff's additions
    z2 = npm.repmat(z,np.shape(img)[1],1)
    z3 = np.flip(z2)
    #Create radial position img
    rad = primrad(z,R0,Z0)
    rad2 = np.flipud(np.transpose(np.tile(rad,(np.shape(img)[1],1))))
    distortion = np.transpose(rad2-img/1e3)
    z = z[::-1]
    #Compute length for each slice
    length = np.array([(np.sum(~np.isnan(li))-1)*dx \
                       for li in distortion],order='F')
    DR = length*np.sin(graze)
    #Integrate each slice in Fortran
    # psf = scatter.primaryPSF(distortion,z-Z0,length,x0,wave,foc,R0,graze)
    psf = scatter.primaryPSF(distortion,z3-Z0,length,x0,wave,foc,R0,graze)
    return psf
