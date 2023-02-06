###############################################################################
###                                                                         ###
### 22/08/09.                                                               ### 
### Revision of the scatter module in Python.                               ### 
### Calculate half power diameter (HPD) and 68% encircled count fraction    ###
### (E68) merit functions for an X-ray mirror using PSF function derived    ###
### by Raimondi & Spiga (2015).                                             ###
###                                                                         ###
###############################################################################

### Import Statements ###
from numpy import sqrt, pi, exp, sin
import numpy as np

def primaryPSF(d,z,l,x0,wave,foc,R0,graze):
    """
    Calculate PSF using the equation derived by Raimondi & Spiga (2015).
    
    Parameters:
        d: input distortion
        z: position from the mirror (now a matrix)
        l: length of mirror
        x0: axial displacement (x0=0 => on axis)
        wave: wavelength of radiation
        foc: focal length 
        R0: radius of mirror at z=f
        graze: grazing incidence angle
    
    !!!IMPORTANT!!! 
    
    1) Must make sure that input distortion does NOT contain any NaNs! This should
    be handled by both eva.computeMeritFunctions and scattering.primary2DPSF.
    
    2) For this script to work, you must make your input z a matrix. To do this
    using axroOptimization functions, first, from numpy import matlib as npm
    then input: 
    z = np.arange(np.shape(img)[0])*dx*np.cos(graze)+Z0
    z2 = npm.repmat(z,np.shape(img)[1],1)
    z3 = np.flip(z2)
    into the primary2DPSF function of the scattering module and change the
    fucntion call to:
    psf = scatter.primaryPSF(distortion,z3-Z0,length,x0,wave,foc,R0,graze)
    """
    c = complex(0.,1.) # c for complex.  c = i
    lengthsum = sum(l)
    dz = abs(z[0][0]-z[0][1])
    psf = np.zeros(len(x0))
    x2,y2 = d.shape # So as not to confuse these x and y with the geometry of the equation.
    for i in range(len(x0)):
        integrand = 0.
        d2 = sqrt((d-x0[i])**2 + (z+foc)**2)
        integrand = np.sum(exp((-2 * c * pi)/wave * (d2-z)) * sqrt(d/d2),axis=1)
        psf[i] += sum(abs(integrand)**2*sin(graze)*dz**2/(wave*R0*lengthsum))
    return psf