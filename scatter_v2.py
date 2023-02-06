###############################################################################
###                                                                         ###
### 22/06/23.                                                               ###
### Revision of the scatter module in Python.                               ###
### Calculate half power diameter (HPD) and 68% encircled count fraction    ###
### (E68) merit functions for an X-ray mirror using PSF function derived    ###
### by Raimondi & Spiga (2015).                                             ###
###                                                                         ###
###############################################################################

### Import Statements ###
from numpy import sqrt, pi, exp, isnan, sin
import numpy as np

def printer():
    print('Hello, scatter_v2!')

### PSF Function ###
def primaryPSF(d,z,l,x0,wave,foc,R0,graze):
    """
    Calculate PSF using the equation derived by Raimondi & Spiga (2015).
    Parameters:
        d: distortion
        z: position from the mirror
        l: length of mirror
        x0: axial displacement
        wave: wavelength of radiation
        foc: focal length
        R0: radius of mirror at z=f
        graze: grazing incidence angle
    """
    c = complex(0.,1.) # c for complex.  c = i
    lengthsum = sum(l)
    # print('z v2:', z)
    dz = abs(z[1]-z[0])
    psf = np.zeros(len(x0))
    x2,y2 = d.shape # So as not to confuse these x and y with the geometry of the equation.
    for i in range(len(x0)):
        for m in range(x2):
            integrand = 0.
            for n in range(y2):
                if isnan(d[m,n]) == False:
                    d2 = sqrt((d[m,n]-x0[i])**2 + (z[n]+foc)**2)
                    integrand += exp((-2 * c * pi)/wave * (d2-z[n])) * sqrt(d[m,n]/d2)
            psf[i] += abs(integrand)**2*sin(graze)*dz**2/(wave*R0*lengthsum)
    return np.array(psf)
