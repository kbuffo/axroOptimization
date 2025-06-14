import numpy as np
import pdb,os
from scipy.optimize import fmin_slsqp,least_squares
from axroOptimization.nyquist_solver import nyquistOptimizer
import astropy.io.fits as pyfits
import scipy.interpolate as interp
import utilities.transformations as tr
import utilities.fourier as fourier

#from axroOptimization.matlab_funcs import matlab_lsqlin_optimization
def printer():
    print('Hello solver!')

def ampMeritFunction(voltages,distortion,ifuncs):
    """Simple merit function calculator.
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
    res = np.mean((np.dot(ifuncs,voltages)-distortion)**2)
    # print('res: {:.3f}'.format(res))
    return res

def ampMeritFunction2(voltages,**kwargs):
    """Simple merit function calculator.
    voltages is 1D array of weights for the influence functions
    distortion is 2D array of distortion map
    ifuncs is 4D array of influence functions
    shade is 2D array shade mask
    Simply compute sum(ifuncs*voltages-distortion)**2)
    """
    #Numpy way
    distortion = kwargs['inp'][0]
    ifuncs = kwargs['inp'][1]
    res = np.mean((np.dot(ifuncs,voltages)-distortion)**2)
    return res, [], 0

def ampMeritDerivative(voltages,distortion,ifuncs):
    """Compute derivatives with respect to voltages of
    simple RMS()**2 merit function
    """
    res = np.dot(2*(np.dot(ifuncs,voltages)-distortion),ifuncs)/\
           np.size(distortion)
    #print('derivative res shape: {}'.format(res.shape))
    return res

def ampMeritDerivative2(voltages,f,g,**kwargs):
    """Compute derivatives with respect to voltages of
    simple RMS()**2 merit function
    """
    distortion = kwargs['inp'][0]
    ifuncs = kwargs['inp'][1]
    res = np.dot(2*(np.dot(ifuncs,voltages)-distortion),ifuncs)/\
           np.size(distortion)
    return res.tolist(), [], 0

def rawOptimizer(ifs,dist,bounds=None,smin=0.,smax=5.):
    """Assumes ifs and dist are both in slope or amplitude space.
    No conversion to slope will occur."""
    #Create bounds list
    if bounds is None:
        bounds = []
        for i in range(np.shape(ifs)[0]):
            bounds.append((smin,smax))

    #Get ifs in right format
    ifs = ifs.transpose(1,2,0) #Last index is cell number

    #Reshape ifs and distortion
    sh = np.shape(ifs)
    ifsFlat = ifs.reshape(sh[0]*sh[1],sh[2])
    distFlat = dist.flatten()

    #Call optimizer algoritim
    optv = fmin_slsqp(ampMeritFunction,np.zeros(sh[2]),\
                      bounds=bounds,args=(distFlat,ifsFlat),\
                      iprint=2,fprime=ampMeritDerivative,iter=200,\
                      acc=1.e-10)

    #Reconstruct solution
    sol = np.dot(ifs,optv)

    return sol,optv

#def prepareIFs(ifs,dx=None,azweight=.015):
#    """
#    Put IF arrays in format required by optimizer.
#    If dx is not None, apply derivative.
#    """
#    #Apply derivative if necessary
#    #First element of result is axial derivative
#    if dx is not None:
#        """
#        line below returns 4D array of shape (2, N, j, k), where the the first
#        dimension indicates axial [0] or azimuthal [1] slope, N is the number
#        of IFs, and (j, k) is the shape of each IF
#        """
#        ifs = np.array(np.gradient(ifs,*dx,axis=(1,2)))*180/np.pi*60.**2 / 1000.
#        ifs[1] = ifs[1]*azweight # decreases the impact of the azimuthal slope
#        ifs = ifs.transpose(1,0,2,3) # changes 4D array shape to be (N, 2, j, k)
#        sha = np.shape(ifs)
#        for i in range(sha[0]): # subtracts the mean value from each IF
#            for j in range(sha[1]):
#                ifs[i,j] = ifs[i,j] - np.nanmean(ifs[i,j])
#        ifs = ifs.reshape((sha[0],sha[1]*sha[2]*sha[3])) # changes IF array shape to be (N, 2*j*k)
#    else:
#        #ifs = ifs.transpose(1,2,0)
#        sha = np.shape(ifs)
#        for i in range(sha[0]):
#            ifs[i] = ifs[i] - np.nanmean(ifs[i])
#        ifs = ifs.reshape((sha[0],sha[1]*sha[2]))
#    # changes IF array shape to be (2*j*k, N)
#    # Each column is a row-connected axial slope connected with row-connected
#    # azimuthal slope for a single IF.
#    return np.transpose(ifs)

def prepareIFs(ifs,dx=None,azweight=.015,slp_ifs=[None,None]):
    """
    Put IF arrays in format required by optimizer.
    If dx is not None, apply derivative.
    """
    #Apply derivative if necessary
    #First element of result is axial derivative
    if dx is not None:
        """
        line below returns 4D array of shape (2, N, j, k), where the the first
        dimension indicates axial [0] or azimuthal [1] slope, N is the number
        of IFs, and (j, k) is the shape of each IF
        """
        if slp_ifs[0] is None and slp_ifs[1] is None:
            ifs = np.array(np.gradient(ifs,*dx,axis=(1,2)))*180/np.pi*60.**2 / 1000.
        else:
            ifs = np.stack(slp_ifs, axis=0)
            print('stacked ifs shape:', ifs.shape)
        ifs[1] = ifs[1]*azweight # decreases the impact of the azimuthal slope
        ifs = ifs.transpose(1,0,2,3) # changes 4D array shape to be (N, 2, j, k)
        sha = np.shape(ifs)
        for i in range(sha[0]): # subtracts the mean value from each IF
            for j in range(sha[1]):
                ifs[i,j] = ifs[i,j] - np.nanmean(ifs[i,j])
        ifs = ifs.reshape((sha[0],sha[1]*sha[2]*sha[3])) # changes IF array shape to be (N, 2*j*k)
    else:
        #ifs = ifs.transpose(1,2,0)
        sha = np.shape(ifs)
        for i in range(sha[0]):
            ifs[i] = ifs[i] - np.nanmean(ifs[i])
        ifs = ifs.reshape((sha[0],sha[1]*sha[2]))
    #print('ifs shape before final transpose:', ifs.shape)
    # changes IF array shape to be (2*j*k, N)
    # Each column is a row-connected axial slope connected with row-connected
    # azimuthal slope for a single IF.
    return np.transpose(ifs)

def prepareDist(d,dx=None,azweight=0.015,avg_slope_remove = True):
    """
    Put distortion array in format required by optimizer.
    If dx is not None, apply derivative.
    Can also be run on shademasks
    """
    #Apply derivative if necessary
    #First element of result is axial derivative
    if dx is not None:
        """
        line below returns 3D array of shape (2, j, k), where the the first
        dimension indicates axial [0] or azimuthal [1] slope, and (j, k) is the
        shape of the distortion
        """
        d = np.array(np.gradient(d,*dx))*180/np.pi*60.**2 / 1000.
        d[0] = d[0] - np.nanmean(d[0])*avg_slope_remove # subtract out mean axial slope
        d[1] = d[1] - np.nanmean(d[1])*avg_slope_remove # subtract out mean azimuth slope
        d[1] = d[1]*azweight # decreases the impact of the azimuthal slope

    # return row-connected axial slope connected with row-connected azimuthal slope
    # this is a 1D array of shape (2*j*k,)
    return d.flatten()

def optimizer(distortion,ifs,shade,smin=0.,smax=5.,bounds=None,matlab_opt=False, 
              azweight=None, correctionShape=None, v0=None, dx=None, slp_ifs=[None, None]):
    """
    Cleaner implementation of optimizer. ifs and distortion should
    already be in whatever form (amplitude or slope) desired.
    IFs should have had prepareIFs already run on them.
    Units should be identical between the two.
    """
    #Load in data
    if type(distortion)==str:
        distortion = pyfits.getdata(distortion)
    if type(ifs)==str:
        ifs = pyfits.getdata(ifs)
    if type(shade)==str:
        shade = pyfits.getdata(shade)

    #Remove shademask
    # Covering the case with no azimuthal weighting.
    if len(distortion) == len(shade):
        ifs = ifs[shade==1]
        distortion = distortion[shade==1]
        div = 1
    # covering the case with azimuthal weighting
    elif len(distortion) == 2*len(shade):
        ifs = np.vstack((ifs[:len(shade)][shade==1],ifs[len(shade):][shade==1]))
        distortion = np.concatenate((distortion[:len(shade)][shade==1],distortion[len(shade):][shade==1]))
        div = 2
    else:
        print('Distortion not an expected length relative to the shade -- investigation needed.')
        pdb.set_trace()

    #Remove nans
    ind = ~np.isnan(distortion)
    ifs = ifs[ind]
    distortion = distortion[ind]

    #Handle bounds
    if bounds is None:
        bounds = []
        for i in range(np.shape(ifs)[1]):
            bounds.append((smin,smax))

    if v0 is None:
        v0 = np.zeros(np.shape(ifs)[1])

    #np.savetxt('ifs.txt',ifs)
    #np.savetxt('dist.txt',distortion)

    if matlab_opt is True:
        optv = matlab_lsqlin_optimization(ifs,distortion,bounds)

    #Call optimizer algorithm
    else:
        optv = fmin_slsqp(ampMeritFunction,v0,\
                          bounds=bounds,args=(distortion,ifs),\
                          iprint=2,fprime=ampMeritDerivative,iter=1000,\
                          acc=1.e-10, disp=False)
    return optv

def correctDistortion(dist, ifs, shade, dx=None, azweight=.015, smax=5.,\
                      bounds=None, avg_slope_remove=True, matlab_opt=False, 
                      optimizer_function=nyquistOptimizer, meritFunc='nyquistMeritFunction',
                      correctionShape=None, v0=None, slp_ifs=[None,None], nyquistFreq=0.1):
    """
    Wrapper function to apply and evaluate a correction
    on distortion data.
    Distortion and IFs are assumed to already be on the
    same grid size.
    dx should be in mm, dist and ifs should be in microns
    """
    #Make sure shapes are correct
    if not (np.shape(dist)==np.shape(ifs[0])==np.shape(shade)):
        print('Unequal shapes!')
        return None

    #Prepare arrays
    distp = prepareDist(dist,dx=dx,azweight=azweight,avg_slope_remove = avg_slope_remove)
    ifsp = prepareIFs(ifs,dx=dx,azweight=azweight, slp_ifs=slp_ifs)
    shadep = prepareDist(shade)

    #Run optimizer
    res = optimizer_function(-distp,ifsp,shadep,smax=smax,bounds=bounds,matlab_opt=matlab_opt, 
                             dx=dx, azweight=azweight, correctionShape=correctionShape, v0=v0, 
                             meritFunc=meritFunc, nyquistFreq=nyquistFreq)

    return res

def convertFEAInfluence(filename,Nx,Ny,method='cubic',\
                        cylcoords=True):
    """Read in Vanessa's CSV file for AXRO mirror
    Mirror no longer assumed to be cylinder.
    Need to regrid initial and perturbed nodes onto regular grid,
    then compute radial difference.
    """
    #Load FEA data
    d = np.transpose(np.genfromtxt(filename,skip_header=1,delimiter=','))

    if cylcoords is True:
        r0 = d[1]*1e3
        rm = np.mean(r0)
        t0 = d[2]*np.pi/180. * rm #Convert to arc length in mm
        z0 = d[3]*1e3
        #r0 = np.repeat(220.497,len(t0))
        r = r0 + d[4]*1e3
        t = (d[2] + d[5])*np.pi/180. * rm #Convert to arc length in mm
        z = z0 + d[6]*1e3
    else:
        x0 = d[2]*1e3
        y0 = d[3]*1e3
        z0 = d[4]*1e3
        x = x0 + d[5]*1e3
        y = y0 + d[6]*1e3
        z = z0 + d[7]*1e3
        #Convert to cylindrical
        t0 = np.arctan2(x0,-z0)*220.497 #Convert to arc length in mm
        r0 = np.sqrt(x0**2+z0**2)
        z0 = y0
        t = np.arctan2(x,-z)*220.497
        r = np.sqrt(x**2+z**2)
        z = y

    #Construct regular grid
    gy = np.linspace(z0.min(),z0.max(),Nx+2)
    gx = np.linspace(t0.min(),t0.max(),Ny+2)
    gx,gy = np.meshgrid(gx,gy)

    #Run interpolation
    g0 = interp.griddata((z0,t0),r0,(gy,gx),method=method)
    g0[np.isnan(g0)] = 0.
    g = interp.griddata((z,t),r,(gy,gx),method=method)
    g[np.isnan(g)] = 0.

    print(filename + ' done')

    return -(g0[1:-1,1:-1]-g[1:-1,1:-1]),g0[1:-1,1:-1],g[1:-1,1:-1]

def createShadePerimeter(sh,axialFraction=0.,azFraction=0.):
    """
    Create a shademask where a fraction of the axial and
    azimuthal perimeter is blocked.
    Fraction is the fraction of blockage in each axis.
    sh is shape tuple e.g. (200,200)

    First create an array of zeros with the same shape as the
    one supplied.
    Take the number of pixels in a given dimension, multiply
    by how many the fraction supplied to get how many pixels
    will remain in the image.
    The number of pixels remaining is divided by 2 for ease of
    indexing.
    Round the number of pixels remaining to the nearest integer,
    and fill the zero array with 1's in the positions where pixels
    are kept.
    """
    arr = np.zeros(sh)
    axIndex = int(round(sh[0]*axialFraction/2))
    azIndex = int(round(sh[1]*azFraction/2))
    arr[axIndex:-axIndex,azIndex:-azIndex] = 1.
    return arr
