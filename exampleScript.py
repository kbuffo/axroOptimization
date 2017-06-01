import axroOptimization.evaluateMirrors as eva
import numpy as np
import astropy.io.fits as pyfits

#Read data
ifs = pyfits.getdata('/home/rallured/Dropbox/AXRO/InfluenceFunctions/220mmRoC/Conical/4inVlad/170531_4inVlad_IFs.fits')
distortion = pyfits.getdata('/home/rallured/GoogleDrive/AXROMetrology/PCO1S08/161108_PCO1S08_CleanedDistortionData.fits')

#Create shademask
shade = eva.slv.createShadePerimeter(np.shape(ifs[0]),axialFraction=.25,\
                                     azFraction=.25)

#Compute corrected mirror shape
cor,coeff = eva.correctXrayTestMirror(distortion,ifs,shade=shade,dx=[.15])

#You now have the corrected mirror shape in cor and the required
#coefficients in coeff
