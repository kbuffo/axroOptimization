import numpy as np
import matlab_wrapper
import pdb

# When installed, make sure this matches your path to axroOptimization package.
axro_opt_matlab_func = r"C:\Users\kbuffo\OneDrive - University of Iowa\Research\axroOptimization\Matlab_Scripts"
matlab = matlab_wrapper.MatlabSession()

def printer():
    print('Hello matlab functions!')

def fliperroo(python_array):
    '''
    A flip sometimes needed since MATLAB prefers to handle row vectors rather than column vectors.
    '''
    return np.ndarray.transpose(np.array([python_array]))

def matlab_fmincon_optimization(ifs,distortion,bounds,verbose = True):
    matlab = matlab_wrapper.MatlabSession()
    matlab.put('solver_path',axro_opt_matlab_func)
    matlab.eval('addpath(solver_path)')

    matlab.put('ifs',ifs)
    matlab.put('distortion',fliperroo(distortion))
    matlab.put('lb',fliperroo(np.asarray(bounds)[:,0]))
    matlab.put('ub',fliperroo(np.asarray(bounds)[:,1]))

    matlab.eval('[x,fval,exitflag] = matlab_run_fmincon(ifs,distortion,lb,ub)')

    optv = matlab.get('x')
    if verbose is True:
        print('Optimization achieved, Merit Function Value: ' + str(matlab.get('fval')))
    return optv

def matlab_lsqlin_optimization(ifs,distortion,bounds,verbose = True):
    #matlab = matlab_wrapper.MatlabSession()
    matlab.put('solver_path',axro_opt_matlab_func)
    matlab.eval('addpath(solver_path)')

    matlab.put('ifs',ifs)
    matlab.put('distortion',fliperroo(distortion))
    matlab.put('lb',fliperroo(np.asarray(bounds)[:,0]))
    matlab.put('ub',fliperroo(np.asarray(bounds)[:,1]))

    matlab.eval('[x,resnorm,residual,exitflag] = matlab_run_lsqlin(ifs,distortion,lb,ub)')

    optv = matlab.get('x')
    if verbose is True:
        print('Optimization achieved, Merit Function Value: ' + str(matlab.get('resnorm')))

    matlab.eval('clear')
    return optv
