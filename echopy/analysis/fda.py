#!/usr/bin/env python3
"""
Funtional Data Analysis on multifrequency acoustic data. 
Adapted from the R package 'fda.oce' (https://github.com/EPauthenet/fda.oce) 

Copyright (c) 2020 Echopy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__authors__ = ['Alejandro Ariza'   # wrote the code
               ]
__credits__ = ['Etienne Pauthenet' # supervised + provided original code               
               ]

import copy
import numpy as np
from skfda.representation import basis, FDataGrid
from skfda.misc.regularization import TikhonovRegularization
from skfda.misc.operators import LinearDifferentialOperator

def get_fdo(Sv, r, f, var, nb, order):
    """
    Creates a functional data object from Sv data.
    
    Args:
        Sv (float): 3D numpy array with Sv data (dB), with dimensions:
            - 1st: range
            - 2nd: time
            - 3st: frequency
        r (float): 1d numpy array with range data (m).
        f (float): 1d numpy array with frequency data (kHz).
        var (dict): Variables to create the functional data object, where:
            - keys (str): variable names
            - values (int): 3-integers tuple with the following information:
                - 1: frequency (kHz)
                - 2: upper range (m)
                - 3: lower range (m)
        nb (int): Number of basis for the functional data object.
        order (int): Order of basis functions.
        
    Returns:
        object: Functional data object.
        
    Notes:
        Number of elements and range extent must be equivalent among variables.
        
    Examples:    
        # Convert variables 38 kHz from 10 to 200 m, 120 kHz from 10 to 200 m, 
        # and 38 kHz from 200 to 390 m into a functional data objected defined
        # by 20 B-spline third-degree polinomials for every variable:
        var = {'038kHz_010-200m': ( 38,  10, 200),
               '120kHz_010-200m': (120,  10, 200),
               '038kHz_200-390m': ( 38, 200, 390)}
        fdo = get_fdobj(Sv, r, fre, var, 20, 3)
    """
    
    # check for proper inputs
    if not isinstance(Sv, np.ndarray):
        raise Exception('\'Sv\' must be a 3d-numpy array')
    if not isinstance(r, np.ndarray):
        raise Exception('\'r\' must be a 1d-numpy array')
    if not isinstance(f, np.ndarray):
        raise Exception('\'f\' must be a 1d-numpy array')
    if not isinstance(var, dict):
        raise Exception('\'var\' must be a dictionary')
    if not isinstance(nb, int):
        raise Exception('\'nb\' must be an integer')
    if not isinstance(order, int):
        raise Exception('\'order\' must be an integer')
    if Sv.ndim!=3:
        raise Exception('\'Sv\' array must have 3 dimensions')
    if r.ndim!=1:
        raise Exception('\'r\' array must have 1 dimension')
    if f.ndim!=1:
        raise Exception('\'f\' array must have 1 dimension')
    if Sv.shape[0]!=len(r):
        raise Exception('\'r\' and \'Sv\' 1st dimension lengths must be equal')
    if Sv.shape[-1]!=len(f):
        raise Exception('\'f\' and \'Sv\' 3rd dimension lenghts must be equal')  
    x0 = list(var.values())[0]
    for xi in var.values():
        if not isinstance(xi, tuple):
            raise Exception('variable values must contain only 3-elements tuples')
        if len(xi)!=3:
            raise Exception('variable values must contain only 3-elements tuples')
        if xi[0] not in f:
            raise Exception('%s kHz frequency is not available' % xi[0])
        if x0[2]-x0[1] != xi[2]-xi[1]:
            raise Exception('range extent of variables might be consistent') 
    
    # interate through variables
    for i, key in enumerate(var.keys()):
        
        # get frequency and range indexes to extract variables
        fi = var[key][0]
        r0 = var[key][1]
        r1 = var[key][2] 
        k  = np.where(f==fi)[0][0]
        i0 = np.argmin(abs(r-r0))
        i1 = np.argmin(abs(r-r1))
        
        # get Sv and range domain for every variable
        y  = Sv[i0:i1, :, k]
        x  = r [i0:i1      ] - r0
        x0 = x[ 0]
        x1 = x[-1]
                     
        # get functional data object for every variable and joind them in 
        # an unique functional data object
        bspline = basis.BSpline(domain_range=(x0, x1), n_basis=nb, order=order)
        fdo_i   = FDataGrid(y.T, sample_points=x, domain_range=(x0, x1))
        fdo_i   = fdo_i.to_basis(bspline)    
        if i==0:
            fdo              = copy.deepcopy(fdo_i)
            fdo.dim_names    = [key]
            fdo.domain_depth = [(r0, r1)]
        else:
            fdo.domain_depth.append((r0, r1))
            fdo.dim_names.append(key)
            fdo.coefficients = np.dstack((fdo.coefficients,fdo_i.coefficients))
            
    return fdo

def get_fpca(fdo):
    """
    Performs functional PCA on funtional data object and returns results.
    
    Args:
        fdo (object): Functional data object.
        
    Returns:
        dict: Functional PCA results, includes the following keys:
            - 'C'       : Centered coefficients matrix 'C'
            - 'Cm'      : Mean coefficients 'Cm' 
            - 'inertia' : Inertia
            - 'W'       : Function-to-discrete metric equivalence matrix 'W'
            - 'M'       : Weighting matrix 'M'
            - 'values'  : PCs values
            - 'pval'    : Percentage of variance of PCs.
            - 'vecnotWM': PCs vectors
            - 'vectors' : PCs weighted vectors
            - 'axes'    : ? #TODO Etienne?
            - 'pc'      : PCs projected on the modes
    """
    
    # get number of samples, basis, and dimensions
    nsam = fdo.n_samples
    nbas = fdo.n_basis
    ndim = len(fdo.dim_names)
    
    # if 3d-array coefficients, convert to a 2d-array (n_samples, n_basis)
    if ndim>1:
        C = np.zeros((nsam, nbas*ndim))
        for k in range(ndim):
            j0          = nbas *  k
            j1          = nbas * (k+1)
            C[:, j0:j1] = fdo.coefficients[:, :, k]
    else:
        C = fdo.coefficients
    
    # compute centered coefficients matrix by subtractig the mean
    Cm = np.mean(C, axis=0)
    Cc = C - Cm[np.newaxis,:]
    
    # get basis penalty matrix
    regularization = TikhonovRegularization(LinearDifferentialOperator(0))
    penalty        = regularization.penalty_matrix(fdo.basis)
    
    # compute crossed-covariance matrix of C and Inertia
    inertia = np.zeros(ndim)
    for k in range(ndim):
        j0         = nbas *  k
        j1         = nbas * (k+1)
        V          = Cc[:, j0:j1].T @ Cc[:, j0:j1] @ penalty / nsam
        inertia[k] = np.trace( V )
        
    # compute weighting matrix 'M' to balance variables of different units
    M       = np.zeros((ndim*nbas,ndim*nbas))
    Mdeminv = M.copy()
    W       = M.copy()
    aux     = np.diag(np.ones(nbas))
    for k in range(ndim):
        i0, j0                = nbas *  k   , nbas *  k
        i1, j1                = nbas * (k+1), nbas * (k+1)    
        M      [i0:i1, j0:j1] = aux/inertia[k]
        Mdeminv[i0:i1, j0:j1] = aux*np.sqrt(inertia[k])
        W      [i0:i1, j0:j1] = penalty
    Mdem = np.sqrt(M);
    
    # compute function-to-discrete metric equivalence matrix 'W
    W       = (W+W.T)/2.
    Wdem    = np.linalg.cholesky(W).T
    Wdeminv = np.linalg.inv(Wdem)
    
    # compute crossed-covariance matrix 'V'
    V = Mdem @ Wdem @ Cc.T @ Cc @ Wdem.T @ Mdem / nsam
    
    # compute eigenvalues and eigenvectors
    pca_values, pca_vectors = np.linalg.eig(V)
    idx                     = pca_values.argsort()[::-1]
    pca_values              = pca_values[idx]
    pca_vectors             = pca_vectors[:,idx]
    pca_vectors_notWM       = pca_vectors
    pca_vectors             = Mdeminv @ Wdeminv @ pca_vectors
    
    # compute principal components projected on the modes
    pc = Cc @ W.T @ M @ pca_vectors
    
    # build PCA dictionary
    fpca              = {}
    fpca['C'        ] = C
    fpca['Cm'       ] = Cm
    fpca['inertia'  ] = inertia
    fpca['W'        ] = W
    fpca['M'        ] = M
    fpca['values'   ] = pca_values
    fpca['pval'     ] = 100 * pca_values.real / np.sum(pca_values.real)
    fpca['vecnotWM' ] = pca_vectors_notWM
    fpca['vectors'  ] = pca_vectors
    fpca['axes'     ] = pca_vectors * np.sqrt(pca_values)[:,np.newaxis].T
    fpca['pc'       ] = pc.real
    
    # Warn if eigen values are not orthogonal
    v1 = fpca['vectors'][:,0].T @ W @ M @ fpca['vectors'][:,0] - 1.
    v2 = fpca['vectors'][:,0].T @ W @ M @ fpca['vectors'][:,1]
    if v1>1.e-10 or v2>1.e-10:
        print('Warning : Eigen values not orthogonal (%s, %s)' % (v1, v2))
        
    return fpca

def get_fvar(fdo, fpca, pc, res):
    """
    Provides information about the along-depth Sv variance contained in a given
    principal component. Results are given for each variable defined in the 
    functional data object.
    
    Args:
        fdo (object): Functional data object.
        fpca (dict): Results from functional PCA.
        pc (int): Principal component to be evaluated.
        res (float): depth resolution of results (m).
        
    Returns:
        dict: along-depth Sv variance results, including the following keys:
            - 'x'     : 1d- array with range dimension (m)
            - 'y'     : 2d-array with Sv profiles (dB)
            - 'ymean' : 1d-array with Sv mean profile (dB)
            - 'yplus' : 1d-array with effect of adding PC variance (dB)
            - 'yminus': 1d-array with effect of subtracting PC variance (dB)
            - '%dim'  : total variance explained by each variable (%) #TODO: consider new name
            - '%pc'   : total variance explained by the PC (%).       
    """
    
    # load variables
    nbas = fdo.n_basis
    ndim = len(fdo.dim_names)
    C    = fpca['C'   ]
    Cm   = fpca['Cm'  ]
    ppc  = fpca['pval'][pc]
    
    # iterate through dimensions
    data = {}
    for k in range(ndim):
        i0 =  k    * nbas
        i1 = (k+1) * nbas
        
        # compute percentage of variance within each dimension
        pdim = np.round(100*np.sum(fpca['vecnotWM'][i0:i1, pc]**2))
        
        # load C mean matrix and it's perturbation from the chosen PC 
        Call  = C           [:,i0:i1]
        Cmean = Cm          [i0:i1    ][np.newaxis,:]
        Cplus = fpca['axes'][i0:i1, pc][np.newaxis,:]
        
        # create new functional data objects by adding and subtracting the PC 
        # effect on the mean coefficients
        fdo_k = copy.deepcopy(fdo)
        if ndim>1:
            fdo_k.coefficients   =  fdo_k.coefficients[:, :, k]
            fdo_k.dim_names      = [fdo_k.dim_names[k]]  
        fdobj_all                =  fdo_k.copy()
        fdobj_mean               =  fdo_k.copy()
        fdobj_plus               =  fdo_k.copy()
        fdobj_minus              =  fdo_k.copy()
        fdobj_all.coefficients   =  Call
        fdobj_mean.coefficients  =  Cmean
        fdobj_plus.coefficients  =  Cmean + Cplus
        fdobj_minus.coefficients =  Cmean - Cplus
        
        # get x positions at which y function will evaluated
        xrange = fdo_k.domain_range[0]
        x      = np.arange(xrange[0], xrange[1], res)
                 
        # evaluate 'y' mean and plus/minus perturbation 
        y      = fdobj_all.evaluate(x).T
        ymean  = fdobj_mean.evaluate (x).ravel()
        yplus  = fdobj_plus.evaluate (x).ravel().real
        yminus = fdobj_minus.evaluate(x).ravel().real
        x      = x + fdo_k.domain_depth[k][0]
        
        # store data
        data.update({fdo_k.dim_names[0]: 
                     {'x'     : x      , 
                      'y'     : y      , 
                      'ymean' : ymean  , 
                      'yplus' : yplus  , 
                      'yminus': yminus , 
                      '%dim'  : pdim   , 
                      '%pc'   : ppc    }})
    
    # return
    return data
        
def get_fmean(fdo, res):
    """
    Computes mean profiles for every variable defined within the Functional 
    data object.
    
    Args:
        fdo (object): Functional data object
        res (float): depth resolution of mean profile (m)
        
    Returns:
        dict: Mean profile results, including the following keys:
            - 'x'     : 1d- array with range dimension (m)
            - 'y'     : 2d-array with Sv profiles (dB)
            - 'ymean' : 1d-array with Sv mean profile (dB)
            - 'yplus' : 1d-array with effect of adding PC variance (dB)
            - 'yminus': 1d-array with effect of subtracting PC variance (dB)
    """ 
    
    # iterate through dimension variables
    data = {}
    ndim   = len(fdo.dim_names)
    for k in range(ndim):
        
        # create single-dimension fdobjs
        fdo_k = copy.deepcopy(fdo)
        if ndim>1:
            fdo_k.coefficients =  fdo_k.coefficients[:, :, k]
            fdo_k.dim_names    = [fdo_k.dim_names[k]]        
    
        # evaluate y at x positions in every single-dimension fdobj
        xrange = fdo_k.domain_range[0]
        x      = np.arange(xrange[0], xrange[1] + res, res)
        y      = fdo_k.evaluate(x)[:,:, 0].T
        x      = x + fdo_k.domain_depth[k][0]
        
        # compute 'y' mean and standard deviations
        ymean  = np.mean(y, axis=1)
        ystd   = np.std (y, axis=1)
        yplus  = ymean + ystd
        yminus = ymean - (yplus-ymean) 
        
        # store data
        data.update({fdo_k.dim_names[0]: 
                     {'x'     : x     , 
                      'y'     : y     , 
                      'ymean' : ymean , 
                      'yplus' : yplus , 
                      'yminus': yminus}})       
   
    # return data
    return data

def reduce_fdo(fdo, fpca, pcs=(1,2,3), profiles=None):
    """
    Reduce functional data object to certain principal components (PCs) and
    profiles.
    
    Args:
        fdo (object): Functional data object.
        fpca (dict): Functional PCA results.
        pcs (tuple): Sequence with the numbers of PCs considered.
        profiles (tuple): Sequence with numbers of profiles considered.
        
    Returns:
        object: New functional data object.
        
    Notes:
        Default settings will select the 3 first PCs and all profiles in the
        functional data object. Set 'pcs' as 'None' and profiles as a tuple
        if you prefer to keep all PCs and select profiles instead. PCs and 
        and profiles can be selected at the same time.
    """
    
    # get number of basis, samples, and dimensions
    nbas = fdo.n_basis
    nsam = fdo.n_samples
    ndim = len(fdo.dim_names)
    
    # create new funtional data object 
    newfdo = copy.deepcopy(fdo)
    
    # select PCs
    if pcs:
        
        # compute new coefficient matrix with selected principal components
        if ndim>1:
            coef = np.zeros((nsam, nbas, ndim))
            for k in range(ndim):
                i0            =  k    * nbas
                i1            = (k+1) * nbas
                cm            = np.tile(fpca['Cm'][i0:i1], (nsam, 1))
                v             = fpca['vectors'][i0:i1, pcs].real
                pc            = fpca['pc'][:, pcs]
                coef[:, :, k] = cm + (v @ pc.T).T
        else:
            cm   = np.tile(fpca['Cm'], (nsam, 1))
            v    = fpca['vectors'][:, pcs]
            pc   = fpca['pc'][:, pcs]
            coef = cm + (v @ pc.T).T
        
        # return functional data object with new coefficients    
        newfdo.coefficients = coef
    
    # select profiles
    if profiles is not None:
        newfdo.coefficients = newfdo.coefficients[profiles, :]
    
    return newfdo