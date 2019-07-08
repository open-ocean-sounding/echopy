#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common unit transformations in fisheries acoustics.

Created on Tue Jun 25 17:37:06 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

import numpy as np

def lin(variable):
    """
    Turn variable into the linear domain.     
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float:array of elements transformed
    """
    
    lin    = 10**(variable/10)
    return   lin  

def log(variable):
    """
    Turn variable into the logarithmic domain. This function will return -999
    in the case of values less or equal to zero (undefined logarithm). -999 is
    the convention for empty water or vacant sample in fisheries acoustics. 
    
    Args:
        variable (float): array of elements to be transformed.    
    
    Returns:
        float: array of elements transformed
    """    
    if not isinstance(variable, (np.ndarray)):
        variable = np.array([variable])
        
    if isinstance(variable, int):
        variable = np.float64(variable)
        
    mask           = np.ma.masked_less_equal(variable, 0).mask
    variable[mask] = np.nan
    log            = 10*np.log10(variable)
    log[mask]      = -999
    return           log

def Sv2sa(Sv, r, r0, r1, method='mean'):
    """
    Compute Area backscattering coefficient (m2 m-2), by integrating Sv in a
    given range interval.
    
    Args:
        Sv (float)    : 2D array with Sv data (dB m-1)
        r  (float)    : 1D array with range data (m)
        r0 (int/float): Top range limit (m)
        r1 (int/float): Bottom range limit (m)
        method (str)  : Method for calculating sa. Accepts "mean" or "sum".
        
    Returns:
        float: 1D array with area backscattering coefficient data.
        float: 1D array with the percentage of vertical samples integrated.
    """
    
    # get r0 and r1 indexes
    r0 = np.argmin(abs(r-r0))
    r1 = np.argmin(abs(r-r1))
    
    # get number and height of samples 
    ns = len(r[r0:r1])
    sh = np.r_[np.diff(r), np.nan]
    sh = np.tile(sh.reshape(-1,1), (1,len(Sv[0])))[r0:r1,:]
    
    # compute Sa    
    sv = lin(Sv[r0:r1, :])
    if method=='mean':    
        sa = np.nanmean(sv * sh, axis=0) * ns
    elif method=='sum':
        sa = np.nansum (sv * sh, axis=0)
    else:
        raise Exception('Method not recognised')
    
    # compute percentage of valid values (not NAN) behind every sa integration    
    per = (len(sv) - np.sum(np.isnan(sv*sh), axis=0)) / len(sv) * 100
    
    # correct sa with the proportion of valid values
    sa = sa/(per/100)
        
    return sa, per

def Sv2NASC(Sv, r, r0, r1, method='mean'):
    """
    Compute Nautical Area Scattering Soefficient (m2 nmi-2), by integrating Sv
    in a given range interval.
    
    Args:
        Sv (float)    : 2D array with Sv data (dB m-1)
        r  (float)    : 1D array with range data (m)
        r0 (int/float): Top range limit (m)
        r1 (int/float): Bottom range limit (m)
        method (str)  : Method for calculating sa. Accepts "mean" or "sum"
        
    Returns:
        float: 1D array with Nautical Area Scattering Coefficient data.
        float: 1D array with the percentage of vertical samples integrated.
    """
    
    # get r0 and r1 indexes
    r0 = np.argmin(abs(r-r0))
    r1 = np.argmin(abs(r-r1))
    
    # get number and height of samples 
    ns     = len(r[r0:r1])
    sh = np.r_[np.diff(r), np.nan]
    sh = np.tile(sh.reshape(-1,1), (1,len(Sv[0])))[r0:r1,:]
    
    # compute NASC    
    sv = lin(Sv[r0:r1, :])
    if method=='mean':    
        NASC = np.nanmean(sv * sh, axis=0) * ns * 4*np.pi*1852**2
    elif method=='sum':
        NASC = np.nansum (sv * sh, axis=0)      * 4*np.pi*1852**2
    else:
        raise Exception('Method not recognised')
    
    # compute percentage of valid values (not NAN) behind every NASC integration    
    per = (len(sv) - np.sum(np.isnan(sv*sh), axis=0)) / len(sv) * 100
    
    # correct sa with the proportion of valid values
    NASC = NASC/(per/100)
        
    return NASC, per    