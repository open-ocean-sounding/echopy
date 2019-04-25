#!/usr/bin/env python3
"""
Contains different modules for estimating background noise.
   
Created on Fri Apr 27 14:28:57 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

import numpy as np
from echopy.operations import binv, bin2d, binvback, bin2dback

def derobertis(Sv, r, alpha, m, n,
               m0=2, bgnmax =-125):
    """
    Estimate background noise as in:
        
        De Robertis and Higginbottom (2007) ‘A post-processing technique to 
        estimate the signal-to-noise ratio and remove echosounder background 
        noise’, ICES Journal of Marine Science, 64: 1282–1291.
    
    Args:
        Sv (float): 2D numpy array with volume backscattering strength data (dB)
        r (float):  1D numpy array with range data (m)
        alpha (float): absorption coefficient value (dB m-1)
        m (int): binning vertical resolution (m)
        n (int): binning horizontal resolution (pings)
        m0 (int): vertical smoothing height before noise estimation (m) 
        bgnmax (int): maximum background noise estimation (dB)
    
    Returns:
        float: 2D numpy array with background noise estimation (dB)
    """
    
    # bin Sv vertically to smooth out Sv data    
    Svbinned, rbinned = binv(Sv, r, m0)
    Svsmooth = binvback(Svbinned, rbinned, r)
    
    # estimate TVG 
    r_ = r.copy()
    r_[r<=0] = np.nan
    tvg = 20*np.log10(r_) + 2*alpha*r_
    
    # subtract TVG from Sv  
    Svnotvg = Svsmooth - np.vstack(tvg)
    
    # resample Sv-TVG into m (meters) by n (pings) cells   
    pings = np.arange(0, Sv.shape[1])
    Svnotvgbnd, rbnd, pingsbnd = bin2d(Svnotvg, r, pings, m, n)
    
    # compute background noise as the minimun value per interval
    Svnotvgbnd[:, np.isnan(Svnotvgbnd).all(axis=0)] = 0
    bgnbnd = np.nanmin(Svnotvgbnd, 0)
    bgnbnd[np.isnan(Svnotvgbnd).all(axis=0)] = np.nan
    bgnbnd = np.tile(bgnbnd, [len(Svnotvgbnd), 1])
        
    # Prevent to exceed the maximum background noise expected
    mask = np.ma.masked_greater(bgnbnd, bgnmax).mask
    bgnbnd[mask] = bgnmax
    
    # resample background noise to previous Sv resolution, and add TVG
    bgn = bin2dback(bgnbnd, rbnd, pingsbnd, r, pings)
    bgn = np.vstack(tvg) + bgn
    
    return bgn

def derobertis_mod(Sv, r, alpha, m, n,
                   m0=2, bgnmax=-125, operation='mean'):
    """
    Modified from module "derobertis". This one allows to choose between
    different average operations.
    
    Args:
        Sv (float): 2D array with volume backscattering strength data (dB).
        r (float):  1D array with range data (m).
        alpha (float): absorption coefficient value (dB m-1).
        m (int): binning vertical resolution (m).
        n (int): binning horizontal resolution (pings).
        m0 (int): vertical smoothing height before noise estimation (m). 
        bgnmax (int): maximum background noise estimation (dB).
        operation (str): type of average operation:
            'mean' (default)
            'percentileXX' (XX is the percentile rank. e.g., 'percentile90')
            'median'
            'mode'
    
    Returns:
        float: 2D array with background noise estimation (dB)
    """
    
    # bin Sv vertically to smooth out Sv data    
    Svbinned, rbinned = binv(Sv, r, m0)[0:2]
    Svsmooth = binvback(Svbinned, rbinned, r)
    
    # estimate TVG
    r_ = r.copy()
    r_[r<=0] = np.nan
    tvg = 20*np.log10(r_) + 2*alpha*r_
    
    # subtract TVG from Sv  
    Svnotvg = Svsmooth - np.vstack(tvg)
    
    # resample Sv-TVG into m (meters) by n (pings) cells   
    pings = np.arange(len(Sv[0]))
    Svnotvgbnd, rbnd, pingsbnd = bin2d(Svnotvg, r, pings, m, n,
                                       operation=operation)[0:3]
    
    # compute background noise as the minimun value per interval
    Svnotvgbnd[:, np.isnan(Svnotvgbnd).all(axis=0)] = 0
    bgnbnd = np.nanmin(Svnotvgbnd, 0)
    bgnbnd[np.isnan(Svnotvgbnd).all(axis=0)] = np.nan
    bgnbnd = np.tile(bgnbnd, [len(Svnotvgbnd), 1])
        
    # Prevent to exceed the maximum background noise expected
    mask = np.ma.masked_greater(bgnbnd, bgnmax).mask
    bgnbnd[mask] = bgnmax
    
    # resample background noise to previous Sv resolution, and add TVG
    bgn = bin2dback(bgnbnd, rbnd, pingsbnd, r, pings)
    bgn[:, np.isnan(Sv).all(axis=0)] = np.nan
    bgn = np.vstack(tvg) + bgn
    
    return bgn

def other():
    """
    Note to contributors:
        Other algorithms for estimating background noise must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check DESIGN.md to adhere to our coding style.
    """