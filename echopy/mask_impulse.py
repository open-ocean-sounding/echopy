#!/usr/bin/env python3
"""
Contains different modules for masking Impulse Noise (IN).
    
Created on Fri Apr 27 14:24:44 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

import numpy as np
from echopy.operations import binv, binvback, tolin, tolog
from skimage.morphology import erosion
from skimage.morphology import dilation
from scipy.ndimage.filters import median_filter as medianf

def ryan(Sv, r, m=5, n=1, thr=10, start=0):
    """
    Mask impulse noise following the two-sided comparison method described
    in:        
        Ryan et al. (2015) ‘Reducing bias due to noise and attenuation in 
        open-ocean echo integration data’, ICES Journal of Marine Science,
        72: 2482–2493.

    Args:
        Sv (float): 2D array with Sv data to be masked (dB).
        r (float):  1D array with range data (m).
        m (int): vertical binning length (m).
        n (int): number of pings either side for comparisons.
        thr (int): user-defined threshold value (dB).
        start (int): ping index to start processing.
        
    Returns:
        list: 2D boolean array with IN mask and 2D boolean array with mask
              indicating where IN detection was unfeasible.  
    """    
        
    # resample down vertically into x-meters bins    
    Svbinned, rbinned, missing = binv(Sv, r, m)
    Svbinned[missing>50] = np.nan
    
    # resample back to native resolution
    Svsmoothed = binvback(Svbinned, rbinned, r)
        
    #side comparison (±n)
    dummy = np.zeros((r.shape[0], n)); dummy[:] = np.nan     
    comparison_forward = Svsmoothed - np.c_[Svsmoothed[:, n:], dummy]
    comparison_backward = Svsmoothed - np.c_[dummy, Svsmoothed[:, 0:-n]]
    
    # get mask
    maskf = np.ma.masked_greater(comparison_forward, thr).mask
    maskb = np.ma.masked_greater(comparison_backward, thr).mask
    mask = maskf & maskb
    
    # get mask indicating where IN detection couldn't be carried out    
    mask_ = np.isnan(Svsmoothed)
    mask_[:, 0:n] = True
    mask_[:, -n:] = True    
    
    return [mask[:, start:], mask_[:, start:]]

def ryan_iterable(Sv, r, m=5, n=(1,2), thr=10, start=0):
    """
    Modified from "ryan" so that the parameter "n" can be provided multiple
    times. It enables the algorithm to iterate and perform comparisons at
    different n distances. Resulting masks at each iteration are combined in
    a single mask. By setting multiple n distances the algorithm can detect 
    spikes adjacent each other.
    
    Args:
        Sv (float): 2D array with Sv data to be masked (dB). 
        r (float):  1D array with range data (m).
        m (int): vertical binning length (m).
        n (tuple): number of pings either side for comparisons.
        thr (int): user-defined threshold value (dB).
        start (int): ping index to start processing. 
        
    Returns:
        list: 2D array with IN mask and 2D array with mask indicating where
              IN detection was unfeasible.
    """    
        
    # resample down vertically into x-meters bins    
    Svbinned, rbinned, missing = binv(Sv, r, m)
    Svbinned[missing>50] = np.nan
    
    # resample back to native resolution
    Svsmoothed = binvback(Svbinned, rbinned, r)
    
    # perform side comparisons and combine masks in one unique mask
    mask = np.zeros(Sv.shape, dtype=bool)
    for i in n:
        dummy = np.zeros((r.shape[0], i)); dummy[:] = np.nan     
        comparison_forward = Svsmoothed - np.c_[Svsmoothed[:,i:], dummy]
        comparison_backward = Svsmoothed - np.c_[dummy, Svsmoothed[:, 0:-i]]
        maskf = np.ma.masked_greater(comparison_forward, thr).mask
        maskb = np.ma.masked_greater(comparison_backward, thr).mask
        mask = mask | (maskf&maskb)
    
    # get mask indicating where IN detection couldn't be implemented    
    mask_ = np.isnan(Svsmoothed)
    mask_[:, 0:max(n)] = True
    mask_[:, -max(n):] = True    
    
    return [mask[:, start:], mask_[:, start:]]


def wang(Sv, thr=(-70,-40),
         erode=[(3,3)], dilate=[(5,5),(7,7)], median=[(7,7)]):
    """
    Clean impulse noise from Sv data following the method decribed by:
        
        Wang et al. (2015) ’A noise removal algorithm for acoustic data with
        strong interference based on post-processing techniques’, CCAMLR
        SG-ASAM: 15/02.
        
    This algorithm runs different cycles of erosion, dilation, and median
    filtering to clean up Sv from impulse noise. Note that this function
    returns a clean/corrected Sv array, instead of a boolean array indicating
    the occurrence of impulse noise.
        
    Args:
        Sv (float)  : 2D numpy array with Sv data (dB).
        thr (int)   : 2-elements tupple with bottom and top Sv thresholds (dB).
        erode (int) : list of 2-element tupples indicating the window's size
                      for each erosion cycle.
        dilate (int): list of 2-element tupples indicating the window's size
                      for each dilation cycle.
        median (int): list of 2-element tupples indicating the window's size
                      for each median filter cycle.
                      
    Returns:
        float: 2D numpy array with clean Sv data.
    """
    
    # set values out of threshold to -999
    Sv_thresholded = Sv.copy()
    Sv_thresholded[(Sv<thr[0])|(Sv>thr[1])] = -999
    
    # run erosion cycles
    Sv_eroded = Sv_thresholded.copy()
    for e in erode:
        Sv_eroded = erosion(Sv_eroded, np.ones(e))

    # run dilation cycles
    Sv_dilated = Sv_eroded.copy()
    for d in dilate:
        Sv_dilated = dilation(Sv_dilated, np.ones(d))
    
    # non-vacant values after erosion/dilation are corrected to corresponding
    # values before erosion/dilation
    Sv_corrected1 = Sv_dilated.copy()
    Sv_corrected1[Sv_dilated!=-999] = Sv_thresholded[Sv_dilated!=-999]
    
    # vacant values after erosion/dilation are corrected to median-filtered
    # values before erosion/dilation
    Sv_median = Sv_thresholded
    for m in median:
        Sv_median = tolog(medianf(tolin(Sv_median), footprint=np.ones(m)))    
    Sv_corrected2 = Sv_corrected1.copy()
    Sv_corrected2[Sv_dilated==-999] = Sv_median[Sv_dilated==-999]
    
    return Sv_corrected2

def other():
    """
    Note to contributors:
        Other algorithms for masking impulse noise must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check /DESIGN.md to adhere to our coding style.
    """
    
