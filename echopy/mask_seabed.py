#!/usr/bin/env python3
"""
Contains different modules for masking seabed (SB).

Created on Fri Apr 27 14:45:59 2018
@author: Alejandro Ariza, British Antarctic Survey
"""
import numpy as np
from echopy.operations import tolin, tolog
import cv2
from skimage.morphology import remove_small_objects

def maxSv(Sv, r, r0=10, r1=1000, roff=0, thr=(-40, -60)):
    """
    Initially detects the seabed as the ping sample with the strongest Sv value, 
    as long as it exceeds a dB threshold. Then it searchs up along the ping 
    until Sv falls below a secondary (lower) dB threshold, where the final 
    seabed is set.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).

    Returns:
        bool: 2D array with seabed mask.     
    """
        
    # get offset and range indexes
    roff = np.nanargmin(abs(r-roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1))
    
    # get indexes for maximum Sv along every ping,
    idx = np.int64(np.zeros(Sv.shape[1]))
    idx[~np.isnan(Sv).all(axis=0)] = np.nanargmax(
            Sv[r0:r1, ~np.isnan(Sv).all(axis=0)], axis=0) + r0
    
    # indexes with maximum Sv < main threshold are discarded (=0)
    maxSv = Sv[idx, range(len(idx))]
    maxSv[np.isnan(maxSv)] = -999
    idx[maxSv < thr[0]] = 0
    
    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask  = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i!=0:
            
            # decrease indexes until Sv mean falls below the 2nd threshold
            while (tolog(np.nanmean(tolin(Sv[i-5:i, j])))>thr[1]) & (i>=5):
                i -= 1
                       
            # subtract range offset & mask all the way down 
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True
    
    return mask

def deltaSv(Sv, r, r0=10, r1=1000, roff=0, thr=20):
    """
    Examines the difference in Sv over a 2-samples moving window along
    every ping, and returns the range of the first value that exceeded 
    a user-defined dB threshold (likely, the seabed).
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m).
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): threshold value (dB).
        start (int): ping index to start processing.

    Returns:
        bool: 2D array with seabed mask. 
    """
    # get offset as number of samples 
    roff = np.nanargmin(abs(r-roff))
    
    # compute Sv difference along every ping
    Svdiff = np.diff(Sv, axis=0)
    dummy = np.zeros((1, Svdiff.shape[1])) * np.nan
    Svdiff = np.r_[dummy, Svdiff]
    
    # get range indexes  
    r0 = np.nanargmin(abs(r-r0))
    r1 = np.nanargmin(abs(r-r1))
    
    # get indexes for the first value above threshold, along every ping
    idx = np.nanargmax((Svdiff[r0:r1, :]>thr), axis=0) + r0
    
    # mask seabed, proceed only with acepted seabed indexes (!=0)
    idx = idx
    mask = np.zeros(Sv.shape, dtype=bool)
    for j, i in enumerate(idx):
        if i != 0: 
            
            # subtract range offset & mask all the way down
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True        

    return mask

def experimental(Sv, r,
                 r0=10, r1=1000, roff=0, thr=(-30, -70), ns=150, nd=3):
    """
    Mask Sv above a threshold to get a potential seabed mask. Then, the mask is
    dilated to fill seabed breaches, and small objects are removed to prevent 
    masking high Sv features that are not seabed (e.g. fish schools or spikes).    
    Once this is done, the mask is built up until Sv falls below a 2nd
    threshold, Finally. the mask is extended all the way down. 
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (tuple): 2 integers with 1st and 2nd Sv threshold (dB).
        ns (int): maximum number of samples for an object to be removed.
        nd (int): number of dilations performed to the seabed mask.
           
    Returns:
        bool: 2D array with seabed mask.
    """

    # get indexes for range offset and range limits 
    roff = np.nanargmin(abs(r - roff))
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1)) + 1
    
    # mask Sv above the first Sv threshold
    mask = Sv[r0:r1, :] > thr[0]
    maskabove = np.zeros((r0, mask.shape[1]), dtype =bool)
    maskbelow = np.zeros((len(r) - r1, mask.shape[1]), dtype=bool)
    mask  = np.r_[maskabove, mask, maskbelow]     
    
    # remove small to prevent other high Sv features to be masked as seabed 
    # (e.g fish schools, impulse noise not properly masked. etc)
    mask = remove_small_objects(mask, ns)
    
    # dilate mask to fill seabed breaches
    # (e.g. attenuated pings or gaps from previous masking) 
    kernel = np.ones((3,5))
    mask = cv2.dilate(np.uint8(mask), kernel, iterations=nd)
    mask = np.array(mask, dtype = 'bool')
        
    # proceed with the following only if seabed was detected
    idx = np.argmax(mask, axis=0)
    for j, i in enumerate(idx):
        if i != 0:
            
            # rise up seabed until Sv falls below the 2nd threshold
            while (tolog(np.nanmean(tolin(Sv[i-5:i, j]))) > thr[1]) & (i>=5):
                i -= 1
                   
            # subtract range offset & mask all the way down 
            i -= roff
            if i<0:
                i = 0
            mask[i:, j] = True  
    
#    # dilate again to ensure not leaving seabed behind
#    kernel = np.ones((3,3))
#    mask = cv2.dilate(np.uint8(mask), kernel, iterations = 2)
#    mask = np.array(mask, dtype = 'bool')

    return mask

def bestcandidate():
    """
    Echoview best bottom candidate
    TODO: need to understand echoview manual to implement the algorithm!
    """

def other():
    """    
    Note to contributors:
        Alternative algorithms for masking seabed must be named with the
        author or method name. If already published, the full citation must
        be provided.
        
        Please, check /DESIGN.md to adhere to our coding style.
    """