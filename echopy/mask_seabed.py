#!/usr/bin/env python3
"""
Contains different modules for masking seabed (SB).

Created on Fri Apr 27 14:45:59 2018
@author: Alejandro Ariza, British Antarctic Survey
"""
import numpy as np
from echopy.transform import lin, log
import cv2
from skimage.morphology import remove_small_objects
from skimage.morphology import erosion
from skimage.morphology import dilation
from scipy.signal import convolve2d
import scipy.ndimage as nd

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
            if np.isnan(Sv[i-5:i, j]).all():
                Svmean = thr[1]+1
            else:      
                Svmean = log(np.nanmean(lin(Sv[i-5:i, j])))
            
            while (Svmean>thr[1]) & (i>=5):
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

def blackwell(Sv, theta, phi, r,
              r0=10, r1=1000, ttheta=702, tphi=282, wtheta=28 , wphi=52):
    """
    Detects and mask seabed using the split-beam angle and Sv, based in 
    "Blackwell et al (2019), Aliased seabed detection in fisheries acoustic
    data". Complete article here: https://arxiv.org/abs/1904.10736
    
    Args:
        Sv (float): 2D numpy array with Sv data (dB)
        theta (float): 2D numpy array with the along-ship angle (degrees)
        phi (float): 2D numpy array with the athwart-ship angle (degrees)
        r (float): 1D range array (m)
        r0 (int): minimum range below which the search will be performed (m) 
        r1 (int): maximum range above which the search will be performed (m)
        ttheta (int): Theta threshold above which seabed is pre-selected (dB)
        tphi (int): Phi threshold above which seabed is pre-selected (dB)
        wtheta (int): window's size for mean square operation in Theta field
        wphi (int): window's size for mean square operation in Phi field
                
    Returns:
        bool: 2D array with seabed mask
    """
    
    # delimit the analysis within user-defined range limits 
    r0 = np.nanargmin(abs(r - r0))
    r1 = np.nanargmin(abs(r - r1)) + 1
    Svchunk = Sv[r0:r1, :]
    thetachunk = theta[r0:r1, :]
    phichunk = phi[r0:r1, :]
    
    # get blur kernels with theta & phi width dimensions 
    ktheta = np.ones((wtheta, wtheta))/wtheta**2
    kphi   = np.ones((wphi, wphi))/wphi**2
    
    # perform mean square convolution and mask if above theta & phi thresholds
    thetamaskchunk = convolve2d(thetachunk, ktheta, 'same',
                                boundary = 'symm')**2 > ttheta
    phimaskchunk   = convolve2d(phichunk  ,  kphi , 'same',
                                boundary = 'symm')**2 > tphi
    anglemaskchunk = thetamaskchunk | phimaskchunk
        
    # if aliased seabed, mask Sv above the Sv median of angle-masked regions
    Svmaskchunk = Svchunk > log(np.nanmedian(lin(Svchunk[anglemaskchunk])))
    
    # label connected features in Svmaskchunk  
    f = nd.label(Svmaskchunk, nd.generate_binary_structure(2,2))[0]
    
    # get features intercepted by the anglemaskchunk (likely, the seabed)
    fintercepted = list(set(f[anglemaskchunk]))  
    if 0 in fintercepted: fintercepted.remove(fintercepted == 0)
        
    # combine angle-intercepted features in a single mask 
    maskchunk = np.zeros(Svchunk.shape, dtype = 'bool')
    for i in fintercepted:
        maskchunk = maskchunk | (f==i)

    # add data above r0 and below r1 (removed in first step)
    above = np.zeros((r0, maskchunk.shape[1]), dtype = 'bool')
    below = np.zeros((len(r) - r1, maskchunk.shape[1]), dtype = 'bool')
    mask  = np.r_[above, maskchunk, below]     

    return mask

def experimental(Sv, r,
                 r0=10, r1=1000, roff=0, thr=(-30,-70), ns=150, nd=3):
    """
    Mask Sv above a threshold to get a potential seabed mask. Then, the mask is
    dilated to fill seabed breaches, and small objects are removed to prevent 
    masking high Sv features that are not seabed (e.g. fish schools or spikes).    
    Once this is done, the mask is built up until Sv falls below a 2nd
    threshold, Finally, the mask is extended all the way down. 
    
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
            while (log(np.nanmean(lin(Sv[i-5:i, j]))) > thr[1]) & (i>=5):
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

def ariza(Sv, r, r0=20, r1=1000, roff=0,
          thr=-40, ec=1, ek=(1,3), dc=10, dk=(3,7)):
   
    """
    Mask Sv above a threshold to get potential seabed features. These features
    are eroded first to get rid of fake seabeds (spikes, schools, etc.) and
    dilated afterwards to fill in seabed breaches. Seabed detection is coarser
    than other methods (it removes water nearby the seabed) but the seabed line
    never drops when a breach occurs. Suitable for pelagic assessments and
    reconmended for non-supervised processing.
    
    Args:
        Sv (float): 2D Sv array (dB).
        r (float): 1D range array (m).
        r0 (int): minimum range below which the search will be performed (m). 
        r1 (int): maximum range above which the search will be performed (m).
        roff (int): seabed range offset (m).
        thr (int): Sv threshold above which seabed might occur (dB).
        ec (int): number of erosion cycles.
        ek (int): 2-elements tuple with vertical and horizontal dimensions
                  of the erosion kernel.
        dc (int): number of dilation cycles.
        dk (int): 2-elements tuple with vertical and horizontal dimensions
                  of the dilation kernel.
           
    Returns:
        bool: 2D array with seabed mask.
    """
    
    # get indexes for range offset and range limits
    r0   = np.nanargmin(abs(r - r0))
    r1   = np.nanargmin(abs(r - r1))
    roff = np.nanargmin(abs(r - roff))
    
    # set to -999 shallow and deep waters (prevents seabed detection)
    Sv_ = Sv.copy()
    Sv_[ 0:r0, :] = -999
    Sv_[r1:  , :] = -999
    
    # return empty mask if there is nothing above threshold
    if not (Sv_>thr).any():
        
        mask = np.zeros_like(Sv_, dtype=bool)
        return mask
    
    # search for seabed otherwise    
    else:
        
        # potential seabed will be everything above the threshold, the rest
        # will be set as -999
        seabed          = Sv_.copy()
        seabed[Sv_<thr] = -999
        
        # run erosion cycles to remove fake seabeds (e.g: spikes, small shoals)
        for i in range(ec):
            seabed = erosion(seabed, np.ones(ek))
        
        # run dilation cycles to fill seabed breaches   
        for i in range(dc):
            seabed = dilation(seabed, np.ones(dk))
        
        # mask as seabed everything greater than -999 
        mask = seabed>-999        
        
        # if seabed occur in a ping...
        idx = np.argmax(mask, axis=0)
        for j, i in enumerate(idx):
            if i != 0:
                
                # ...apply range offset & mask all the way down 
                i -= roff
                if i<0:
                    i = 0
                mask[i:, j] = True 
                
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