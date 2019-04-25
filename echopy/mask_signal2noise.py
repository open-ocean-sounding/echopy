#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mask Sv based on the signal-to-noise values.

Created on Thu Jul  5 14:13:15 2018
@author: Alejandro Ariza, British Antartic Survey
"""

import numpy as np
from echopy.operations import tolin, tolog

def derobertis(Sv, bgn, thr):
    """
    Mask Sv values when lower than background noise by a user-defined
    threshold, following:
        
        De Robertis and Higginbottom (2007) ‘A post-processing technique to 
        estimate the signal-to-noise ratio and remove echosounder background 
        noise’, ICES Journal of Marine Science, 64: 1282–1291.
    
    Args:
        Sv (float): 2D array with Sv data to be masked (dB)
        background (float): 2 array with background noise data (dB)
        thr (int): threshold value (dB)
        
    Returns:
        bool:  2D array mask (Sv < background = True)
    """
    
    # subtract background noise
    Svclean = tolog(tolin(Sv) - tolin(bgn))
    
    # signal to noise ratio
    s2n = Svclean - bgn
    
    # mask where Sv is less than background noise by a user-defined threshold
    mask1 = np.ma.masked_less(s2n, thr).mask
    mask2 = np.ma.masked_less(tolin(Sv) - tolin(bgn), 0).mask
    mask = mask1| mask2
    
    return mask

def fielding(bgn, thr=-80):
    """
    Mask where the background noise estimation is above the minimum Sv value
    expected for the target being surveyed. The mask is applied in the Sv array
    to indicate where the surveyed targets won't be visible due to background
    noise levels. Method proposed by Sophie Fielding (unpub.).
    
    Args:
        bgn: 2D array with the background noise estimation (dB).
        thr: Target Sv threshold (dB).
    Returns:
        bool: 2D array mask (background noise < expeted target Sv = True) 
    """
    
    return np.ma.masked_greater(bgn, thr).mask

def other():
    """   
    Note to contributors:
        Other signal-to-noise masks must be named with the author or method
        name. If already published, the full citation must be provided. Please,
        add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check /DESIGN.md to adhere to our coding style.
    """