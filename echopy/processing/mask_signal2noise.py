#!/usr/bin/env python3
"""
Algorithms for masking data based on signal-to-noise ratio.
    
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

__authors__ = ['Alejandro Ariza'   # wrote derobertis(), and fielding()
               ]
__credits__ = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas               
               ]

import numpy as np
from echopy.utils import transform as tf

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
    Svclean = tf.log(tf.lin(Sv) - tf.lin(bgn))
    
    # signal to noise ratio
    s2n = Svclean - bgn
    
    # mask where Sv is less than background noise by a user-defined threshold
    mask1 = np.ma.masked_less(s2n, thr).mask
    mask2 = np.ma.masked_less(tf.lin(Sv) - tf.lin(bgn), 0).mask
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
    
    # mask indicating where masking is unfeasible due to NAN values
    mask_ = np.isnan(bgn)
    
    # nask where bgn is above the minimum value expected
    bgn_ = bgn.copy()             
    bgn_[mask_] = np.inf      
    mask = bgn_>thr
    
    return mask, mask_

def other():
    """   
    Note to contributors:
        Other signal-to-noise masks must be named with the author or method
        name. If already published, the full citation must be provided. Please,
        add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check contribute.md to follow our coding and documenting style.
    """