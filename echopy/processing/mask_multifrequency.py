#!/usr/bin/env python3
"""
Algorithms for masking acoustic features based on frequency response.
    
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

__authors__ = ['Alejandro Ariza'   # wrote dbdiff()
               ]                                 
__credits__ = []

import numpy as np

def dbdiff(Sv1, Sv2, thr):
    """
    The db-difference method computes the difference of a pair of frequencies, 
    and mask where the values fall within a Sv threshold range.
    
    Args:
        Sv1 (float): 2D array with Sv data, the minuend frequency (dB)
        Sv2 (float): 2D array with Sv data, subtrahend frequency (dB)
        thr (int): tupple containing the Sv threshold range (dB)
        
    Returns:
        bool: 2D array mask (values within threshold range = True)
    """
    
    # dB difference
    dbdiff = Sv1-Sv2
    
    # mask where dB difference falls within the threshold range
    mask = np.ma.masked_greater(dbdiff, thr[0]).mask
    mask = np.ma.masked_less   (dbdiff, thr[1]).mask    
    mask = mask & mask
    
    return mask

def other():
    """   
    Note to contributors:
        Alternative multifrequency masking methods must be named with
        the author or method name. If already published, the full citation must
        be provided.
        
        Please, check contribute.md to follow our coding and documenting style.
    """   