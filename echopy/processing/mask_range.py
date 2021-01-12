#!/usr/bin/env python3
"""
Filters for masking data based on depth range.
    
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

__authors__ = ['Alejandro Ariza'   # wrote above(), below(), inside(), outside()
               ]                                 
__credits__ = []

import numpy as np

def above(Sv, r, r0):
    """
    Mask data above a given range.
    
        Args:
            Sv (float): 2D array with data to be masked.
            r (float): 1D array with range data.
            r0 (int):  range above which data will be masked.
        
        Returns:
            bool: 2D array mask (above range = True).
    """
       
    idx = np.where(np.ma.masked_less(r, r0).mask)[0]
    mask = np.zeros((Sv.shape), dtype=bool)
    mask[idx,:] = True    
    return mask

def below(Sv, r, r0):
    """
    Mask data below a given range.
    
        Args:
            Sv (float): 2D array with data to be masked.
            r (float): 1D array with range data.
            r0 (int):  range below which data will be masked.
        
        Returns:
            bool: 2D array mask (below range = True).
    """
       
    idx = np.where(np.ma.masked_greater(r, r0).mask)[0]
    mask = np.zeros((Sv.shape), dtype=bool)
    mask[idx,:] = True    
    return mask

def inside(Sv, r, r0, r1):
    """
    Mask data inside a given range.
    
        Args:
            Sv (float): 2D array with data to be masked.
            r (float): 1D array with range data.
            r0 (int): Upper range limit.
            r1 (int): Lower range limit.
        
        Returns:
            bool: 2D array mask (inside range = True).
    """
    masku = np.ma.masked_greater_equal(r, r0).mask
    maskl = np.ma.masked_less(r, r1).mask    
    idx = np.where(masku & maskl)[0]
    mask = np.zeros((Sv.shape), dtype=bool)
    mask[idx,:] = True    
    return mask

def outside(Sv, r, r0, r1):
    """
    Mask data outside a given range.

        Args:
            Sv (float): 2D array with data to be masked.
            r (float): 1D array with range data.
            r0 (int): Upper range limit.
            r1 (int): Lower range limit.
        
        Returns:
            bool: 2D array mask (out of range = True).   
    """
    masku = np.ma.masked_less(r, r0).mask
    maskl = np.ma.masked_greater_equal(r, r1).mask    
    idx = np.where(masku | maskl)[0]
    mask = np.zeros((Sv.shape), dtype=bool)
    mask[idx,:] = True
    
    return mask

def other():
    """   
    Note to contributors:
        Please, check contribute.md to follow our coding and documenting style.
    """