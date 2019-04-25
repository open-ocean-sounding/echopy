#!/usr/bin/env python3
"""
Contains different depth range masking operations.
 
Created on Tue May  8 14:14:34 2018
@author: Alejandro Ariza, British Antarctic Survey 
"""
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
       
    idx = np.where(np.ma.masked_greater(r, r0).mask)[0]
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
       
    idx = np.where(np.ma.masked_less(r, r0).mask)[0]
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
        Please, check /DESIGN.md and adhere to our coding style
        for other range masking operations.
    """