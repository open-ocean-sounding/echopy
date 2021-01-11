#!/usr/bin/env python3
"""
Contains different modules for masking echograms based on their response 
across two or more frequencies.

Created on Thu Jul  5 16:47:21 2018
@author: Alejandro Ariza, British Antarctic Survey
"""
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
        
        Please, check /DESIGN.md to adhere to our coding style.
    """   