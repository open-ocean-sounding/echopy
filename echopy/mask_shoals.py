#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains different modules for masking biological aggregations such as
shoals, schools, or swarms.

Created on Mon Jun  3 12:58:35 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

import numpy as np
import scipy.ndimage as nd

def weill(Sv, r, thr=-65, maxvgap=4, minlength=2):
    """
    Mask shoals as in "Weill et al. (1993): MOVIES-B — an acoustic detection 
    description software . Application to shoal species' classification".
    
    Contiguous Sv samples above a given threshold will be considered as shoals,
    as long as they meet the following contiguity laws:
        
        * Vertical contiguity: along-ping gaps are allowed to some extent. 
        Tipically, no more than the number of samples equivalent to half of the
        pulse length.
        
        * Horizontal contiguity: above-threshold features from contiguous pings
        will be regarded as the same shoal if there is at least one sample in
        each ping at the same range depth.          
          
    Shoals shorter than a user-defined number of pings will be ruled out.
    
    Args:
        Sv (float): 2D numpy array with Sv data (dB).
        r (float): 1D numpy array with range data (m).
        thr (int): Sv threshold (dB).
        maxvgap (int): maximum vertical gap allowed (n samples).
        minlength (int): minimum length for a shoal to be eligible (n pings).
        
    Returns:
        bool: 2D mask with shoals identified.
    """
    
    # mask Sv above threshold
    mask = Sv>thr
    
    # for each ping in the mask... 
    for jdx, ping in enumerate(list(np.transpose(mask))):    
        
        # find gaps between masked features, and give them a label number
        pinglabelled = nd.label(np.invert(ping))[0]    
        
        # proceed only if the ping presents gaps
        if (not (pinglabelled==0).all()) & (not (pinglabelled==1).all()):
            
            # get list of gap labels and iterate through gaps       
            labels = np.arange(1, np.max(pinglabelled)+1)
            for label in labels:
                
                # if gaps are equal/shorter than maxgap...
                gap= pinglabelled==label
                if np.sum(gap)<=maxvgap:
                    
                    # get gap indexes and fill in the mask
                    idx= np.where(gap)[0]
                    if (not 0 in idx) & (not len(mask)-1 in idx): #(exclude edges)
                        mask[idx, jdx] = True
    
    # label connected features in the mask
    masklabelled = nd.label(mask)[0]  
    
    # get list of features labelled and iterate through them
    labels = np.arange(1, np.max(masklabelled)+1)
    for label in labels:
            
        # target feature & calculate its maximum length
        feature = masklabelled==label
        featurelength = np.max(np.sum(feature, axis=1))
        
        # remove feature from mask if length < minlength
        if featurelength<minlength:
            idx, jdx = np.where(feature)
            mask[idx, jdx] = False
                    
    # the remaining features in the mask are your shoals:
    return mask

def other():
    """
    Note to contributors:
        Other algorithms for masking shoals must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check DESIGN.md to adhere to our coding style.
    """