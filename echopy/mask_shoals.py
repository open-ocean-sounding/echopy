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

def weill(Sv, thr=-70, maxvgap=5, maxhgap=0, minvlen=0, minhlen=0, start=0):
    """
    Detects and masks shoals following the algorithm decribed in:
        
        "Weill et al. (1993): MOVIES-B — an acoustic detection description
        software . Application to shoal species' classification".
    
    Contiguous Sv samples above a given threshold will be considered as the
    same shoal, as long as it meets the contiguity criteria described by Weill
    et al. (1993):
        
        * Vertical contiguity: along-ping gaps are allowed to some extent. 
        Tipically, no more than the number of samples equivalent to half of the
        pulse length.
        
        * Horizontal contiguity: above-threshold features from contiguous pings
        will be regarded as the same shoal if there is at least one sample in
        each ping at the same range depth.
        
    Although the default settings strictly complies with Weill's contiguity
    criteria, other contiguity arguments has been enabled in this function to
    increase operability. For instance, the posibility to link non-contiguous
    pings to certain extent, or to set minimum vertical and horizontal lengths
    for a feature to be regarded as a shoal.
    
    Args:
        Sv (float): 2D numpy array with Sv data (dB).
        r (float): 1D numpy array with range data (m).
        thr (int): Sv threshold (dB).
        maxvgap (int): maximum vertical gap allowed (n samples).
        maxhgap (int): maximum horizontal gap allowed (n pings).
        minvlen (int): minimum vertical length for a shoal to be eligible
                       (n samples).
        minhlen (int): minimum horizontal length for a shoal to be eligible
                       (n pings).
        start (int): ping index to start processing. If greater than cero, it
                     means that Sv carries data from a preceeding file and the
                     the algorithm needs to know where to start processing.
        
    Returns:
        bool: 2D mask with shoals identified.
    """
                 
    # mask Sv above threshold
    mask = np.ma.masked_greater(Sv, thr).mask
    
    # for each ping in the mask... 
    for jdx, ping in enumerate(list(np.transpose(mask))):    
        
        # find gaps between masked features, and give them a label number
        pinglabelled = nd.label(np.invert(ping))[0]    
        
        # proceed only if the ping presents gaps
        if (not (pinglabelled==0).all()) & (not (pinglabelled==1).all()):
            
            # get list of gap labels and iterate through gaps       
            labels = np.arange(1, np.max(pinglabelled)+1)
            for label in labels:
                
                # if vertical gaps are equal/shorter than maxvgap...
                gap= pinglabelled==label
                if np.sum(gap)<=maxvgap:
                    
                    # get gap indexes and fill in with True values (masked)
                    idx= np.where(gap)[0]
                    if (not 0 in idx) & (not len(mask)-1 in idx): #(exclude edges)
                        mask[idx, jdx] = True
                        
                # if horizontal gaps are equal/shorter than maxhgap...
                # TODO: implement this bit
    
    # label connected features in the mask
    masklabelled = nd.label(mask)[0]  
    
    # get list of features labelled and iterate through them
    labels = np.arange(1, np.max(masklabelled)+1)
    for label in labels:
            
        # target feature & calculate its maximum vertical/horizontal length
        feature     = masklabelled==label
        featurehlen = np.max(np.sum(feature, axis=1))
        featurevlen = np.max(np.sum(feature, axis=0))
        
        # remove feature from mask if its maximum vertical lenght < minvlen
        if featurevlen<minvlen:
            idx, jdx       = np.where(feature)
            mask[idx, jdx] = False
            
        # remove feature from mask if its maximum horizontal lenght < minhlen
        if featurehlen<minhlen:
            idx, jdx       = np.where(feature)
            mask[idx, jdx] = False                    
    
    # get mask_ indicating the valid samples for mask   
    mask_                      = np.zeros_like(mask, dtype=bool)
    mask_[minvlen:len(mask_)-minvlen, minhlen:len(mask_[0])-minhlen] = True
    
    # return masks, from the start ping onwards           
    return mask, mask_

def other():
    """
    Note to contributors:
        Other algorithms for masking shoals must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check DESIGN.md to adhere to our coding style.
    """