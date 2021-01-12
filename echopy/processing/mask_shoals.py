#!/usr/bin/env python3
"""
Algorithms for masking shoals.
    
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

__authors__ = ['Alejandro Ariza'   # wrote weill(), echoview()
               ]                                 
__credits__ = ['Rob Blackwell'     # supervised the code and provided ideas
               'Sophie Fielding'   # supervised the code and provided ideas               
               ]

import numpy as np
import scipy.ndimage as nd
import pandas as pd

def weill(Sv, thr=-70, maxvgap=5, minvlen=0, minhlen=0):
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
    increase operability. For instance, the posibility to set minimum vertical
    and horizontal lengths for a feature to be regarded as a shoal.
    
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
    
    # label connected features in the mask
    masklabelled = nd.label(mask)[0]  
    
    # get list of features labelled and iterate through them
    labels = np.arange(1, np.max(masklabelled)+1)
    for label in labels:
            
        # target feature & calculate its maximum vertical/horizontal length
        feature     = masklabelled==label
        idx, jdx    = np.where(feature)
        featurehlen = max(idx+1) - min(idx)
        featurevlen = max(jdx+1) - min(jdx)
        
        # remove feature from mask if its maximum vertical lenght < minvlen
        if featurevlen<minvlen:
            mask[idx, jdx] = False
            
        # remove feature from mask if its maximum horizontal lenght < minhlen
        if featurehlen<minhlen:
            mask[idx, jdx] = False                    
    
    # get mask_ indicating the valid samples for mask   
    mask_                      = np.zeros_like(mask, dtype=bool)
    mask_[minvlen:len(mask_)-minvlen, minhlen:len(mask_[0])-minhlen] = True
    
    # return masks, from the start ping onwards           
    return mask, mask_


def echoview(Sv, idim, jdim,
             thr=-70, mincan=(3,10), maxlink=(3,15), minsho=(3,15)):
    """
    Shoals detection algorithm as described in echoview.
    
    In progress.
    
    Args:
        Sv      (float    ): 2D array with Sv data (dB)
        idim    (int/float): i vertical dimension (n samples, range, etc.)
        jdim    (int/float): j horizontal dimension (n pings, time, distance, etc.)
        thr     (int/float): threshold value above which Sv will be masked (dB)
        mincan  (int/float): 2-element tuple with minimum allowed height and 
                             width for shoal candidates before linking.
        maxlink (int/float): 2-element tuple with maximum allowed height and 
                             width distances to link neighbour shoals.
        minsho  (int/float): 2-element tuple with minimum allowed height and
                             width for shoals after linking. 
    
    Returns:
        bool: 2D array with shoals identified.
        
    Notes:
        i/j dimensions must be the same in all arguments. For example if i/j
        dimensions refer to range in metres and distance in nautical miles, the
        height and width in mincan, maxlink and minsho might be metres and
        nautical miles as well.
    """
    
    # check aptness of i/j dimensions
    if np.isnan(idim).any():
        raise Exception('Can not proceed with NAN values in i dimension')
    if np.isnan(jdim).any():
        raise Exception('Can not proceed with NAN values in j dimension')
        
    # get mask with candidate shoals by masking Sv above threshold
    mask = np.ma.masked_greater(Sv, thr).mask
    if isinstance(mask, np.bool_):
        mask = np.zeros_like(Sv, dtype=bool)
    
    # iterate through shoal candidates
    candidateslabeled= nd.label(mask, np.ones((3,3)))[0]
    candidateslabels = pd.factorize(candidateslabeled[candidateslabeled!=0])[1]
    for cl in candidateslabels:
        
        #measure candidate's height and width
        candidate       = candidateslabeled==cl 
        idx             = np.where(candidate)[0]
        jdx             = np.where(candidate)[1]
        candidateheight = idim[max(idx+1)] - idim[min(idx)]
        candidatewidth  = jdim[max(jdx+1)] - jdim[min(jdx)]
        
        # remove candidate from mask if larger than min candidate size
        if (candidateheight<mincan[0]) | (candidatewidth<mincan[1]):
            mask[idx, jdx] = False
    
    # declare linked-shoals array
    linked    = np.zeros(mask.shape, dtype=int)

    # iterate through shoals
    shoalslabeled = nd.label(mask, np.ones((3,3)))[0]
    shoalslabels  = pd.factorize(shoalslabeled[shoalslabeled!=0])[1]
    for fl in shoalslabels:
        shoal = shoalslabeled==fl

        # get i/j frame coordinates for the shoal
        i0 = min(np.where(shoal)[0])
        i1 = max(np.where(shoal)[0])
        j0 = min(np.where(shoal)[1])
        j1 = max(np.where(shoal)[1])
        
        # get i/j frame coordinates including linking distance around the shoal
        i00 = np.nanargmin(abs(idim-(idim[i0]-(maxlink[0]+1))))
        i11 = np.nanargmin(abs(idim-(idim[i1]+(maxlink[0]+1))))+1
        j00 = np.nanargmin(abs(jdim-(jdim[j0]-(maxlink[1]+1))))
        j11 = np.nanargmin(abs(jdim-(jdim[j1]+(maxlink[1]+1))))+1
        
        # find neighbours around shoal
        around                  = np.zeros_like(mask, dtype=bool)
        around[i00:i11,j00:j11] = True       
        neighbours              = around & mask # & ~feature      
        neighbourlabels         = pd.factorize(shoalslabeled[neighbours])[1]
        neighbourlabels         = neighbourlabels[neighbourlabels!=0]
        neighbours              = np.isin(shoalslabeled, neighbourlabels)
        
        # link neighbours by naming them with the same label number
        if (pd.factorize(linked[neighbours])[1]==0).all():
            linked[neighbours] = np.max(linked)+1
        
        # if some are already labeled, rename all with the minimum label number
        else:
            formerlabels        = pd.factorize(linked[neighbours])[1]
            minlabel            = np.min(formerlabels[formerlabels!=0])
            linked[neighbours] = minlabel
            for fl in formerlabels[formerlabels!=0]:
                linked[linked==fl] = minlabel
    
    # iterate through linked shoals
    linkedlabels   = pd.factorize(linked[linked!=0])[1]
    for ll in linkedlabels:
        
        # measure linked shoal's height and width
        linkedshoal       = linked==ll
        idx               = np.where(linkedshoal)[0]
        jdx               = np.where(linkedshoal)[1]
        linkedshoalheight = idim[max(idx+1)] - idim[min(idx)]
        linkedshoalwidth  = jdim[max(jdx+1)] - jdim[min(jdx)]
        
        # remove linked shoal from mask if larger than min linked shoal size
        if (linkedshoalheight<minsho[0]) | (linkedshoalwidth<minsho[1]):
            mask[idx, jdx] = False
    
    # get mask indicating mask edges where shoals coudn't be evaluated due to
    # shoals chopped at data borders.
    mask_               = np.ones(mask.shape, dtype=bool)    
    edgeheight          = np.max([mincan[0], maxlink[0], minsho[0]])
    edgewidth           = np.max([mincan[1], maxlink[1], minsho[1]])    
    i0                  = np.where((idim-idim[ 0]) - edgeheight >= 0)[0][ 0]
    i1                  = np.where((idim-idim[-1]) + edgeheight <  0)[0][-1]+1    
    j0                  = np.where((jdim-jdim[ 0]) - edgewidth  >= 0)[0][ 0]
    j1                  = np.where((jdim-jdim[-1]) + edgewidth  <  0)[0][-1]+1
    mask_[i0:i1, j0:j1] = False
        
    return mask, mask_
        
def other():
    """
    Note to contributors:
        Other algorithms for masking shoals must be named with the
        author or method name. If already published, the full citation must be
        provided. Please, add "unpub." otherwise. E.g: Smith et al. (unpub.)
        
        Please, check contribute.md to follow our coding and documenting style.
    """