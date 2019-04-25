#!/usr/bin/env python3
"""
Set of modules for binning and related operations.

Created on Tue Sep 25 11:20:00 2018
@author: Alejandro Ariza, British Antarctic Survey
"""
import numpy as np
from scipy import stats

def bin2d(data, iaxis, jaxis, istep, jstep,
          operation='mean', log=True, start=0):
    """
    Bin a 2D array according to given steps in i and j axes. It can deal
    with non-constant incrementing axes.
    
    Example:        
        Bin the following data array every 2 and 4 increments in the i and j
        dimension, respectively.
        
        i\j| 0  2  4  8 12 16
        ---|-----------------
         0 | 0  1  2  3  4  5
         1 | 6  7  8  9 10 11   (note that j axis is not a contant
         2 |12 13 14 15 16 17    incrementing array) 
         3 |18 19 20 21 22 23
         4 |24 25 26 27 28 29
         
        >>> bin2d(data, iaxis, jaxis, 2, 4)
        (array([[ 3.5,  5. ,  6. ,  7. ],
                [15.5, 17. , 18. , 19. ]]),    
        array([1., 3.]),
        array([ 2.,  6., 10., 14.]))
         
        this how the operation was done... 
                
        i/j| 0  2 | 4 | 8 |12 |16
        ---|---------------------
         0 | 0  1 | 2 | 3 | 4 | x         i/j|  2.0  6.0  10.0  14.0
         1 | 6  7 | 8 | 9 |10 | x         ---|---------------------- 
        -------------------------   =    1.0 | 3.5   5.0   6.0   7.0
         2 |12 13 |14 |15 |16 | x        3.0 |15.5  17.0  18.0  19.0
         3 |18 19 |20 |21 |22 | x
        -------------------------     (last row and column are not included
         4 | x  x | x |x |  x | x      in the results because ranges in python
                                       are exclusive)                                 
    Args:
        data (float): 2D array with data for binning.
        iaxis (float): 1D array with i vertical axis.   
        jaxis (float): 1D array with j horizontal axis.
        istep (int): step intervals for segmenting the i axis.
        jstep (int): step intervals for segmenting the j axis.
        operation (str): type of binning operation:
            'mean' (mean, default)
            'percentileXX' (percentile, XX is rank. e.g., 'percentile90')
            'median' (median)
            'mode'(mode)
            'std' (standard deviation)
            'var' (variance)
            'sum' (sum)
        log (bool): if True, data is considered as logarithmic and converted to
                    linear before any operation.
                    
    Returns:
        float: 2D binned data array
        float: 1D i binned array
        float: 1D j binned array
        float: 2D array with percentage of data missing before binning.
    """
    
    # covert to linear, if logarithmic
    if log is True:
        data = tolin(data)
    
    # get indexes and binned (bnd) i/j axes
    idx, iaxisbnd = get_intervals(iaxis, istep)
    jdx, jaxisbnd = get_intervals(jaxis, jstep)
        
    # vertically split data into slices
    slices = np.vsplit(data, idx[1:])
    
    # horizontally split slices into chunks
    chunks = [0]*len(slices)        
    for i in range(len(slices)):        
        chunks[i] = np.hsplit(slices[i], jdx[1:])
    
    # average chunks and allocate results in a data binned (bnd) array    
    databnd = []
    missing = []    
    for chunk in chunks[:-1]:
        for chunk in  chunk[:-1]:
            if (np.isnan(chunk).all()) | (chunk.size==0):
                databnd.append(np.nan)
            elif 'mean' in operation:
                databnd.append(np.nanmean(chunk))
            elif 'percentile' in operation:
                databnd.append(np.nanpercentile(chunk, int(operation[-2:])))
            elif 'median' in operation:
                databnd.append(np.nanmedian(chunk))
            elif 'mode' in operation:
                databnd.append(np.float64(stats.mode(chunk, axis=None)[0]))
            elif 'std' in operation:
                databnd.append(np.nanstd(chunk))
            elif 'var' in operation:
                databnd.append(np.nanvar(chunk))
            elif 'sum' in operation:
                databnd.append(np.nansum(chunk))
            else:
                raise Exception('Operation not recognised') 
            missing.append(np.isnan(chunk).sum()/chunk.size*100)
            
    databnd = np.array(databnd).reshape(len(chunks)-1, len(chunks[0])-1)
    missing = np.array(missing).reshape(len(chunks)-1, len(chunks[0])-1)
    
    # convert back to logarithmic, if logarithmic
    if log is True:
        databnd = tolog(databnd)
        
    # return data and axes binned
    return databnd, iaxisbnd, jaxisbnd, missing

def binh(data, jaxis, jstep,
         operation='mean', log=True):
    """
    Same as bin2d, but binning is only performed along the horizontal dimension
    
    Args:
        data (float): 2D array with data for binning.   
        jaxis (float): 1D array with j horizontal axis.
        jstep (int): step intervals for segmenting the j axis.
        operation (str): type of binning operation:
            'mean' (default)
            'percentileXX' (XX is the percentile rank. e.g., 'percentile90')
            'median'
            'mode'
        log (bool): if True, data is considered as logarithmic and converted to
                    linear before any operation.
                    
    Returns:
        float: binned 2D data array
        float: binned 1D j array
    """
    
    # convert to linear, if logarithmic
    if log is True:
        data = tolin(data)
    
    # get indexes and middle values for j axis
    jdx, jaxisbnd = get_intervals(jaxis, jstep)
            
    # split data horizontally into slices
    slices = np.hsplit(data,jdx[1:])
    
    # average slices and return them allocated in a 2D array
    databnd = np.zeros([len(data), len(slices)-1])*np.nan
    missing = np.zeros([len(data), len(slices)-1])*np.nan      
    for j, slc in enumerate(slices[:-1]):
        if (np.isnan(slc).all()) | (slc.size==0):
            databnd[:, j] = np.nan
        elif 'mean' in operation:
            databnd[~np.isnan(slc).all(axis=1), j] = np.nanmean(
                    slc[~np.isnan(slc).all(axis=1), :], 1)               
        elif 'percentile' in operation:
            databnd[~np.isnan(slc).all(axis=1), j] = np.nanpercentile(
                    slc[~np.isnan(slc).all(axis=1), :], int(operation[-2:]), 1)                
        elif 'median' in operation:
            databnd[~np.isnan(slc).all(axis=1), j] = np.nanmedian(
                    slc[~np.isnan(slc).all(axis=1), :], 1)                
        elif 'mode' in operation:
            databnd[~np.isnan(slc).all(axis=1), j] = np.float64(stats.mode(
                    slc[~np.isnan(slc).all(axis=1), :], axis = 1)[0])                
        else:
            raise Exception('Operation not recognised')
        missing[:, j] = np.isnan(slc).sum(axis=1)/slc.shape[1]*100
                
    # convert back to logarithmic, if logarithmic
    if log is True:
        databnd = tolog(databnd)
            
    return databnd, jaxisbnd, missing

def binv(data, iaxis, istep,
         operation='mean', log=True):
    """
    Same as bin2d, but binning is only performed along the vertical dimension.
    
    Args:
        data (float): 2D array with data for binning.
        iaxis (float): 1D array with i vertical axis.   
        istep (int): step intervals for segmenting the i axis.
        operation (str): type of binning operation:
            'mean' (default)
            'percentileXX' (XX is the percentile rank. e.g., 'percentile90')
            'median'
            'mode'
        log (bool): if True, data is considered as logarithmic and converted to
                    linear before any operation.
                    
    Returns:
        float: binned 2D data array
        float: binned 1D i array
    """
    
    # convert to linear, if logarithmic
    if log is True:
        data = tolin(data)
        
    # get indexes and middle values for i axis
    idx, iaxisbnd = get_intervals(iaxis, istep)
            
    # split data vertically into slices
    slices = np.vsplit(data, idx[1:])
    
    # average slices and return them allocated in a 2D array
    databnd = np.zeros([len(slices)-1,len(data[0])])*np.nan
    missing = np.zeros([len(slices)-1,len(data[0])])*np.nan                   
    for i, slc in enumerate(slices[:-1]):
        if (np.isnan(slc).all()) | (slc.size==0):
            databnd[i, :] = np.nan
        elif 'mean' in operation:
            databnd[i, ~np.isnan(slc).all(axis=0)] = np.nanmean(
                    slc[:, ~np.isnan(slc).all(axis=0)], 0)
        elif 'percentile' in operation:
            databnd[i, ~np.isnan(slc).all(axis=0)] = np.nanpercentile(
                    slc[:, ~np.isnan(slc).all(axis=0)], int(operation[-2:]), 0)                
        elif 'median' in operation:
            databnd[i, ~np.isnan(slc).all(axis=0)] = np.nanmedian(
                    slc[:, ~np.isnan(slc).all(axis=0)], 0)                
        elif 'mode' in operation:
            databnd[i, ~np.isnan(slc).all(axis=0)] = np.float64(stats.mode(
                    slc[:, ~np.isnan(slc).all(axis=0)],axis=0)[0])                
        else:
            raise Exception('Operation not recognised')
        missing[i,:] = np.isnan(slc).sum(axis=0)/slc.shape[0]*100
                
    # convert back to logarithmic, if logarithmic
    if log is True:
        databnd = tolog(databnd)
                
    # return data and i axis binned
    return databnd, iaxisbnd, missing

def bin2dback(databnd, iaxisbnd, jaxisbnd, iaxis, jaxis):
    """
    Turn binned data back to full resolution, according to original i and j
    full resolution axes. Array elements will be left as nan, if i and j 
    coordinates are not found within the binned data.
    
    Args:
        databnd (float): 2D array with binned data.
        iaxisbnd (float): 1D array with binned i axis.
        jaxisbnd (float): 1D array with binned j axis.
        iaxis (float): 1D array with full resolution i axis.
        jaxis (float): 1D array with full resolution j axis.
        
    Returns:
        float: 2D array with full resolution data binned.
    """
       
    # get distance radius for i and j binned intervals
    irad = np.diff(iaxisbnd[0:2])*.5
    jrad = np.diff(jaxisbnd[0:2])*.5
    
    # create full resolution array & fill with binned data, if match intervals
    data = np.zeros((len(iaxis), len(jaxis)))*np.nan            
    for i, ival in enumerate(iaxisbnd):
        idx = np.where((ival-irad<=iaxis) & (iaxis<ival+irad))[0]
        for j, jval in enumerate(jaxisbnd):        
            jdx = np.where((jval-jrad<=jaxis) & (jaxis<jval+jrad))[0]
            if idx.size*jdx.size > 0:
                data[idx[0]:idx[-1]+1,:][:,jdx[0]:jdx[-1]+1]= databnd[i,j]
                        
    return data

def binhback(databnd, jaxisbnd, jaxis):
    """
    Turn horizontally binned data back to full resolution, according to the 
    original full resolution j axis. Array elements will be left as nan, if j 
    coordinates are not found within the binned data.
    
    Args:
        databnd (float): 2D array with binned data.
        jaxisbnd (float): 1D array with binned j axis.
        jaxis (float): 1D array with full resolution j axis.
        
    Returns:
        float: 2D array with full resolution data.
    """
       
    # get distance radius for j binned intervals
    jrad = np.diff(jaxisbnd[0:2])*.5
    
    # create full resolution array & fill with binned data, if match intervals
    data = np.zeros((databnd.shape[0], len(jaxis)))*np.nan            
    for j, jval in enumerate(jaxisbnd):        
        jdx = np.where((jval-jrad<=jaxis) & (jaxis<jval+jrad))[0]
        if jdx.size > 0:
            data[:, jdx[0]:jdx[-1]+1] = databnd[:,j]
                    
    return data

def binvback(databnd, iaxisbnd, iaxis):
    """
    Turn vertically binned data back to full resolution, according to the 
    original full resolution i axis. Array elements will be left as nan, if i
    coordinates are not found within the binned data.
    
    Args:
        databnd (float): 2D array with binned data.
        iaxisbnd (float): 1D array with binned i axis.
        iaxis (float): 1D array with full resolution i axis.
        
    Returns:
        float: 2D numpy array with full resolution data.
    """
       
    # get distance radius for i and j binned intervals
    irad = np.diff(iaxisbnd[0:2])*.5
    
    # create full resolution array & fill with binned data, if match intervals
    data = np.zeros((len(iaxis), databnd.shape[1]))*np.nan            
    for i, ival in enumerate(iaxisbnd):
        idx = np.where((ival-irad<=iaxis) & (iaxis<ival+irad))[0]
        if idx.size > 0:
            data[idx[0]:idx[-1]+1, :] = databnd[i,:]
                    
    return data

def get_intervals(array, step, error=5):
    """
    Given a 1D incrementing numpy array, it returns interval indexes and 
    their middle-range values as requested by the user. It works with normal
    numpy arrays and datetime numpy arrays.
    
    Example:        
        1) Get indexes and middle-range values every 10 seconds for the time
         sequence t = np.array(['2009-12-15T12:38:00', '2009-12-15T12:38:02',
                                '2009-12-15T12:38:04', '2009-12-15T12:38:08',
                                '2009-12-15T12:38:10', '2009-12-15T12:38:20',
                                '2009-12-15T12:38:30'], dtype='datetime64[ms]')    
            
            >>> get_intervals(t, np.timedelta64(10, 's'))
            ([0, 4, 5], array(['2009-12-15T12:38:05','2009-12-15T12:38:15']))
        
        2) Get indexes and middle-range values every 3 meters for the 
        range depth array r = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            
            >>> get_intervals(r, 3)
            ([0, 3, 6], array([1.5, 4.5]))

    Note:
        By default, indexes will be computed as long as differences between the
        elements of the array and the requested intervals are below 5%. 
        Otherwise, it raises an error. If the array is numpy datetime64 format,
        the interval must be timedelta64 format.
        
    Args:
        array (float): incrementing numpy array to be segmented in intervals
        interval (float): interval width or array with desired intervals
        error (int): difference error admitted for computation (default=5%)
    
    Returns:
        int: list with interval indexes
        float: 1D numpy array with middle-range interval values
    """

    # if array and step are datetime format, convert to float
    if 'datetime64' in str(array[0].dtype):
        arraytype= [str(array[0].dtype), array[0].dtype]
        step = np.float64(step.astype((array[0]-array[0]).dtype))
        array = np.float64(array)
    else:
        arraytype= [str(array[0].dtype), array[0].dtype]        
        
    # use step to create an array of intervals       
    intervals = np.arange(array[0], array[-1]+.00001*step, step)
            
    # get interval indexes
    idx = []    
    for i in range(len(intervals)):   
        if (min(abs(array-intervals[i]))/step<error/100) & (len(intervals)>1):        
            idx.append(int(np.argmin(abs(array - intervals[i]))))            
        else:
            raise Exception('unable to get accurate intervals: ' + 
                            'tweak step argument to fit array\'s resolution')
    
    # get mid-range values
    val = np.float64(intervals)[1:] - (intervals[1]-intervals[0])/2
    
    # turn values back to datetime if that was the original format
    if 'datetime64' in arraytype[0]:
        val = val.astype(arraytype[1])
         
    return idx, val
    
def tolin(variable):
    """
    Turn variable into the linear domain.     
    Args:
        variable (float): array of elements to be transformed.    
    Returns:
        float:array of elements transformed
    """    
    return 10**(variable/10)

def tolog(variable):
    """
    Turn variable into the logarithmic domain. Negative elements are masked
    beforehand to avoid infinite values in return.    
    Args:
        variable (float): array of elements to be transformed.    
    Returns:
        float: array of elements transformed
    """    
    variable = np.ma.masked_less_equal(variable, 0)
    variable[variable.mask] = np.nan    
    return 10*np.log10(variable.data)

def fill_nans(array, ws=(1,1)):
    """
    Fill nan elements in an array with the median of surrounding values.
    
    Args:
        array (float): array containing nan values.
        ws (tuple): window's vertical and horizontal size.
    Returns:
        arrayout (float): array after infilling the nan values
    """
    
    # declare array output
    arrayout = array.copy()
    
    # locate nan values indexes
    idx, jdx = np.where(np.isnan(array))
    
    # work out window's indexes and calculate window's median
    for i, j in zip(idx, jdx):        
        
        i0 = i-ws[0]
        i1 = i+ws[0]
        j0 = j-ws[1]
        j1 = j+ws[1]
        
        if i0<0            : i0 = 0                    
        if i1>len(array)   : i1 = len(array)                    
        if j0<0            : j0 = 0                    
        if j1>len(array[0]): j1 = len(array[0])
        
        arrayout[i, j] = np.nanmedian(array[i0:i1, j0:j1])
        
    return arrayout