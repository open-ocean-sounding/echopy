#!/usr/bin/env python3
"""
Filters for masking stationary or near stationary data based on vessel speed

"""

def mask_stationary(Sv, speed, threshold):
    """
    Mask stationary or near stationary data based on vessel speed
    
    Args:
        Sv (float): 2D numpy array with Sv data to be masked (dB)
        speed (float): 1D numpy array with vessel speed data (knots)
        threshold (int): speed below which Sv data will be masked (knots)
        
    Returns:
            bool: 2D numpy array mask (stationary = True)
            float: 2D numpy array with Sv data masked with NAN values
    """
    
    print('TODO')
    # TODO: need to implement distance and speed retrieval in PyEchoLab 
    #       which seems not to be working yet?