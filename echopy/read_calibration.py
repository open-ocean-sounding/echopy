#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read calibration parameters from different calibration files and return them 
in a object, following PyEcholab's nomenclature
 
Created on Tue Jul 17 12:18:51 2018
@author: Alejandro Ariza, British Antarctic Survey 
"""
import numpy as np
import toml

def ices(calfile, frequency):
    """
    Read calibration parameters from a ICES metadata toml file 
    
    Args:
        calfile (str): path/to/calibration_file.
        frequency (int): frequency you want to read.
                
    Returns:
        object with calibration parameters 
    """
    class params(object):
        pass
    
    #find data_processing attributes for the requested frequency
    data = toml.load(calfile)   
    data = [x for x in data['data_processing'] if x['frequency'] == frequency][0]
    
    # populate params object with data_processing attributes
    params.frequency = np.float64(data['frequency']*1000) # Hz
    params.transmit_power = np.float64(data['transceiver_power']) # watts
    params.pulse_length = np.float64(data['transmit_pulse_length']/1000) # s
    params.gain =  np.float64(data['on_axis_gain']) # dB
    params.sa_correction = np.float64(data['Sacorrection']) # dB   
    params.absorption_coefficient = np.float64(data['absorption']) # dB m-1
    params.sound_velocity = np.float64(data['soundspeed']) # m s-1
    params.equivalent_beam_angle = np.float64(data['psi']) # dB
    params.angle_beam_athwartship = np.float64(data['beam_angle_major']) # deg
    params.angle_beam_alongship = np.float64(data['beam_angle_minor']) # deg
    params.angle_offset_athwartship = np.float64(data['beam_angle_offset_major']) # deg  
    params.angle_offset_alongship = np.float64(data['beam_angle_offset_minor']) # deg
        
    return params
 
def lobes(calfile, channel):
    """
    Read calibration parameters from a SIMRAD lobes calibration file
    
    Args:
        calfile (str): path/to/calibration_file.
        channel (int): channel you want to read.
                
    Returns:
        object with calibration parameters
    """
    # TODO: 

def echoview(calfile, channel):
    """
    Read calibration parameters from an echoview calibration file
    
    Args:
        calfile (str): path/to/calibration_file.
        channel (int): channel you want to read.
                
    Returns:
        object with calibration parameters
    """
    
    # create object to populate parameters
    class params(object):
        pass
    
    # open calibration file
    f = open(calfile, 'r')
    
    # read all lines in the file
    line = ' '    
    while line:       
        line = f.readline()
        
        # look for parameters after finding the requested transducer channel
        if line == 'SourceCal T' + str(channel) + '\n':           
            while line:                
                line = f.readline()
                
                if line != '\n':
                    
                    if line.split()[0] == 'AbsorptionCoefficient':        
                        params.absorption_coefficient = np.float64(line.split()[2]) # dB s-1
                        
                    if line.split()[0] == 'EK60SaCorrection':        
                        params.sa_correction = np.float64(line.split()[2]) # dB
                        
                    if line.split()[0] == 'Ek60TransducerGain':        
                        params.gain = np.float64(line.split()[2]) # dB
                        
                    if line.split()[0] == 'Frequency':        
                        params.frequency = np.float64(line.split()[2])*1000 # Hz
                        
                    if line.split()[0] == 'MajorAxis3dbBeamAngle':        
                        params.angle_beam_athwartship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MajorAxisAngleOffset':        
                        params.angle_offset_athwartship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MajorAxisAngleSensitivity':        
                        params.angle_sensitivity_athwartship = np.float64(line.split()[2]) #
                        
                    if line.split()[0] == 'MinorAxis3dbBeamAngle':        
                        params.angle_beam_alongship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MinorAxisAngleOffset':        
                        params.angle_offset_alongship = np.float64(line.split()[2]) # deg
                        
                    if line.split()[0] == 'MinorAxisAngleSensitivity':        
                        params.angle_sensitivity_alongship = np.float64(line.split()[2]) #
                        
                    if line.split()[0] == 'SoundSpeed':        
                        params.sound_velocity = np.float64(line.split()[2]) # m s-1
                        
                    if line.split()[0] == 'TransmittedPower':        
                        params.transmit_power = np.float64(line.split()[2]) # watts
                        
                    if line.split()[0] == 'TransmittedPulseLength':        
                        params.pulse_length = np.float64(line.split()[2])/1000 # s
                        
                    if line.split()[0] == 'TvgRangeCorrection':        
                        params.tvg_range_correction = line.split()[2] # str
                        
                    if line.split()[0] == 'TwoWayBeamAngle':        
                        params.equivalent_beam_angle = np.float64(line.split()[2]) # dB
                
                # break when empty line accours 
                else:                    
                    break
            
            # stop reading    
            break
    
    # return parameters    
    return params

def other():
    """    
    Note to contributors:
        Further calibration file readers must be named with the file's
        name or format.
        
        Please, check /DESIGN.md to adhere to our coding style.
    """