#!/usr/bin/env python3
"""
Example script that read, calibrate and display acoustic data

Notes: Get files in ftp://ftp.bas.ac.uk/rapidkrill/ and allocate them in the
corresponding directory for this to work

Created on Mon Jul 16 13:11:44 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy.reading import read_calibration as readCAL 
from echopy.plotting.cmaps import cmaps

# =============================================================================
# load raw file
# =============================================================================
print('Loading raw file...')
rawfile = os.path.abspath('../data/JR230-D20091215-T121917.raw')
ek60 = EK60.EK60()
ek60.read_raw(rawfile)

# =============================================================================
# get calibration parameters
# =============================================================================
print('Getting calibration parameters...')
calfile = os.path.abspath('../data/JR230_metadata.toml')
params = readCAL.ices(calfile, 38)

# =============================================================================
# read 38 kHz non-calibrated data
# =============================================================================
print('Reading 38 kHz non-calibrated data...')
raw38 = ek60.get_raw_data(channel_number = 1)
Sv38  = np.transpose(raw38.get_Sv().data)
t38   = raw38.get_Sv().ping_time
r38   = raw38.get_Sv().range

# =============================================================================
# read 38 kHz calibrated data
# =============================================================================
print('Reading 38 kHz calibrated data...')
raw38   = ek60.get_raw_data(channel_number = 1)
Sv38toml = np.transpose(raw38.get_Sv(calibration = params).data)
t38toml  = raw38.get_Sv(calibration = params).ping_time
r38toml  = raw38.get_Sv(calibration = params).range

# =============================================================================
# plot non-calibrated and calibrated data
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(211).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('Sv 38 kHz (dB)')
plt.title('Calibration NOT applied')
plt.ylabel('Depth (m)')
plt.tick_params(labelbottom = False)

plt.subplot(212).invert_yaxis()
plt.pcolormesh(t38toml, r38toml, Sv38toml, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('Sv 38 kHz (dB)')
plt.title('Calibration applied')
plt.ylabel('Depth (m)')
plt.xlabel('Time (dd HH:MM)')

plt.show()
#plt.savefig('example_calibrate.png', dpi = 150)