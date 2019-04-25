#!/usr/bin/env python3
"""
Example script that resample Sv data and display results.

Notes: Get files in ftp://ftp.bas.ac.uk/rapidkrill/ and allocate them in the
corresponding directory for this to work

Created on Mon Aug 20 17:57:49 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy import read_calibration as readCAL
from echopy import operations as op
from echopy.cmaps import cmaps

# =============================================================================
# load raw file
# =============================================================================
print('Loading raw file...')
rawfile = os.path.abspath('../data/JR161-D20061127-T144557.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

# =============================================================================
# get calibration parameters
# =============================================================================
print('Getting calibration parameters...')
calfile = os.path.abspath('../data/JR161_metadata.toml')
params  = readCAL.ices(calfile, 38)

# =============================================================================
# read 38 kHz calibrated data
# =============================================================================
print('Reading 38 kHz calibrated data...')
raw38 = ek60.get_raw_data(channel_number = 1)
Sv38  = np.transpose(raw38.get_Sv(calibration = params).data)
t38   = raw38.get_Sv(calibration = params).ping_time
r38   = raw38.get_Sv(calibration = params).range

# =============================================================================
# Bin and bin back Sv
# =============================================================================
print('Resampling down and resampling back...')
m, n = 5, np.timedelta64(50, 's') # (metres, seconds)

Sv38bnd, r38bnd, t38bnd = op.bin2d(Sv38, r38, t38, m, n)[0:3]
Sv38bndback = op.bin2dback(Sv38bnd, r38bnd, t38bnd, r38, t38)

# =============================================================================
# plot results
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(311).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.title('Sv 38 kHz')
plt.tick_params(labelbottom = False)

plt.subplot(312).invert_yaxis()
plt.pcolormesh(t38bnd, r38bnd, Sv38bnd, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.title('Sv 38 kHz - resampled down')
plt.ylabel('Depth (m)')
plt.tick_params(labelbottom = False)

plt.subplot(313).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38bndback, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.title('Sv 38 kHz - resampled back')
plt.xlabel('Time (dd HH:MM)')

plt.show()
#plt.savefig('example_resample.png', dpi = 150)