#!/usr/bin/env python3
"""
Example script that reads Sv, mask attenuated signal, and display results.

Notes: Get files in ftp://ftp.bas.ac.uk/rapidkrill/ and allocate them in the
corresponding directory for this to work

Created on Thu Jul 19 17:39:00 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy import read_calibration as readCAL
from echopy import mask_attenuated as maskAS
from echopy.cmaps import cmaps

# =============================================================================
# load raw file
# =============================================================================
print('Loading raw file...')
rawfile = os.path.abspath('../data/JR161-D20061118-T010645.raw')
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
# mask attenuated signal
# =============================================================================
print('Masking attenuated signal...')
r0, r1, n, threshold = 180, 280, 30, -6 # (m, m, pings, dB)
mask38as = maskAS.ryan(Sv38, r38, r0, r1, n, threshold)
Sv38asoff = Sv38.copy()
Sv38asoff[mask38as[0]|mask38as[1]] = np.nan
# =============================================================================
# plot results
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(311).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.tick_params(labelbottom = False)
plt.title('Sv 38 kHz') 

plt.subplot(312).invert_yaxis()
plt.pcolormesh(t38, r38, np.int64(mask38as[0]|mask38as[1]), cmap = 'Greys')
plt.colorbar().set_label('boolean')
plt.tick_params(labelbottom = False)
plt.title('mask for attenuated signal')
plt.ylabel('Depth (m)')

plt.subplot(313).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38asoff, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.title('Sv 38 kHz - attenuated signal masked')
plt.xlabel('Time (dd HH:MM)')

plt.show()
#plt.savefig('example_mask_attenuated.png', dpi = 150)