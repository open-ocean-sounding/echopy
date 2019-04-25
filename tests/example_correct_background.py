#!/usr/bin/env python3
"""
Example script where background noise is corrected and
signal-to-noise ratio > 6 dB masked

Notes: Get files in ftp://ftp.bas.ac.uk/rapidkrill/ and allocate them in the
corresponding directory for this to work

Created on Thu Jul 19 20:29:49 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy import read_calibration as readCAL
from echopy import get_background as getBGN
from echopy import mask_signal2noise as maskS2N
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
params  = readCAL.ices(calfile, 120)

# =============================================================================
# read 120 kHz calibrated data
# =============================================================================
print('Reading 120 kHz calibrated data...')
raw120 = ek60.get_raw_data(channel_number = 2)
Sv120  = np.transpose(raw120.get_Sv(calibration = params).data)
t120   = raw120.get_Sv(calibration = params).ping_time
r120   = raw120.get_Sv(calibration = params).range

# =============================================================================
# estimate background noise
# =============================================================================
print('Estimating background noise...')
m, n, operation = 10, 20, 'percentile90' # (m, npings, str)   
bgn120 = getBGN.derobertis_mod(Sv120, r120,
                               raw120.absorption_coefficient[0],
                               m, n, operation = operation)

# =============================================================================
# clean background noise
# =============================================================================
print('Cleaning signal-to-noise < 6 dB...')    
threshold = 6 # (decibels)
mask120 = maskS2N.derobertis(Sv120, bgn120, threshold)
Sv120clean = op.tolog(op.tolin(Sv120) - op.tolin(bgn120))
Sv120clean[mask120] = -999
# =============================================================================
# plot results
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(221).invert_yaxis()
plt.pcolormesh(t120, r120, Sv120, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar()
plt.tick_params(labelbottom=False)
plt.title('Sv 120 kHz')
plt.ylabel('Depth (m)')

plt.subplot(223).invert_yaxis()
plt.pcolormesh(t120, r120, bgn120, cmap=c.ek500)
plt.colorbar()
plt.title('BGN')
plt.ylabel('Depth (m)')
plt.xlabel('Time (dd HH:MM)')

plt.subplot(222).invert_yaxis()
plt.pcolormesh(t120, r120, np.int64(mask120), cmap='Greys')
plt.colorbar()
plt.tick_params(labelbottom=False)
plt.title('S2N mask')

plt.subplot(224).invert_yaxis()
plt.pcolormesh(t120, r120, Sv120clean, vmin=-80, vmax=-50, cmap=c.ek500)
plt.colorbar()
plt.title('BGN corrected & S2N applied')
plt.xlabel('Time (dd HH:MM)')

plt.show()
#plt.savefig('example_correct_background.png', dpi = 150)