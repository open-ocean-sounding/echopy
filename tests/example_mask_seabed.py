#!/usr/bin/env python3
"""
Example script that mask seabed

Notes: Get files in ftp://ftp.bas.ac.uk/rapidkrill/ and allocate them in the
corresponding directory for this to work

Created on Thu Aug  2 17:16:19 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy import operations as op #TODO: deprecated, change to resample()
from echopy.processing import mask_seabed as  maskSB
from echopy.plotting.cmaps import cmaps

# =============================================================================
# load raw file
# =============================================================================
print('Loading raw file...')
rawfile = os.path.abspath('../data/JR260_-D20120111-T045121.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

# =============================================================================
# read 38 kHz data
# =============================================================================
print('Reading 38 kHz data...')
raw38 = ek60.get_raw_data(channel_number = 1)
Sv38  = np.transpose(raw38.get_Sv().data)
theta38 = np.transpose(raw38.angles_alongship_e)
phi38 = np.transpose(raw38.angles_athwartship_e)
t38   = raw38.get_Sv().ping_time
r38   = raw38.get_Sv().range

# =============================================================================
# smooth out Sv horizontally to facilitate seabed masking
# =============================================================================
print('Resampling down...')
m, n = 2, 4 # (m, npings)

p38 = np.arange(len(Sv38[0])) 
Sv38bnd, r38bnd, p38bnd = op.bin2d(Sv38, r38, p38, m, n)[0:3]
Sv38smooth = op.bin2dback(Sv38bnd, r38bnd, p38bnd, r38, p38)

# =============================================================================
# Seabed masking
# =============================================================================
print('Masking seabed...')
r0, r1, roff, thr = 10, 1000, 10, (-40, -60)  # (m, m, m, dB)
mask38sb = maskSB.maxSv(Sv38smooth, r38, r0=r0, r1=r1, roff=roff, thr=thr)
Sv38sboff = Sv38.copy()
Sv38sboff[mask38sb] = np.nan

# =============================================================================
# plot results
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(211).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.ylabel('Depth (m)')
plt.tick_params(labelbottom = False)
plt.title('Sv 38 kHz')

plt.subplot(212).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38sboff, vmin = -80, vmax = -50, cmap = c.ek500)
plt.colorbar().set_label('dB')
plt.ylabel('Depth (m)')
plt.xlabel('Time (dd HH:MM)')
plt.title('Sv 38 kHz - seabed masked')

plt.show()
#plt.savefig('example_mask_seabed.png', dpi = 150)