#!/usr/bin/env python3
"""
Example cleaning background noise.
"""

__authors__ = ['Alejandro Ariza'
               ]
 
#------------------------------------------------------------------------------
# import modules
import os
import numpy as np
import matplotlib.pyplot as plt
from echolab2.instruments import EK60
from echopy.utils import transform as tf
from echopy.processing import get_background as gBN
from echopy.processing import mask_signal2noise as mSN
from echopy.plotting.cmaps import cmaps

#------------------------------------------------------------------------------
# load rawfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 120 kHz data
raw120   = ek60.get_raw_data(channel_number=2)
alpha120 = raw120.absorption_coefficient[0]    # water absorption
Sv120    = np.transpose(raw120.get_Sv().data)  # Sv array
t120     = raw120.get_Sv().ping_time           # time array
r120     = raw120.get_Sv().range               # range array
p120     = np.arange(len(t120))                # ping array
s120     = np.arange(len(r120))                # sample array

#------------------------------------------------------------------------------
# get background noise estimation, resampling at 5x20 (samples x pings) bins
bn120, m120bn_ = gBN.derobertis(Sv120, s120, p120, 5, 20,
                                r120, alpha120, bgnmax=-125)

# Maximum background noise estimation is -125 dB

#------------------------------------------------------------------------------
# clean background noise from Sv
Sv120clean = tf.log(tf.lin(Sv120) - tf.lin(bn120))

# -----------------------------------------------------------------------------
# mask low signal-to-noise Sv samples
m120sn = mSN.derobertis(Sv120, bn120, thr=12)

# In this case, Sv samples 12 dB above the background noise estimation, or
# lower, will be masked. 

# -----------------------------------------------------------------------------
# use the mask to turn the Sv samples into "empty water" (-999)
Sv120clean[m120sn] = -999

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(8,7))

# Sv original
plt.subplot(311).invert_yaxis()
plt.pcolormesh(p120, r120, Sv120, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.tick_params(labelbottom=False)
plt.colorbar()
plt.title('Sv original')

# Background noise estimation
plt.subplot(312).invert_yaxis()
plt.pcolormesh(p120, r120, bn120, cmap=cmaps().ek500)
plt.tick_params(labelbottom=False)
plt.ylabel('Depth (m)')
plt.colorbar()
plt.title('Background noise')

# Sv after correcting for background noise
plt.subplot(313).invert_yaxis()
plt.pcolormesh(p120, r120, Sv120clean, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.colorbar()
plt.xlabel('Number of pings')
plt.title('Sv clean')

# Show and save
plt.tight_layout()
plt.show()
# plt.savefig('cleaning_background.png', figsize=(8,7), dpi=150)