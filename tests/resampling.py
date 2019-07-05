#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example resampling acoustic data.

Created on Thu Jul  4 16:27:55 2019
@author: Alejandro Ariza, British Antarctic Survey
"""
#------------------------------------------------------------------------------
# import modules
import os
import numpy as np
from echolab2.instruments import EK60
from echopy import resample as rs
import matplotlib.pyplot as plt
from echopy.cmaps import cmaps

#------------------------------------------------------------------------------
# load rawfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 38 kHz data
raw120   = ek60.get_raw_data(channel_number=1)
Sv120    = np.transpose(raw120.get_Sv().data)
theta120 = np.transpose(raw120.angles_alongship_e)
t120     = raw120.get_Sv().ping_time
r120     = raw120.get_Sv().range

#------------------------------------------------------------------------------
# Resample Sv in 2D, every 10 m in range and 100 seconds in time.
r120rs              = np.arange(r120[0], r120[-1], 10)
t120rs              = np.arange(t120[0], t120[-1], np.timedelta64(100, 's'))
Sv120rs, Sv120rsper = rs.twod(Sv120, r120, t120, r120rs, t120rs, log=True)

# By typing "log=True" the algorith knows that the variable to be resampled
# is in logarithmic scale (Sv) and need to be converted into linear before
# resampling and back to logarithmic after the resampling.

# "Sv120rs" is the resampled array, and "Sv120per" is the percentage of valid
# samples used in each resampling bin. It's a proxy of data quality.

#------------------------------------------------------------------------------
# Resample theta in 1D, every 10 m in the vertical.
r120rs                    = np.arange(r120[0], r120[-1], 10)
theta120rs, theta120rsper = rs.oned(theta120, r120, r120rs, log=False)

# This time we set "log=False" becaue theta is already at linear scale.
# "log=False" is the default, so might skip this setting. 

#------------------------------------------------------------------------------
# Resample Sv and theta back to full resolution.
Sv120rsf   , m120rsf_ = rs.full(Sv120rs   , r120rs, t120rs, r120, t120)
theta120rsf, m120rsf_ = rs.full(theta120rs, r120rs, t120  , r120, t120)

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(8,6))

plt.subplot(221).invert_yaxis()
plt.pcolormesh(t120, r120, Sv120, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.tick_params(labelbottom=False)
plt.ylabel('Depth (m)')
plt.title('Sv')

plt.subplot(222).invert_yaxis()
plt.pcolormesh(t120, r120, theta120, cmap=cmaps().coolwarm)
plt.tick_params(labelbottom=False)
plt.tick_params(labelleft=False)
plt.title('Theta')

plt.subplot(223).invert_yaxis()
plt.pcolormesh(t120, r120, Sv120rsf, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.ylabel('Depth (m)')
plt.xlabel('Time (dd HH:MM)')
plt.title('Sv resampled')

plt.subplot(224).invert_yaxis()
plt.pcolormesh(t120, r120, theta120rsf, cmap=cmaps().coolwarm)
plt.tick_params(labelleft=False)
plt.xlabel('Time (dd HH:MM)')
plt.title('Theta resampled')

plt.show()
#plt.savefig('resampling.png', figsize=(8,6), dpi=150)