#!/usr/bin/env python3
"""
Display Sv together with along-ship (theta) and athwart-ship (phi) angles.
"""

__authors__ = ['Alejandro Ariza'
               ]
 
# import modules
import numpy as np
import matplotlib.pyplot as plt
import os
from echolab2.instruments import EK60
from echopy.plotting.cmaps import cmaps

# =============================================================================
# load raw file
# =============================================================================
print('Loading raw file...')
rawfile = os.path.abspath('../data/JR230-D20091215-T121917.raw')
ek60 = EK60.EK60()
ek60.read_raw(rawfile)            
 
# =============================================================================
# read 38 kHz data (Sv, angles, time and range)
# =============================================================================
print('Reading 38 kHz data...')
raw38 = ek60.get_raw_data(channel_number = 1)
Sv38 = np.transpose(raw38.get_Sv().data)
theta38 = np.transpose(raw38.angles_alongship_e)
phi38 = np.transpose(raw38.angles_athwartship_e)
t38   = raw38.get_Sv().ping_time
r38   = raw38.get_Sv().range

# =============================================================================
# plot Sv, theta and phi
# =============================================================================
print('Displaying results...')
plt.figure(figsize = (8, 6))
c = cmaps()

plt.subplot(311)
plt.pcolormesh(t38, r38, Sv38, vmin = -80, vmax = -50, cmap = c.ek500)
plt.gca().invert_yaxis()
plt.tick_params(labelbottom = False)
plt.colorbar().set_label('dB')
plt.title('Sv 38 kHz')

plt.subplot(312)
plt.pcolormesh(t38, r38, theta38, cmap = 'coolwarm')
plt.gca().invert_yaxis()
plt.tick_params(labelbottom = False)
plt.colorbar().set_label('degrees')
plt.title('Along-ship angle')
plt.ylabel('Depth (m)')

plt.subplot(313)
plt.pcolormesh(t38, r38, phi38, cmap = 'coolwarm')
plt.gca().invert_yaxis()
plt.colorbar().set_label('degrees')
plt.title('Athwart-ship angle')
plt.xlabel('Time (dd HH:MM)')

plt.show()
#plt.savefig('display_Sv_and_angles.png', dpi = 150)