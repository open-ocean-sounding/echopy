#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example getting cumulated distance and speed from longitude and latitude
positions.

Created on Fri Jul 12 12:30:55 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

#------------------------------------------------------------------------------
# import modules
import os
import matplotlib.pyplot as plt
from echolab2.instruments import EK60
from echopy import transform as tf

#------------------------------------------------------------------------------
# load rawfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 120 kHz data
raw38 = ek60.get_raw_data(channel_number=1)
t38   = ek60.nmea_data.interpolate(raw38.get_Sv(), 'GGA')['ping_time']
lon38 = ek60.nmea_data.interpolate(raw38.get_Sv(), 'GGA')['longitude']
lat38 = ek60.nmea_data.interpolate(raw38.get_Sv(), 'GGA')['latitude']

dis38    = tf.pos2dis(lon38, lat38, units='nm')
knots38 = tf.dis2speed(t38, dis38)

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(6,4))

# Distance
plt.subplot(211)
plt.plot(t38, dis38, 'g')
plt.tick_params(labelbottom=False)
plt.ylabel('Distance (nmi)')

# Speed
plt.subplot(212)
plt.plot(t38, knots38, 'r')
plt.ylabel('Speed (Knots)')
plt.xlabel('Time (dd HH:MM)')

# Show and save
plt.tight_layout()
plt.show()
plt.savefig('getting_distance_and_speed.png', figsize=(6,4), dpi=150)