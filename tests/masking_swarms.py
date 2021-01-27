#!/usr/bin/env python3
"""
Example masking krill swarms.
"""

__authors__ = ['Alejandro Ariza'
               ] 

#------------------------------------------------------------------------------
# import modules
import os
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from echolab2.instruments import EK60
from echopy.processing import mask_impulse as mIN
from echopy.processing import mask_range as mRG
from echopy.utils import transform as tf
from echopy.processing import mask_shoals as mSH
from echopy.plotting.cmaps import cmaps

#------------------------------------------------------------------------------
# load rawfile
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60    = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 120 kHz data
raw120 = ek60.get_raw_data(channel_number=2)
Sv120  = np.transpose(raw120.get_Sv().data)
t120   = raw120.get_Sv().ping_time
r120   = raw120.get_Sv().range

#------------------------------------------------------------------------------
# Remove impulse noise and signal below/above the target of interest using
# Wang's algorithm.
Sv120clean = mIN.wang(Sv120, thr=(-70,-40), erode=[(3,3)],
                      dilate=[(7,7)], median=[(7,7)])[0]

#------------------------------------------------------------------------------
# Remove data outside the range 20-250 m 
m120rg             = mRG.outside(Sv120clean, r120, 20, 250)
Sv120clean[m120rg] = np.nan

#------------------------------------------------------------------------------
# Convolute Sv prior to mask swarms, with 3x3 moving window
k = np.ones((3, 3))/3**2
Sv120cvv = tf.log(convolve2d(tf.lin(Sv120clean), k, 'same', boundary='symm'))

#------------------------------------------------------------------------------
# Mask swarms using Weill's algorithm
m120sh = mSH.weill(Sv120cvv, thr=-70, maxvgap=15, maxhgap=0,
                   minhlen= 3, minvlen=15)[0]

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(8,5))
plt.subplots_adjust(left=0.08, right=0.91, bottom=0.08, top=0.95, wspace=0.08)
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, .05])

# Sv original
plt.subplot(gs[0]).invert_yaxis()
im= plt.pcolormesh(t120, r120, Sv120, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.ylabel('Depth (m)')
plt.xlabel('Time (dd HH:MM)')
plt.title('Sv')

# Swarms mask
plt.subplot(gs[1]).invert_yaxis()
plt.pcolormesh(t120, r120, m120sh*1, cmap='Greys')
plt.tick_params(labelleft=False)
plt.xlabel('Time (dd HH:MM)')
plt.title('Swarms')

# colorbar
ax=plt.subplot(gs[2])
plt.colorbar(im, cax=ax).set_label('dB re m$^{-1}$')

# Show and save
plt.show()
# plt.savefig('masking_swarms.png', figsize=(8,5), dpi=150)