#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example integrating Sv to sa (m^2 m^-2) and NASC (m^2 nmi^-2).

Created on Sun Jul  7 16:15:32 2019
@author: Alejandro Ariza, British Antarctic Survey
"""

#------------------------------------------------------------------------------
# import modules
import os
import numpy as np
import matplotlib.pyplot as plt
from echolab2.instruments import EK60
from echopy.processing import mask_impulse as mIN
from echopy.utils import transform as tf
from echopy.plotting.cmaps import cmaps

#------------------------------------------------------------------------------
# load rawfile
path     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
rawfile  = os.path.join(path, 'JR230-D20091215-T121917.raw')
ek60     = EK60.EK60()
ek60.read_raw(rawfile)

#------------------------------------------------------------------------------
# get 38 kHz data
raw38  = ek60.get_raw_data(channel_number=1)
Sv38  = np.transpose(raw38.get_Sv().data)   # Sv array
t38    = raw38.get_Sv().ping_time           # time array
r38    = raw38.get_Sv().range               # range array

#------------------------------------------------------------------------------
# remove impulse noise
m38in            = mIN.ryan(Sv38, r38, m=5, n=1, thr=10)[0]
Sv38inoff        = Sv38.copy()
Sv38inoff[m38in] = np.nan

# We remove noise so we can see how integration deals with missing samples.

#------------------------------------------------------------------------------
# integrate area scattering coefficient (sa), from 20 to 250 metres
sa  , saper   = tf.Sv2sa  (Sv38inoff, r38, 20, 250, method='mean')

# note the mean method is used.

# the second output is the percentange vertical samples integrated behind every
# computation of sa.

#------------------------------------------------------------------------------
# integrate Nautical Area Scattering Coefficient (NASC), from 20 to 250 metres
NASC, NASCper = tf.Sv2NASC(Sv38inoff, r38, 20, 250, method='sum' )

# note the sum method is used.

# the second output is the percentange vertical samples integrated behind every
# computation of NASC.

#------------------------------------------------------------------------------
# Figures
plt.figure(figsize=(8,6))

# Sv with impulse noise removed
plt.subplot(311).invert_yaxis()
plt.pcolormesh(t38, r38, Sv38inoff, vmin=-80, vmax=-50, cmap=cmaps().ek500)
plt.tick_params(labelbottom=False)
plt.ylabel('Depth (m)')
plt.title('Sv with impulse noise removed')

# integrated sa
ax =plt.subplot(312)
ax = [ax, ax.twinx()]
ax[0].plot(t38, sa,'-r')
ax[0].set_xlim(t38[0], t38[-1])
ax[0].tick_params(axis='y', colors='r')
ax[0].yaxis.tick_left()
ax[0].yaxis.set_label_position("left")
ax[0].set_ylabel('s$_a$ (m$^2$ m$^{-2}$)', color='r')
ax[0].tick_params(labelbottom=False)
ax[1].plot(t38, saper,'-b')
ax[1].set_xlim(t38[0], t38[-1])
ax[1].set_ylim(0, 100)
ax[1].tick_params(axis='y', colors='b')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].set_ylabel('% integrated samples', color='b')
ax[1].tick_params(labelbottom=False)
ax[1].set_title('Area scattering coefficient, integrated from 20 to 250 m')

# integrated NASC
ax =plt.subplot(313)
ax = [ax, ax.twinx()]
ax[0].plot(t38, NASC,'-r')
ax[0].set_xlim(t38[0], t38[-1])
ax[0].tick_params(axis='y', colors='r')
ax[0].yaxis.tick_left()
ax[0].yaxis.set_label_position("left")
ax[0].set_ylabel('NASC (m$^2$ nmi$^{-2}$)', color='r')
ax[1].plot(t38, NASCper,'-b')
ax[1].set_xlim(t38[0], t38[-1])
ax[1].set_ylim(0, 100)
ax[1].tick_params(axis='y', colors='b')
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[1].set_ylabel('% integrated samples', color='b')
ax[1].set_title('Nautical area scattering coefficient, integrated from 20 to 250 m')

# Show and save
plt.tight_layout()
plt.show()
plt.savefig('integrating_sa_and_NASC.png', figsize=(8,4), dpi=150)