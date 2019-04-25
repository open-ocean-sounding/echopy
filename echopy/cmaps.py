#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load in-house colormaps

Created on Thu Aug 16 20:25:59 2018
@author: Alejandro Ariza, British Antarctic Survey
"""


class cmaps():
    
    def __init__(self):
        
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        
        # simrad EK500 default colormap
        ek500 = [(1.0000, 1.0000, 1.0000), # white
                 (0.6235, 0.6235, 0.6235), # light grey
                 (0.3725, 0.3725, 0.3725), # dark grey
                 (0.0000, 0.0000, 1.0000), # dark blue
                 (0.0000, 0.0000, 0.5000), # light blue
                 (0.0000, 0.7490, 0.0000), # light green
                 (0.0000, 0.5000, 0.0000), # dark green
                 (1.0000, 1.0000, 0.0000), # yellow
                 (1.0000, 0.5000, 0.0000), # orange
                 (1.0000, 0.0000, 0.7490), # pink
                 (1.0000, 0.0000, 0.0000), # red
                 (0.6509, 0.3255, 0.2353), # light brown
                 (0.4705, 0.2353, 0.1568)] # dark brown
        self.ek500 = colors.LinearSegmentedColormap.from_list('ek500', ek500)
        self.ek500.set_bad(color = 'k', alpha = 1)
        
        # coolwarm colormap with nan values set to black
        self.cw = cm.coolwarm
        self.cw.set_bad(color = 'k', alpha = 1)