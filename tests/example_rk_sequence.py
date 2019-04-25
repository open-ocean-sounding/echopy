#!/usr/bin/env python3
"""
Example script for running the rapidkrill processing and analysis routine in a
sequence of files.

Created on Fri Nov 30 09:15:11 2018
@author: Alejandro Ariza, British Antarctic Survey
"""

import os, io
from rapidkrill import rapidkrill as rk
import matplotlib.pyplot as plt

# file list
path     = os.path.abspath('../data/ek60/')
rawfiles = [path + '/JR179-D20080410-T143945.raw',
            path + '/JR179-D20080410-T145311.raw',
            path + '/JR179-D20080410-T150637.raw']
calfile  =  path + '/JR179_metadata.toml'

# loop through your file list
for i, rawfile in enumerate(rawfiles):        
        
        # Process and analyse 
        data    = rk.process(rawfile, calfile=calfile, sequence=True)
        nasc, t = rk.analyse(data['Sv38clean'], data['Sv120clean'],
                             data['t38']      , data['r38'       ],
                             sequence=True)        

        # display results            
        plt.close()
        rk.show(data)        
        
        # Print results in tab separated format
        table = io.StringIO()
        table.write(" %s\t\t\t\t%s\n" % ('Time:', 'NASC (m^2 nmi^-2):'))
        for t, nasc in zip(t, nasc):
            table.write(" %s\t%2.2f\n" % (t, nasc))
        table = table.getvalue()        
        print(table)