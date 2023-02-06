#!/usr/bin/env python3
"""
Example script to classify acoustic seascape using functional data analysis.
"""

__authors__ = ['Alejandro Ariza',
               'Jérémie Habasque',
               'Etienne Pauthenet'
               ] 

# import modules
from os.path import dirname, join, exists
import requests, netCDF4
from echopy.analysis.fda import get_fdo,get_fpca,get_fvar,get_fmean,reduce_fdo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Download acoustic data from www.seanoe.org 
filename = '70042.nc'
url      = 'https://www.seanoe.org/data/00602/71379/data/%s'%filename
ncfile   = join(dirname(dirname(__file__)), 'data', filename) # run from script
if not exists(ncfile):
    print('Downloading netCDF file \'%s\' ...'%filename)
    r  = requests.get(url)
    with open(ncfile, 'wb') as f:
        f.write(r.content)
    print('Saved in \'%s\''%ncfile)

# Read Sv, range, frequency, latitude, and longitude data    
nc  = netCDF4.Dataset(ncfile, 'r')
Sv  = nc['Sv'       ][:].data.transpose(2,1,0)
r   = nc['depth'    ][:].data
f   = nc['channel'  ][:].data
lat = nc['latitude' ][:].data
lon = nc['longitude'][:].data

# Keep only 18 and 38 kHz and only down to 800 m depth
Sv = Sv[:,:,(f==18)|(f==38)][r<800,:,:]
f  = f [    (f==18)|(f==38)]
r  = r[r<800]

# Keep only full pings (no NANs)
fullping18 = ~np.isnan(Sv[:,:,f==18]).any(axis=0).flatten()
fullping38 = ~np.isnan(Sv[:,:,f==38]).any(axis=0).flatten()
Sv         = Sv [:,fullping18&fullping38,:]
lat        = lat[  fullping18&fullping38  ]
lon        = lon[  fullping18&fullping38  ]

# Create functional data object using two variables, 18 and 38 kHz, 
# from 8 to 798 m. Both profiles will be approximated to a function system
# of 20 b-splines of order 4  
var   = {'18kHz': ( 18,  8, 798), '38kHz': ( 38,  8, 798)}
fdo   = get_fdo(Sv, r, 20, order=4, f=f, var=var)

# Get principal components through functional PCA
fpca  = get_fpca(fdo)
pcs   = fpca['pc']

# Plot the percentage of variance contained within each PC 
pvar = fpca['pval']
plt.subplot(321)
x = np.arange(len(pvar))+1
plt.bar(x, pvar           , color='k'   , label='Specific'   , zorder=1)
plt.bar(x, np.cumsum(pvar), color='grey', label='Accumulated', zorder=0)
plt.legend()
plt.xlabel('Principal componets')
plt.ylabel('Explained variance (%)')

# Classify the acoustic seascape, using the 5 first PCs (>70% of 18 and 38 kHz 
# variance). We will create 2 clusters using K-means and they will be projected
# in the principal components space (1st and 2nd PCs in X and Y axes).  
kmeans = KMeans(n_clusters=2)
kmeans.fit(pcs[:, :5]) 
clusters = kmeans.labels_
plt.subplot(322)
colormap = mpl.colors.ListedColormap(['b','r'])
plt.scatter(pcs[:, 0], pcs[:, 1], c=clusters, cmap=colormap)
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# let's have a look how the classification looks like in the geographical space
plt.subplot(323)
plt.scatter(lon, lat, c=clusters, cmap=colormap)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# now let's see the classification along the echogram
plt.subplot(324).invert_yaxis()
Sv18 = Sv[:,:,0]
pings = np.arange(Sv18.shape[1])
plt.pcolormesh(pings, r, Sv18, vmin=-80, vmax=-50, cmap='Greys')
plt.xlabel('Ping number')
plt.ylabel('Depth (m)')
plt.scatter(pings, pings*0, c=clusters, cmap=colormap)


# as expected, the classification captures mainly the nocturnal and diurnal
# acoustic seascapes. However, we may ask how acoustic profiles change between 
# day and night. We can get a glance of it by looking at the profile variance 
# contained at each principal component, let's see the first PC, in a profile 
# of 5 m vertical resolution. We will plot the negative and positive deviations 
# of the 18 kHz mean profile.     
fvar   = get_fvar(fdo, fpca, 0, 5)
fvar18 = fvar['18kHz']
plt.subplot(325).invert_yaxis() 
plt.plot(fvar18['ymean' ], fvar18['x'], '-k', label='18 kHz mean profile',lw=2)
plt.plot(fvar18['yminus'], fvar18['x'], '_k', label='Negative deviation')
plt.plot(fvar18['yplus' ], fvar18['x'], '+k', label='Positive deviation')
plt.legend()
plt.xlabel('Sv (dB)')
plt.ylabel('Depth (m)')

# Looking at the deviation of the 18 kHz mean profile, observations with 
# negative and positive values in the 1st PC dimension seems to correspond 
# to day and night profiles, respectively. Let's extract the average profile,
# with 5 meters resolution, representing each cluster to confirm this 
# hypothesis. To avoid noise from other minor sources of variance, we will use 
# a functional data object with the variability reduced to the five first PCs:
fdor_cluster0 = reduce_fdo(fdo, fpca, pcs=[0,1,2,3,4], profiles=clusters==0)
fdor_cluster1 = reduce_fdo(fdo, fpca, pcs=[0,1,2,3,4], profiles=clusters==1)
fmeanc0       = get_fmean(fdor_cluster0, 5)
fmeanc1       = get_fmean(fdor_cluster1, 5)
fmean18c0     = fmeanc0['18kHz']
fmean18c1     = fmeanc1['18kHz']
plt.subplot(326).invert_yaxis() 
plt.plot(fmean18c0['ymean' ], fmean18c0['x'],'-', color='b',
         lw=3, label='Night 18 kHz mean profile')
plt.plot(fmean18c0['yminus'], fmean18c0['x'],':', color='b')
plt.plot(fmean18c0['yplus' ], fmean18c0['x'],':', color='b')
plt.plot(fmean18c1['ymean' ], fmean18c1['x'],'-', color='r',
         lw=3, label='Day 18 kHz mean profile')
plt.plot(fmean18c1['yminus'], fmean18c1['x'],':', color='r')
plt.plot(fmean18c1['yplus' ], fmean18c1['x'],':', color='r')
plt.xlabel('Sv (dB)')
plt.ylabel('Depth (m)')
plt.legend()