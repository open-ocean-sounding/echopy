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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Download acoustic data from www.seanoe.org 
filename = '70042.nc'
url      = 'https://www.seanoe.org/data/00602/71379/data/%s'%filename
ncfile   = join(dirname(dirname(__file__)), 'data', filename)
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
Sv  = Sv[:,:,(f==18)|(f==38)][r<800,:,:]
f   = f [    (f==18)|(f==38)]
r   = r[r<800]

# Keep only full pings (no NANs)
fullping18 = ~np.isnan(Sv[:,:,f==18]).any(axis=0).flatten()
fullping38 = ~np.isnan(Sv[:,:,f==38]).any(axis=0).flatten()
Sv  = Sv[:,fullping18&fullping38,:]
lat = lat[fullping18&fullping38]
lon = lon[fullping18&fullping38]

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
plt.bar(np.arange(len(pvar))+1, pvar           , label='Specific'   , zorder=1)
plt.bar(np.arange(len(pvar))+1, np.cumsum(pvar), label='Accumulated', zorder=0)
plt.legend()
plt.xlabel('Principal componets')
plt.ylabel('Explained variance (%)')

# Classify the acoustic seascape, using the 5 first PCs (>70% of 18 and 38 kHz 
# variance). We will create 3 clusters using K-means and they will be projected
# in the principal components space (1st and 2nd PCs in X and Y axes).  
kmeans = KMeans(n_clusters=3)
kmeans.fit(pcs[:, :5]) 
clusters = kmeans.labels_
plt.subplot(322)
plt.scatter(pcs[:, 0], pcs[:, 1], c=clusters)
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# let's have a look how the classification looks like in the geographical space
plt.subplot(323)
plt.scatter(lon, lat, c=clusters)
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# now let's see the classification along the echogram
plt.subplot(324).invert_yaxis()
Sv18 = Sv[:,:,0]
pings = np.arange(Sv18.shape[1])
plt.pcolormesh(pings, r, Sv18, vmin=-80, vmax=-50, cmap='Greys')
plt.xlabel('Ping number')
plt.ylabel('Depth (m)')
plt.scatter(pings, pings*0, c=clusters)


# as expected, the classification captures mainly the nocturnal and diurnal
# acoustic seascapes, but how acoustic profiles changes between day and
# night? We can get a glance of it by looking at the profile variance contained
# at each principal component, let's see the first PC, in a profile of 5 m
# vertical resolution. We will plot the negative and positive deviations of 
# the 18 kHz mean profile.     
fvar = get_fvar(fdo, fpca, 0, 5)
plt.subplot(325).invert_yaxis() 
plt.plot(fvar['18kHz']['ymean' ],fvar['18kHz']['x'],label='Mean')
plt.plot(fvar['18kHz']['yminus'],fvar['18kHz']['x'],label='Negative deviation')
plt.plot(fvar['18kHz']['yplus' ],fvar['18kHz']['x'],label='Positive deviation')
plt.legend()

# Looking at the deviation of the mean 18 kHz profile, observations with 
# negative and positive values in the 1st PC dimension seems to correspond 
# to day and night profiles, respectively. Let's extract the average profile
# representing each cluster to confirm this hypothesis. To avoid noise from
# other minor sources of variance, with 5 meters resolution. We will do this, 
# using a functional data object with the variability reduced to the five
# first PCs:

fdor_cluster0 = reduce_fdo(fdo, fpca, pcs=[0,1,2,3,4], profiles=clusters==0)
fdor_cluster1 = reduce_fdo(fdo, fpca, pcs=[0,1,2,3,4], profiles=clusters==1)
fdor_cluster2 = reduce_fdo(fdo, fpca, pcs=[0,1,2,3,4], profiles=clusters==2)
fmean0  = get_fmean(fdor_cluster0, 5)
fmean1  = get_fmean(fdor_cluster1, 5)
fmean2  = get_fmean(fdor_cluster2, 5)
plt.subplot(326).invert_yaxis() 
plt.plot(fmean0['18kHz']['ymean' ],fmean0['18kHz']['x'],'-' , color='C0')
plt.plot(fmean0['18kHz']['yminus'],fmean0['18kHz']['x'],'--', color='C0')
plt.plot(fmean0['18kHz']['yplus' ],fmean0['18kHz']['x'],'--', color='C0')
plt.plot(fmean1['18kHz']['ymean' ],fmean1['18kHz']['x'],'-' , color='C1')
plt.plot(fmean1['18kHz']['yminus'],fmean1['18kHz']['x'],'--', color='C1')
plt.plot(fmean1['18kHz']['yplus' ],fmean1['18kHz']['x'],'--', color='C1')
plt.plot(fmean2['18kHz']['ymean' ],fmean2['18kHz']['x'],'-' , color='C2')
plt.plot(fmean2['18kHz']['yminus'],fmean2['18kHz']['x'],'--', color='C2')
plt.plot(fmean2['18kHz']['yplus' ],fmean2['18kHz']['x'],'--', color='C2')