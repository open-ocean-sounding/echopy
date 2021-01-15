#!/usr/bin/env python3
"""
Write and read NMEA and configuration summary data from a list of RAW files.           

Copyright (c) 2020 Echopy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__authors__ = ['Alejandro Ariza'
               ]

# import modules
import os, glob, xarray
import warnings 
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
from geopy.distance import distance
from echolab2.instruments import EK60
import ephem

def write(rawdir, save=False, txtfile='filereport.txt',
           maxspeed=20, precision=8, bathymetry=None):
    """
    Write a report with NMEA and CONFIG information for each RAW file contained 
    in a directory.
    
    Args:
        rawdir   (str ): Directory were the RAW files are stored.
        save     (bool): Whether or not to save data as a text file in rawdir.
        txtfile  (str ): Name of text file created.
        maxspeed (int ): Max speed permited in knots. Above this, it will
                         be considered an error and set to NAN.
        precision (int): Decimal precision of data printed.
        bathyrmetry (str): path/to/bathymetry/GEBCO.nc (default: None)
        
    Returns:
        Pandas.DataFrame: Report with the following tabular data:
            * file  : Name of RAW file
            * y0    : start year (years after AD epoch)
            * y1    : end year (years after AD epoch)
            * d0    : start day (day of year)
            * d1    : end day (day of year)
            * h0    : start time (decimal hours)
            * h1    : end time (decimal hours)
            * sun0  : sun altitude at the start (degreses)
            * sun1  : sun altitude at the end (degrees)
            * seabed: seabed depth, estimated from bathymetry (m)
            * lonmin: longitude minimum (decimal degrees)
            * lonp25: longitude percentile 25 (decimal degrees)
            * lonmed: longitude median (decimal degrees)
            * lonp75: longitude percentile 75 (decimal degrees)
            * lonmax: longitude maximum (decimal degrees)
            * latmin: latitude minimum (decimal degrees)
            * latp25: latitude percentile 25 (decimal degrees)
            * latmed: latitude median (decimal degrees)
            * latp75: latitude percentile 75 (decimal degrees)
            * latmax: latitude maximum (decimal degrees)
            * kntmin: speed minimum (knots)
            * kntp25: speed percentile 25 (knots)
            * kntmed: speed median (knots)
            * kntp75: speed percentile 75 (knots)
            * kntmax: speed maximum (knots)
            * f18   : Presence of 18 kHz frequency (boolean)
            * p18   : Power at 18 kHz (w)
            * l18   : Pulse length at 18 kHz (m)
            * i18   : Sample interval at 18 kHz (m)
            * c18   : Sample count at 18 kHz)
            * f38   : Presence of 38 kHz frequency (boolean)
            * p38   : Power at 38 kHz (w)
            * l38   : Pulse length at 38 kHz (m)
            * i38   : Sample interval at 38 kHz (m)
            * c38   : Sample count at 38 kHz
            * f70   : Presence of 70 kHz frequency (boolean)
            * p70   : Power at 70 kHz (w)
            * l70   : Pulse length at 70 kHz (m)
            * i70   : Sample interval at 70 kHz (m)
            * c70   : Sample count at 70 kHz
            * f120  : Presence of 120 kHz frequency (boolean)
            * p120  : Power at 120 kHz (w)
            * l120  : Pulse length at 120 kHz (m)
            * i120  : Sample interval at 120 kHz (m)
            * c120  : Sample count at 120 kHz
            * f200  : Presence of 200 kHz frequency (boolean)
            * p200  : Power at 200 kHz (w)
            * l200  : Pulse length at 200 kHz (m)
            * i200  : Sample interval at 200 kHz (m)
            * c200  : Sample count at 200 kHz
            * f333  : Presence of 333 kHz frequency (boolean)
            * p333  : Power at 333 kHz (w)
            * l333  : Pulse length at 333 kHz (m)
            * i333  : Sample interval at 333 kHz (m)
            * c333  : Sample count at 333 kHz         
    """
    
    # load bathymetry file and preallocate bathymetry data
    if bathymetry:
        bfile = xarray.open_dataset(bathymetry, chunks={'lon':360,'lat':180})                  
        bdata = None
        
    # look for RAW files in the directory
    rawfiles = np.sort(glob.glob(os.path.join(rawdir, '*.raw')))
    if not len(rawfiles):
        raise Exception('No RAW files in directory %s' % rawdir)
    
    # create data frame and declare columns names
    pd.set_option('precision', precision)
    df = pd.DataFrame(columns=[
                      'file'  ,
                      'y0'    ,'y1'    ,
                      'd0'    ,'d1'    , 
                      'h0'    ,'h1'    ,
                      'sun0'  ,'sun1'  ,
                      'seabed',                                 
                      'lonmin','lonp25','lonmed','lonp75','lonmax',
                      'latmin','latp25','latmed','latp75','latmax',
                      'kntmin','kntp25','kntmed','kntp75','kntmax',                               
                      'f18'   ,'f38'   ,'f70'   ,'f120'  ,'f200'  ,'f333',
                      'p18'   ,'p38'   ,'p70'   ,'p120'  ,'p200'  ,'p333',
                      'l18'   ,'l38'   ,'l70'   ,'l120'  ,'l200'  ,'l333',
                      'i18'   ,'i38'   ,'i70'   ,'i120'  ,'i200'  ,'i333',
                      'c18'   ,'c38'   ,'c70'   ,'c120'  ,'c200'  ,'c333'])
    
    # iterate through RAW files
    errors = []
    for i, rawfile in enumerate(rawfiles):
        print('%s of %s: Collecting summary data from file %s' %
             (i+1, len(rawfiles), os.path.split(rawfile)[-1]))
        
        # load RAW file
        try:
            ek60 = EK60.EK60()
            ek60.read_raw(rawfile)
        
        # if error, add empty row to the data frame, log error, and continue
        except Exception:
             print('Unable to read file %s' % 
                   os.path.split(rawfile)[-1])
             df.loc[i] = [os.path.split(rawfile)[-1]]+['']*(df.shape[1]-1)
             errors.append(os.path.split(rawfile)[-1])
             continue         
        
        # get year, day of year, and decimal hour from  start datetime
        dt0 = ek60.start_time
        dt0 = pd.DatetimeIndex([dt0])
        y0  = np.int64(dt0.year)[0]
        d0  = np.int64(dt0.dayofyear)[0]
        h0  = np.float64(dt0.hour
                         + dt0.minute     /60 
                         + dt0.second     /3600
                         + dt0.microsecond/3600e6)[0]
        
        # get year, day of year, and decimal hour from end datetime
        dt1 = ek60.end_time
        dt1 = pd.DatetimeIndex([dt1])
        y1  = np.int64(dt1.year)[0]
        d1  = np.int64(dt1.dayofyear)[0]
        h1  = np.float64(dt1.hour
                         + dt1.minute     /60 
                         + dt1.second     /3600
                         + dt1.microsecond/3600e6)[0]
        
        # explore NMEA messages to get position data, break when got it
        for msg in ['GGA', 'GLL', 'RMC']:
            time= ek60.nmea_data.get_datagrams(msg,return_fields=['Time'])
            time= time[msg]['time']    
            lon = ek60.nmea_data.get_datagrams(msg,return_fields=['longitude'])
            lon = lon[msg]['longitude']
            lat = ek60.nmea_data.get_datagrams(msg,return_fields=['latitude'])
            lat = lat[msg]['latitude' ]
            if (isinstance(time, np.ndarray)&
                isinstance(lon , np.ndarray)&
                isinstance(lat , np.ndarray)):
                break
        
        # set metrics to NAN if NMEA data was not found   
        if (time is None)|(lon is None)|(lat is None):
            nan= np.nan
            sun0  , sun1                           = nan, nan
            seabed                                 = nan
            lonmin, lonp25, lonmed, lonp75, lonmax = nan, nan, nan, nan, nan
            latmin, latp25, latmed, latp75, latmax = nan, nan, nan, nan, nan
            kntmin, kntp25, kntmed, kntp75, kntmax = nan, nan, nan, nan, nan
            warnings.warn('NMEA position data not found', Warning)
        
        # calculate metrics otherwise
        else:
        
            # allocate longitude metrics    
            lonmin = np.min(lon)
            lonp25 = np.percentile(lon, 25)
            lonmed = np.median(lon)
            lonp75 = np.percentile(lon, 75)
            lonmax = np.max(lon)
            
            # allocate latitude metrics    
            latmin = np.min(lat)
            latp25 = np.percentile(lat, 25)
            latmed = np.median(lat)
            latp75 = np.percentile(lat, 75)
            latmax = np.max(lat)
            
            # calculate speed in knots
            nmi = np.zeros(len(lon))*np.nan
            for j in range(len(lon)-1):
                if np.isnan(np.array([lon[j],lon[j+1],lat[j],lat[j+1]])).any():
                    nmi[j+1] = np.nan
                else:
                    nmi[j+1] = distance((lat[j],lon[j]),(lat[j+1],lon[j+1])).nm
            knt = np.r_[np.nan, nmi[1:]/(np.float64(np.diff(time))/1000)*3600]
            
            # remove values above max speed
            if (knt[~np.isnan(knt)]>maxspeed).any():
                warnings.warn('%s speeds out of %s above max speed: set to NAN'
                              % (sum(knt[~np.isnan(knt)]>maxspeed), len(nmi)),
                              Warning)        
                knt[np.ma.masked_greater(knt, maxspeed).mask] = np.nan
            
            # allocate speed metrics    
            kntmin = np.nanmin(knt)
            kntp25 = np.nanpercentile(knt, 25)
            kntmed = np.nanmedian(knt)
            kntp75 = np.nanpercentile(knt, 75)
            kntmax = np.nanmax(knt)
            
            # get sun altitude in degrees with respect to platform positions
            # at the first and last timestamp.
            p0      = ephem.Observer()
            p0.lon  = str(lon[0])    
            p0.lat  = str(lat[0])   
            p0.date = pd.to_datetime(time[0]).strftime('%Y/%m/%d %H:%M:%S')
            sun0str = str(ephem.Sun(p0).alt)
            deg0    = float(sun0str.split(':')[0])
            min0    = float(sun0str.split(':')[1])
            sec0    = float(sun0str.split(':')[2])
            sun0    = deg0 + min0/60 + sec0/3600
            p1      = ephem.Observer()
            p1.lon  = str(lon[-1])    
            p1.lat  = str(lat[-1])   
            p1.date = pd.to_datetime(time[-1]).strftime('%Y/%m/%d %H:%M:%S')
            sun1str = str(ephem.Sun(p1).alt)
            deg1    = float(sun1str.split(':')[0])
            min1    = float(sun1str.split(':')[1])
            sec1    = float(sun1str.split(':')[2])
            sun1 = deg1 + min1/60 + sec1/3600
            
            # estimate seabed depth
            if bathymetry:
                
                # load bathymetry data if not done yet
                if bdata is None:
                    c = (lonmed-1,lonmed+1,latmed+1,latmed-1) # W, E, N, S
                    print('Loading bathymetry [%2.2fW, %2.2fE, %2.2fN, %2.2fS]...'%c)
                    bdata = bfile.where((c[0]<bfile.lon)&(bfile.lon<c[1])&
                                        (c[3]<bfile.lat)&(bfile.lat<c[2]),
                                        drop=True)
                    blon = bdata.lon.data
                    blat = bdata.lat.data
                    bele = bdata.elevation.data.compute()
                
                # use bathymetry to estimate seabed at RAW file's position                    
                f = interp2d(blon, blat, bele, bounds_error=True)                        
                try:
                    seabed = int(f(lonmed,latmed)[0])
                
                # reload  bathymetry if file's position falls outside the area
                except ValueError:
                    c = (lonmed-1,lonmed+1,latmed+1,latmed-1) # W, E, N, S
                    print('Reloading bathymetry [%sW, %sE, %sN, %sS]...'%c)
                    bdata = bfile.where((c[0]<bfile.lon)&(bfile.lon<c[1])&
                                        (c[3]<bfile.lat)&(bfile.lat<c[2]),
                                        drop=True)
                    blon = bdata.lon.data
                    blat = bdata.lat.data
                    bele = bdata.elevation.data.compute()
                    f = interp2d(blon, blat, bele, bounds_error=True)
                    seabed = int(f(lonmed,latmed)[0])
            
            # fill seabed depth with NAN, if bathymetry is not provided
            else:
                seabed=np.nan
        
        # collect configuration for each frequency
        f18,f38,f70,f120,f200,f333 =  0, 0, 0, 0, 0, 0
        p18,p38,p70,p120,p200,p333 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        l18,l38,l70,l120,l200,l333 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        i18,i38,i70,i120,i200,i333 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan
        c18,c38,c70,c120,c200,c333 = np.nan,np.nan,np.nan,np.nan,np.nan,np.nan        
        
        for ch, freq in enumerate(ek60.channel_ids):    
            
            # at 18 kHz
            if '18 kHz' in freq:
                f18 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f18.transmit_power))==1:
                    p18 = f18.transmit_power[0]
                else:
                    p18 = -1
                if len(np.unique(f18.pulse_length))==1:
                    l18 = f18.pulse_length[0]
                else:
                    l18 = -1
                if len(np.unique(f18.sample_interval))==1:
                    i18 = f18.sample_interval[0]
                else:
                    i18 = -1
                if len(np.unique(f18.sample_count))==1:
                    c18 = f18.sample_count[0]
                else:
                    c18 = -1
                f18 = 1
            
            # at 38 kHz
            if '38 kHz' in freq:
                f38 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f38.transmit_power))==1:
                    p38 = f38.transmit_power[0]
                else:
                    p38 = -1
                if len(np.unique(f38.pulse_length))==1:
                    l38 = f38.pulse_length[0]
                else:
                    l38 = -1
                if len(np.unique(f38.sample_interval))==1:
                    i38 = f38.sample_interval[0]
                else:
                    i38 = -1
                if len(np.unique(f38.sample_count))==1:
                    c38 = f38.sample_count[0]
                else:
                    c38 = -1
                f38 = 1
            
            # at 70 kHz
            if '70 kHz' in freq:
                f70 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f70.transmit_power))==1:
                    p70 = f70.transmit_power[0]
                else:
                    p70 = -1
                if len(np.unique(f70.pulse_length))==1:
                    l70 = f70.pulse_length[0]
                else:
                    l70 = -1
                if len(np.unique(f70.sample_interval))==1:
                    i70 = f70.sample_interval[0]
                else:
                    i70 = -1
                if len(np.unique(f70.sample_count))==1:
                    c70 = f70.sample_count[0]
                else:
                    c70 = -1
                f70 = 1
            
            # at 120 kHz
            if '120 kHz' in freq:
                f120 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f120.transmit_power))==1:
                    p120 = f120.transmit_power[0]
                else:
                    p120 = -1
                if len(np.unique(f120.pulse_length))==1:
                    l120 = f120.pulse_length[0]
                else:
                    l120 = -1
                if len(np.unique(f120.sample_interval))==1:
                    i120 = f120.sample_interval[0]
                else:
                    i120 = -1
                if len(np.unique(f120.sample_count))==1:
                    c120 = f120.sample_count[0]
                else:
                    c120 = -1
                f120 = 1
            
            # at 200 kHz
            if '200 kHz' in freq:
                f200 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f200.transmit_power))==1:
                    p200 = f200.transmit_power[0]
                else:
                    p200 = -1
                if len(np.unique(f200.pulse_length))==1:
                    l200 = f200.pulse_length[0]
                else:
                    l200 = -1
                if len(np.unique(f200.sample_interval))==1:
                    i200 = f200.sample_interval[0]
                else:
                    i200 = -1
                if len(np.unique(f200.sample_count))==1:
                    c200 = f200.sample_count[0]
                else:
                    c200 = -1
                f200 = 1
            
            # at 333 kHz
            if '333 kHz' in freq:
                f333 = ek60.get_raw_data(channel_number=ch+1)
                if len(np.unique(f333.transmit_power))==1:
                    p333 = f333.transmit_power[0]
                else:
                    p333 = -1
                if len(np.unique(f333.pulse_length))==1:
                    l333 = f333.pulse_length[0]
                else:
                    l333 = -1
                if len(np.unique(f333.sample_interval))==1:
                    i333 = f333.sample_interval[0]
                else:
                    i333 = -1
                if len(np.unique(f333.sample_count))==1:
                    c333 = f333.sample_count[0]
                else:
                    c333 = -1
                f333 = 1
        
        # add new row of data to dataframe 
        df.loc[i] = [os.path.split(rawfile)[-1],
                     y0    , y1    ,
                     d0    , d1    , 
                     h0    , h1    ,
                     sun0  , sun1  ,
                     seabed,
                     lonmin, lonp25, lonmed, lonp75, lonmax,
                     latmin, latp25, latmed, latp75, latmax,
                     kntmin, kntp25, kntmed, kntp75, kntmax,
                     f18   , f38   , f70   , f120  , f200  , f333,
                     p18   , p38   , p70   , p120  , p200  , p333,
                     l18   , l38   , l70   , l120  , l200  , l333,
                     i18   , i38   , i70   , i120  , i200  , i333,
                     c18   , c38   , c70   , c120  , c200  , c333]
    
    # save data frame as a text file
    if save:
        with open(os.path.join(rawdir, txtfile), 'w+') as f:
            
            # write header
            f.write('# NMEA AND CONFIGURATION DATA SUMMARY:\n')
            f.write('# Variable names per column, from left to right:\n')
            f.write('# * file  : Name of RAW file\n')
            f.write('# * y0    : start year (years after AD epoch)\n')
            f.write('# * y1    : end year (years after AD epoch)\n')
            f.write('# * d0    : start day (day of year)\n')
            f.write('# * d1    : end day (day of year)\n')
            f.write('# * h0    : start time (decimal hours)\n')
            f.write('# * h1    : end time (decimal hours)\n')
            f.write('# * sun0  : sun altitude at the start (degreses)\n')
            f.write('# * sun1  : sun altitude at the end (degrees)\n')
            f.write('# * seabed: seabed depth, estimated from bathymetry (m)\n')
            f.write('# * lonmin: longitude minimum (decimal degrees)\n')
            f.write('# * lonp25: longitude percentile 25 (decimal degrees)\n')
            f.write('# * lonmed: longitude median (decimal degrees)\n')
            f.write('# * lonp75: longitude percentile 75 (decimal degrees)\n')
            f.write('# * lonmax: longitude maximum (decimal degrees)\n')
            f.write('# * latmin: latitude minimum (decimal degrees)\n')
            f.write('# * latp25: latitude percentile 25 (decimal degrees)\n')
            f.write('# * latmed: latitude median (decimal degrees)\n')
            f.write('# * latp75: latitude percentile 75 (decimal degrees)\n')
            f.write('# * latmax: latitude maximum (decimal degrees)\n')
            f.write('# * kntmin: speed minimum (knots)\n')
            f.write('# * kntp25: speed percentile 25 (knots)\n')
            f.write('# * kntmed: speed median (knots)\n')
            f.write('# * kntp75: speed percentile 75 (knots)\n')
            f.write('# * kntmax: speed maximum (knots)\n')
            f.write('# * f18   : Presence of 18 kHz frequency (boolean)\n')            
            f.write('# * f38   : Presence of 38 kHz frequency (boolean)\n')            
            f.write('# * f70   : Presence of 70 kHz frequency (boolean)\n')            
            f.write('# * f120  : Presence of 120 kHz frequency (boolean)\n')            
            f.write('# * f200  : Presence of 200 kHz frequency (boolean)\n')
            f.write('# * f333  : Presence of 333 kHz frequency (boolean)\n')
            f.write('# * p18   : Power at 18 kHz (w). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * p38   : Power at 38 kHz (w). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * p70   : Power at 70 kHz (w). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * p120  : Power at 120 kHz (w). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * p200  : Power at 200 kHz (w). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * p333  : Power at 333 kHz (w). Fill value \'-1\' stands for \'variable\'\n')                    
            f.write('# * l18   : Pulse length at 18 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * l38   : Pulse length at 38 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * l70   : Pulse length at 70 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * l120  : Pulse length at 120 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * l200  : Pulse length at 200 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * l333  : Pulse length at 333 kHz (m). Fill value \'-1\' stands for \'variable\'\n')      
            f.write('# * i18   : Sample interval at 18 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * i38   : Sample interval at 38 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * i70   : Sample interval at 70 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * i120  : Sample interval at 120 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * i200  : Sample interval at 200 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * i333  : Sample interval at 333 kHz (m). Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c18   : Sample count at 18 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c38   : Sample count at 38 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c70   : Sample count at 70 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c120  : Sample count at 120 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c200  : Sample count at 200 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('# * c333  : Sample count at 333 kHz. Fill value \'-1\' stands for \'variable\'\n')
            f.write('\n')
            
            # write data as comma-separated values (csv)
            data = df.to_string(header=False, index=False).split('\n')
            data = [','.join(x.split()) for x in data]
            data = [x.replace('NaN','') for x in data]
            for d in data: 
                f.write(d+'\n')
    
    # report files not read due to reading errors
    if errors:
        print('Unable to read the following RAW files: %s' % errors)
        print('%s files out of %s couldn\' be read!' %
             (len(errors), len(rawfiles)))
    
    # return dataframe        
    return df

def read(txtfile):
    """
    Read NMEA and configuration summary data from the text file generated by 
    report.write().  
    
    Args:
        filereport (str): path/to/report.txt
        
    Returns:
         pandas.DataFrame: file report content.
    """
    
    # open file report and count header lines
    f = open(txtfile, "r")
    columnnames = []
    headerlines = 0
    while 1:
        line = f.readline()
        if '*' in line:
            columnnames.append(line.split('*')[1].split(':')[0].strip())
        if not ('#' in line[0]) | ('\n' in line[:2]):
            break
        headerlines +=1
    
    # read csv and add column names (after skipping header lines)     
    df = pd.read_csv(txtfile, skiprows=headerlines, names=columnnames)
    df = df.sort_values('file')
    
    return df

if __name__=='__main__':
    """
    If run the script as the main program, it will ask the user for inputs
    in the console which will be used as arguments in report.write().
    """
    
    print('Make NMEA report...')
    
    # input directory path where the RAW files are stored
    while 1:
        rawdir = input('Directory where RAW files are stored' +
                       ' (\'Ctrl-C\' to exit):')
        if os.path.isdir(rawdir):
            break
        else:
            print('\'%s\' is not a valid directory' % rawdir)
    
    # input whether you'd like to save the NMEA report
    while 1:
        save   = input('Save data in text file (y/n, \'Ctrl-C\' to exit)? ')
        if (save.lower()=='n') | (save.lower()=='no'):
            report_table = write(rawdir, save=False)
            break                          
        elif (save.lower()=='y') | (save.lower()=='yes'):
            report_table = write(rawdir, save=True)
            break
        else:
            print('answer \'%s\' not understood' % save)