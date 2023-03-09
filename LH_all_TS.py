#!/usr/bin/env python
# coding: utf-8
"""
Created on Thu Mar  9 17:15:12 2023

@author: Xiaoting CHEN
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
import datetime
import matplotlib.dates as mdates
import itertools
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER,mticker

import warnings
warnings.filterwarnings("ignore")

start_time = datetime.datetime.now()



levels = np.array([106.27, 
                   131.20, 
                   161.99, 
                   200.00, 
                   222.65,
                   247.87, 
                   275.95, 
                   307.20, 
                   341.99, 
                   380.73, 
                   423.85,
                   471.86, 
                   525.00, 
                   584.80, 
                   651.04, 
                   724.78, 
                   800.00, 
                   848.69, 
                   900.33,
                   955.12,
                   1013.00
                  ])

deltaPs = levels[1:] - levels[:-1]
print(np.size(deltaPs))
#print('deltaPs:',deltaPs)


#ampm ='AM'
#sat = 'AIRS'
#time = '0130'
model_type = '_'  # '_' for with ML, '_noML_' for without ML



"""
Read all data from 2004 - 2018 for AIRS, or 2008 - 2018 for IASI

"""

f_path = f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/RRind3{model_type}LH_nan0_phase2_data-day/'  #loholt 

plot_path = '/home/xchen/combined_LH_HR_Vert/plot/'



"""
Function definition 

"""

################################# Read all data #################################
#Read all data from 2004 - 2018 for AIRS, or 2008 - 2018 for IASI

def read_data(sat,time, ampm, year_range):
    
    ds_list = []
    ds_input = []

    for year in year_range: 

        for month in range(1,12+1): 

            path = f_path + f'20{year:02d}/{month:02d}/' 
            
            try:
                files = glob.glob(path+f'LH_{sat}_*{time}{ampm}.nc')
                ds_input = xr.open_mfdataset(files) 

                ds_list.append(ds_input)

            except OSError:
                pass
                    
            continue
    
    ds = xr.concat(ds_list,'time')
    
    return ds


################################# Time series - Integral #################################
def plot_TSint(ds,sat):
    #mean(dim=['time', 'lat', 'lon'])
    ds_t = ds.LH.mean(dim=['longitude', 'latitude']).transpose('time','level') #downscaling

    ds_int = ds_t*deltaPs
    
    # check the date of terrible Anomalies
    ds_anomaly =  ds_int.sum('level').where(ds_int.sum('level') > 1000, drop=True)
    print('Day of Terrible Anomalies:',ds_anomaly.time)

    fig1, ax1 = plt.subplots(1,1,figsize=(18,5),dpi=300)

    ds_int.sum('level').plot(ax=ax1,c='navy',label=sat)

    ax1.set_title('Time series - Integrated',fontsize=16)
    ax1.set_ylabel('LH',fontsize=14)
    ax1.set_xlabel('time',fontsize=14)
    ax1.set_ylim(200,1000)
    ax1.legend()
    
    fig1.savefig(plot_path + f'time_series{model_type}int_{sat}_tot.png')
    plt.close()
    
    return 


def plot_TSint_all(ds1, ds2):
    #mean(dim=['time', 'lat', 'lon'])
    ds1_t = ds1.LH.mean(dim=['longitude', 'latitude']).transpose('time','level') #downscaling
    ds2_t = ds2.LH.mean(dim=['longitude', 'latitude']).transpose('time','level')

    ds1_int = ds1_t*deltaPs
    ds2_int = ds2_t*deltaPs

    fig1, ax1 = plt.subplots(1,1,figsize=(18,5),dpi=300)

    ds1_int.sum('level').plot(ax=ax1,c='firebrick',label='AIRS')
    ds2_int.sum('level').plot(ax=ax1,c='navy',label='IASI')

    ax1.set_title('Time series - Integrated',fontsize=16)
    ax1.set_ylabel('LH',fontsize=14) #LH (K/data-day)
    ax1.set_xlabel('time',fontsize=14) 
    ax1.set_ylim(200,2000)
    ax1.legend()
    
    fig1.savefig(plot_path + f'time_series{model_type}int_AIRS_vs_IASI_tot.png')
    plt.close()
    
    return 


year_AIRS = range(4,18+1)
year_IASI = range(8,18+1)


ds_AIRS = read_data('AIRS','0130', 'AM', year_AIRS)
ds_IASI = read_data('IASI-A','0930', 'AM', year_IASI)
# In[6]:

plot_TSint(ds_AIRS, 'AIRS')
plot_TSint(ds_IASI, 'IASI-A')
plot_TSint_all(ds_AIRS, ds_IASI)


print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds') 





