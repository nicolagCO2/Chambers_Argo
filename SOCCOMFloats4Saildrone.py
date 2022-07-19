#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 14:54:36 2022

@author: nicolausf
"""

### Import SOCCOM Fleet came from Nancy Williams in Walliam's Lab Github

# ### Import the entire SOCCOM fleet snapshot (low-res LIAR_TA files) and:
# * Bring in satellite sea ice extent and mask out SSIZ floats
# * Add mixed layer depth and other criteria (depth of certain oxygen or temperature layers, etc) for picking zones
# * do CO2SYS calculations to get Omega_Ar
# * pickle into a dataset to be used elsewhere
#
# TO DO:
# * Add "ice" field for actual profiles that are under ice. Use monthly ice product for this.
# * switch from pickle to something like yaml or json
# * do replace '//' with larger chunks
# * update code to use new SOCCOM snapshot format (files not called *QC.txt anymore?)
# * Add AOU
# * calculate entrainment term properly (currently the vertical spacing is not accounted for- maybe interpolate to 1m spacing)
# * Unzip soccom float file into a different folder to not clutter the home folder
# * REMOVE DATA NORTH OF 20S

# +
import glob, os
import pandas as pd
import numpy as np
import gsw
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import urllib.request
from zipfile import ZipFile
from re import search
from scipy import ndimage
import PyCO2SYS as pyco2

#from dask.distributed import Client #client is the interface to
#from dask.distributed import LocalCluster #Start the cluster locally
#cluster = LocalCluster()
#client = Client(cluster) #Connect the client to the c
#client

# +
output_dir = 'generated/'
data_dir = 'data/' 
# This is the folder where all the float data live

MLDmethod = 'density_gradient' 
# options are "threshold" or "density_gradient"

# +
# You can switch this to turn debugging on or off (True or False)
# When debugging is True, only process 20 floats and so does not pickle a new dataset
DEBUG = True

def debug (*args):
    """Call print() with arguments if DEBUG is True""" # preserves formatting
    if DEBUG:
        print(*args)


# -

# ## Import sea ice data
#
# Gray et al. (2018) uses 15% contour from September 2014-2016 of these data: https://nsidc.org/data/NSIDC-0051/versions/1
#
# I'm using monthly data in netcdf files I found ... Somewhere...
#

def import_nsidc_datasets(years, months):
    filenames = []
    for year in years:
        for month in months:
            ff = glob.glob(f'ice/*_{year}{month:02d}_*.nc')
            filenames.append(ff[0])
    print(filenames)
    return xr.open_mfdataset(filenames)#, parallel=True


# +
years = [2014, 2015, 2016]
months = [9]
max_conc = 0.15

ds = import_nsidc_datasets(years, months)
ds = ds.mean(dim='time')
ds.seaice_conc_monthly_cdr.where(
    (ds.seaice_conc_monthly_cdr <= 1) &
    (ds.seaice_conc_monthly_cdr >= max_conc)
).plot()
ds

# +
# Test method on one point
# Looks for the sea ice concentration at one location given by lat/lon 
# then we check if it's >.15
# Near prime meridian and 180, this needs to be lat < 0.2 and lon > 0.5 to capture enough data.
lat = -76
lon = 0
if lon > 180: # translate to negative longitudes where > 180
    lon = lon - 360

ds.seaice_conc_monthly_cdr.where((abs(ds.latitude - lat) < 0.2) &
                                       (abs(ds.longitude - lon) <.5)).mean(skipna=True).plot()
ice_conc = float(ds.seaice_conc_monthly_cdr.where((abs(ds.latitude - lat) < 0.2) &
                                       (abs(ds.longitude - lon) <0.5)).mean(skipna=True))
ice_conc

# +
# download the last snapshot (still need to find a way to make the name general..)
os.chdir(data_dir)
SOCCOM_snap    = 'SOCCOM_LRQC_LIAR_odvtxt_20210622'
SOCCOM_ODV_zip = os.path.join("http://soccompu.princeton.edu/www/dl.php?file=/FloatViz_Data",SOCCOM_snap + ".zip")

urllib.request.urlretrieve(SOCCOM_ODV_zip, SOCCOM_snap + ".zip" )

# unzip the file
with ZipFile(SOCCOM_snap + ".zip", 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
os.chdir('..')
# -

floatsuffix = '*QC.TXT' # all of the Southern Ocean floats have the same suffix
# Note that in the future snapshots, the float numbers will change to the WMOID numbers
filelist = glob.glob(data_dir + floatsuffix)
if DEBUG == True:
    filelist = filelist[:5]
print(filelist)

# +
# Create new dataframe called "SOCCOM" and append data float by float
SOCCOM = pd.DataFrame()
import time
tic = time.perf_counter()

# loop through float files
for f in filelist:
    with open(f, 'rt', encoding='UTF-8') as fin:
        with open(data_dir + 'fltrem.txt','wt') as fout:
            data = ''.join(line.replace('//','#') for line in fin)
            fout.write(data)
            

    flt = pd.read_csv((data_dir + 'fltrem.txt'),
                      delimiter='\t',
                      comment='#',
                      na_values=-1E10,
                     )
    flt['date'] = pd.to_datetime(flt['mon/day/yr'] + ' ' + flt['hh:mm'])
    flt['POTTMP'] = gsw.pt0_from_t(flt['Salinity[pss]'],
                                  flt['Temperature[°C]'],
                                  flt['Pressure[dbar]'])
    
    if flt['Lat [°N]'].mean() <= -20:
        # Loop through columns and apply all Quality Flags (QFs)
        # Make exception for Lat/Lon which are often flagged 4 if interpolated
        for column in range(len(flt.columns)):
            name=flt.columns[column]
            if 'QF' in name: # if the column is a QF column, apply it to the preceeding column, otherwise go on to next column
                var=flt.columns[column-1]
                if (var == 'Lat [°N]') | (var == 'Lon [°E]'):
                    continue
                flt[var] = np.where(flt.iloc[:,flt.columns.get_loc(var) + 1] ==0, flt[var], np.nan)

        flt['MLD'] = np.NaN
        flt['ent_DIC'] = np.NaN
        flt['zO2min'] = np.NaN
        flt['zSmax'] = np.NaN
        flt['zTmax'] = np.NaN
        flt['zone'] = np.NaN
        flt['sector'] = np.NaN
        flt['DIC_T_min_upper_500'] = np.NaN
        flt['O2_T_min_upper_500'] = np.NaN
        last_MLD = np.NaN
        last_ML_DIC = np.NaN
        last_subML_DIC = np.NaN
        last_t = np.datetime64('NaT')

        # flt['ice'] = np.NaN # Add back later to check for actual ice profiles
        if 'DIC_LIAR[µmol/kg]' in flt.columns:
            for station in flt['Station'].unique():
                diurnal_layer = 20
                if MLDmethod == 'threshold':
                    # Add MLD using density threshold criterion 
                    dens_threnshold = 0.03 # According to Holte et al. (2017) this number should be variable and possibly smaller in the Southern Ocean
                    surfacedepth = flt.loc[(
                        flt['Station'] == station) & (flt['Depth[m]'] > diurnal_layer), 'Depth[m]'].min()
                    # calculate surface density
                    surfacedens = flt.loc[(
                        flt['Depth[m]'] == surfacedepth) & (
                        flt['Station'] == station), 'Sigma_theta[kg/m^3]'].min()
                    surface_t = flt.loc[(
                        flt['Depth[m]'] == surfacedepth) & (
                        flt['Station'] == station), 'Temperature[°C]'].min()
                    MLD = flt.loc[(
                        flt['Station'] == station)  & (flt['Depth[m]'] > diurnal_layer) & (
                        flt['Sigma_theta[kg/m^3]'] - surfacedens > dens_threnshold), 'Depth[m]'].min()
                    flt.loc[(flt['Station'] == station),'MLD'] = MLD

                if MLDmethod == 'density_gradient':
                    # Add MLD using max density gradient
                    # calculate density gradient at each step
                    dens_grad = flt.loc[(
                        flt['Station'] == station), 'Sigma_theta[kg/m^3]'].diff() / flt.loc[(
                        flt['Station'] == station), 'Depth[m]'].diff()
                    flt.loc[(flt['Station'] == station), 'dens_grad'] = dens_grad
                    MLD = flt.loc[(
                        flt['Station'] == station) & (flt['Depth[m]'] > diurnal_layer) & (
                        flt['dens_grad'] == dens_grad.max()), 'Depth[m]'].min()
                    flt.loc[(flt['Station'] == station),'MLD'] = MLD

                # calculate entrainment of DIC
                dMLD = MLD - last_MLD
                deltat = (flt.loc[(flt['Station'] == station), 'date'] - last_t).min().total_seconds() / (24 * 60 * 60)

                if (dMLD > 0) & (deltat > 0): # removing " & ((last_subML_DIC - last_ML_DIC) > 0)" because I don't think this is correct. Talk with Seth
                    flt.loc[(flt['Station'] == station), 'ent_DIC'] = dMLD * last_subML_DIC * last_subML_dens / deltat * 1E-6

                    if (dMLD * last_subML_DIC * last_subML_dens / deltat * 1E-6) > 1500:
                        print('Entrainment is too large for float',
                              flt.loc[(flt['Station'] == station), 'Cruise'],
                              'station', station)

                #Set values for next entrainment calculation
                last_t = flt.loc[(flt['Station'] == station), 'date'].min()
                last_MLD = MLD
                last_subML_DIC = flt.loc[(flt['Station'] == station) &
                                         (flt['Pressure[dbar]'] >= MLD) & 
                                         (flt['Pressure[dbar]'] <= MLD + 30),
                                         'DIC_LIAR[µmol/kg]'].mean(skipna=True)
                last_subML_dens = flt.loc[(flt['Station'] == station) &
                                         (flt['Pressure[dbar]'] >= MLD) & 
                                         (flt['Pressure[dbar]'] <= MLD + 30),
                                         'Sigma_theta[kg/m^3]'].mean(skipna=True)+1000
                last_ML_DIC = flt.loc[(flt['Station'] == station) & 
                                      (flt['Pressure[dbar]'] < MLD),
                                      'DIC_LIAR[µmol/kg]'].mean(skipna=True)


                # calculate 100m potential temp
                T_100 = flt.loc[(flt['Depth[m]'] > 97.5) &
                                (flt['Depth[m]'] < 102.5) &
                                (flt['Station'] == station),
                                'POTTMP'].mean(skipna=True)
                # calculate 400m potential temp
                T_400 = flt.loc[(flt['Depth[m]'] > 390) &
                                (flt['Depth[m]'] < 410) &
                                (flt['Station'] == station),
                                'POTTMP'].mean(skipna=True)
                T_min_upper_200 = flt.loc[(flt['Depth[m]'] < 200) &
                                          (flt['Station'] == station),
                                          'POTTMP'].min() 

                # Add oxygen min and depth of oxygen min
                O2min = flt.loc[(flt['Station'] == station),'Oxygen[µmol/kg]'].min()
                zO2min = flt.loc[(flt['Station'] == station)&
                                 (flt['Oxygen[µmol/kg]'] == O2min),
                                 'Depth[m]'].min()
                flt.loc[(flt['Station'] == station), 'zO2min'] = zO2min

                # Add salinity max and depth of salinity max
                Smax = flt.loc[(flt['Station'] == station), 'Salinity[pss]'].max()
                zSmax = flt.loc[(flt['Station'] == station)&
                                (flt['Salinity[pss]'] == Smax),
                                'Depth[m]'].max()
                flt.loc[(flt['Station'] == station), 'zSmax'] = zSmax

                # Add temp max and depth of temp max
                Tmax = flt.loc[(flt['Station'] == station), 'Temperature[°C]'].max()
                zTmax = flt.loc[(flt['Station'] == station)&
                                (flt['Oxygen[µmol/kg]'] == O2min),
                                'Depth[m]'].min()
                flt.loc[(flt['Station'] == station), 'zTmax'] = zTmax

                # Looks for the sea ice concentration at the float location, 
                # then we check if it's >.15
                lat = flt.loc[(flt['Station'] == station), 'Lat [°N]'].min()
                lon = flt.loc[(flt['Station'] == station), 'Lon [°E]'].min()
                if lon > 180: # translate to negative longitudes where > 180
                    lon = lon - 360

                if lat >- 50:
                    ice_conc = 0
                else:
                    ice_conc = float(ds.seaice_conc_monthly_cdr.where((abs(ds.latitude - lat) < 0.2) &
                                                                      (abs(ds.longitude - lon) <0.5)).mean(skipna=True))

                if ice_conc > max_conc: # Assign SSIZ if satellite observed ice is greater than max_conc as defined above
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'SSIZ'
                # Assign STZ
                elif T_100 >= 11:
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'STZ'
                # Assign SAZ
                elif (T_100 < 11) & (T_400 >= 5):
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'SAZ'
                # Assign PFZ
                elif (T_400 < 5) & (T_min_upper_200 >= 2):
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'PFZ'
                    T_min_upper_500 = flt.loc[(flt['Depth[m]'] < 500) &
                                              (flt['Station'] == station),
                                              'POTTMP'].min()
                    DIC_T_min_upper_500 = flt.loc[(flt['POTTMP'] == T_min_upper_500) &
                                                  (flt['Station'] == station),
                                                  'DIC_LIAR[µmol/kg]'].mean()
                    O2_T_min_upper_500 = flt.loc[(flt['POTTMP'] == T_min_upper_500) &
                                                  (flt['Station'] == station),
                                                  'Oxygen[µmol/kg]'].mean()
                    flt.loc[(flt['Station'] == station), 'DIC_T_min_upper_500'] = DIC_T_min_upper_500
                    flt.loc[(flt['Station'] == station), 'O2_T_min_upper_500'] = O2_T_min_upper_500

                # Assign ASZ
                elif (T_min_upper_200 < 2):
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'ASZ'
                    T_min_upper_500 = flt.loc[(flt['Depth[m]'] < 500) &
                                              (flt['Station'] == station),
                                              'POTTMP'].min()
                    DIC_T_min_upper_500 = flt.loc[(flt['POTTMP'] == T_min_upper_500) &
                                                  (flt['Station'] == station),
                                                  'DIC_LIAR[µmol/kg]'].mean()
                    O2_T_min_upper_500 = flt.loc[(flt['POTTMP'] == T_min_upper_500) &
                                                  (flt['Station'] == station),
                                                  'Oxygen[µmol/kg]'].mean()
                    flt.loc[(flt['Station'] == station), 'DIC_T_min_upper_500'] = DIC_T_min_upper_500
                    flt.loc[(flt['Station'] == station), 'O2_T_min_upper_500'] = O2_T_min_upper_500

                else:
                    flt.loc[(flt['Station'] == station),(['zone'])] = 'none'

    SOCCOM = SOCCOM.append(flt, ignore_index=True)
    
SOCCOM.loc[(SOCCOM['Lon [°E]'] >= 146) | (SOCCOM['Lon [°E]'] <= -62), (['sector'])] = 'Pacific'
SOCCOM.loc[(SOCCOM['Lon [°E]'] >= 22) & (SOCCOM['Lon [°E]'] <= 146), (['sector'])] = 'Indian'
SOCCOM.loc[(SOCCOM['Lon [°E]'] < 22) & (SOCCOM['Lon [°E]'] > -62), (['sector'])] = 'Atlantic'

debug(SOCCOM)
toc = time.perf_counter()
print(f"Loop ran in {toc - tic:0.4f} seconds")
# -

SOCCOM

SOCCOM[['Lat [°N]','MLD','pHinsitu[Total]','DIC_LIAR[µmol/kg]','TALK_LIAR[µmol/kg]','ent_DIC']].describe()

results = pyco2.sys(
    par1=SOCCOM['pHinsitu[Total]'], # total scale is default
    par2=SOCCOM['TALK_LIAR[µmol/kg]'],
    par1_type=3,
    par2_type=1,
    # Don't need to specify output conditions if same
    temperature=SOCCOM['Temperature[°C]'], 
    pressure=SOCCOM['Pressure[dbar]'],
    salinity=SOCCOM['Salinity[pss]'],
    opt_k_carbonic=10,
    opt_k_bisulfate=1, # this is the default
    opt_k_fluoride=1, # this is the default
    buffers_mode='auto',
)
SOCCOM['OmegaAr']=results['saturation_aragonite']
SOCCOM['OmegaAr'].describe()

# Add fields for year, month, and day
SOCCOM['year'] = SOCCOM['date'].dt.year
SOCCOM['month'] = SOCCOM['date'].dt.month
SOCCOM['day'] = SOCCOM['date'].dt.day

if DEBUG == False:
    # pickle dataframe only if rerunning the whole snapshot
    SOCCOM.to_pickle('SOCCOM_snapshot.pkl')
    # to read later use: SOCCOM = pd.read_pickle(SOCCOM_snapshot.pkl)

SOCCOMbyzone = SOCCOM.groupby(['zone']).count()
SOCCOMbyzone

SOCCOMbysector = SOCCOM.groupby(['sector']).count()
SOCCOMbysector

SOCCOMbyyear = SOCCOM.groupby(['year']).mean()
SOCCOMbyyear['ent_DIC']
