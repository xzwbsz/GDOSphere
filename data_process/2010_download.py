import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt

year = 2010
base = dt.datetime(year,1,1,0)
date_list = [base + dt.timedelta(hours=x) for x in range(365*24)]

surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']

pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']
resolution= ['1', '1']

def download_plevel(time, path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable':  upper_variables,
            'pressure_level': pressure_levels,
            'year': time.year,
            'month': time.month,
            'day': time.day,
            'time': time.hour,
            'format': 'netcdf',
            'grid': resolution,
        }
        ,
        path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.nc")
    )

def download_surface(time, path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable':  surface_variables,
            'year': time.year,
            'month': time.month,
            'day': time.day,
            'time': time.hour,
            'format': 'netcdf',
            'grid': resolution,
        }
        ,
        path+os.sep+time.strftime("ERA5_surface_%Y_%m_%d_%H.nc")
    )

path='/public/1_d_data/'+str(year)

for time in date_list:
    if not os.path.exists(path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.nc")):
        download_plevel(time, path)
    if not os.path.exists(path+os.sep+time.strftime("ERA5_surface_%Y_%m_%d_%H.nc")):
        download_surface(time, path)