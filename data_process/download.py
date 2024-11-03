import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt


year=2020
main='/xuzhewen/climate/'
base = dt.datetime(year,1,1,0)
date_list = [base + dt.timedelta(hours=x) for x in range(24*365)]

vars = ['geopotential', 'specific_humidity', 'temperature','u_component_of_wind', 'v_component_of_wind']
surface_vars=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature','mean_sea_level_pressure']

levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300','350', '400',
          '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975','1000']
levels.reverse()

def download_plevel(time, path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable':  vars,
            'pressure_level': levels,
            'year': time.year,
            'month': time.month,
            'day': time.day,
            'time': time.hour,
            'format': 'grib',
        }
        ,
        path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.grib")
    )

def download_surface(time, path):
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable':  surface_vars,
            'pressure_level': levels,
            'year': time.year,
            'month': time.month,
            'day': time.day,
            'time': time.hour,
            'format': 'grib',
        }
        ,
        path+os.sep+time.strftime("ERA5_surface_%Y_%m_%d_%H.grib")
    )

path=main+str(year)

for time in date_list:
    if not os.path.exists(path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.grib")):
        download_plevel(time, path)
    if not os.path.exists(path+os.sep+time.strftime("ERA5_surface_%Y_%m_%d_%H.grib")):
        download_surface(time, path)

