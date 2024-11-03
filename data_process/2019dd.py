import xarray as xr
from datetime import timedelta, date,datetime
import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt

dataset = 'gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/'

year=2017
main='/home/gnn/2019052223'
base = dt.datetime(year,5,22,0)
date_list = [base + dt.timedelta(days=x) for x in range(1)]

vars = ['geopotential', 'specific_humidity', 'temperature','u_component_of_wind', 'v_component_of_wind']

#     gsutil -m cp -r \
#   "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/2010/01/05/geopotential" \
#   "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/2010/01/05/specific_humidity" \
#   "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/2010/01/05/temperature" \
#   "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/2010/01/05/u_component_of_wind" \
#   "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level/2010/01/05/v_component_of_wind" \
#   .

for time in date_list:
    for varr in vars:
        down_target = dataset+str(time.strftime('%Y'))+'/'+str(time.strftime('%m'))+'/'+str(time.strftime('%d'))+'/'+str(varr)
        file_path = main+'/'+str(time.strftime('%Y'))+'/'+str(time.strftime('%m'))+'/'+str(time.strftime('%d'))+'/'
        # print(file_path)
        if not os.path.exists(file_path):
            os.system('mkdir -p {0}'.format(file_path))
        print(os.system("gsutil -m cp -r {0} {1}".format(down_target,file_path)))

    # if not os.path.exists(path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d.grib")):
