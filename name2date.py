import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt
from tqdm import tqdm

year = 2021
base = dt.datetime(year,1,1,0)
date_list = [base + dt.timedelta(hours=x) for x in range(365*24)]
data_idx = 0

for time in tqdm(date_list):
    temp = np.load('/home/gnn/workplace2/zw/exp4/icos2021/'+str(data_idx)+'.npy')
    np.save('/home/gnn/workplace2/zw/exp4/icos/'+time.strftime("ERA5_plevel_%Y_%m_%d_%H.npy"),temp)
    data_idx += 1