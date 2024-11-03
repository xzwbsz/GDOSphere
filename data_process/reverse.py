import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt
from tqdm import tqdm

year = 2019
base = dt.datetime(year,1,1,0)
date_list = [base + dt.timedelta(hours=x) for x in range(365*24)]
path = '/climate/ERA5/icos6/'
path1 = '/climate/ERA5/icosnew/'

for time in tqdm(date_list):
    # int(time.strftime('%H'))
    file = path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_")+str(int(time.strftime('%H')))+'.npy'

    AA = np.load(file)
    AA=AA.squeeze() #5,37,40962
    BB = AA
    for plevel in range(BB.shape[1]):
        BB[:,plevel,...] = AA[:,36-plevel,...]
    file1 = path1+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.npy")
    np.save(file1,BB)




