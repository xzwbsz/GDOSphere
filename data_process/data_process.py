import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt
import xarray as xr

count = 0
for year in [2021,2022]:
    main='/home/gnn/data2/ERA5/'
    base = dt.datetime(year,1,1,0)
    date_list = [base + dt.timedelta(hours=x) for x in range(0,24*365,6)]
    path='/xuzhewen/climate/'+str(year)+'/nc2'
    #path+os.sep+time.strftime("ERA5_surface_%Y_%m_%d_%H.nc")
    max_temp = np.random.rand(5,37)#最大
    min_temp = np.random.rand(5,37)#最小
    mean_temp = np.random.rand(5,37)#均值
    std_temp = np.random.rand(5,37)#方差
    k=721*1440
    for time in date_list:
        data = xr.open_dataset(path+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.nc"))
        data = np.array(data.to_array())
        data = data.squeeze() #5,37,721,1440
        file = main+str(count)+'.npy' #time.strftime("ERA5_plevel_%Y_%m_%d_%H.npy")
        file2 = main+"mean_std_max_min.npz"
        np.save(file,data)
        print(time)
        for var in range(data.shape[0]):
            for plevel in range(data.shape[1]):
                
                if count==0:
                    max_temp[var,plevel] = np.max(data[var,plevel,...])# max
                    min_temp[var,plevel] = np.min(data[var,plevel,...])# min
                    mean_temp[var,plevel] = np.mean(data[var,plevel,...]) # mean
                    std_temp[var,plevel] = np.std(data[var,plevel,...]) # std
                else:
                    if max_temp[var,plevel] < np.max(data[var,plevel,...]):
                        max_temp[var,plevel] = np.max(data[var,plevel,...])
                    if min_temp[var,plevel] > np.min(data[var,plevel,...]):
                        min_temp[var,plevel] = np.min(data[var,plevel,...])
                    mean_temp[var,plevel] = (count*mean_temp[var,plevel]+np.mean(data[var,plevel,...]))/(count+1)
                    std_temp[var,plevel] =  np.sqrt(count/(count+1)*std_temp[var,plevel]**2+np.sum((data[var,plevel,...]-np.expand_dims(np.expand_dims(mean_temp[var,plevel], axis=-1),axis=-1))**2)/((count+1)*k)) 
                    #np.sqrt(count/(count+1)*std_temp[var,plevel]**2+(np.mean(data[var,plevel,...])-mean_temp[var,plevel])**2/(count+1))# std

                np.savez(file2,max=max_temp,min=min_temp,mean=mean_temp,std=std_temp)
        count+=1
print('OK')





