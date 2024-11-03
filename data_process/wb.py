import xarray as xr
import numpy as np
from tqdm import tqdm
climatology = xr.open_zarr('gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr')
# climatology
print('下载位势中')
geopotential=climatology.geopotential
for idx in tqdm(range(geopotential.shape[1])):
    temp=geopotential[:,idx]
    file_name = 'geopotential/geopotential'+'_'+str(idx)+'.npy'
    np.save(file_name,temp)
print('下载完成')
print('下载湿度中')
specific_humidity=climatology.specific_humidity
for idx in tqdm(range(specific_humidity.shape[1])):
    temp=specific_humidity[:,idx]
    file_name = 'specific_humidity/specific_humidity'+'_'+str(idx)+'.npy'
    np.save(file_name,temp)
print('下载完成')
print('下载温度中')
temperature=climatology.temperature
for idx in tqdm(range(temperature.shape[1])):
    temp=temperature[:,idx]
    file_name = 'temperature/temperature'+'_'+str(idx)+'.npy'
    np.save(file_name,temp)
print('下载完成')
print('下载U风中')
u_component_of_wind=climatology.u_component_of_wind
for idx in tqdm(range(u_component_of_wind.shape[1])):
    temp=u_component_of_wind[:,idx]
    file_name = 'u_component_of_wind/u_component_of_wind'+'_'+str(idx)+'.npy'
    np.save(file_name,temp)
print('下载完成')
print('下载V风中')
v_component_of_wind=climatology.v_component_of_wind
for idx in tqdm(range(v_component_of_wind.shape[1])):
    temp=v_component_of_wind[:,idx]
    file_name = 'v_component_of_wind/v_component_of_wind'+'_'+str(idx)+'.npy'
    np.save(file_name,temp)
print('下载完成')