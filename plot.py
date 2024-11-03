import os
import cmaps
import pickle
import numpy as np
import xarray as xr
# import netCDF4 as nc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
# from global_land_mask import globe
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

def weighted_rmse(x,real):
    rmse = 0
    sum=0
    for lat_idx in range(x.shape[0]):
        sum+=np.sin(lat_idx/x.shape[0]*np.pi)
    
    for lat in range(x.shape[0]):
        weight = np.sin(lat/x.shape[0]*np.pi)
        rmse+=weight*np.sqrt(mean_squared_error(x[lat],real[lat]))/sum
    return rmse

deg = 1
msmm = np.load('mean_std_max_min.npz')
mean_=msmm['mean'].astype(np.float32)
std_=msmm['std'].astype(np.float32)

lon = np.arange(-180, 180, deg) #经度，分辨率
lat = np.arange(-90.1,90,deg) #纬度，分辨率
# points = np.concatenate([lon_all.reshape(-1, 1), lat_all.reshape(-1, 1)], axis = 1)
aka_path = '/data2/xzwdir/inferdata/'
# bt_all = xr.open_dataset('ERA5_plevel_2014_01_01_13.nc')
# bt_all = np.array(bt_all.to_array())
bt_all = np.load('result_temp.npy')
# bt_all = np.load(aka_path+'result_temp_12218.npy')
at_all = np.load('label_temp.npy')

var = 0
k = 1
pred = bt_all[0]
real = at_all[0]
pred = pred[var,k,...]*std_[var,15,0]+mean_[var,15]
real = real[var,k,...]*std_[var,15,0]+mean_[var,15]
# bt_all = (bt_all-mean_[var,15])/std_[var,15]
# at_all = (at_all-mean_[var,15])/std_[var,15]
# bt_all = abs(bt_all - at_all) #(((bt_all - at_all)**2).mean())**0.5
print((weighted_rmse(pred,real)))
lon_grid, lat_grid = np.meshgrid(lon, lat)

# bt = griddata(points, bt_all.reshape(-1, 1), (lon_grid, lat_grid), method = 'linear')[:, :, 0]

# fig = plt.figure(figsize = (8, 8))
# ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
# ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
# # ax.set_extent([lon[0], lon[-1], lat[-1], lat[0]], crs = ccrs.PlateCarree())
# # ax.set_xticks(np.arange(lon[0], lon[-1] + 0.1, 20))
# # ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
# # ax.xaxis.set_major_formatter(LongitudeFormatter())
# # ax.yaxis.set_major_formatter(LatitudeFormatter())
# ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
# contourf = plt.contourf(lon, lat, bt_all, transform = ccrs.PlateCarree(), cmap = 'coolwarm')
# colorbar = fig.colorbar(contourf, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'Geopotential')

fig, axs = plt.subplots(2, figsize=(8, 8))
axs[0].imshow(pred, cmap='viridis')
axs[1].imshow(real, cmap='viridis')

# fig, axs = plt.subplots(1, figsize=(8, 8))
# axs.imshow(bt_all, cmap='viridis')

plt.title('Z500', loc = 'center', fontsize = 25)
plt.savefig('re24.png', dpi = 1000)

# import pandas as pd

# df = pd.DataFrame(bt_all)
# df.to_csv("out.csv", index=False)

# df2 = pd.DataFrame(at_all)
# df2.to_csv("label.csv", index=False)
