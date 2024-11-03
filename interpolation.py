import os
import cmaps
import pickle
import numpy as np
import xarray as xr
import netCDF4 as nc
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
# from global_land_mask import globe
from scipy.interpolate import griddata
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

bt_all = np.load('datatest.npy') # array()
bt_all = bt_all[0,0,5,:]
lat_all = 90-np.load('LatLon.npz')['lat']/4
lon_all = np.load('LatLon.npz')['lon']/4 -180
points = np.concatenate([lon_all.reshape(-1, 1), lat_all.reshape(-1, 1)], axis = 1)
lon = np.arange(-180, 180, 0.25) #经度，分辨率
lat = np.arange(-90.1,90,0.25) #纬度，分辨率
lon_grid, lat_grid = np.meshgrid(lon, lat)
bt = griddata(points, bt_all.reshape(-1, 1), (lon_grid, lat_grid), method = 'linear')[:, :, 0]

fig = plt.figure(figsize = (8, 8))
ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
ax.set_extent([lon[0], lon[-1], lat[-1], lat[0]], crs = ccrs.PlateCarree())
ax.set_xticks(np.arange(lon[0], lon[-1] + 0.1, 20))
ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
contourf = plt.contourf(lon, lat, bt, transform = ccrs.PlateCarree(), cmap = 'coolwarm')
colorbar = fig.colorbar(contourf, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'Geopotential')
# plt.quiver(lon[::50], lat[::50], data_u[::50, ::50], data_v[::50, ::50])
plt.title('Z', loc = 'center', fontsize = 25)
plt.savefig('output.png', dpi = 100)

bt_all = np.load('data_global.npy')
lat_all = np.load('LatLon.npz')['lat']/4-90
lon_all = np.load('LatLon.npz')['lon']/4 -180
bt_all = bt_all[0,0,5,:]
points = np.concatenate([lon_all.reshape(-1, 1), lat_all.reshape(-1, 1)], axis = 1)
at = griddata(points, bt_all.reshape(-1, 1), (lon_grid, lat_grid), method = 'linear')[:, :, 0]
fig = plt.figure(figsize = (8, 8))
ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
ax.set_extent([lon[0], lon[-1], lat[-1], lat[0]], crs = ccrs.PlateCarree())
ax.set_xticks(np.arange(lon[0], lon[-1] + 0.1, 20))
ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
contourf = plt.contourf(lon, lat, at, transform = ccrs.PlateCarree(), cmap = 'coolwarm')
colorbar = fig.colorbar(contourf, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'Geopotential')
# plt.quiver(lon[::50], lat[::50], data_u[::50, ::50], data_v[::50, ::50])
plt.title('Z', loc = 'center', fontsize = 25)
plt.savefig('output_ori.png', dpi = 100)

bt_all = np.load('grid.npy')
bt_all = bt_all[0,0,5,:]
lat_all = np.load('LatLon.npz')['lat']/4 - 90
lon_all = np.load('LatLon.npz')['lon']/4 -180
#at = griddata(points, bt_all.reshape(-1, 1), (lon_grid, lat_grid), method = 'linear')[:, :, 0]
fig = plt.figure(figsize = (8, 8))
ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
ax.set_extent([lon[0], lon[-1], lat[-1], lat[0]], crs = ccrs.PlateCarree())
ax.set_xticks(np.arange(lon[0], lon[-1] + 0.1, 20))
ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
contourf = plt.contourf(lon, lat, bt_all, transform = ccrs.PlateCarree(), cmap = 'coolwarm')
colorbar = fig.colorbar(contourf, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'Geopotential')
# plt.quiver(lon[::50], lat[::50], data_u[::50, ::50], data_v[::50, ::50])
plt.title('Z', loc = 'center', fontsize = 25)
plt.savefig('output_grid.png', dpi = 100)