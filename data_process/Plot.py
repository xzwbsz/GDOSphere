import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib import rcParams
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

config = {'font.family':'Times New Roman', 'font.size':12}
rcParams.update(config)
data = np.random.rand(5, 37, 721, 1440)
std = np.random.rand(5)
mean = np.random.rand(5)
for n in range(5):
    data[n] = data[n] * std[n] + mean[n]
data_t = data[0]
data_p = data[1]
data_q = data[2]
data_u = data[3]
data_v = data[4]
output = xr.Dataset({'t':(('level', 'lat', 'lon'), data_t), 'p':(('level', 'lat', 'lon'), data_p), 'q':(('level', 'lat', 'lon'), data_q), 'u':(('level', 'lat', 'lon'), data_u), 'v':(('level', 'lat', 'lon'), data_v)}, \
                        coords = {'level':np.arange(37), 'lat':np.arange(90, -90.001, -0.25), 'lon':np.arange(-180, 180, 0.25)})
output.to_netcdf('/output.nc')

output = xr.open_dataset('/output.nc')
# data_t = np.array(output['t'])
# data_p = np.array(output['p'])
# data_q = np.array(output['q'])
data_u = np.array(output['u'])[10]
data_v = np.array(output['v'])[10]
lon = np.array(output['lon'])
lat = np.array(output['lat'])
wnd = (data_u ** 2 + data_v ** 2) ** 0.5
fig = plt.figure(figsize = (8, 8))
ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 1)
ax.set_extent([lon[0], lon[-1], lat[-1], lat[0]], crs = ccrs.PlateCarree())
ax.set_xticks(np.arange(lon[0], lon[-1] + 0.1, 20))
ax.set_yticks(np.arange(lat[-1], lat[0] + 0.1, 20))
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(axis = 'both', which = 'major', labelsize = 10, direction = 'out', length = 5, width = 1, pad = 2, top = False, right = False)
contourf = plt.contourf(lon, lat, wnd, transform = ccrs.PlateCarree(), cmap = 'coolwarm')
colorbar = fig.colorbar(contourf, shrink = 0.7, orientation = 'horizontal', pad = 0.05, label = 'Wind Speed (m/s)')
plt.quiver(lon[::50], lat[::50], data_u[::50, ::50], data_v[::50, ::50])
plt.title('Wind', loc = 'center', fontsize = 25)
plt.savefig('output.png', dpi = 1000)