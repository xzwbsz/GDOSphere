import numpy as np
from scipy.interpolate import Rbf

# 已知的全球坐标值
 # 纬度值
 # 经度值
lats = np.load('../data_process/LatLon6.npz')['lat'].astype(int)
lons = np.load('../data_process/LatLon6.npz')['lon'].astype(int)
values = np.load('../Diffgraph_clima/result_temp.npy')[0,0,0]       # 对应的数值

def Decoding_interpolate(lats,lons,values):
# 创建Rbf对象
    rbf = Rbf(lats, lons, values, function='gaussian')
    
    # 需要插值的新点
    new_lats = np.linspace(-90.1, 90, 181)  # 新的纬度值
    new_lons = np.linspace(-180, 180, 360) # 新的经度值
    
    # 生成网格的新点
    new_lats, new_lons = np.meshgrid(new_lats, new_lons)
    
    # 进行插值
    interpolated_values = rbf(new_lats, new_lons)
    return interpolated_values


aka=Decoding_interpolate(lats,lons,values)
print(aka.shape)
