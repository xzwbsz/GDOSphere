import numpy as np
import torch

lanlon = np.load('LatLon6.npz')
lat_index = np.array(lanlon['lat'], dtype = float)
lon_index = np.array(lanlon['lon'], dtype = float) #10242

R = 6378
edge_src_target = torch.ones([2,1])
edge_weight = torch.ones([1])
edge_idx = 0

print('start to compute index in mesh')
Dis1 = (torch.arange(1440))/1439*360
Dis2 = (torch.arange(721)/720-0.5)*180
latdis_temp = torch.ones((1,721))
londis_temp = torch.ones((1440,1))
Dis1 = torch.flatten(Dis1[:,None]*latdis_temp)
Dis2 = torch.flatten(londis_temp*Dis2[None,:])
lat_index_=torch.tensor(lat_index[:,None])
lon_index_=torch.tensor(lon_index[None,:])

fi = torch.sin(Dis1)*torch.sin(lat_index_)+torch.cos(Dis1)*torch.cos(lat_index_)*torch.cos(Dis2-lon_index_)
fi = np.array(fi)
fi = np.where(fi>1,1,fi)
distance = R*np.arccos(fi)
distance = np.where(distance>20037,40075-distance,distance)
print(fi.shape)

