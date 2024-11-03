import numpy as np
import torch
from tqdm import tqdm
import folium
from mpl_toolkits.basemap import Basemap  #导入Basemap库
import matplotlib.pyplot as plt

mesh_folder = '/xuzhewen/gnn/data_process/meshcnn'
max_level = 4
File1 = str(mesh_folder) + '/lat_' + str(max_level) + '.txt'
File2 = str(mesh_folder) + '/long_' + str(max_level) + '.txt'
data_list = []
with open(File1, encoding='utf-8') as file_obj:
    for line in tqdm(file_obj):
        data_list.append(line.rstrip().split(','))
lat_icos = np.array(data_list).astype(float).squeeze()

data_list = []
with open(File2, encoding='utf-8') as file_obj:
    for line in tqdm(file_obj):
        data_list.append(line.rstrip().split(','))
lon_icos = np.array(data_list).astype(float).squeeze()  

print(lat_icos.min())
k1=66
k2=874
lat_angle = lat_icos/np.pi*180
lon_angle = lon_icos/np.pi*180
# for place in tqdm(range(lat_angle.shape[0])):
#     m = folium.Map(location=[lat_angle[place], lon_angle[place]], zoom_start=10)
# m.save('map.html')

m = Basemap(resolution='l')     # 实例化一个map
m.drawcoastlines()  # 画海岸线
m.drawmapboundary(fill_color='aqua') 
m.drawcountries()
#m.drawrivers()
m.fillcontinents(color='lightgray',lake_color='blue') # 画大洲，颜色填充为褐色

parallels = np.arange(-90., 90., 15.)  # 这两行画纬度，范围为[-90,90]间隔为10
m.drawparallels(parallels,labels=[False, True, True, False])
meridians = np.arange(-180., 180., 30.)  # 这两行画经度，范围为[-180,180]间隔为10
m.drawmeridians(meridians,labels=[True, False, False, True])

# for idx in range(lat_angle.shape[0]):
#     m.plot(lon_angle[idx],lat_angle[idx],marker='D',color='yellow')
m.plot(lon_angle[k1],lat_angle[k1],marker='D',color='yellow')
m.plot(lon_angle[k2],lat_angle[k2],marker='D',color='yellow')

plt.savefig('fig1.png')

adj = './icosphere_idx_'+str(max_level)+'.npz'

lat_array = np.array(lat_icos, dtype = int)
lon_array = np.array(lon_icos, dtype = int)
R = 6378
edge_src_target = torch.ones([2,1])
edge_weight = torch.ones([1])
edge_idx = 0

print('start to compute index in mesh')

edge_num = 30
for level in range(max_level+1):
    if level == 0:
        edge_num = 30
    else:
        edge_num = int(2*edge_num + 3*20*4**(level-1))
print('共有',lat_icos.shape[0],'个点')
print('共有',edge_num,'条边')
print('共有',20*4**(max_level),'个面')
approxi_edge_len = (510067866/(20*4**(max_level))*4/((3)**0.5))**0.5
print('预计棱长为',approxi_edge_len,'km')

fi = np.sin(lat_array[:,None])*np.sin(lat_array[None,:])+np.cos(lat_array[:,None])*np.cos(lat_array[None,:])*np.cos(lon_array[:,None]-lon_array[None,:])
fi[np.where(fi<-1.)] = -1.
fi[np.where(fi>1.)] = 1.
distance_matrix = np.round(R*np.arccos(fi),4)

iidx,iidy = np.unravel_index(np.where(np.argsort(distance_matrix.flatten())<(2*edge_num+lat_icos.shape[0]))[0],distance_matrix.shape)
distance_matrix[np.where(distance_matrix==0.)]=9999999
# np.savetxt('file_path.csv', distance_matrix, delimiter=",")
print('有边',distance_matrix.min())
print('最大距离为',distance_matrix.flatten()[np.where(np.argsort(distance_matrix.flatten())<(2*edge_num+lat_icos.shape[0]))[0]].max())
print('共应有',iidx.shape,'条边')
edge_src_target = np.array([iidx,iidy])

# dis_final1 = np.ones((2,lat_array.shape[0],lat_array.shape[0]))
# for idx1 in tqdm(range(lat_array.shape[0])):
#     for idx2 in range(lat_array.shape[0]):
#         dis_final1[0,idx1,idx2] = R*np.arccos(np.sin(lat_array[idx1])*np.sin(lat_array[idx2])+np.cos(lat_array[idx1])*np.cos(lat_array[idx2])*np.cos(lon_array[idx1]-lon_array[idx2]))
#         distt = np.sqrt((lat_angle[idx1]-lat_angle[idx2])**2+(lon_angle[idx1]-lon_angle[idx2])**2)*110
#         if distt > 40075/2:
#             distt = 40075 - distt
#         dis_final1[1,idx1,idx2] = distt
# np.save('test_for_edge.npy',dis_final1)

# res = dis_final1[0] - distance_matrix
# np.save('test_for_res.npy',res)

print(edge_src_target.shape)
np.savez(adj,edge_src_target=edge_src_target,edge_weight=edge_weight)
print('complete computing index in mesh')