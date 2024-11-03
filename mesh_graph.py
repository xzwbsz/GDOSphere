import numpy as np
from tqdm import tqdm
import torch
mesh_folder = '/xuzhewen/gnn/data_process/meshcnn'
max_level = 5
R = 6371
File1 = str(mesh_folder) + '/lat_' + str(max_level) + '.txt'
File2 = str(mesh_folder) + '/long_' + str(max_level) + '.txt'
data_list = []
with open(File1, encoding='utf-8') as file_obj:
    for line in file_obj:
        data_list.append(line.rstrip().split(','))
lat_icos = np.array(data_list).astype(float).squeeze()
#print(lat_icos)
data_list = []
with open(File2, encoding='utf-8') as file_obj:
    for line in file_obj:
        data_list.append(line.rstrip().split(','))
lon_icos = np.array(data_list).astype(float).squeeze()
#print(lon_icos)
edge_src_target = torch.ones([2,1])
mesh_node = torch.rand(lon_icos.shape[0])
print('start to compute index in mesh')
for idx_icos in tqdm(range(lon_icos.shape[0])):
    lat_index = round((np.pi/2+lat_icos[idx_icos])/np.pi*720)
    lon_index = round((lon_icos[idx_icos]+np.pi)/(2*np.pi)*1439)
    #该icos上的点对应在平面网格上的索引
    mesh_node[idx_icos] = lat_index*1440-1+lon_index #(round((np.pi/2-lat_icos[idx_icos])/np.pi*720))*1440-1+round((lon_icos[idx_icos]+np.pi)/(2*np.pi)*1439)
    edge_idx = 0
    first_handler = 0
    temp_sample = torch.rand(2,1)
    for idx_lat in range(721):
        for idx_lon in range(1440):
            lat_angle = np.pi/2-idx_lat/720*np.pi
            lon_angle = idx_lon/1439*2*np.pi-np.pi
            distance = R*np.arccos(np.sin(lat_angle)*np.sin(lat_icos[idx_icos])+np.cos(lat_angle)*np.cos(lat_icos[idx_icos])*np.cos(lon_angle-lon_icos[idx_icos]))
            #distance = 27.82*np.sqrt((lat_index-idx_lat)**2+(lon_index-idx_lon)**2) #28*np.sqrt((round((np.pi/2-lat_icos[idx_icos])/np.pi*720)-idx_lat)**2+(round((lon_icos[idx_icos]+np.pi)/(2*np.pi)/(2*np.pi)*1439)-idx_lon)**2) #27.8是1度网格长度
            if distance>20037 : #超过一半周长
                distance = 40075-distance
            if (150.0-distance)>0:
                #print(edge_idx)
                #print('lat_angle=',lat_angle,'lat_icos=',lat_icos[idx_icos],'lon_angle=',lon_angle,'lon_icos=',lon_icos[idx_icos])
                # if edge_idx > 2000:
                #     sys.exit()
                if first_handler == 0:
                    temp = torch.rand(2,1)
                    temp[0,0]=1440*idx_lat-1+idx_lon #给出flatten后的索引
                    #计算icos上的点的位置
                    temp[1,0]=mesh_node[idx_icos] #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*720)+round((lon_icos[idx_icos]+np.pi)/np.pi)*1399
                    temp_sample = temp
                    first_handler = 1
                else:
                    #temp = torch.rand(2,1)
                    temp[1,0] = 1440*idx_lat+idx_lon #给出flatten后的索引
                    temp[0,0] = idx_icos #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*720)+round((lon_icos[idx_icos]+np.pi)/(2*np.pi))*1399
                    temp_sample = torch.cat((temp_sample,temp),1)
    if idx_icos==0:
        edge_src_target = temp_sample
    else:
        sample_rate = round(temp_sample.shape[1]/50)
        #print(temp_sample.shape[1])
        for num in range(temp_sample.shape[1]):
            #num_list = []
            if num%sample_rate==0:
                #num_list.append = num
                sampler = temp_sample[:,num].unsqueeze(1)
                #print(sampler.shape)
                edge_src_target = torch.cat((edge_src_target,sampler),1)
                    #print(edge_src_target.shape)
    edge_idx = edge_src_target.shape[1]
print(edge_src_target.shape)
adj = 'Mesh_5_edge.npz'
np.savez(adj,edge_src_target=edge_src_target,mesh_node=mesh_node)

print('complete computing index in mesh')