import torch
from tqdm import tqdm
import numpy as np

lanlon = np.load('LatLon.npz')
lat_index = np.array(lanlon['lat'], dtype = int)
lon_index = np.array(lanlon['lon'], dtype = int) #10242

R = 6378
# File1 = str(mesh_folder) + '/lat_' + str(max_level) + '.txt'
# File2 = str(mesh_folder) + '/long_' + str(max_level) + '.txt'
# data_list = []
# with open(File1, encoding='utf-8') as file_obj:
#     for line in file_obj:
#         data_list.append(line.rstrip().split(','))
# lat_icos = np.array(data_list).astype(float).squeeze()

# data_list = []
# with open(File2, encoding='utf-8') as file_obj:
#     for line in file_obj:
#         data_list.append(line.rstrip().split(','))
# lon_icos = np.array(data_list).astype(float).squeeze()

edge_src_target = torch.ones([2,1])
edge_weight = torch.ones([1])
edge_idx = 0

print('start to compute index in mesh')
for idx_icos in tqdm(range(lat_index.shape[0])):
    # lat_index = round((np.pi/2+lat_icos[idx_icos])/np.pi*180)
    # lon_index = round((lon_icos[idx_icos]+np.pi)/(2*np.pi)*359)
    distance_temp = []
    first_handler = 0
    temp_sample = torch.ones([2,1])
    for idx_lat in range(181):
        for idx_lon in range(360):
            lat_angle = np.pi/2-idx_lat/180*np.pi
            lon_angle = idx_lon/359*2*np.pi-np.pi

            lat_icos = np.pi/2-lat_index[idx_icos]/180*np.pi
            lon_icos = lon_index[idx_icos]/359*2*np.pi-np.pi
            fi = np.sin(lat_angle)*np.sin(lat_icos)+np.cos(lat_angle)*np.cos(lat_icos)*np.cos(lon_angle-lon_icos)
    
            if fi>1 and fi<1.1: 
                fi = 1
            elif fi<-1 and fi>-1.1:
                fi=-1
            distance = R*np.arccos(fi)
                # print(fi)
                # print(idx_icos)
                # print(lat_angle,)
            #distance = 27.82*np.sqrt((lat_index-idx_lat)**2+(lon_index-idx_lon)**2) #28*np.sqrt((round((np.pi/2-lat_icos[idx_icos])/np.pi*180)-idx_lat)**2+(round((lon_icos[idx_icos]+np.pi)/(2*np.pi)/(2*np.pi)*359)-idx_lon)**2) #27.8是1度网格长度
            if distance>20037 : #超过一半周长
                distance = 40075-distance
            if (450.0-distance)>0:
                if edge_idx == 0:
                    temp = torch.ones([2,1])
                    temp[1,0]=360*idx_lat+idx_lon #给出flatten后的索引
                    #计算icos上的点的位置
                    temp[0,0]=idx_icos #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*180)+round((lon_icos[idx_icos]+np.pi)/np.pi)*1399
                    edge_src_target = temp
                    distance_temp.append(distance)
                    edge_idx += 1
                else:
                    temp = torch.ones([2,1])
                    temp[1,0] = 360*idx_lat+idx_lon #给出flatten后的索引
                    temp[0,0] = idx_icos #mesh_node[idx_icos] #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*180)+round((lon_icos[idx_icos]+np.pi)/(2*np.pi))*1399
                    edge_src_target = torch.cat((edge_src_target,temp),1)
                    distance_temp.append(distance)
                    edge_idx += 1
    distance_temp = np.array(distance_temp)
    # print(distance_temp.shape)
    distance_temp = torch.tensor((np.sum(distance_temp)-distance_temp)/((distance_temp.shape[0]-1)*np.sum(distance_temp)))
    if idx_lat == 0:
        edge_weight = distance_temp
    else:
        edge_weight = torch.cat((edge_weight,distance_temp),0)
    edge_weight = edge_weight[1:]

    # if idx_icos==0:
    #     edge_src_target = temp_sample
    # else:
    #     sample_rate = round(temp_sample.shape[1]/30)
    #     #print(temp_sample.shape[1])
    #     for num in range(temp_sample.shape[1]):
    #         #num_list = []
    #         if num%sample_rate==0:
    #             #num_list.append = num
    #             sampler = temp_sample[:,num].unsqueeze(1)
    #             #print(sampler.shape)
    #             edge_src_target = torch.cat((edge_src_target,sampler),1)
    #                 #print(edge_src_target.shape)
    edge_idx = edge_src_target.shape[1]
print(edge_src_target.shape)
adj = 'mesh_gen.npz'
np.savez(adj,edge_src_target=edge_src_target,edge_weight=edge_weight)

print('complete computing index in mesh')
