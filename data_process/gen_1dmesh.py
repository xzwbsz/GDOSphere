import os
import torch
import numpy as np
from tqdm import tqdm
mesh_folder = 'meshcnn'
max_level = 6
adj = './mesh_idx_'+str(max_level)+'.npz'
deg = 0.25
lat_C,lon_C = round(180/deg)+1, round(360/deg)
# edge_src_target = torch.ones([2,1])
edge_weight = torch.ones([1])
if not os.path.exists(adj):
    R = 6371
    File1 = str(mesh_folder) + '/lat_' + str(max_level) + '.txt'
    File2 = str(mesh_folder) + '/long_' + str(max_level) + '.txt'
    data_list = []
    with open(File1, encoding='utf-8') as file_obj:
        for line in file_obj:
            data_list.append(line.rstrip().split(','))
    lat_icos = np.array(data_list).astype(float).squeeze()

    data_list = []
    with open(File2, encoding='utf-8') as file_obj:
        for line in file_obj:
            data_list.append(line.rstrip().split(','))
    lon_icos = np.array(data_list).astype(float).squeeze()

    print('start to compute index in mesh')
    for idx_icos in tqdm(range(lon_icos.shape[0])):
        Est = []
        distance_temp = []
        edge_idx = 0
        first_handler = 0
        temp_sample = torch.rand(2,1)
        for idx_lat in range(lat_C):
            for idx_lon in range(lon_C):
                lat_angle = np.pi/2-idx_lat/(lat_C-1)*np.pi
                lon_angle = idx_lon/(lon_C-1)*2*np.pi-np.pi
                fi = np.sin(lat_angle)*np.sin(lat_icos[idx_icos])+np.cos(lat_angle)*np.cos(lat_icos[idx_icos])*np.cos(lon_angle-lon_icos[idx_icos])
                fi = np.clip(fi, -1, 1) #fi有可能等于1.000002
                distance = R*np.arccos(fi)
                #distance = 27.82*np.sqrt((lat_index-idx_lat)**2+(lon_index-idx_lon)**2) #28*np.sqrt((round((np.pi/2-lat_icos[idx_icos])/np.pi*720)-idx_lat)**2+(round((lon_icos[idx_icos]+np.pi)/(2*np.pi)/(2*np.pi)*1439)-idx_lon)**2) #27.8是1度网格长度
                if distance>20037 : #超过一半周长
                    distance = 40075-distance
                if (98.0-distance)>0:
                    if edge_idx==0:
                        temp = torch.ones([2,1])
                        temp[1,0]=lon_C*idx_lat+idx_lon #给出flatten后的索引
                        #计算icos上的点的位置
                        temp[0,0]=idx_icos #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*720)+round((lon_icos[idx_icos]+np.pi)/np.pi)*1399
                        Est.append(temp)
                        edge_idx+=1
                        distance_temp.append(distance)
        distance_temp = np.array(distance_temp)
        # print(distance_temp.shape)
        if distance_temp.shape[0] !=1:
            # print(lat_icos[idx_icos],lon_icos[idx_icos],lat_angle,lon_angle)
            # print(distance_temp)
            distance_temp = torch.tensor((np.sum(distance_temp)-distance_temp)/((distance_temp.shape[0]-1)*np.sum(distance_temp)))
        else:
            distance_temp = torch.tensor(distance_temp)
        edge_weight = torch.cat((edge_weight,distance_temp),0)       
        edge_src_target = np.array(torch.cat(Est,-1))
    print(edge_src_target.shape)
    print(edge_idx)
    print(edge_weight)
    edge_weight = edge_weight[1:]
    np.savez(adj,edge_src_target=edge_src_target,edge_weight=edge_weight)

    print('complete computing index in mesh')

