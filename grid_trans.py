import math
import argparse
import sys
sys.path.append("../../meshcnn")
import numpy as np
import pickle, gzip
import os
import shutil
import logging
from collections import OrderedDict
from tqdm import tqdm
import sys

from ops import MeshConv
from loader import ClimateSegLoader
from model import SphericalUNet, Multi_uunet
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from utils import Spmm_for_grid2mesh,SPmm
import collections 
import collections.abc

import os

device = torch.device('cuda')

def pre_load_target(data,edge_src_target,edge_weight,nshape_edge):
    # data = torch.tensor(data)
    data = data.to(device)
    shape = data.shape
    edge_src_target = torch.tensor(edge_src_target).type(torch.long).to(device)
    data = torch.flatten(data,start_dim=1, end_dim=2) # bs,185,721,1440
    data = torch.flatten(data,start_dim=-2, end_dim=-1) # bs,370,721x1440
    nv = torch.tensor(edge_weight).to(device)#边权重
    data = SPmm(data,edge_src_target,nv,nshape_edge)# bs,370,10242
    data = data.view(shape[0],shape[1],shape[2],data.shape[-1])
    return data

def adjinit(max_level,mesh_folder):

    adj = './mesh_idx_'+str(max_level)+'.npz'
    edge_sample = 5
    if not os.path.exists(adj):
        lanlon = np.load('LatLon.npz')
        lat_index = np.array(lanlon['lat'], dtype = int)
        lon_index = np.array(lanlon['lon'], dtype = int) #10242
        R = 6378
        edge_src_target = torch.ones([2,1])
        edge_weight = torch.ones([1])
        edge_idx = 0

        print('start to compute index in mesh')
        for idx_icos in tqdm(range(lat_index.shape[0])):
            # lat_index = round((np.pi/2+lat_icos[idx_icos])/np.pi*720)
            # lon_index = round((lon_icos[idx_icos]+np.pi)/(2*np.pi)*1439)
            distance_temp = []
            first_handler = 0
            temp_sample = torch.ones([2,1])
            for idx_lat in range(721):
                for idx_lon in range(1440):
                    lat_angle = np.pi/2-idx_lat/720*np.pi
                    lon_angle = idx_lon/1439*2*np.pi-np.pi

                    lat_icos = np.pi/2-lat_index[idx_icos]/720*np.pi
                    lon_icos = lon_index[idx_icos]/1439*2*np.pi-np.pi
                    fi = np.sin(lat_angle)*np.sin(lat_icos)+np.cos(lat_angle)*np.cos(lat_icos)*np.cos(lon_angle-lon_icos)
                    if fi>1 and fi<1.1: 
                        fi = 1
                    elif fi<-1 and fi>-1.1:
                        fi=-1
                    distance = R*np.arccos(fi)
                        # print(fi)
                        # print(idx_icos)
                        # print(lat_angle,)
                    #distance = 27.82*np.sqrt((lat_index-idx_lat)**2+(lon_index-idx_lon)**2) #28*np.sqrt((round((np.pi/2-lat_icos[idx_icos])/np.pi*720)-idx_lat)**2+(round((lon_icos[idx_icos]+np.pi)/(2*np.pi)/(2*np.pi)*1439)-idx_lon)**2) #27.8是1度网格长度
                    if distance>20037 : #超过一半周长
                        distance = 40075-distance
                    if (110.0-distance)>0:
                        if edge_idx == 0:
                            temp = torch.ones([2,1])
                            temp[1,0]=1440*idx_lat+idx_lon #给出flatten后的索引
                            #计算icos上的点的位置
                            temp[0,0]=idx_icos #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*720)+round((lon_icos[idx_icos]+np.pi)/np.pi)*1399
                            edge_src_target = temp
                            distance_temp.append(distance)
                            edge_idx += 1
                        else:
                            temp = torch.ones([2,1])
                            temp[1,0] = 1440*idx_lat+idx_lon #给出flatten后的索引
                            temp[0,0] = idx_icos #mesh_node[idx_icos] #721*(round((np.pi/2-lat_icos[idx_icos])/np.pi)*720)+round((lon_icos[idx_icos]+np.pi)/(2*np.pi))*1399
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

        print(edge_src_target.shape)
        np.savez(adj,edge_src_target=edge_src_target,edge_weight=edge_weight)
        print('complete computing index in mesh')
    else:
        edge_src_target,edge_weight = np.load(adj)['edge_src_target'],np.load(adj)['edge_weight']
    return edge_src_target,edge_weight

max_level = 5
mesh_folder = '/xuzhewen/gnn/data_process/meshcnn/'
path = '/xuzhewen/gnn/data2/ERA5/'
path2 = '/xuzhewen/gnn/data2/icos/'
edge_src_target,edge_weight = adjinit(max_level,mesh_folder)
nshape_edge = (10*4**max_level+2,721*1440)

for idx in tqdm(range(9000)):
    file = path+str(idx)+'.npy'
    Ff = np.load(file)
    Ff = torch.tensor(Ff).unsqueeze(0)
    # print(Ff.shape)
    NEW_file = pre_load_target(Ff,edge_src_target, edge_weight, nshape_edge)
    NEW_file.squeeze()
    NEW_file = np.array(NEW_file.cpu())
    np.save(path2+str(idx)+'.npy',NEW_file)