import numpy as np
from glob import glob
import os
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import xarray as xr

# precomputed mean and std of the dataset
level_13 = [0,3,6,11,13,15,17,19,20,22,24,26,28]
level_8 = [0,1,2,3,4,5,6,7]
level_4 = [4,5,6,7]
mesh_folder = '/home/gnn/data_process/meshcnn'
class ClimateSegLoader(Dataset):
    """Data loader for Climate Segmentation dataset."""

    def __init__(self, data_dir, max_level, fl, climatology_dir, mean_, std_, partition="train"):
        """
        Args:
            data_dir: path to data directory
            partition: train or test
        """
        self.partition=partition
        assert(partition in ["train", "test", "val"])
        with open(partition+"_split.txt", "r") as f:
            lines = f.readlines()
        self.flist = [os.path.join(data_dir, l.replace('\n', '')) for l in lines]
        self.level = max_level
        self.mean = mean_  #np.load('mean_std_max_min.npz')
        self.std = std_
        self.forecastlength = fl  #预测h数
        self.climatology_dir = climatology_dir

    def __len__(self):
        length = len(self.flist)-self.forecastlength-2 #len(self.flist)-self.forecastlength-1
        return length

    def __getitem__(self, idx):
        Lat = np.load('LatLon6.npz')['lat']
        Lon = np.load('LatLon6.npz')['lon']
        climatology_dir = self.climatology_dir
        vars = ['geopotential','specific_humidity','temperature','u_component_of_wind','v_component_of_wind']
        Fname = self.flist[idx]
        # Fname1 = self.flist[idx+1]
        # Fname2 = self.flist[idx+2]
        if self.partition == 'train':
            Lname =self.flist[idx+int(self.forecastlength)]
        else:
            Lname =self.flist[idx+int(self.forecastlength/6)]

        ## load climatology
        day_idx = ((idx+self.forecastlength+1)//4)%366
        hour_idx = int(((idx+self.forecastlength+1)%(24/6)))
        clidata = []
        for var in vars:
            CliFile = climatology_dir+str(var)+'/'+str(var)+'_'+str(day_idx)+'.npy'
            temp_var = np.load(CliFile)[...,::4,::4] #采样181 360
            temp_var = np.flip(temp_var[hour_idx,...][None,...],1) #weatherbench2的plevel数据和google的反过来了
            clidata.append(temp_var)
        climatology = np.concatenate(clidata,0)#[...,Lat.astype(int)-1,Lon.astype(int)-1]
        data1 = np.array((xr.open_dataset(Fname)).to_array()).squeeze()
        # data2 = np.array((xr.open_dataset(Fname1)).to_array()).squeeze()
        # data3 = np.array((xr.open_dataset(Fname2)).to_array()).squeeze()
        labels = (np.array((xr.open_dataset(Lname)).to_array())).squeeze()

        data1 = (data1- self.mean[...,None,None][:,level_13]) / self.std[...,None][:,level_13]
        # data2 = (data2- self.mean[...,None,None][:,level_13]) / self.std[...,None][:,level_13]
        # data3 = (data3- self.mean[...,None,None][:,level_13]) / self.std[...,None][:,level_13]
        labels = (labels- self.mean[...,None,None][:,level_13]) / self.std[...,None][:,level_13]

        # data1 = (np.array((xr.open_dataset(Fname)).to_array()).squeeze()- self.msmm['min'][...,None,None][:,level_13]) / (self.msmm['max'][...,None,None][:,level_13]-self.msmm['min'][...,None,None][:,level_13])
        # data2 = (np.array((xr.open_dataset(Fname1)).to_array()).squeeze()- self.msmm['min'][...,None,None][:,level_13]) / (self.msmm['max'][...,None,None][:,level_13]-self.msmm['min'][...,None,None][:,level_13])
        # data3 = (np.array((xr.open_dataset(Fname2)).to_array()).squeeze()- self.msmm['min'][...,None,None][:,level_13]) / (self.msmm['max'][...,None,None][:,level_13]-self.msmm['min'][...,None,None][:,level_13])
        # labels = (np.array((xr.open_dataset(Lname)).to_array()).squeeze()- self.msmm['min'][...,None,None][:,level_13]) / (self.msmm['max'][...,None,None][:,level_13]-self.msmm['min'][...,None,None][:,level_13])

        # data2 = np.expand_dims(data2, axis=0)
        # data1 = np.expand_dims(data1, axis=0)
        # data3=np.expand_dims(data3, axis=0)
        # data = np.concatenate([data1,data2,data3],axis=0)
        
        # label = np.concatenate(label_list,axis=0)
        label = np.concatenate((labels,climatology),0)
        # data = data[:,:,5]
        
        data = data1[:,level_4]
        label = label[:,level_4]

        return data.astype(np.float32), label.astype(np.float32)

