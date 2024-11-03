import xarray as xr
from datetime import timedelta, date,datetime
import cdsapi
import numpy as np
import pandas as pd
import calendar
import os
import datetime as dt
from tqdm import tqdm
import torch
import torch
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from torch.utils.data import Dataset
import pickle, gzip
from torch_sparse import spmm

year=2019
main = '/climate/ERA5'
base = dt.datetime(year,1,1,0)
date_list = [base + dt.timedelta(days=x) for x in range(365)]
vars = ['geopotential', 'specific_humidity', 'temperature','u_component_of_wind', 'v_component_of_wind']
levels = ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400','350', '300',
          '250', '225', '200', '175', '150', '125', '100', '70', '50', '30', '20', '10', '7', '5', '3', '2','1']

# levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300','350', '400',
#           '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975','1000']
device = torch.device('cuda:1')

def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor(np.array([m.row, m.col]))
    v = torch.FloatTensor(m.data)

    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))

def s2IV(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor(np.array([m.row, m.col]))
    v = torch.FloatTensor(m.data)

    return i,v

def dense2sparseMM(m,n):
    """
    稠密x稀疏矩阵转稀疏矩阵乘,输出稠密矩阵
    """
    #m稀疏 n稠密
    m = m.squeeze()
    M1 = m.shape[0]
    N1 = m.shape[1]
    M2 = n.shape[0]
    N2 = n.shape[1]
    n=n.T
    m=m.T
    # ED1 = sparse.coo_matrix(m)
    # ED2 = sparse.coo_matrix(n)
    # i2 = torch.nonzero(n).T
    i2 = torch.where(n!=0)
    v2  = n[i2]
    i2 = torch.stack(i2)
    # v2 = n[i2[0],i2[1]]
    out = spmm(i2,v2,N2,M2,m)
    out = out.T
    out = out.unsqueeze(0)
    return out

def Spmm_for_grid2mesh(m,ni,nv,nshape):
    out = []
    for batch in range(m.shape[0]):
        out.append(spmm(ni,nv,nshape[0],nshape[1],m[batch,...]).unsqueeze(0))
    out = torch.cat(out,0)
    return out

def SPmm(m,ni,nv,nshape):
    """
    稠密x稀疏矩阵转稀疏矩阵乘,输出稠密矩阵
    """
    #m稀疏 n稠密
    # bs, feat,10242   x 10242,61440

    bs = m.shape[0]
    m = torch.flatten(m,start_dim=0, end_dim=1)
    # out = []
    # for batch in range(m.shape[0]):
    #     out.append(spmm(ni,nv,nshape[0],nshape[1],m[batch,...].T).T.unsqueeze(0))
    out = spmm(ni,nv,nshape[0],nshape[1],m.T)
    out = out.T
    out = out.view(bs,-1,nshape[0])
    # out = torch.cat(out,0)
    return out

def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape #V x newlen
    """

    return torch.matmul(den, sp)


def xyz2latlong(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    long = np.arctan2(y, x)
    xy2 = x**2 + y**2
    lat = np.arctan2(z, np.sqrt(xy2))
    return lat, long

def interp_r2tos2(sig_r2, V, method="linear", dtype=np.float32):
    """
    sig_r2: rectangular shape of (lat, long, n_channels)
    V: array of spherical coordinates of shape (n_vertex, 3)
    method: interpolation method. "linear" or "nearest"
    """
    ele, azi = xyz2latlong(V)
    nlat, nlong = sig_r2.shape[0], sig_r2.shape[1]
    dlat, dlong = np.pi/(nlat-1), 2*np.pi/nlong
    lat = np.linspace(-np.pi/2, np.pi/2, nlat)
    long = np.linspace(-np.pi, np.pi, nlong+1)
    sig_r2 = np.concatenate((sig_r2, sig_r2[:, 0:1]), axis=1)
    intp = RegularGridInterpolator((lat, long), sig_r2, method=method)
    s2 = np.array([ele, azi]).T
    sig_s2 = intp(s2).astype(dtype)
    return sig_s2

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
    if not os.path.exists(adj):
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
        np.savez(adj,edge_src_target=edge_src_target,edge_weight=edge_weight)
        print('complete computing index in mesh')
    else:
        edge_src_target,edge_weight = np.load(adj)['edge_src_target'],np.load(adj)['edge_weight']
    return edge_src_target,edge_weight

max_level = 6 #5
mesh_folder = '/home/gnn/data_process/meshcnn'
path2 = '/climate/ERA5/icos6/'
edge_src_target,edge_weight = adjinit(max_level,mesh_folder)
nshape_edge = (10*4**max_level+2,721*1440)

for time in tqdm(date_list): #每天的5个变量，37层
    day_list = []
    for varr in vars: #5变量
        plevel_list = []
        for plevel in levels: #37层
            file_path = main+'/'+str(time.strftime('%Y'))+'/'+str(time.strftime('%m'))+'/'+str(time.strftime('%d'))+'/'+str(varr)+'/'+str(plevel)+'.nc'
            temp = xr.open_dataset(file_path)
            Te = np.array(temp.to_array())
            plevel_list.append(Te)
            var_temp = np.concatenate(plevel_list,0)
        day_list.append(var_temp[None,...])
    day_temp = np.concatenate(day_list,0) # 5,37,24,721,1440
    for hour in range(day_temp.shape[2]):
        inputing = day_temp[:,:,hour,...]            
        Ff = torch.tensor(inputing).unsqueeze(0)
        NEW_file = pre_load_target(Ff,edge_src_target, edge_weight, nshape_edge)
        NEW_file.squeeze()
        NEW_file = np.array(NEW_file.cpu())
        file_name = time.strftime("ERA5_plevel_%Y_%m_%d")+'_'+str(hour)
        np.save(path2+file_name+'.npy',NEW_file)

date_list2 = [base + dt.timedelta(hours=x) for x in range(365*24)]
path_ = '/climate/ERA5/icos6/'
path1_ = '/climate/ERA5/icosnew/'

for time in tqdm(date_list2):
    # int(time.strftime('%H'))
    file = path_+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_")+str(int(time.strftime('%H')))+'.npy'
    BB=np.load(file)
    BB=BB.squeeze()
    file1 = path1_+os.sep+time.strftime("ERA5_plevel_%Y_%m_%d_%H.npy")
    np.save(file1,BB)



        
