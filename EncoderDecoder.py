import numpy as np
import torch
from torch_geometric.nn import GCNConv,GATConv
import torch.nn as nn

# Lat = torch.tensor(np.load('LatLon6.npz')['lat'],dtype=torch.long)
# Lon = torch.tensor(np.load('LatLon6.npz')['lon'],dtype=torch.long)

def Grid2Mesh(x,factor,Lat,Lon):
    x_ = x[...,:-1,:]
    x_pole_ = x[...,-1,0].unsqueeze(-1).unsqueeze(-1) #南极
    x_pole = torch.ones(x.shape[0],x.shape[1],1,x.shape[-1]*factor, device=x.device)

    x_pole = x_pole*x_pole_
    x = torch.nn.functional.interpolate(x_, scale_factor=factor, mode='bilinear')
    x = torch.cat((x,x_pole),-2)
    Lat = Lat.to(x.device)
    Lon = Lon.to(x.device)
    icos_x = x[...,Lat-1,Lon-1]
    return icos_x

class Grid2Mesh_Encoder(nn.Module):

    def __init__(self, fl, bs, in_ch, out_ch): #12 42 162 642 2562 10242
        super(Grid2Mesh_Encoder, self).__init__()
        self.fl = fl
        self.bs = bs
        self.gnn_layer = 2
        self.in_ch = in_ch
        hidden_size = 256
        self.out_ch = out_ch
        plevel = 13 #37
        self.uu_level0 = 32 # in_ch * plevel * 2
        self.gelu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        # self.meshnode = torch.zeros((bs/2,self.in_ch,40962),dtype=torch.float32)
        self.encoder1 = GCNConv(self.in_ch, hidden_size)
        self.encoder2 = GCNConv(hidden_size, hidden_size)
        self.Lmodel1 = LinearRegressionModel(hidden_size, self.uu_level0*8)
        self.Lmodel2 = LinearRegressionModel(hidden_size, out_ch)
        self.Lat = torch.tensor(np.load('LatLon6.npz')['lat'],dtype=torch.long)
        self.Lon = torch.tensor(np.load('LatLon6.npz')['lon'],dtype=torch.long)

    def forward(self, x, edge_index):
        # x = torch.flatten(x,1,2) #5*13
        # meshnode = torch.zeros((x.shape[0],x.shape[1],40962))
        meshnode = Grid2Mesh(x,4,self.Lat,self.Lon)
        # meshnode = meshnode.to(x.device)
        x = torch.flatten(x,-2,-1) #181*360
        x = torch.cat((x,meshnode),-1) #181*360+40962
        x = x.permute(0,2,1)
        x = self.encoder1(x, edge_index)
        x = self.gelu(x)
        x = self.encoder2(x, edge_index)
        x = x.permute(0,2,1)
        x = x[...,-40962:]
        x = x.permute(0,2,1)
        x = self.Lmodel2(x)
        x = self.gelu(x)
        x = x.permute(0,2,1)

        return x 

class Mesh2Grid_Decoder(nn.Module):

    def __init__(self, fl, bs, in_ch, out_ch, w=181, h=360, fdim=48): #12 42 162 642 2562 10242
        super(Mesh2Grid_Decoder, self).__init__()
        self.fl = fl
        self.fdim = fdim
        self.bs = bs
        self.gnn_layer = 2
        self.in_ch = in_ch
        hidden_size = 256
        self.out_ch = out_ch
        plevel = 13 #37
        self.uu_level0 = 32 # in_ch * plevel * 2
        self.size = h*w
        self.h = h
        self.w = w
        self.gelu = nn.GELU()
        self.relu = nn.ReLU(inplace=True)
        # self.gridnode = torch.zeros((bs,self.in_ch,self.size),dtype=torch.float32)
        self.batchnorm1 = nn.BatchNorm1d(in_ch*plevel*4)
        self.decoder1 = GCNConv(self.in_ch, hidden_size)
        self.decoder2 = GCNConv(hidden_size, hidden_size)
        self.Lmodel1 = LinearRegressionModel(hidden_size, self.uu_level0*8)
        self.Lmodel2 = LinearRegressionModel(hidden_size, out_ch)

    def forward(self, x, x_res_grid, edge_index):
        # x = torch.flatten(x,1,2) #5*13
        # x = torch.flatten(x,-2,-1) #181*360
        # gridnode = torch.zeros((x.shape[0],x.shape[1],self.size),dtype=torch.float32)
        gridnode = x_res_grid.to(x.device)
        x = torch.cat((gridnode,x),-1) #181*360+40962
        x = x.permute(0,2,1)
        x = self.decoder1(x, edge_index)
        x = self.gelu(x)
        x = self.decoder2(x, edge_index)
        x = x.permute(0,2,1)
        x = x[...,:self.size]
        x = x.permute(0,2,1)
        x = self.Lmodel2(x)
        x = x.permute(0,2,1) 
        # x = x.view(self.bs,self.out_ch,self.h,self.w)

        return x 

class LinearRegressionModel(nn.Module):
    def __init__(self,shape,out):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(shape, 256)
        self.linear1 = nn.Linear(256, out)
        torch.nn.init.xavier_normal_(self.linear.weight)
        torch.nn.init.xavier_normal_(self.linear1.weight)
        # torch.nn.init.kaiming_uniform_()

    def forward(self, x):
        
        out = self.linear(x)
        out = self.linear1(out)

        return out