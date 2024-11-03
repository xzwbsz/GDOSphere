from torch import nn
from ops import MeshConv, MeshConv_transpose, ResBlock
import torch.nn.functional as F
import os
import torch
import pickle
import numpy as np
from GCNmodel import GraphGNN, GCNLayer
from torch_geometric.nn import GCNConv
from utils import SPmm
from EncoderDecoder import Grid2Mesh_Encoder,Mesh2Grid_Decoder, Grid2Mesh
import os
import time
from Unet import UNet
# from DDDUnet import UNet3d
from AttUnet import attUNet

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
        use mesh_file for the mesh of one-level up
        """
        super().__init__()
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(level))
        half_in = int(in_ch/2)
        self.up = MeshConv_transpose(half_in, half_in, mesh_file, stride=2)
        self.conv = ResBlock(in_ch, out_ch, out_ch, level, False, mesh_folder)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, level, mesh_folder, bias=True):
        """
        use mesh_file for the mesh of one-level down
        """
        super().__init__()
        self.conv = ResBlock(in_ch, in_ch, out_ch, level+1, True, mesh_folder)

    def forward(self, x):
        x = self.conv(x)
        return x


class SphericalUNet(nn.Module):
    def __init__(self, fl, mesh_folder, in_ch, out_ch, max_level, min_level=0, fdim=64):
        super().__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level
        self.down = []
        self.up = []
        self.in_conv = MeshConv(in_ch, fdim, self.__meshfile(max_level), stride=1)
        self.out_conv = MeshConv(fdim, out_ch, self.__meshfile(max_level), stride=1)
        # Downward path
        for i in range(self.levels-1):
            idx = i
            self.down.append(Down(fdim*(2**idx), fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), min_level, mesh_folder))
        #     self.down.append(Down(fdim*(2**idx), fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        # self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), min_level, mesh_folder))
        # Upward path
        for i in range(self.levels-1):
            self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        self.up.append(Up(fdim*2, fdim, max_level, mesh_folder))
        #     self.up.append(Up(fdim*(2**(self.levels-i)), fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        # self.up.append(Up(fdim*2, fdim, max_level, mesh_folder))
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):

        #x初始是[bs, 16, 10242]
        x_ = [self.in_conv(x)] #[bs, 185, 10242]

        for i in range(self.levels): #不同程度的icos循环down
            x_.append(self.down[i](x_[-1]))

        x = self.up[0](x_[-1], x_[-2]) #整体一起up #[bs, 64, 42]

        for i in range(self.levels-1): # -3-i不是很能理解
            x = self.up[i+1](x, x_[-3-i]) #([bs, 8, 10242])等，第二三个数字变化

        x = self.out_conv(x) #[bs, 3, 10242]

        return x 

    def __meshfile(self, i):
        return os.path.join(self.mesh_folder, "icosphere_{}.pkl".format(i))
    
class Multi_uunet(nn.Module):
    # fl=args.forecast_len, mesh_folder=args.mesh_folder, in_ch=5, out_ch=5, 
    #     max_level=args.max_level, min_level=args.min_level, fdim=args.feat
    def __init__(self, fl, bs, mesh_folder, in_ch, out_ch, max_level, min_level, fdim=48): #12 42 162 642 2562 10242
        super(Multi_uunet, self).__init__()
        self.Lat = torch.tensor(np.load('LatLon6.npz')['lat'],dtype=torch.long)
        self.Lon = torch.tensor(np.load('LatLon6.npz')['lon'],dtype=torch.long)
        self.fl = fl
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.bs = bs
        self.gnn_layer = 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.relu = nn.GELU()
        self.g2m = torch.tensor(np.load('g2m_6_graphcast.npy'),dtype=torch.long)
        self.m2g = torch.tensor(np.load('m2g_6_graphcast.npy'),dtype=torch.long)
        plevel = 4 #4 #13 #37
        self.uu_level0 = plevel*in_ch # in_ch * plevel * 2
        self.hidden_ = 64
        self.spheric_aluNet2 = SphericalUNet(self.fl,self.mesh_folder, self.hidden_, self.hidden_, self.max_level, self.min_level, self.fdim)
        # self.Lmodel1 = LinearRegressionModel(in_ch*plevel*2, self.uu_level0)
        self.Lmodel1 = LinearRegressionModel(3, 1, 64)
        self.Lmodel2 = LinearRegressionModel(self.hidden_, self.hidden_)  #*2是为了残差
        self.Lmodel3 = LinearRegressionModel(self.hidden_, self.hidden_)
        self.Lmodel4 = LinearRegressionModel(self.hidden_, self.hidden_)
        self.Lmodel_north = LinearRegressionModel(self.hidden_, in_ch*plevel)
        self.Lmodel_res = LinearRegressionModel(in_ch*plevel, self.hidden_)
        self.g2mnet = Grid2Mesh_Encoder(fl, bs, self.hidden_, self.hidden_)
        self.m2gnet = Mesh2Grid_Decoder(fl, bs, self.hidden_, self.hidden_, 181, 360)
        self.drop = 0
        self.dropout = nn.Dropout(0.2) #nn.functional.dropout(0.6)
        self.Unet = attUNet(self.hidden_,in_ch*plevel)
        # self.UNet3d = UNet3d(in_ch, in_ch)
        # self.linearnet1 = []
        # for forecast_len in range(self.fl):
        #     self.linearnet1.append(LinearRegressionModel(in_ch*plevel*2, self.uu_level0))
        #     self.linearnet1.append(SphericalUNet(self.fl,self.mesh_folder, self.uu_level0, self.uu_level0, self.max_level, self.min_level, self.fdim))
        #     self.linearnet1.append(LinearRegressionModel(self.uu_level0*2, in_ch*plevel))
        # self.linearnet1 = nn.Sequential(*self.linearnet1)
    def forward(self,x):
        shape = x.shape #bs,2,5,13,181*360
        self.g2m = self.g2m.to(x.device)
        self.m2g = self.m2g.to(x.device)
        # x = torch.flatten(x,1,3)
        # x = x.permute(0,2,3,4,5,1) #bs,5,2,181,360,3

        # x = x.permute(0,2,3,4,1) #bs,5,181,360,3
        # x = self.Lmodel1(x)
        # x = x.squeeze() #bs,5,2,181,360
        # # x = torch.flatten(x,1,2) #bs,10,181,360
        # x = self.BN_1_4(x)
        x = torch.flatten(x,1,2) #bs,20,181,360
        x = x.permute(0,2,3,1)

        # x = self.relu(x) 
        x = self.Lmodel_res(x) 
        x = x.permute(0,3,1,2) #bs,32,181,360
        x = self.BN_1_4(x)
        x = self.relu(x)
        x_res_grid = x #torch.zeros(x.shape) #torch.flatten(x,-2,-1)#.clone()
        x_res_grid = torch.flatten(x_res_grid,-2,-1)
        #x = self.conv1d_down(x)
        # x = x.permute(0,2,1)
        x = self.g2mnet(x,self.g2m) #bs,32,40962
        x = self.BN_1_3(x)
        # x = self.relu(x)
        # x = Grid2Mesh(x,4,self.Lat,self.Lon)
        # x_res = x.clone() #bs,32,40962
        x = self.spheric_aluNet2(x) #[bs,C,Vertex]
        x = self.BN_1_3(x)
        x = self.relu(x) #bs,65,40962
        # x = torch.cat((x,x_res),1) #bs,64,40962
        x = x.permute(0,2,1) 
        x = self.Lmodel2(x) 
        x = x.permute(0,2,1) #bs,32,40962

        x = self.m2gnet(x,x_res_grid,self.m2g) #bs,65,181*360
        x = self.BN_1_3(x)
        x = self.relu(x)
        x = x.permute(0,2,1) 
        x = self.Lmodel3(x)
        x = self.BN_1_3(x)
        x = self.relu(x)
        # if self.drop:
        #     x = self.dropout(x) #dropout
        # x = x.permute(0,2,3,1) 
        x = self.Lmodel4(x)
        if self.drop:
            x = self.dropout(x) #dropout
        x = x.permute(0,2,1)
        x = x.view(shape[0],-1,shape[-2],shape[-1])
         #bs,20,181,360
        
        north_polar = x[:,:,0,:].unsqueeze(-2)
        x = self.Unet(x[:,:,1:,:]) #裁掉北极
        north_polar = north_polar.permute(0,2,3,1) 
        north_polar = self.Lmodel_north(north_polar)
        north_polar = north_polar.permute(0,3,1,2) 
        x = torch.cat((north_polar,x),-2)
        x = x.view(shape[0],self.out_ch,-1,shape[-2],shape[-1])

        return x
    
    def BN_1_4(self,x):
        pre_x = x.permute(1,0,2,3)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
    def BN_1_3(self,x):
        pre_x = x.permute(1,0,2)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(0)+1e-5)
        return x

class LinearRegressionModel(nn.Module):
    def __init__(self,shape,out,hs=64):
        super(LinearRegressionModel, self).__init__()
        self.linout = nn.Sequential(
            nn.Linear(shape, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, hs),
            nn.ReLU(inplace=True),
            nn.Linear(hs, out),
        )

    def forward(self, x):
        out = self.linout(x) 
        # out = self.linear(x)
        # out = self.linear1(out)

        return out

class vertical_down(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_down, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1ddown1 = nn.Conv1d(self.in_ch, 256, 1, stride=1) # (input_channels, output_channels, kernel_size, stride)
        self.conv1ddown2 = nn.Conv1d(256, self.out_ch, 1, stride=1)
    def forward(self, x): #x (batch_size,channels,length)
        x = self.conv1ddown1(x)
        x = self.conv1ddown2(x)
        return x

class vertical_up(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_up, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.conv1dup1 = nn.ConvTranspose1d(self.in_ch, 256, 1, stride=1) # (input_channels, output_channels, kernel_size, stride)
        self.conv1dup2 = nn.ConvTranspose1d(256, self.out_ch, 1, stride=1)
    def forward(self, x): #x (batch_size,channels,length)
        x = self.conv1dup1(x)
        x = self.conv1dup2(x)
        return x

class vertical_down3d(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_down3d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        # self.kernel_size1 = kernel_1
        # self.kernel_size2 = kernel_2
        self.conv2ddown1 = nn.Conv2d(self.in_ch, 256, (1, 1)) # Conv2d[ channels, output, height_2, width_2 ]
        self.conv2ddown2 = nn.Conv2d(256, self.out_ch, (1, 1))#只能用于37层的大气层
    def forward(self, x): #x (batch_size,channels,length)
        out = self.conv2ddown1(x)
        out = self.conv2ddown2(out)
        return out

class vertical_up2d(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(vertical_up2d, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        # self.kernel_size1 = kernel_1
        # self.kernel_size2 = kernel_2
        self.conv2dup1 = nn.ConvTranspose2d(self.in_ch, 256, (1, 1)) # Conv2d[ channels, output, height_2, width_2 ]
        self.conv2dup2 = nn.ConvTranspose2d(256, self.out_ch, (1, 1))
    def forward(self, x): #x (batch_size,channels,length)
        out = self.conv2dup1(x)
        out = self.conv2dup2(out)
        return out

