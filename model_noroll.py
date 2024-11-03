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
import os
import time

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
    def __init__(self, fl, mesh_folder, in_ch, out_ch, max_level, min_level=0, fdim=32):
        super().__init__()
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.levels = max_level - min_level
        self.down = []
        self.up = []
        self.in_conv = MeshConv(in_ch, 2*fdim, self.__meshfile(max_level), stride=1)
        self.out_conv = MeshConv(2*fdim, out_ch, self.__meshfile(max_level), stride=1)
        # Downward path
        for i in range(self.levels-1):
            idx = i
            self.down.append(Down(2*fdim*(2**idx), 2*fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        self.down.append(Down(2*fdim*(2**(self.levels-1)), 2*fdim*(2**(self.levels-1)), min_level, mesh_folder))
        #     self.down.append(Down(fdim*(2**idx), fdim*(2**(idx+1)), max_level-i-1, mesh_folder))
        # self.down.append(Down(fdim*(2**(self.levels-1)), fdim*(2**(self.levels-1)), min_level, mesh_folder))
        # Upward path
        for i in range(self.levels-1):
            self.up.append(Up(2*fdim*(2**(self.levels-i)), 2*fdim*(2**(self.levels-i-2)), min_level+i+1, mesh_folder))
        self.up.append(Up(2*fdim*2, 2*fdim, max_level, mesh_folder))
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
    def __init__(self, fl, bs, mesh_folder, in_ch, out_ch, max_level, min_level, fdim=32): #12 42 162 642 2562 10242
        super(Multi_uunet, self).__init__()
        self.fl = fl
        self.mesh_folder = mesh_folder
        self.fdim = fdim
        self.max_level = max_level
        self.min_level = min_level
        self.bs = bs
        self.gnn_layer = 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        plevel = 37
        self.uu_level0 = 32 # in_ch * plevel * 2

        self.spheric_aluNet2 = SphericalUNet(self.fl,self.mesh_folder, self.uu_level0, self.uu_level0, self.max_level, self.min_level, self.fdim)
        self.Lmodel1 = LinearRegressionModel(in_ch*plevel*2, self.uu_level0)
        self.Lmodel2 = LinearRegressionModel(self.uu_level0*2, in_ch*plevel)
    def forward(self,x):
        x = torch.flatten(x,1,3)
        shape = x.shape #bs,370,10242
        x = x.permute(0,2,1)
        x = self.Lmodel1(x)
        x = x.permute(0,2,1)
        x = F.relu(x)
        x_res = x.clone()
        #x = self.conv1d_down(x) 
        x = self.spheric_aluNet2(x) #[bs,8,10242]
        x = F.relu(x)
        x = torch.cat((x,x_res),1)
        x = x.permute(0,2,1)
        # x = self.conv1d_up(x,int(shape[1]/2))
        x = self.Lmodel2(x)
        x = x.permute(0,2,1)
        # x = self.conv1d_up(x,int(shape[1]))
        x = x.view(shape[0],self.out_ch,-1,shape[2]) # bs,10,37,10242
        
        return x

    # def Squeeze_redundancy(self,x):
    #     if x.shape[0]==1:
    #         x = x.squeeze()
    #         x = x.unsqueeze(0)
    #     else:
    #         x = x.squeeze()
    #     return x

    # def conv1d_down(self,x):
    #     model = vertical_down(x.shape[1], self.uu_level0)
    #     model = model.to(x.device)
    #     x = model(x)

    #     return x

    # def conv1d_up(self,x,out_ch):
    #     model = vertical_up(self.uu_level0, out_ch)
    #     model = model.to(x.device)
    #     x = model(x)

        return x


class LinearRegressionModel(nn.Module):
    def __init__(self,shape,out):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(shape, 256) 
        self.linear1 = nn.Linear(256, out)

    def forward(self, x):
        
        out = self.linear(x)
        out = self.linear1(out)

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