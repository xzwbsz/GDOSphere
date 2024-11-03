import math
import pickle, gzip
import os

import torch
from torch import nn
from torch.nn.parameter import Parameter

from utils import sparse2tensor, dense2sparseMM, spmatmul, s2IV, SPmm


class _MeshConv(nn.Module):#定义参数用
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        assert stride in [1, 2]
        super(_MeshConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.ncoeff = 4
        self.coeffs = Parameter(torch.Tensor(out_channels, in_channels, self.ncoeff))
        self.set_coeffs()
        # load mesh file
        pkl = pickle.load(open(mesh_file, "rb"))
        self.pkl = pkl
        self.nv = self.pkl['V'].shape[0]
        #稀疏矩阵转换成稠密矩阵
        # G = sparse2tensor(pkl['G']).to_dense().permute(1,0)  # gradient matrix V->F, 3#F x #V
        Gi,Gv = s2IV(pkl['G'])
        self.Gshape = torch.Size(pkl['G'].shape)
        #读取出NS和EW作为x与y
        NS = torch.tensor(pkl['NS'], dtype=torch.float32)  # north-south vector field, #F x 3 torch.float32
        EW = torch.tensor(pkl['EW'], dtype=torch.float32)  # east-west vector field, #F x 3 torch.float32
        #人为定义的参数，不参与到训练中去
        # self.register_buffer("G", G)
        self.register_buffer("Gi", Gi)
        self.register_buffer("Gv", Gv)
        self.register_buffer("NS", NS)
        self.register_buffer("EW", EW)
        
    def set_coeffs(self):
        n = self.in_channels * self.ncoeff
        stdv = 1. / math.sqrt(n)
        self.coeffs.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class MeshConv(_MeshConv): #继承_MeshConv类的内容，并定义一阶导二阶导与相关系数的乘法
    def __init__(self, in_channels, out_channels, mesh_file, stride=1, bias=True):
        super(MeshConv, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        pkl = self.pkl
        if stride == 2:
            self.nv_prev = pkl['nv_prev']
            # 转coo COO格式是一种基于三元组表示的稀疏矩阵格式，
            #它将矩阵中非零元素的行列坐标和值分别存储在三个一维数组中。
            #这种格式适用于需要频繁地修改矩阵中的元素的场景。
            # L = sparse2tensor(pkl['L'].tocsr()[:self.nv_prev].tocoo()) # laplacian matrix V->V
            Li,Lv = s2IV(pkl['L'].tocsr()[:self.nv_prev].tocoo())
            self.Lshape = pkl['L'].tocsr()[:self.nv_prev].tocoo().shape
            # F2V = sparse2tensor(pkl['F2V'].tocsr()[:self.nv_prev].tocoo())
            F2Vi, F2Vv = s2IV(pkl['F2V'].tocsr()[:self.nv_prev].tocoo())  # F->V, #V x #F
            self.F2Vshape = pkl['F2V'].tocsr()[:self.nv_prev].tocoo().shape
        else: # stride == 1
            self.nv_prev = pkl['V'].shape[0]
            #稀疏转稠密，L不是拉普拉斯矩阵，F2V是面转点矩阵
            # L = sparse2tensor(pkl['L'].tocoo())
            Li,Lv = s2IV(pkl['L'].tocoo())
            self.Lshape = pkl['L'].tocoo().shape
            # F2V = sparse2tensor(pkl['F2V'].tocoo())
            F2Vi, F2Vv = s2IV(pkl['F2V'].tocoo())
            self.F2Vshape = pkl['F2V'].tocoo().shape
        #print('Lshape',L.shape)
        # L = L.to_dense().permute(1,0) #permute是矩阵转置，相当于reshape
        #print('F2Vshape',F2V.shape)
        # F2V = F2V.to_dense().permute(1,0)
        #这两个也是自定义不训练参数
        
        self.register_buffer("Li", Li)
        self.register_buffer("Lv", Lv)
        self.register_buffer("F2Vi", F2Vi)
        self.register_buffer("F2Vv", F2Vv)
        
    def forward(self, input):
        # compute gradient

        # grad_face = dense2sparseMM(input, self.G)#输入与梯度相乘
        grad_face =SPmm(input, self.Gi, self.Gv, self.Gshape)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        
        # laplacian = dense2sparseMM(input, self.L)#输入与拉普拉斯矩阵相乘
        laplacian = SPmm(input, self.Li, self.Lv, self.Lshape)
        identity = input[..., :self.nv_prev]#array[...] 就是array[:,:,:]
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)

        grad_vert_ew = SPmm(grad_face_ew, self.F2Vi, self.F2Vv, self.F2Vshape)# xy与F2V相乘
        grad_vert_ns = SPmm(grad_face_ns, self.F2Vi, self.F2Vv, self.F2Vshape) 

        feat = [identity, laplacian, grad_vert_ew, grad_vert_ns]

        out = torch.stack(feat, dim=-1)
        #128x16(随U-net变化)x10242(随level变化)x4(四个feat，即普通量，两个梯度，一个拉普拉斯)
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        #128x8(随U-net变化)x10242
        out += self.bias.unsqueeze(-1)
        #128x8(随U-net变化)x10242
        return out


class MeshConv_transpose(_MeshConv):
    def __init__(self, in_channels, out_channels, mesh_file, stride=2, bias=True):
        assert(stride == 2)
        super(MeshConv_transpose, self).__init__(in_channels, out_channels, mesh_file, stride, bias)
        pkl = self.pkl
        self.nv_prev = self.pkl['nv_prev']
        self.nv_pad = self.nv - self.nv_prev
        # L = sparse2tensor(pkl['L'].tocoo()) # laplacian matrix V->V
        # F2V = sparse2tensor(pkl['F2V'].tocoo()) # F->V, #V x #F
        # L = L.to_dense().permute(1,0)
        # F2V = F2V.to_dense().permute(1,0)
        Li,Lv = s2IV(pkl['L'].tocoo())
        self.Lshape = pkl['L'].tocoo().shape
        F2Vi, F2Vv = s2IV(pkl['F2V'].tocoo())
        self.F2Vshape = pkl['F2V'].tocoo().shape

        #PyTorch中定义模型时，有时候会遇到self.register_buffer('name', Tensor)的操作，
        #该方法的作用是定义一组参数，该组参数的特别之处在于：
        #模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        #但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        self.register_buffer("Li", Li)
        self.register_buffer("Lv", Lv)
        self.register_buffer("F2Vi", F2Vi)
        self.register_buffer("F2Vv", F2Vv)
        # self.register_buffer("L", L)
        # self.register_buffer("F2V", F2V)
        
    def forward(self, input):
        # pad input with zeros up to next mesh resolution
        ones_pad = torch.ones(*input.size()[:2], self.nv_pad).to(input.device)
        input = torch.cat((input, ones_pad), dim=-1)
        # compute gradient
        # grad_face = dense2sparseMM(input, self.G) Gshape
        grad_face =SPmm(input, self.Gi, self.Gv, self.Gshape)
        grad_face = grad_face.view(*(input.size()[:2]), 3, -1).permute(0, 1, 3, 2) # gradient, 3 component per face
        laplacian = SPmm(input, self.Li, self.Lv, self.Lshape)
        # laplacian = dense2sparseMM(input, self.L)

        # identity = input
        
        #一阶导乘后求和
        grad_face_ew = torch.sum(torch.mul(grad_face, self.EW), keepdim=False, dim=-1)
        grad_face_ns = torch.sum(torch.mul(grad_face, self.NS), keepdim=False, dim=-1)
        grad_vert_ew = SPmm(grad_face_ew, self.F2Vi, self.F2Vv, self.F2Vshape)# xy与F2V相乘
        grad_vert_ns = SPmm(grad_face_ns, self.F2Vi, self.F2Vv, self.F2Vshape)
        # grad_vert_ew = dense2sparseMM(grad_face_ew, self.F2V)
        # grad_vert_ns = dense2sparseMM(grad_face_ns, self.F2V)

        feat = [input, laplacian, grad_vert_ew, grad_vert_ns]
        out = torch.stack(feat, dim=-1)#在新的维度拼接这几个矩阵
        out = torch.sum(torch.sum(torch.mul(out.unsqueeze(1), self.coeffs.unsqueeze(2)), dim=2), dim=-1)
        out += self.bias.unsqueeze(-1)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_chan, neck_chan, out_chan, level, coarsen, mesh_folder):
        super().__init__()
        l = level-1 if coarsen else level
        self.coarsen = coarsen
        mesh_file = os.path.join(mesh_folder, "icosphere_{}.pkl".format(l))
        self.conv1 = nn.Conv1d(in_chan, neck_chan, kernel_size=1, stride=1)
        self.conv2 = MeshConv(neck_chan, neck_chan, mesh_file=mesh_file, stride=1)
        self.conv3 = nn.Conv1d(neck_chan, out_chan, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm1d(neck_chan)
        self.bn2 = nn.BatchNorm1d(neck_chan)
        self.bn3 = nn.BatchNorm1d(out_chan)
        self.nv_prev = self.conv2.nv_prev
        self.down = DownSamp(self.nv_prev)
        self.diff_chan = (in_chan != out_chan)

        if coarsen:
            self.seq1 = nn.Sequential(self.down, self.conv1, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)#序列容器，三次卷积，其中第二次是meshconv
        else:
            self.seq1 = nn.Sequential(self.conv1, self.bn1, self.relu, 
                                      self.conv2, self.bn2, self.relu, 
                                      self.conv3, self.bn3)#序列容器，不要降采样

        if self.diff_chan or coarsen:
            self.conv_ = nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=1)
            self.bn_ = nn.BatchNorm1d(out_chan)
            if coarsen:
                self.seq2 = nn.Sequential(self.conv_, self.down, self.bn_)
            else:
                self.seq2 = nn.Sequential(self.conv_, self.bn_)

    def forward(self, x):
        if self.diff_chan or self.coarsen:
            x2 = self.seq2(x) #如果不需要降采样就执行序列2
        else:
            x2 = x
        x1 = self.seq1(x)  #执行序列一
        out = x1 + x2
        out = self.relu(out)
        return out


class DownSamp(nn.Module):
    def __init__(self, nv_prev):
        super().__init__()
        self.nv_prev = nv_prev

    def forward(self, x):
        return x[..., :self.nv_prev] #只取self.nv_prev的部分以进行降采样
