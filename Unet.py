# import torch
# from torch import nn
# from torch.nn import functional as F

# # 上采样+拼接
# class Up(nn.Module):
#     def __init__(self,in_channels,out_channels,bilinear=False):
#         '''
#         :param in_channels: 输入通道数
#         :param out_channels:  输出通道数
#         :param bilinear: 是否采用双线性插值，默认采用
#         '''
#         super(Up, self).__init__()
#         if bilinear:
#             # 双线性差值
#             self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#             self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
#         else:
#             # 转置卷积实现上采样
#             # 输出通道数减半，宽高增加一倍
#             self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
#             self.conv = doubleConv(in_channels,out_channels)

#     def forward(self,x1,x2):
#         # 上采样
#         x1 = self.up(x1)
#         # print('check1',x1.shape)
#         # print('check2',x2.shape)
#         # 拼接
#         x = torch.cat([x1,x2],dim=1)

#         # 经历双卷积
#         x = self.conv(x)
#         return x

# # 双卷积层
# class doubleConv(nn.Module):
#     def __init__(self,in_channels,out_channels,mid_channels=None):
#         '''
#         :param in_channels: 输入通道数 
#         :param out_channels: 双卷积后输出的通道数
#         :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
#         :return: 
#         '''
#         super(doubleConv, self).__init__()
#         if mid_channels is None:
#             mid_channels = out_channels
#         # layer = []
#         # layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True))
#         self.net1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True)
#         self.reluu = nn.ReLU(inplace=True)
#         self.net2 = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True)
#         # layer.append(nn.BatchNorm2d(mid_channels))
#         # layer.append(nn.ReLU(inplace=True))
#         # layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True))
#         # layer.append(nn.BatchNorm2d(out_channels))
#         # layer.append(nn.ReLU(inplace=True))
#         # self.layernet = nn.Sequential(*layer)
#     def forward(self,x):
#         out = self.net1(x)
#         out = self.BN_(out)
#         out = self.reluu(out)
#         out = self.net2(out)
#         out = self.BN_(out)
#         out = self.reluu(out)
#         return out
    
#     def BN_(self,x):
#         pre_x = x.permute(1,0,2,3)
#         pre_x = torch.flatten(pre_x,1,-1)
#         mean = pre_x.mean(dim=1)
#         std = pre_x.std(dim=1)
#         x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
#         return x
    
# # 下采样
# class down(nn.Module):
#     def __init__(self,in_channels,out_channels,mid_channels=None):
#         '''
#         :param in_channels: 输入通道数 
#         :param out_channels: 双卷积后输出的通道数
#         :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
#         :return: 
#         '''
#         super(down, self).__init__()
#         layer = []
#         layer.append(nn.MaxPool2d(2,stride=2))
#         layer.append(doubleConv(in_channels,out_channels))
#         self.layernet = nn.Sequential(*layer)
#     def forward(self,x):
#         out = self.layernet(x)
#         return out


# # 整个网络架构
# class UNet(nn.Module):
#     def __init__(self,in_channels,out_channels,bilinear=False,base_channel=64):
#         '''
#         :param in_channels: 输入通道数，一般为3，即彩色图像
#         :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
#         :param bilinear: 是否采用双线性插值来上采样，这里默认采取
#         :param base_channel: 第一个卷积后的通道数，即64
#         '''
#         super(UNet, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.bilinear = bilinear
#         factor = 1 #2  if self.bilinear else 1
#         self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
#         self.downsample = nn.Upsample(scale_factor=1/2,mode='bilinear',align_corners=True)
#         # 输入
#         self.in_conv = doubleConv(base_channel,base_channel,mid_channels=base_channel*2)
#         # 下采样
#         # self.down1 = down(base_channel,base_channel*2) # 64,128
#         # self.down2 = down(base_channel*2,base_channel*4// factor) # 128,256
#         # self.down3 = down(base_channel*4,base_channel*8) # 256,512
#         # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
#         # 当然，是否采取双线新差值，还是由我们自己决定
  
#         # self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512
#         # 上采样 + 拼接
#         # self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
#         # self.up2 = Up(base_channel*8 ,base_channel*4 // factor, self.bilinear)
#         # self.up3 = Up(base_channel*4 ,base_channel*2 // factor, self.bilinear)
#         # self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)
#         # 输出
#         self.out = doubleConv(base_channel,base_channel,mid_channels=base_channel*2) #nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)
#         self.conv_layer_in = nn.Conv2d(in_channels=in_channels,out_channels=base_channel,kernel_size=(2, 1),stride=(1, 1),bias=False)
#         self.conv_layer_out = nn.Conv2d(in_channels=base_channel,out_channels=in_channels,kernel_size=(2, 1),stride=(1, 1),padding=(1,0),bias=False)
    
#     def forward(self,x):

#         x = self.conv_layer_in(x)
#         # x = self.upsample(x)
#         x1 = self.in_conv(x)
#         # x2 = self.down1(x1)
#         # x3 = self.down2(x2)
#         # x_mid = self.down3(x3)
#         # 不要忘记拼接
#         # x = self.up2(x_mid, x3)
#         # x = self.up3(x3, x2)
#         # x = self.up4(x, x1)
#         out = self.out(x)
#         # out = self.out(self.downsample(x))
#         out = self.conv_layer_out(out)
#         return out #{'out':out}
# # class UNet(nn.Module):
# #     def __init__(self,in_channels,out_channels,bilinear=False,base_channel=64):
# #         '''
# #         :param in_channels: 输入通道数，一般为3，即彩色图像
# #         :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
# #         :param bilinear: 是否采用双线性插值来上采样，这里默认采取
# #         :param base_channel: 第一个卷积后的通道数，即64
# #         '''
# #         super(UNet, self).__init__()
# #         self.in_channels = in_channels
# #         self.out_channels = out_channels
# #         self.bilinear = bilinear
# #         factor = 1 #2  if self.bilinear else 1
# #         self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
# #         self.downsample = nn.Upsample(scale_factor=1/2,mode='bilinear',align_corners=True)
# #         # 输入
# #         self.in_conv = doubleConv(self.in_channels,base_channel)
# #         # 下采样
# #         self.down1 = down(base_channel,base_channel*2) # 64,128
# #         self.down2 = down(base_channel*2,base_channel*4// factor) # 128,256
# #         self.down3 = down(base_channel*4,base_channel*8) # 256,512
# #         # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
# #         # 当然，是否采取双线新差值，还是由我们自己决定
  
# #         # self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512
# #         # 上采样 + 拼接
# #         # self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
# #         self.up2 = Up(base_channel*8 ,base_channel*4 // factor, self.bilinear)
# #         self.up3 = Up(base_channel*4 ,base_channel*2 // factor, self.bilinear)
# #         self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)
# #         # 输出
# #         self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)
# #         self.conv_layer_in = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(2, 1),stride=(1, 1),bias=False)
# #         self.conv_layer_out = nn.Conv2d(in_channels=out_channels,out_channels=in_channels,kernel_size=(2, 1),stride=(1, 1),padding=(1,0),bias=False)
    
# #     def forward(self,x):

# #         x = self.conv_layer_in(x)
# #         x = self.upsample(x)
# #         x1 = self.in_conv(x)
# #         x2 = self.down1(x1)
# #         x3 = self.down2(x2)
# #         x_mid = self.down3(x3)
# #         # 不要忘记拼接
# #         x = self.up2(x_mid, x3)
# #         x = self.up3(x3, x2)
# #         x = self.up4(x, x1)
# #         out = self.out(self.downsample(x))
# #         out = self.conv_layer_out(out)
# #         return out #{'out':out}
import torch
from torch import nn
from torch.nn import functional as F

# 上采样+拼接
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=False):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        # 上采样
        x1 = self.up(x1)
        # print('check1',x1.shape)
        # print('check2',x2.shape)
        # 拼接
        x = torch.cat([x1,x2],dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x

# 双卷积层
class doubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        '''
        :param in_channels: 输入通道数 
        :param out_channels: 双卷积后输出的通道数
        :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
        :return: 
        '''
        super(doubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        # layer = []
        # layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True))
        self.net1 = nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=True)
        self.reluu = nn.ReLU(inplace=True)
        self.net2 = nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True)
        # layer.append(nn.BatchNorm2d(mid_channels))
        # layer.append(nn.ReLU(inplace=True))
        # layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=True))
        # layer.append(nn.BatchNorm2d(out_channels))
        # layer.append(nn.ReLU(inplace=True))
        # self.layernet = nn.Sequential(*layer)
    def forward(self,x):
        out = self.net1(x)
        out = self.BN_(out)
        out = self.reluu(out)
        out = self.net2(out)
        out = self.BN_(out)
        out = self.reluu(out)
        return out
    
    def BN_(self,x):
        pre_x = x.permute(1,0,2,3)
        pre_x = torch.flatten(pre_x,1,-1)
        mean = pre_x.mean(dim=1)
        std = pre_x.std(dim=1)
        x = (x-mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(0))/(std.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)+1e-5)
        return x
    
# 下采样
class down(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        '''
        :param in_channels: 输入通道数 
        :param out_channels: 双卷积后输出的通道数
        :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
        :return: 
        '''
        super(down, self).__init__()
        layer = []
        layer.append(nn.MaxPool2d(2,stride=2))
        layer.append(doubleConv(in_channels,out_channels))
        self.layernet = nn.Sequential(*layer)
    def forward(self,x):
        out = self.layernet(x)
        return out


# 整个网络架构
class UNet(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=False,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        factor = 1 #2  if self.bilinear else 1
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.downsample = nn.Upsample(scale_factor=1/2,mode='bilinear',align_corners=True)
        # 输入
        self.in_conv = doubleConv(base_channel,base_channel,mid_channels=base_channel*2)
        # 下采样
        # self.down1 = down(base_channel,base_channel*2) # 64,128
        # self.down2 = down(base_channel*2,base_channel*4// factor) # 128,256
        # self.down3 = down(base_channel*4,base_channel*8) # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
  
        # self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512
        # 上采样 + 拼接
        # self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
        # self.up2 = Up(base_channel*8 ,base_channel*4 // factor, self.bilinear)
        # self.up3 = Up(base_channel*4 ,base_channel*2 // factor, self.bilinear)
        # self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)
        # 输出
        self.out = doubleConv(base_channel,base_channel,mid_channels=base_channel*2) #nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)
        self.conv_layer_in = nn.Conv2d(in_channels=in_channels,out_channels=base_channel,kernel_size=(2, 1),stride=(1, 1),bias=False)
        self.conv_layer_out = nn.Conv2d(in_channels=base_channel,out_channels=in_channels,kernel_size=(2, 1),stride=(1, 1),padding=(1,0),bias=False)
    
    def forward(self,x):

        x = self.conv_layer_in(x)
        # x = self.upsample(x)
        x1 = self.in_conv(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x_mid = self.down3(x3)
        # 不要忘记拼接
        # x = self.up2(x_mid, x3)
        # x = self.up3(x3, x2)
        # x = self.up4(x, x1)
        out = self.out(x)
        # out = self.out(self.downsample(x))
        out = self.conv_layer_out(out)
        return out #{'out':out}
