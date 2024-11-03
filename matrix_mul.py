import torch
import torch_sparse
import time
from utils import s2IV,dense2sparseMM,SPmm

data1 = torch.rand(5550,2000)
data2 = torch.rand(2000,1006)

device = torch.device('cuda:1')

data1[torch.where(data1<0.5)] = 0 #造稀疏

data1 = data1.to(device)
data2 = data2.to(device)

start1 = time.time()

result = torch.mm(data1,data2)

end1 = time.time()-start1
print('稠密相乘计算耗时',end1)

start1 = time.time()

reult = dense2sparseMM(data1,data2)

end1 = time.time()-start1
print('稀疏相乘计算耗时',end1)