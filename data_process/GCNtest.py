from torch_geometric.nn import GCNConv
import torch
from torch.nn import functional as F

use_cuda = 1
#device = torch.device("cuda:1" if use_cuda else "cpu")
device = 'cuda:1'
class GCN(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GCN, self).__init__()
        self.gconv1 = GCNConv(in_ch, 32)
        self.gconv2 = GCNConv(32, out_ch)
        self.norm = torch.nn.BatchNorm1d(32)

    def forward(self, data, edge_index):
        x, edge_index = data, edge_index
        x = self.gconv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gconv2(x, edge_index)

        return x
A = 100
Net = GCN(16,16).to(device)
V = torch.rand(A,16).to(device)
E = (torch.rand(2,560)*V.shape[0]).type(torch.long).to(device)
outlist = []
for batch_idx in range(5): 
    output = Net(V,E).unsqueeze(0)
    outlist.append(output)
res = torch.cat(outlist,dim=0)
print(res.shape)