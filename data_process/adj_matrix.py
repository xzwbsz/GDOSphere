import numpy as np
from tqdm import tqdm
import torch
file_path = '/home/gnn/data_process/meshcnn/'
R = 6378
# def sphere_distance(a1,a2,b1,b2): #a1 a2 经度角 b1 b2 纬度角
#     tq = torch.cos(b1)*torch.cos(b2)*torch.cos(a1-a2)+torch.sin(b1)*torch.sin(b2)
#     tq = torch.clip(tq , -1, 1)
#     r = R*torch.arccos(tq)
#     # r = R*np.arccos(np.cos(b1)*np.cos(b2)*np.cos(a1-a2)+np.sin(b1)*np.sin(b2))
#     return r
def sphere_distance(a1,a2,b1,b2): #a1 a2 经度角 b1 b2 纬度角
    tq = np.cos(b1)*np.cos(b2)*np.cos(a1-a2)+np.sin(b1)*np.sin(b2)
    tq = np.clip(tq,-1,1)
    r = R*np.arccos(tq)
    return r
with open(file_path+"lat_6.txt", "r") as f:
    lines = f.readlines()
lat = [ l.replace('\n', '') for l in lines]
for idx in range(len(lat)):
    lat[idx]=float(lat[idx])
with open(file_path+"long_6.txt", "r") as f2:
    lines1 = f2.readlines()
lon = [ l.replace('\n', '') for l in lines1]
for idx in range(len(lon)):
    lon[idx]=float(lon[idx])

device = torch.device('cuda')

# adj_matrix = np.ones((len(lon),len(lon)))
# lat = torch.tensor(np.array(lat)).to(device)
# lon = torch.tensor(np.array(lon)).to(device)
# adj_matrix = sphere_distance(lon.unsqueeze(0),lon.unsqueeze(-1),lat.unsqueeze(0),lat.unsqueeze(-1))
# adj_matrix = np.array(adj_matrix.cpu())
# edge_list = []
edge_idx = 0
skip_mark = np.zeros(len(lon))
for iidx in tqdm(range(len(lon))):
    if skip_mark[iidx]>=6:
        break
    # sub_edge_idx = 0
    for iidx2 in range(iidx,len(lat)):
        if skip_mark[iidx2]>=6:
            break
        adj_temp = np.ones((2,1))
        distance = sphere_distance(lon[iidx],lon[iidx2],lat[iidx],lat[iidx2])
        if distance>10 and distance<150:
            adj_temp[0,0]=iidx
            adj_temp[1,0]=iidx2
            skip_mark[iidx]+=1
            skip_mark[iidx2]+=1
            if edge_idx==0:
                edge_list = adj_temp
            else:
                edge_list = np.concatenate((edge_list,adj_temp),1)
            edge_idx+=1
            # sub_edge_idx+=1
        # if sub_edge_idx==6:
        #     break

np.save('adj_matrix6.npy',edge_list)
