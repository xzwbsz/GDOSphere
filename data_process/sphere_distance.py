import numpy as np
from tqdm import tqdm
file_path = '/home/gnn/data_process/meshcnn/'
R = 6378
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

adj_matrix = np.ones((len(lon),len(lon)))
for iidx in tqdm(range(len(lon))):
    for iidx2 in range(len(lat)):
        adj_matrix[iidx2,iidx] = sphere_distance(lon[iidx],lon[iidx2],lat[iidx],lat[iidx2])

adj_matrix = np.where(adj_matrix<150,adj_matrix,0)
adj_matrix = np.where(adj_matrix>10,1,0)
print(adj_matrix)
np.save('adj_matrix6.npy',adj_matrix)
