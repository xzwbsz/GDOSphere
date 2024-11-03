import numpy as np
from tqdm import tqdm

src_tgt_temp = np.random.rand(2,1)
edge_src_target = np.random.rand(2,1)
#向左的边
for idx_lat in tqdm(range(721)):
    for idx_lon in range(1439):
        src_tgt_temp[0,0] = idx_lat*1440+idx_lon+1 
        src_tgt_temp[1,0] = idx_lat*1440+idx_lon
        if idx_lat==0 and idx_lon==0:
            edge_src_target = src_tgt_temp #第一条边
        else:
            edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)
for idx_lat in tqdm(range(721)): #补一下最左侧和最右侧的连接
    src_tgt_temp[0,0] = idx_lat*1440
    src_tgt_temp[1,0] = idx_lat*1440+1439
    edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)
#向右的边
for idx_lat in tqdm(range(721)):
    for idx_lon in range(1439):
        src_tgt_temp[0,0] = idx_lat*1440+idx_lon
        src_tgt_temp[1,0] = idx_lat*1440+idx_lon+1
        edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)
for idx_lat in tqdm(range(721)): #补一下最左侧和最右侧的连接
    src_tgt_temp[0,0] = idx_lat*1440+1439
    src_tgt_temp[1,0] = idx_lat*1440
    edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)
#向上的边
for idx_lat in tqdm(range(720)):
    for idx_lon in range(1440):
        src_tgt_temp[0,0] = (idx_lat+1)*1440+idx_lon
        src_tgt_temp[1,0] = idx_lat*1440+idx_lon
        edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)
#向下的边
for idx_lat in tqdm(range(720)):
    for idx_lon in range(1440):
        src_tgt_temp[0,0] = idx_lat*1440+idx_lon
        src_tgt_temp[1,0] = (idx_lat+1)*1440+idx_lon
        edge_src_target = np.concatenate((edge_src_target,src_tgt_temp), axis=-1)

np.save('global_graph.npy',edge_src_target)