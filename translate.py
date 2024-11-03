import numpy as np

A = np.load('mesh_idx_5.npz')
meshnode = A['mesh_node']
C = A['edge_src_target']
B = np.ones([C.shape[0],C.shape[1]])
B[0] = np.array(A['edge_src_target'][1])
B[1] = np.array(A['edge_src_target'][0])
#2,350090
count = 0
C = np.ones(B.shape[1])

for idx in range(B.shape[1]):
    if idx == 0:
        C[idx] = count
    else:
        a = B[0,idx]
        b = B[0,idx-1]
        if a != b:
            count=count+1
        C[idx]=count

B[0] = C
np.savez('mesh_idx_5.npz',edge_src_target=B,mesh_node=meshnode)
    