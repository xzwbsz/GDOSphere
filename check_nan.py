import numpy as np
from tqdm import tqdm
path = '/xuzhewen/gnn/data2/ERA5/'
nan = []
for idx in tqdm(range(8500)):
    file = path+str(idx)+'.npy'
    npyfile = np.load(file)
    X = np.any(np.isnan(npyfile))
    if X:
        nnn = np.array(nan.append(idx))
        np.save('nanfile',nnn)