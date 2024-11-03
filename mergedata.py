import numpy as np
from glob import glob
import os
from tqdm import tqdm
import pickle

partition = 'train'
data_dir = '../icos/'
num = 8760 #len(flist)
with open(partition+"_split.txt", "r") as f:
    lines = f.readlines()
flist = [os.path.join(data_dir, l.replace('\n', '')) for l in lines]
shape1 = (np.load(flist[0]).squeeze()).shape
var = []

for file in tqdm(flist[:8760]):
    var.append(np.load(file))

with open('test.pkl', 'wb') as f:
    pickle.dump(var, f)