import numpy as np
train = []
test = []
for idx in range(6500):
    text = str(idx)+'.npy'
    train.append(text)

for idx in range(2000):
    idx = idx+6500
    text = str(idx)+'.npy'
    test.append(text)
np.savetxt('train_split.txt',train,fmt="%s")
np.savetxt('val_split.txt',test,fmt="%s")