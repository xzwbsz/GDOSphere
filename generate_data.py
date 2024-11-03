import numpy as np


#shape = '[2,3]'
A = np.zeros([16,1], dtype=float)
B = np.expand_dims(A,1)
print (B.shape)
np.savetxt('data.txt',A)