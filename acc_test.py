import numpy as np

def gen_weight():
    latitudes = np.linspace(-90, 90, 181)
    lon = np.linspace(0, 359, 360)
    weights = np.cos(latitudes*np.pi/180)
    W=np.tile(weights,(len(lon),1)).T
    return W

def weighted_average(x):
    W = gen_weight()
    z=np.average(x,weights=W)
    return z

def acc(pre,true):
    a= weighted_average(true*pre)
    b=np.sqrt(weighted_average(true*true))
    c=np.sqrt(weighted_average(pre*pre))
    return a/b/c

p1 = np.random.rand(181,360)
p2 = np.random.rand(181,360)

ACC = acc(p1,p2)
print(ACC)