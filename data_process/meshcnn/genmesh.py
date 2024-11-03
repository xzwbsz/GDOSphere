import numpy as np
import mesh
from mesh import icosphere as icos

levellist = []
level = 9#设定level，5是指分形5次，这里可以有多个
for i in range(level):
    levellist.append(i)

for idx in levellist: 
    mesh.export_spheres([idx],'./')
    lat = np.array(icos(idx).lat2).tolist() #纬度
    lon = np.array(icos(idx).long2).tolist() #精度
    x = np.array(icos(idx).xx)
    y = np.array(icos(idx).yy)
    z = np.array(icos(idx).zz)
    latname = 'lat_' + str(idx) + '.txt'
    longname = 'long_' + str(idx) + '.txt'
    xyz = 'xyz'
    np.savetxt(latname, lat)
    np.savetxt(longname, lon)
    np.savez(xyz+str(idx),x=x,y=y,z=z)


    
