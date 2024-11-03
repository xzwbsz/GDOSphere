import numpy as np
mesh_folder = '/xuzhewen/gnn/data_process/meshcnn'
max_level = 5
File1 = str(mesh_folder) + '/lat_' + str(max_level) + '.txt'
File2 = str(mesh_folder) + '/long_' + str(max_level) + '.txt'
data_list = []
with open(File1, encoding='utf-8') as file_obj:
    for line in file_obj:
        data_list.append(line.rstrip().split(','))
lat_icos = np.array(data_list).astype(float).squeeze()
#print(lat_icos)
data_list = []
with open(File2, encoding='utf-8') as file_obj:
    for line in file_obj:
        data_list.append(line.rstrip().split(','))
lon_icos = np.array(data_list).astype(float).squeeze()  
lat_index = np.ones(lon_icos.shape[0])
lon_index = np.ones(lon_icos.shape[0])
for idx_icos in range(lon_icos.shape[0]):
    lat_index[idx_icos] = round((np.pi/2+lat_icos[idx_icos])/np.pi*180) #720
    lon_index[idx_icos] = round((lon_icos[idx_icos]+np.pi)/(2*np.pi)*359) #1439

np.savez('lat_lon1d.npz',lat=lat_index,lon=lon_index)