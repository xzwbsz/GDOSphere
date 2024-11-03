import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt
t = (1.0 + np.sqrt(5.0)) / 2.0
vertices = np.array([
    [-1,  t,  0],
    [ 1,  t,  0],
    [-1, -t,  0],
    [ 1, -t,  0],
    [ 0, -1,  t],
    [ 0,  1,  t],
    [ 0, -1, -t],
    [ 0,  1, -t],
    [ t,  0, -1],
    [ t,  0,  1],
    [-t,  0, -1],
    [-t,  0,  1]
])

faces = np.array([
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1]
])

def create_icosphere(order):
    faces_=faces
    vertices_ = vertices
    
    def normalize(v):
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        return v / norm

    for _ in range(order):
        faces_subdiv = []
        midpoints = {}

        def midpoint(i0, i1):
            key = tuple(sorted([i0, i1]))
            if key not in midpoints:
                mid = (vertices_[i0] + vertices_[i1]) / 2.0
                midpoints[key] = len(vertices_)
                vertices_ = np.vstack([vertices_, mid])
            return midpoints[key]

        for tri in faces_:
            v0, v1, v2 = tri
            a = midpoint(v0, v1)
            b = midpoint(v1, v2)
            c = midpoint(v2, v0)

            faces_subdiv.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c]
            ])

        faces_ = np.array(faces_subdiv)

    vertices_ = normalize(vertices_)
    return vertices, faces_

# Step 1: 生成6阶Icosphere
vertex, faces = create_icosphere(6)
print(vertex)

# # Step 2: 生成示例全球温度数据（181x360）
# latitude = np.linspace(-90, 90, 181)
# longitude = np.linspace(-180, 180, 360)
# temperature_data = np.random.rand(181, 360) * 30  # 生成随机温度数据，范围0-30度

# # 将温度数据转换为PyTorch张量
# temperature_data = torch.tensor(temperature_data, dtype=torch.float32)

# # Step 3: 定义神经网络
# class TempToIcoNet(nn.Module):
#     def __init__(self):
#         super(TempToIcoNet, self).__init__()
#         self.fc1 = nn.Linear(2, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 初始化神经网络
# model = TempToIcoNet()

# # Step 4: 将经纬度网格转换为输入数据
# lon_grid, lat_grid = np.meshgrid(longitude, latitude)
# lon_lat_pairs = np.vstack([lon_grid.ravel(), lat_grid.ravel()]).T
# lon_lat_pairs = torch.tensor(lon_lat_pairs, dtype=torch.float32)

# # 预测Icosphere顶点上的温度
# ico_temps = []

# with torch.no_grad():
#     for vertex in vertices:
#         # 将顶点的(x, y, z)转换为经纬度
#         lat = torch.asin(vertex[2]) * 180.0 / np.pi
#         lon = torch.atan2(vertex[1], vertex[0]) * 180.0 / np.pi
#         # 查找最近的经纬度点
#         lat_idx = torch.abs(torch.tensor(latitude) - lat).argmin()
#         lon_idx = torch.abs(torch.tensor(longitude) - lon).argmin()
#         temp = temperature_data[lat_idx, lon_idx]
#         ico_temps.append(temp)

# ico_temps = torch.tensor(ico_temps, dtype=torch.float32)

# # Step 5: 绘制Icosphere温度分布图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=ico_temps, cmap='coolwarm')
# plt.colorbar(sc, label='Temperature (°C)')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Icosphere Temperature Distribution')
# plt.show()
