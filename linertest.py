import numpy as np

# 房屋面积
areas = np.random.rand(128,3,10242)

# 房价
prices = np.random.rand(128,3,10242)

# 数据规范化
areas = (areas - np.mean(areas)) / np.std(areas)
prices = (prices - np.mean(prices)) / np.std(prices)
