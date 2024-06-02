import numpy as np

# 假设你有以下numpy数组
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# 计算每行的平均值
row_means = np.mean(data, axis=0)

print(row_means)