import numpy as np
from sklearn.decomposition import PCA
from random import randint

# 生成1~30之间整数的10个样本，5个特征的数据集
data = np.array([[randint(1, 30) for _ in range(5)] for _ in range(10)])
print("原始数据集：")
print(data)

# 使用PCA进行降维（降至2维）
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print("降维后的数据集：")
print(reduced_data)