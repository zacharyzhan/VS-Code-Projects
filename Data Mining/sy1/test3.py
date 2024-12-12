import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 数据样本集合 D
data = np.array([[1, 0],
                 [4, 0],
                 [0, 1],
                 [1, 1],
                 [2, 1],
                 [3, 1],
                 [4, 1],
                 [5, 1],
                 [0, 2],
                 [1, 2],
                 [4, 2],
                 [1, 3]])

# (1) 画出 D 的散点图
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('散点图')
plt.show()

# (2) 利用 KMeans 进行聚类 (k=2)
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(data)

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap='viridis', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('KMeans 聚类 (k=2)')
plt.show()

# (3) 利用 DBSCAN 进行聚类 (eps=1, min_samples=4)
dbscan = DBSCAN(eps=1, min_samples=4)
dbscan_labels = dbscan.fit_predict(data)

plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], c=dbscan_labels, cmap='viridis', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('DBSCAN 聚类 (eps=1, min_samples=4)')
plt.show()
