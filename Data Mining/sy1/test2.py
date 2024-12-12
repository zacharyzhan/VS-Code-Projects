import numpy as np
from collections import Counter

# 先验数据
train_features = np.array([[1.0, 1.1],
                           [1.0, 1.0],
                           [0.9, 0.8],
                           [0.0, 0.0],
                           [0.1, 0.1]])

train_labels = np.array(['A', 'A', 'A', 'B', 'B'])

# 未知类别数据
test_data = np.array([[0.1, 0.3],
                     [1.1, 1.2]])

# K值
k = 3


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def predict(test_point):
    distances = [euclidean_distance(test_point, train_point) for train_point in train_features] 
    k_nearest_indices = np.argsort(distances)[:k]
    k_nearest_labels = [train_labels[i] for i in k_nearest_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]


# 预测表二数据的类别
for test_point in test_data:
    predicted_label = predict(test_point)
    print(f"属性1: {test_point[0]}, 属性2: {test_point[1]}, 预测类别: {predicted_label}")


