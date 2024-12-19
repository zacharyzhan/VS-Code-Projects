from sklearn import svm

# 数据集和类别标签
X = [[2, 0, 1], [1, 1, 2], [2, 3, 3]]
y = [0, 0, 1]

# 创建支持向量机分类器，使用线性核函数
clf = svm.SVC(kernel='linear')

# 训练分类模型
clf.fit(X, y)

# 预测新数据的类别
new_data = [[2, 0, 3]]
prediction = clf.predict(new_data)

print(f"Predicted class for {new_data} is {prediction[0]}")