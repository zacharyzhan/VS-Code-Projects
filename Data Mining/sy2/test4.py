import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

# 房子的面积（平方米）和房价（万元）数据
area = np.array([50, 60, 70, 80, 90, 100]).reshape(-1, 1)
price = np.array([150, 180, 210, 240, 270, 300])

# 创建线性回归模型
model = LinearRegression()
model.fit(area, price)

# 预测88平方米房子的房价
predicted_price = model.predict(np.array([[88]]))

print(f"88平方米房子的预测房价为: {predicted_price[0]:.2f} 万元")

# 可视化
plt.scatter(area, price, color='blue', label='实际数据')
plt.plot(area, model.predict(area), color='red', label='回归线')
plt.scatter(88, predicted_price, color='green', label='预测点')
plt.xlabel('房子面积（平方米）')
plt.ylabel('房价（万元）')
plt.legend()
plt.show()