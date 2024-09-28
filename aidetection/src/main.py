import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

# 加载预训练模型
model = tf.keras.models.load_model('./best_model.h5')

# 读取图片并进行预测
def predict_images(image_folder):
    results = []

    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 可以根据需要调整图片格式
            img_path = os.path.join(image_folder, filename)
            
            # 读取并处理图片
            img = image.load_img(img_path, target_size=(128, 128))  # 根据模型调整大小
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # 归一化
            
            # 进行预测
            predictions = model.predict(img_array)
            # 根据预测结果确定预测类别
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # 保存结果
            results.append({'filename': filename, 'predicted_class': predicted_class})

    return results

# 指定图片文件夹
image_folder = '../testdata'
predicted_results = predict_images(image_folder)

# 将结果写入output.csv
df = pd.DataFrame(predicted_results)
df.to_csv('../cla_pre.csv', index=False , header=False)
