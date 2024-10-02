#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 加载预训练模型
model = tf.keras.models.load_model('../output/model3.h5')


# 读取图片并进行预测
def predict_images(image_folder):
    results = []

    # 获取文件名并按字典序升序排序，区分大小写
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg') or 
                    f.endswith('.png')], key=lambda s: s.lower())
    
    for filename in image_files:
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 可以根据需要调整图片格式
            img_path = os.path.join(image_folder, filename)
            
            # 读取并处理图片
            img = image.load_img(img_path, target_size=(128, 128))  # 根据模型调整大小
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # 归一化
            
            # 进行预测
            predictions = model.predict(img_array)
            predicted_class = 1 if predictions[0][0] >= 0.5 else 0  # 根据概率判断类别
            
            # 去掉文件扩展名
            img_name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
            # 保存结果
            results.append({'filename': img_name_without_ext, 'predicted_class': predicted_class})

    return results

# 指定图片文件夹
image_folder = '../testdata'
predicted_results = predict_images(image_folder)

# 将结果写入output.csv
df = pd.DataFrame(predicted_results)
df.to_csv('../cla_pre.csv', index=False, header=False)
print('ok')
