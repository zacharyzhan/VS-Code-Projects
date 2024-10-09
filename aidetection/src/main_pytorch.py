#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np
from transformers import ViTForImageClassification, ViTImageProcessor

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[27]:


# 加载预训练模型
# model = torch.load('../output/vit_model.pth')
# model = model.to(device)
# model.eval()

# 创建模型实例
local_model_path = "../input/pretrained_vit_model"

model = ViTForImageClassification.from_pretrained(local_model_path, num_labels=2)

# 加载状态字典
state_dict = torch.load('../output/vit_model.pth', map_location=device)

# 将状态字典加载到模型中
model.load_state_dict(state_dict)

# 将模型移动到设备（GPU或CPU）
model.to(device)

# 设置为评估模式
model.eval()

# 加载图像处理器
image_processor = ViTImageProcessor.from_pretrained(local_model_path)

def predict_images(image_folder):
    predicted_results = {}
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        # 检查文件扩展名，确保只处理图像文件
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, IOError):
            print(f"Cannot identify image file {image_path}. Skipping.")
            continue
        
        # 调整图像大小为224x224
        image = image.resize((224, 224))
        
        inputs = image_processor(images=image, return_tensors="pt")
        
        # 将输入数据移动到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            predicted_class = predicted.item()
        
        predicted_results[image_name] = predicted_class
    return predicted_results

# 指定图片文件夹
image_folder = '../testdata'
predicted_results = predict_images(image_folder)


# 将结果写入output.csv
import pandas as pd
import os

# 去掉文件格式后缀并对字典的键（文件名）进行排序
sorted_predicted_results = sorted(
    (os.path.splitext(image_name)[0], predicted_class) for image_name, predicted_class in predicted_results.items()
)

# 将排序后的结果转换为DataFrame
df = pd.DataFrame(sorted_predicted_results, columns=['ImageName', 'PredictedClass'])

# 将结果写入output.csv
df.to_csv('../cla_pre.csv', index=False, header=False)
print('ok')




