#!/usr/bin/env python
# coding: utf-8

import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import numpy as np

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型
model = torch.load('../output/model3.pth')
model = model.to(device)
model.eval()

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),  # 根据模型调整大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 根据预训练模型的要求调整
])

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
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
            
            # 进行预测
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_class = predicted.item()
            
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
