#!/usr/bin/env python
# coding: utf-8

import os
import torch
import timm
import pandas as pd
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例并加载预训练权重
model = timm.create_model('vit_base_patch16_224', pretrained=False)

# 修改最后的分类层，使其输出 2 个类别
num_classes = 2
model.head = torch.nn.Linear(model.head.in_features, num_classes)

# 加载状态字典
state_dict = torch.load('../output/vit_base16_mixed.pth', weights_only=True)#, map_location=device)

# 将状态字典加载到模型中
model.load_state_dict(state_dict)

# 将模型移动到设备（GPU或CPU）
model.to(device)

# 设置为评估模式
model.eval()

# 定义图像预处理函数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), ###resize 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)  # 增加批次维度

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
        
        # 预处理图像
        image_tensor = preprocess_image(image)
        
        # 将输入数据移动到设备
        image_tensor = image_tensor.to(device)
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        predicted_results[image_name] = predicted_class
    return predicted_results

# 指定图片文件夹
image_folder = '../testdata'
predicted_results = predict_images(image_folder)

# 将结果写入output.csv
# 去掉文件格式后缀并对字典的键（文件名）进行排序
sorted_predicted_results = sorted(
    (os.path.splitext(image_name)[0], predicted_class) for image_name, predicted_class in predicted_results.items()
)

# 将排序后的结果转换为DataFrame
df = pd.DataFrame(sorted_predicted_results, columns=['ImageName', 'PredictedClass'])

# 将结果写入output.csv
df.to_csv('../cla_pre_vit_base16_mixed.csv', index=False, header=False, encoding='utf-8')
print('ok')
