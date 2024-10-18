#!/usr/bin/env python
# coding: utf-8

# # Import libraries, load and transform data

import warnings
warnings.filterwarnings("ignore")  # Ignore warnings during execution
import os
import gc
import timm
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import accelerate
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import TrainingArguments, Trainer, DefaultDataCollator
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomRotation, RandomResizedCrop, RandomHorizontalFlip, RandomAdjustSharpness, Resize, ToTensor
from PIL import ImageFile
from pathlib import Path
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'#防止内存碎片化

# Load and prepare data
file_names, labels = [], []
for file in tqdm(sorted((Path('../input/Human Faces Dataset/').glob('*/*.*')))):
    file_names.append(str(file))
    label = ' '.join(str(file).split('/')[-2].split('_')[:2])
    labels.append(label)

df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

# Random oversampling of all minority classes
y = df[['label']]
df = df.drop(['label'], axis=1)
ros = RandomOverSampler(random_state=83)
df, y_resampled = ros.fit_resample(df, y)
df['label'] = y_resampled
gc.collect()

dataset = Dataset.from_pandas(df).cast_column("image", Image())

# Create label mappings
labels_list = sorted(list(set(labels)))
label2id = {label: i for i, label in enumerate(labels_list)}
id2label = {i: label for i, label in enumerate(labels_list)}

ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)
dataset = dataset.cast_column('label', ClassLabels)
dataset = dataset.train_test_split(test_size=0.4, shuffle=True, stratify_by_column="label")
train_data, test_data = dataset['train'], dataset['test']

# Image processing
local_model_path = "../input/pretrained-vit-hybrid"

processor = ViTHybridImageProcessor.from_pretrained(local_model_path)
image_mean, image_std = processor.image_mean, processor.image_std
# size = processor.size["height"]
size = 384  #input 

normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose([
    Resize((size, size)),
    RandomRotation(20),
    RandomAdjustSharpness(2),
    RandomHorizontalFlip(0.5),
    ToTensor(),
    normalize
])

_val_transforms = Compose([
    Resize((size, size)),
    ToTensor(),
    normalize
])

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

train_data.set_transform(train_transforms)
test_data.set_transform(val_transforms)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example['label'] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Load, train, and evaluate model
model = ViTHybridForImageClassification.from_pretrained(local_model_path, num_labels=len(labels_list), ignore_mismatched_sizes=True)

model.config.id2label = id2label
model.config.label2id = label2id

print(f"Number of trainable parameters: {model.num_parameters(only_trainable=True) / 1e6:.2f}M")

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.argmax(axis=1)
    label_ids = eval_pred.label_ids
    acc_score = accuracy_metric.compute(predictions=predictions, references=label_ids)['accuracy']
    f1 = f1_score(label_ids, predictions, average='macro')
    return {"accuracy": acc_score, "f1": f1}

args = TrainingArguments(
    output_dir="../output/training_vit_hybrid_model",
    logging_dir='./logs',
    evaluation_strategy="epoch",
    learning_rate=1e-5,  # 调整学习率以加快模型收敛
    per_device_train_batch_size=8,  # 降低batch size以减少显存占用
    gradient_accumulation_steps=4,  # 累积4步相当于batch size = 32
    per_device_eval_batch_size=4,    # 同样降低评估时的batch size
    num_train_epochs=30,
    weight_decay=0.02,
    warmup_steps=50,
    remove_unused_columns=False,
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)

trainer.evaluate()
trainer.train()
trainer.evaluate()

outputs = trainer.predict(test_data)
print(outputs.metrics)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

if len(labels_list) <= 120:
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

print("\nClassification report:\n")
print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

torch.save(model.state_dict(), '../output/vit_hybrid_model.pth')
