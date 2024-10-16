#!/usr/bin/env python
# coding: utf-8

# # 1. Environment

# ## 1.1 Kaggle environment

import os
import shutil

os.mkdir("/kaggle/working/code")
os.mkdir("/kaggle/working/model")
os.mkdir("/kaggle/working/output")
shutil.copyfile(src="/kaggle/input/models-with-code/code/swin_transformer_v2.py", 
                dst="/kaggle/working/code/swin_transformer_v2.py")
shutil.copyfile(src="/kaggle/input/models-with-code/model/swinv2_tiny_patch4_window16_256.pth", 
                dst="/kaggle/working/model/swinv2_tiny_patch4_window16_256.pth")
os.chdir("/kaggle/working/code")

%config Completer.use_jedi = False

import gc
import random

import torch
import warnings
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch import optim
from torchvision import models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, sampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from swin_transformer_v2 import SwinTransformerV2


# set random seeds to make results reproducible
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

SEED = 42
seed_everything(SEED)


warnings.filterwarnings("ignore")
plt.rcParams.update({'axes.titlesize': 20})


# ## 1.3 Arguments

# We put hyperparameter together for easy modification.

class Args:
    def __init__(self) -> None:
        # data arguments
        self.num_classes = 2
        self.img_size = 256
        self.num_train_data = 10000
        self.num_test_data = 2000
        self.dataset_path = "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/"
        
        # training arguments
        self.learning_rate =  1e-4
        self.epochs = 10
        self.scheduler = True
        self.sch_step_size = 2
        self.sch_gamma = 0.1
        
        # model arguments
        self.drop_path_rate = 0.2
        self.embed_dim = 96
        self.depths = (2, 2, 6, 2)
        self.num_heads = (3, 6, 12, 24)
        self.window_size = 16
        self.load_model_path = "/kaggle/working/model/swinv2_tiny_patch4_window16_256.pth"
        self.save_model_path = "/kaggle/working/output/swinv2_tiny_patch4_window16_256.pth"
        
        # output arguments
        self.output_path = "/kaggle/working/output/"


args = Args()


# # 2. Data

# ## 2.1 Augmentation

# Take some augmentation actions on the images to improve training effectiveness.

train_augmentations = transforms.Compose([
    transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0), ratio=(3./ 4., 4. / 3.)),
    transforms.RandomHorizontalFlip(0.5),
    # transforms.RandomVerticalFlip(0.1),  # no VerticalFlip
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_augmentations = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
basic_augmentations = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.CenterCrop(args.img_size),
    transforms.ToTensor()
])


# ## 2.2 Read and split

# Read the datasets and split it into training and testing sets. It should be noted that due to the large size of the original dataset, we will only randomly select a portion of the data to complete our task.


# read train and test dataset
train_dataset = datasets.ImageFolder(root=args.dataset_path + "train/", transform=train_augmentations)
test_dataset = datasets.ImageFolder(root=args.dataset_path + "test/", transform=test_augmentations)


# select a subset of the dataset
train_fake_all_indices = np.arange(len(train_dataset) / 2, dtype=np.int32)
train_fake_indices = np.random.choice(train_fake_all_indices, size=int(args.num_train_data / 2), replace=False)
train_real_indices = train_fake_indices + int(len(train_dataset) / 2)
train_indices = np.append(train_fake_indices, train_real_indices)

test_fake_all_indices = np.arange(len(test_dataset) / 2, dtype=np.int32)
test_fake_indices = np.random.choice(test_fake_all_indices, size=int(args.num_test_data / 2), replace=False)
test_real_indices = test_fake_indices + int(len(test_dataset) / 2)
test_indices = np.append(test_fake_indices, test_real_indices)


train_sampler = sampler.SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=2, sampler=train_sampler)
test_sampler = sampler.SubsetRandomSampler(test_indices)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, num_workers=2, sampler=test_sampler)


classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx
idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))


# ## 2.3 Visualization

# Visualize the images we read in, as well as the augmented data.


raw_dataset = datasets.ImageFolder(root=args.dataset_path + "train/", transform=basic_augmentations)
valid_dataset = datasets.ImageFolder(root=args.dataset_path + "valid/", transform=basic_augmentations)



# we randomly select real and fake face images
indices = [random.randint(0, len(train_dataset)) for i in range(16)]


# show raw training data
figure = plt.figure(figsize=(16, 16))
for i in range(16):
    index = indices[i]
    img = raw_dataset[index][0].permute(1, 2, 0)
    label = idx_to_class[raw_dataset[index][1]]
    figure.add_subplot(4, 4, i + 1)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img)



# show augmented training data 
figure = plt.figure(figsize=(16, 16))
for i in range(16):
    index = indices[i]
    img = train_dataset[index][0].permute(1, 2, 0)
    label = idx_to_class[train_dataset[index][1]]
    figure.add_subplot(4, 4, i + 1)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img)


# # 3. Model

# ## 3.1 Network

# We will use pre-trained swin-transformer V2 model.


# load pretrained model
model = SwinTransformerV2(img_size=args.img_size,
                          drop_path_rate=args.drop_path_rate, 
                          embed_dim=args.embed_dim,
                          depths=args.depths,
                          num_heads=args.num_heads,
                          window_size=args.window_size)
state_dict = torch.load(args.load_model_path)
model.load_state_dict(state_dict["model"])


# change the last linear layer to fit our classification problem
model.head = torch.nn.Linear(model.head.in_features, args.num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = model.to(device)


# ## 3.2 Optimizer


optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)



if args.scheduler:
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=args.sch_step_size, 
                                          gamma=args.sch_gamma)


# ## 3.3 Loss function


loss_fn = torch.nn.CrossEntropyLoss()


# # 4. Train

# Start training the model and record the intermediate results, including loss, accuracy, precision, recall and f1-score.


train_acc, test_acc = [], []
train_precision, test_precision = [], []
train_recall, test_recall = [], []
train_f1, test_f1 = [], []
train_loss, test_loss = [], []


class LossBuffer:
    """
    We hope to record all losses over a period of time 
    and calculate their average value,
    which is smooth and does not have too much jitter.
    In fact, we don't need to record the entire array, 
    only the current average and number of records are enough.
    """
    
    def __init__(self, mean=0, n=0):
        self.mean = mean
        self.n = n
        
    def add(self, num):
        self.mean = (self.mean * self.n + num) / (self.n + 1)
        self.n += 1


def train(model, dataloader, epoch):
    model.train()
    correct, cursum = 0, 0
    loop = tqdm(dataloader, total=len(dataloader))
    y_true, y_pred = [], []
    loss_buffer = LossBuffer()
    for idx, (data, label) in enumerate(loop):
        data, label = data.to(device), label.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        y_true.extend(label.cpu())
        y_pred.extend(pred.cpu())
        acc = accuracy_score(y_true, y_pred)
        
        optimizer.zero_grad()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        loss_buffer.add(loss.item())
          
        loop.set_description(f"[Epoch {epoch}/{args.epochs}]")
        loop.set_postfix(LOSS="{:.6f}".format(loss_buffer.mean), ACC="{:.2f}%".format(100 * acc))
    
    if args.scheduler:
        scheduler.step()
        
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    torch.save(model.state_dict(), args.save_model_path)
    
    train_acc.append(acc)
    train_precision.append(precision)
    train_recall.append(recall)
    train_f1.append(f1)
    train_loss.append(loss_buffer.mean)


def test(model, dataloader, epoch):
    model.eval()
    correct = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        total_len = len(dataloader.dataset)
        loss_buffer = LossBuffer()
        for idx, (data, label) in enumerate(dataloader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            y_true.extend(label.cpu())
            y_pred.extend(pred.cpu())
            loss = loss_fn(output, label)
            loss_buffer.add(loss.item())
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    print("\n" + "-" * 60)
    print("[Epoch {}/{}]:  Test -> LOSS: {:.6f}  |  Accuracy: {:.2f}%".format(epoch, args.epochs, loss_buffer.mean, 100 * acc))
    print("-" * 60 + "\n")
    
    test_acc.append(acc)
    test_precision.append(precision)
    test_recall.append(recall)
    test_f1.append(f1)
    test_loss.append(loss_buffer.mean)


for epoch in range(1, args.epochs + 1):
    train(model, train_loader, epoch)
    test(model, test_loader, epoch)


# # 5. Visualization

# ## 5.1 Print


dic = {
    "train_loss"     : train_loss,
    "test_loss"      : test_loss,
    "train_acc"      : train_acc,
    "test_acc"       : test_acc,
    "train_precision": train_precision,
    "test_precision" : test_precision,
    "train_recall"   : train_recall,
    "test_recall"    : test_recall,
    "train_f1"       : train_f1,
    "test_f1"        : test_f1,
}



# print and save results
for key, value in dic.items():
    print(key + ": " + str(value), "\n")
    np.savetxt(args.output_path + key + ".txt", value)


# ## 5.2 Plot


# plot loss
plt.plot(np.arange(1, args.epochs + 1), np.array(train_loss), 'go-')
plt.plot(np.arange(1, args.epochs + 1), np.array(test_loss), 'ro-')
plt.xticks(np.arange(2, args.epochs + 1, 2))
plt.title("Loss")
plt.grid(True)
plt.legend(["train", "test"], loc="upper right")
plt.savefig("../output/Loss.png", dpi=600)


figure = plt.figure(figsize=(16,16))

# plot accuracy
figure.add_subplot(2, 2, 1)
plt.plot(np.arange(1, args.epochs + 1), np.array(train_acc), 'go-')
plt.plot(np.arange(1, args.epochs + 1), np.array(test_acc), 'ro-')
plt.xticks(np.arange(2, args.epochs + 1, 2))
plt.title("Accuracy")
plt.grid(True)
plt.legend(["train", "test"], loc="lower right")
plt.savefig("../output/Accuracy.png", dpi=600)

# plot precision
figure.add_subplot(2, 2, 2)
plt.plot(np.arange(1, args.epochs + 1), np.array(train_precision), 'go-')
plt.plot(np.arange(1, args.epochs + 1), np.array(test_precision), 'ro-')
plt.xticks(np.arange(2, args.epochs + 1, 2))
plt.title("Precision")
plt.grid(True)
plt.legend(["train", "test"], loc="lower right")
plt.savefig("../output/Precision.png", dpi=600)

# plot recall
figure.add_subplot(2, 2, 3)
plt.plot(np.arange(1, args.epochs + 1), np.array(train_recall), 'go-')
plt.plot(np.arange(1, args.epochs + 1), np.array(test_recall), 'ro-')
plt.xticks(np.arange(2, args.epochs + 1, 2))
plt.title("Recall")
plt.grid(True)
plt.legend(["train", "test"], loc="lower right")
plt.savefig("../output/Recall.png", dpi=600)

# plot F1-score
figure.add_subplot(2, 2, 4)
plt.plot(np.arange(1, args.epochs + 1), np.array(train_f1), 'go-')
plt.plot(np.arange(1, args.epochs + 1), np.array(test_f1), 'ro-')
plt.xticks(np.arange(2, args.epochs + 1, 2))
plt.title("F1-score")
plt.grid(True)
plt.legend(["train", "test"], loc="lower right")
plt.savefig("../output/F1-score.png", dpi=600)

for i in range(10):
    torch.cuda.empty_cache()

