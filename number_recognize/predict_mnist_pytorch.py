import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型类
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载MNIST测试数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = torchvision.datasets.MNIST(root='auto', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=20, shuffle=True)

# 加载最优模型状态
model = ConvNet().to(device)
model.load_state_dict(torch.load('mnist_model.pth', map_location=device, weights_only=True)) # 确保加载的是最佳模型

model.eval()

# 从测试集中获取20个样本进行预测
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 移动数据到设备
images, labels = images.to(device), labels.to(device)

# 进行预测
outputs = model(images)
_, predicted_classes = torch.max(outputs.data, 1)

# 计算预测正确率
correct = (predicted_classes == labels).sum().item()
accuracy = correct / len(labels) * 100

# 显示图片和预测结果
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.flatten()
for i in range(20):
    img = images[i].cpu().numpy().squeeze()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {labels[i].item()}\nPred: {predicted_classes[i].item()}')
    axes[i].axis('off')

# 显示预测正确率
plt.suptitle(f'Accuracy: {accuracy:.2f}%', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
