import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据集路径
train_dir = './dataset1/train'
val_dir = './dataset1/val'
test_dir = './dataset1/test'


# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, uav_dir, background_dir, transform=None):
        self.uav_images = [os.path.join(uav_dir, img) for img in os.listdir(uav_dir)]
        self.background_images = [os.path.join(background_dir, img) for img in os.listdir(background_dir)]
        self.images = self.uav_images + self.background_images
        self.labels = [1] * len(self.uav_images) + [0] * len(self.background_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path  

# 数据增强和转换
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 创建数据集和数据加载器
train_dataset = ImageDataset(os.path.join(train_dir, 'UAV'), os.path.join(train_dir, 'background'), transform=transform)
val_dataset = ImageDataset(os.path.join(val_dir, 'UAV'), os.path.join(val_dir, 'background'), transform=transform)
test_dataset = ImageDataset(os.path.join(test_dir, 'UAV'), os.path.join(test_dir, 'background'), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

