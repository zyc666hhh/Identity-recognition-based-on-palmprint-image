# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/25
# @file     :model2.py
# @contact  :17866548902@163.com
# @info     :只计算tizhong

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import torch.utils.data as data
from PIL import Image

# 是否在GPU上面跑
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义超参
batch_size = 16
epochs = 30

###数据加载模块
class MydataSet(data.Dataset):
    def __init__(self, path, label_path, transform):
        super(MydataSet, self).__init__()
        self.path = path
        self.label_path = label_path
        self.transform = transform
        self.imgs, self.labels = self._get_images()
        # self.labels = self._get_labels()

    def __len__(self):
        return len(self.imgs[0])

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        label = self.labels[item]
        return img, label

    def _get_images(self):  ##只训练性别和年龄的这两个识别器
        df = pd.read_csv(self.label_path)
        # indexMap1 = {n: i for i, n in enumerate(sorted(df["gender"].unique()))}
        indexMap2 = {n: i for i, n in enumerate(sorted(df["weight"].unique()))}
        # print(indexMap2)
        # indexMap3 = {n: i for i, n in enumerate(sorted(df["nation"].unique()))}
        # indexMap4 = {n: i for i, n in enumerate(sorted(df["height"].unique()))}
        # indexMap5 = {n: i for i, n in enumerate(sorted(df["weight"].unique()))}
        # labels1 = df["gender"].to_list()
        labels2 = df["weight"].to_list()
        # labels3 = df["nation"].to_list()
        # labels4 = df["height"].to_list()
        # labels5 = df["weight"].to_list()
        # labels = [(indexMap1[i], indexMap2[j], indexMap3[z], indexMap4[h], indexMap5[w]) for i, j, z, h, w in zip(labels1, labels2, labels3, labels4, labels5)]
        # imgs = [self.path + i for i in df["filename"].to_list()]
        labels = [indexMap2[i] for i in labels2]
        imgs = [i for i in df["filename"].to_list()]
        return imgs, labels


transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dir = '../datasets2/train/'
val_dir = '../datasets2/val/'
train_labeldir = '../datasets2/train_label.csv'
val_labeldir = '../datasets2/val_label.csv'

traindata = MydataSet(train_dir, train_labeldir, transform)
valdata = MydataSet(val_dir, val_labeldir, transform)

train_loader = data.DataLoader(traindata, batch_size=16, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
val_loader = data.DataLoader(valdata, batch_size=16, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

train_data_size = len(traindata)
val_data_size = len(valdata)

# 创建网络
resnet18 = models.resnet18()

# 添加自己的全连接层
fc_x = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(fc_x, 256),
    nn.Dropout(0.5, inplace=True),
    nn.ReLU(),
    nn.Linear(256, 20), #7种年龄 , 20种体重
)


# 使用自己的resnet18
# resnet18 = ResNet18()
resnet18 = resnet18.to(device)

# 定义损失函数和优化器
criteon = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters(), lr=1e-3)


# 训练和验证
def train_and_valid(model, loss_function, optimizer, epochs=100):
    history = []
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        # 转换为训练模式
        model.train()

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        for batch_idx, (x, label) in enumerate(train_loader):
            x = x.to(device)
            label = label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算loss，✖batch_size是因为loss是平均值
            train_loss += loss.item() * x.size(0)

            # 计算acc
            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            acc = correct / x.size(0)
            train_acc += acc * x.size(0)

        # 测试
        with torch.no_grad():
            model.eval()

            for batch_idx, (x, label) in enumerate(val_loader):
                # 在gpu跑
                x = x.to(device)
                label = label.to(device)

                # 损失
                logits = model(x)
                loss = loss_function(logits, label)
                valid_loss += loss.item() * x.size(0)

                # 计算acc
                pred = logits.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                acc = correct / x.size(0)
                valid_acc += acc * x.size(0)

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        avg_valid_loss = valid_loss / val_data_size
        avg_valid_acc = valid_acc / val_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                epoch_end - epoch_start
            ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        model_path = 'models_resnet_ep' + str(epochs)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        # 保存10个模型
        if epoch >= epochs-10:
            torch.save(model, model_path + '/' + 'A_B_model_' + str(epoch + 1) + '.pth')
    return model, history, best_acc, best_epoch


start_time = time.time()

model, history, best_acc, best_epoch = train_and_valid(resnet18, criteon, optimizer, epochs)

end_time = time.time()
print("total_time:", end_time-start_time)

# for img,label in val_loader:
#     print(label)

