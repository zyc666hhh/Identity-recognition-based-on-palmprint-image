# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/23
# @file     :ageman.py
# @contact  :17866548902@163.com
# @info     :分类年龄、性别（男1女2）、民族

# import pytorch_lightning as pl
import torch
import torch.nn as nn
# from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import resnet18
import torch.optim as optim
# from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###模型定义模块
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # self.modle = resnet18(pretrained=True, progress=True) # 测试的时候注释掉
        self.dropout = nn.Dropout2d(0.5)
        self.classifier1 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 2)  # 性别
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 7)  # 7种年龄
        )
        self.classifier3 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 6)  # 共有六个民族
        )
        self.classifier4 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 17)#shengao
        )
        self.classifier5 = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            self.dropout,
            nn.Linear(256, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, 20)# tizhong
        )

    def forward(self, imgs):
        return self.classifier1(self.modle(imgs)), self.classifier2(self.modle(imgs)), self.classifier3(
            self.modle(imgs)), self.classifier4(self.modle(imgs)), self.classifier5(self.modle(imgs))
        # return self.classifier1(self.modle(imgs))


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
        indexMap1 = {n: i for i, n in enumerate(sorted(df["gender"].unique()))}
        indexMap2 = {n: i for i, n in enumerate(sorted(df["age"].unique()))}
        indexMap3 = {n: i for i, n in enumerate(sorted(df["nation"].unique()))}
        indexMap4 = {n: i for i, n in enumerate(sorted(df["height"].unique()))}
        indexMap5 = {n: i for i, n in enumerate(sorted(df["weight"].unique()))}
        labels1 = df["gender"].to_list()
        labels2 = df["age"].to_list()
        labels3 = df["nation"].to_list()
        labels4 = df["height"].to_list()
        labels5 = df["weight"].to_list()
        labels = [(indexMap1[i], indexMap2[j], indexMap3[z], indexMap4[h], indexMap5[w]) for i, j, z, h, w in zip(labels1, labels2, labels3, labels4, labels5)]
        # imgs = [self.path + i for i in df["filename"].to_list()]
        imgs = [i for i in df["filename"].to_list()]
        return imgs, labels


###模型训练模块

transform = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dir = '../datasets/train/'
val_dir = '../datasets/val/'
train_labeldir = '../datasets/train_label.csv'
val_labeldir = '../datasets/val_label.csv'
traindata = MydataSet(train_dir, train_labeldir, transform)
valdata = MydataSet(val_dir, val_labeldir, transform)
train_loader = data.DataLoader(traindata, batch_size=16, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
val_loader = data.DataLoader(valdata, batch_size=16, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)

train_data_size = len(traindata)
val_data_size = len(valdata)

resnet18 = ResNet18().to(device)
# 定义损失函数和优化器
criteon = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters())

# 定义超参
batch_size = 16
epochs = 100
lr = 1e-3


scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 训练和验证
def train_and_valid(model, loss_function, optimizer, epochs=100, ):
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
            # print(label)
            # print(len(label))
            label1, label2, label3, label4, label5 = label  # label = label.to(device)
            label1 = label1.to(device)
            label2 = label2.to(device)
            label3 = label3.to(device)
            label4 = label4.to(device)
            label5 = label5.to(device)
            # print(x, label)
            out1, out2, out3, out4, out5 = model(x)
            out1 = out1.to(device)
            out2 = out2.to(device)
            out3 = out3.to(device)
            out4 = out4.to(device)
            out5 = out5.to(device)
            loss1 = criteon(out1, label1)
            loss2 = criteon(out2, label2)
            loss3 = criteon(out3, label3)
            loss4 = criteon(out4, label4)
            loss5 = criteon(out5, label5)
            loss = loss1 + loss2 + loss3 + loss4 + loss5

            # baclkprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 计算loss，✖batch_size是因为loss是平均值
            train_loss += loss.item() * x.size(0)

            # 计算acc
            # pred = logits.argmax(dim=1)
            # correct = torch.eq(pred, label).float().sum().item()
            # acc = correct / x.size(0)
            # train_acc += acc * x.size(0)
            # 计算acc
            _, pred1 = out1.max(1)
            _, pred2 = out2.max(1)
            _, pred3 = out3.max(1)
            _, pred4 = out4.max(1)
            _, pred5 = out5.max(1)
            correct1 = pred1.eq(label1).sum().item()
            correct2 = pred2.eq(label2).sum().item()
            correct3 = pred3.eq(label3).sum().item()
            correct4 = pred4.eq(label4).sum().item()
            correct5 = pred5.eq(label5).sum().item()
            acc = (correct1 + correct2 + correct3 + correct4 + correct5) / (5 * x.size(0))
            train_acc += acc * x.size(0)

        # 测试
        with torch.no_grad():
            model.eval()

            for batch_idx, (x, label) in enumerate(val_loader):
                x = x.to(device)
                # print(label)
                # print(len(label))
                label1, label2, label3, label4, label5 = label  # label = label.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                label3 = label3.to(device)
                label4 = label4.to(device)
                label5 = label5.to(device)
                # print(x, label)
                out1, out2, out3, out4, out5 = model(x)
                out1 = out1.to(device)
                out2 = out2.to(device)
                out3 = out3.to(device)
                out4 = out4.to(device)
                out5 = out5.to(device)
                loss1 = loss_function(out1, label1)
                loss2 = loss_function(out2, label2)
                loss3 = loss_function(out3, label3)
                loss4 = loss_function(out4, label4)
                loss5 = loss_function(out5, label5)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                # loss = loss1
                valid_loss += loss.item() * x.size(0)

                # 计算loss，✖batch_size是因为loss是平均值
                # 计算acc
                _, pred1 = out1.max(1)
                _, pred2 = out2.max(1)
                _, pred3 = out3.max(1)
                _, pred4 = out4.max(1)
                _, pred5 = out5.max(1)
                correct1 = pred1.eq(label1).sum().item()
                correct2 = pred2.eq(label2).sum().item()
                correct3 = pred3.eq(label3).sum().item()
                correct4 = pred4.eq(label4).sum().item()
                correct5 = pred5.eq(label5).sum().item()
                acc = (correct1 + correct2 + correct3 + correct4 + correct5) / (5 * x.size(0))
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
        if epoch >= 20:
            torch.save(model, model_path + '/' + 'age_gender_model_' + str(epoch + 1) + '.pth')
    return model, history, best_acc, best_epoch

"""
start_time = time.time()

model, history, best_acc, best_epoch = train_and_valid(resnet18, criteon, optimizer, epochs)

end_time = time.time()
print("total_time:", end_time - start_time)

model_path = 'models_resnet_ep' + str(epochs)
history = np.array(history)
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 8)
plt.title('best_acc:' + str(best_acc)[0:6] + ' best_epoch' + str(best_epoch))
plt.savefig('Palmprint' + model_path + '_loss_curve.png')
plt.close()

plt.plot(history[:, 2:])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('best_acc:' + str(best_acc)[0:6] + ' best_epoch' + str(best_epoch))
plt.savefig('Palmprint' + model_path + '_accuracy_curve.png')"""
# plt.close()
