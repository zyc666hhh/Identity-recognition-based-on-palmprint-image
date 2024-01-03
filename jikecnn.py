# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/25
# @file     :jikecnn.py
# @contact  :17866548902@163.com

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
import numpy as np

device = torch.device('cuda:0')


class MultiLabelCNN(nn.Module):
    def __init__(self, num_labels, size):
        super(MultiLabelCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (size[0] // 8) * (size[1] // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 分类
        self.sex = nn.Linear(256, 2)
        self.nation = nn.Linear(256, 53)
        self.right_left = nn.Linear(256, 2)
        self.who = nn.Linear(256, num_labels)
        # 回归
        self.age = nn.Linear(256, 1)
        self.high = nn.Linear(256, 1)
        self.weight = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(f'x:{x,x.shape},sex:{self.sex(x),self.sex(x).shape}')
        sex = torch.softmax(self.sex(x), dim=1)
        nation = torch.softmax(self.nation(x), dim=1)
        right_left = torch.softmax(self.right_left(x), dim=1)
        who = torch.softmax(self.who(x), dim=1)
        # 回归
        age = self.age(x)
        high = self.high(x)
        weight = self.weight(x)
        return sex, age, high, weight, nation, right_left, who


from torchvision.transforms import transforms


# 加载模型
def load_model(model_class, size, numble_class, path='model.pth'):
    model = model_class(numble_class, size)
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.eval()
    return model


# 预测函数
def predict(image_path, model, size=(96, 160)):
    # 图片预处理
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path)
    img_tensor = preprocess(img)

    # 添加batch维度并在GPU上运行（如果可用）
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor

    # 进行预测并返回结果
    with torch.no_grad():
        output_tuple = model(img_tensor)
        results = []
        for tensor in output_tuple:
            if tensor.numel() == 1:
                results.append(tensor.item())
            else:
                results.append(tensor.cpu().numpy())

        return tuple(results)

def max_prob_index(tensor, dict):

# 转换为NumPy数组
    prob_array = np.array(tensor)

# 找到最大值的索引
    max_index = np.argmax(prob_array)
    result = dict[max_index]
    return result



sex_dict = {0:'男', 1:'女'}
left_right = {0:'右利手', 1:'左利手'}
nation_dict = {0: '汉族', 1: '蒙古族', 2: '回族', 3: '藏族', 4: '维吾尔族', 5: '苗族', 6: '彝族', 7: '壮族', 8: '布依族', 9: '朝鲜族', 10: '满族', 11: '侗族', 12: '瑶族', 13: '白族', 14: '土家族', 15: '哈尼族', 16: '哈萨克族', 17: '傣族', 18: '黎族', 19: '傈僳族', 20: '佤族', 21: '畲族', 22: '高山族', 23: '拉祜族', 24: '水族', 25: '东乡族', 26: '纳西族', 27: '景颇族', 28: '柯尔克孜族', 29: '土族', 30: '达斡尔族', 31: '仫佬族', 32: '羌族', 33: '布朗族', 34: '撒拉族', 35: '毛难族', 36: '仡佬族', 37: '锡伯族', 38: '阿昌族', 39: '普米族', 40: '塔吉克族', 41: '怒族', 42: '乌孜别克族', 43: '俄罗斯族', 44: '鄂温克族', 45: '崩龙族', 46: '保安族', 47: '裕固族', 48: '京族', 49: '塔塔尔族', 50: '独龙族', 51: '鄂伦春族', 52: '赫哲族', 53: '门巴族', 54: '珞巴族', 55: '基诺族'}


num_labels = 31  # 身高，体重，年龄，性别, 民族, 左右利手
size = [96,160]
# image_path = "数据集+信息表+说明/身份识别任务数据集/身份识别任务数据集/训练集/00024/2023-03-20-1.jpg"
imgpath = "/Users/fariy/Desktop/MUC/数字图像处理/期末/处理好的掌纹数据/00006/02_28_r0.jpg"
pt_path = '/Users/fariy/Desktop/MUC/数字图像处理/期末/datasets/cnn_model_zhang(1).pth'

model = load_model(MultiLabelCNN,size,num_labels, pt_path)
result = predict(imgpath, model)
#print(result)

#print(f'年龄为：{sex_dict[torch.max(result[0])]} 岁')
print(f'年龄为：{result[1]} 岁')
print(f'身高为：{result[2]} cm')
print(f'体重为：{result[3]} kg')


print(f'性别为：{max_prob_index(result[0],dict=sex_dict)} 性')
print(f'民族为：{max_prob_index(result[4],dict=nation_dict)}')
print(f'左右利手为：{max_prob_index(result[5],dict=left_right)}')

