# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/22
# @file     :modelPredict.py
# @contact  :17866548902@163.com
# @info     :输入图片路径,身份识别，并求得置信度

import classifyHand
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import sys
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        # self.modle = resnet18()
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


def recognizeID(image):
    """
    :info :识别手掌id
    :param image:
    :return:
    """
    # 是否在GPU上面跑
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 模型路径
    pth_path = '../datasets/A_B_model_493.pth'
    model = models.resnet18() #pretrained=True
    model = torch.load(pth_path,map_location=torch.device('cpu'))
    #
    # # 加载模型参数
    # state_dict = torch.load(pth_path, map_location=device)
    # model.load_state_dict(state_dict)
    # model = model.to(device)

    # 定义图像的预处理转换
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 对图像进行预处理
    input_pil = Image.fromarray(image)
    input_tensor = preprocess(input_pil)
    # 添加批次维度
    input_batch = input_tensor.unsqueeze(0)
    # 加载预训练的ResNet-18模型
    model.eval()  # 设置为评估模式

    # 运行图像通过ResNet-18模型
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # 标签列表
    # labels = sorted(os.listdir("../预处理掌纹数据/"))
    labels = ['00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009', '00010', '00011', '00012', '00014', '00015', '00016', '00017', '00018', '00019', '00020', '00021', '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029', '00030', '00031', '00032', '00033']
    # print(labels)

    # 获取结果的置信度
    # print(output)
    # print(probabilities)
    pred = probabilities.argmax().item()
    confidence_score = torch.max(probabilities).item()
    predicted_class = labels[pred]
    # print("置信度:", confidence_score)
    return predicted_class,confidence_score,probabilities


def recognizeAttribute(image):
    """
    :param image:
    :return:
    :info:通过手掌图像来识别身高、体重、民族、年龄、性别
    """
    # # 是否在GPU上面跑
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    pt_path = '../datasets/age_gender_model_36.pth'
    model = models.resnet18()
    model= torch.load(pt_path, map_location=torch.device('cpu'))

    transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 对图像进行预处理
    input_pil = Image.fromarray(image)
    input_tensor = transform(input_pil)
    # 添加批次维度
    input_batch = input_tensor.unsqueeze(0)
    # 加载预训练的ResNet-18模型
    model.eval()  # 设置为评估模式

    # 运行图像通过ResNet-18模型
    with torch.no_grad():
        output1,output2,output3,output4,output5 = model(input_batch)
        # print(output) #输出五元组
        prob1 = torch.nn.functional.softmax(output1, dim=1)
        prob2 = torch.nn.functional.softmax(output2, dim=1)
        prob3 = torch.nn.functional.softmax(output3, dim=1)
        prob4 = torch.nn.functional.softmax(output4, dim=1)
        prob5 = torch.nn.functional.softmax(output5, dim=1)
        # print(prob1,prob2,prob3,prob4,prob5)

    #真实标签
    label1 = {0: "男", 1: "女"} #性别
    label2 = {0: 22, 1: 23, 2: 24, 3: 25, 4: 26, 5: 27, 6: 29} #年龄
    label3 = {0: "汉族", 1: "蒙古族", 2: "回族", 3: "藏族", 4: "彝族", 5: "土家族"} #民族
    label4 = {0: 160, 1: 163, 2: 164, 3: 165, 4: 166, 5: 168, 6: 169, 7: 170, 8: 171, 9: 172, 10: 175, 11: 177, 12: 178, 13: 180,14: 182, 15: 185} #身高
    label5 = {0: 42, 1: 45, 2: 52, 3: 54, 4: 55, 5: 56, 6: 58, 7: 60, 8: 62, 9: 65, 10: 68, 11: 70, 12: 72, 13: 73, 14: 76,15: 79, 16: 83, 17: 88, 18: 95, 19: 100} #体重
    #将预测结合和真实标签对应
    pred1 = prob1.argmax().item()
    conf1 = torch.max(prob1).item()
    gender = label1[pred1]

    pred2 = prob2.argmax().item()
    conf2 = torch.max(prob2).item()
    age = label2[pred2]

    pred3 = prob3.argmax().item()
    conf3 = torch.max(prob3).item()
    nation = label3[pred3]

    pred4 = prob4.argmax().item()
    conf4 = torch.max(prob4).item()
    height = label4[pred4]

    pred5 = prob5.argmax().item()
    conf5 = torch.max(prob5).item()
    weight = label5[pred5]
    # print(pred1,pred2,pred3,pred4,pred5)
    return [[gender,prob1],[age,prob2],[nation,prob3],[height,prob4],[weight,prob5]]

def main():
    imgpath = "/Users/fariy/Desktop/MUC/数字图像处理/期末/身份识别任务数据集/训练集/00029/2023-3-13-0.jpg"
    img = cv2.imread(imgpath)
    imagename = imgpath.split('/')[-1].split('.')[0]
    newimg = classifyHand.clssifyHand(img,imagename)
    if newimg == "error":
        print("输入图像有误，重新编辑运行")
        sys.exit()
    handsdata = classifyHand.clssifyHand(newimg,imagename,1)

    handsdata = np.squeeze(np.array(handsdata)[:, :, 1:])
    # handsdata = np.squeeze(handsdata)
    points = [tuple(sublist) for sublist in handsdata]
    # print(points)
    # 根据索引选取点
    selected_points = [points[i] for i in [0, 1, 5, 9, 13, 17]]
    # print(selected_points)
    region = classifyHand.cropPalmprint(newimg, selected_points) #掌心区域
    labelPre,conf,probabilities= recognizeID(region)
    print("该手掌的身份id：",labelPre)
    print("置信度：{:.4f}".format(conf))
    print("每个类别的置信度集合：",probabilities)

    attribute = recognizeAttribute(region)
    # print(f"性别 : {attribute[0]}, 年龄 : {attribute[1]}, 民族 : {attribute[2]}, 身高 : {attribute[3]} , 体重 : {attribute[4]}")
    print(f"性别：{attribute[0]}")
    print(f"年龄：{attribute[1]}")
    print(f"民族：{attribute[2]}")
    print(f"身高：{attribute[3]}")
    print(f"体重：{attribute[4]}")
if __name__ == '__main__':
    main()

