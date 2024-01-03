# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/15
# @file     :imageCrop.py
# @contact  :17866548902@163.com
# @info     :从单只手掌数据集中提取掌纹区域

import cv2, imutils as im, argparse
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import math
import os
import  mediapipe as mp

def clssifyHand(imgpath):
    #
    # :param rootpath: 图像根路径
    # :return: 保存新路径
    #
    img = cv2.imread(imgpath)
    # print(imgpath)
    # 生成手部对象，注意的是,在后面处理的是RGB格式图像 所以使用 hands.process()处理的图像必须是RGB格式
    myHands = mp.solutions.hands
    hands = myHands.Hands()
    # mpDraw = mp.solutions.drawing_utils

    # handsdata = []
    handspoint = []
    # 必须是RGB格式 而得到的图像默认是BGR格式所以要转
    if img is None:
        print(imgpath)
    img_R = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_R)
    # 检测所有手的列表,对列表进行访问可以获得 手的位置信息
    if (result.multi_hand_landmarks):
        for handLms in result.multi_hand_landmarks:
            # 每一个标志位都有一个id 获取并将其显示
            # handsdata.append([])
            handspoint.append([])
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # 获取界面中的坐标 ,这里经过测试是获取的小数需要*界面获取真正坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                # handsdata[len(handsdata) - 1].append([id, cx, h - cy]) # handsdata中存储着[id,横坐标,纵坐标]
                # cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                handspoint[len(handspoint) - 1].append([id, cx, cy])
            #然后进行画图
            # mpDraw.draw_landmarks(img, handLms, myHands.HAND_CONNECTIONS)
            # cv2.imshow("Hand Landmarks", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    handspoint = np.array(handspoint)

    return handspoint

def cropPalmprint(imgpath, points):
    """
    根据给定的坐标截取图片中的区域

    :param img: 输入图像
    :param points: 要截取区域的坐标列表，每个点的坐标为 (x, y)
    :return: 截取的区域图像
    """
    img = cv2.imread(imgpath)
    # 确保至少有两个点来定义区域
    if len(points) < 2:
        raise ValueError("至少需要两点来定义区域")

    # 获取图像的尺寸
    img_height, img_width, _ = img.shape

    # 计算区域的边界
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)

    # 确保区域不超出图像边界
    left = max(0, left)
    top = max(0, top)
    right = min(img_width - 1, right)
    bottom = min(img_height - 1, bottom)

    # 提取区域图像
    region = img[top:bottom, left:right, :]
    # cv2.imshow("提取的掌心区域",region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return region


def main():
    srcpath = "../预处理掌纹数据/"
    outpath = "../处理好的掌纹数据/"
    for fatherpath in os.listdir(srcpath):
        if fatherpath == ".DS_Store":
            continue
        # elif fatherpath in ["00001","00006","00007","00008","00009","00012","00015","00023","00024","00030","00031","00022","00014","00032","00004","00003","00005","00033","00016","00029","00027","00018","00020","00021","00026","00019","00010","00017","00025","00002","00011"]:
        #     continue
        # elif fatherpath in ["00025","00002","00011","00028"]:
        #     continue
        print("开始读取", fatherpath)
        filepath = os.path.join(srcpath, fatherpath)
        for imagename in os.listdir(filepath):
            if imagename == ".DS_Store":
                continue
            image_path = os.path.join(filepath, imagename)
            handsdata = clssifyHand(image_path)
            # 处理异常块
            try:
                handsdata = np.array(handsdata)[:, :, 1:]
            except IndexError as e:
                # 处理索引错误
                print(imagename)
            except Exception as e:
                # 处理其他异常
                print("An error occurred:", e)

            handsdata = np.squeeze(handsdata)
            points = [tuple(sublist) for sublist in handsdata]
            # 根据索引选取点
            selected_points = [points[i] for i in [0, 5, 9, 13, 17]]
            # print(selected_points)
            region = cropPalmprint(image_path, selected_points)

            #保存掌纹区域
            if not os.path.exists(os.path.join(outpath,fatherpath)):
                os.makedirs(os.path.join(outpath,fatherpath))
            new = os.path.join(outpath,fatherpath,imagename)
            cv2.imwrite(new, region)

if __name__ == '__main__':
    main()

    """
    00025: 切取错误
    00002: 切取错误
    00011:
    """