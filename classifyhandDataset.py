# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/15
# @file     :classify_hand.py
# @contact  :17866548902@163.com
# @info     :识别图中有几只手，形成单只手掌的数据集

import  cv2
import math
import  mediapipe as mp
import time
import numpy as np
import os


def clssifyHand(rootpath,newpath,isSave = False):
    # 
    # :param rootpath: 图像根路径
    # :return: 保存新路径
    #
    for fatherpath in os.listdir(rootpath):
        # readlist是第一次运行没有保存下来的文件夹
        # readlist = ["00010","00011","00016","00017","00018","00019","00020","00021","00026","00027","00028","00029","00033"]
        if fatherpath == ".DS_Store":
            continue
        # elif fatherpath not in readlist:
        #     continue
        print("开始读取",fatherpath)
        filepath = os.path.join(rootpath, fatherpath)
        for imagename in os.listdir(filepath):
            imgpath = os.path.join(filepath,imagename)
            img = cv2.imread(imgpath)
            # print(imgpath)
            # 生成手部对象，注意的是,在后面处理的是RGB格式图像 所以使用 hands.process()处理的图像必须是RGB格式
            myHands = mp.solutions.hands
            hands = myHands.Hands()
            mpDraw = mp.solutions.drawing_utils

            handsdata = []
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
                    handsdata.append([])
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        # 获取界面中的坐标 ,这里经过测试是获取的小数需要*界面获取真正坐标
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        handsdata[len(handsdata) - 1].append([id, cx, h - cy])
                        handspoint[len(handsdata) - 1].append([id, cx, cy])
            # handsdata = np.array(handsdata)[:, :, 1:]
            handsdata = np.array(handsdata)
            handspoint = np.array(handspoint)
            final_data = []
            for hand in handsdata:
                final_data.append([])
                for i in [0, 4, 8, 12, 16]:  # 计算关节间的距离，一共15个
                    for j in [1, 2, 3]:
                        final_data[len(final_data) - 1].append(np.linalg.norm(hand[j + i + 1] - hand[j + i], axis=0))
                for i in [1, 5, 9, 13]:  # 计算手指之间的夹角，一共4个
                    final_data[len(final_data) - 1].append(np.dot(hand[i + 4] - hand[0], hand[i] - hand[0]) / (
                                np.linalg.norm(hand[i + 4] - hand[0]) * np.linalg.norm(hand[i] - hand[0])))
            # print(final_data)
            if not os.path.exists(os.path.join(newpath,fatherpath)):
                os.makedirs(os.path.join(newpath,fatherpath))
            new = os.path.join(newpath,fatherpath,imagename)
            # print(new)
            if isSave:
                if len(final_data) == 2:
                    if int(imagename.split('.')[0][-1]) == 0:
                        newimage = imageleftCut(imgpath) #切割出左手
                        cv2.imwrite(new,newimage)
                        if newimage is None:
                            print(new,"保存失败")
                        else:
                            pass
                    elif int(imagename.split('.')[0][-1]) == 1:
                        newimage = imagerightCut(imgpath)  # 切割出右手
                        cv2.imwrite(new,newimage)
                        # print("成功保存右手～")
                elif len(final_data) == 1:
                    # 0是手心，1是手背
                    if int(imagename.split('.')[0][-1]) == 0:
                        cv2.imwrite(new,img) #直接保存原来的图像
                else:
                    print(new," 未检测到手")

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
    cv2.imshow("提取的掌心区域",region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return region

def imageleftCut(imagepath):
    """
    :info :切割出左手手心向上
    :param imagepath: 有两只手
    :return:
    """
    img = cv2.imread(imagepath)
    height,width,_=img.shape
    # 计算切割点的横坐标（中间点）
    split_point = width // 2
    # 得到左侧图像
    # left_image = img.crop((0, 0, split_point, height))
    # left_image = img[, height]
    left_image = img[:, :split_point, :]

    return left_image


def imagerightCut(imagepath):
    """
    :info :切割出右手手心向上
    :param imagepath: 有两只手
    :return:
    """
    img = cv2.imread(imagepath)
    height,width,_ = img.shape
    # 计算切割点的横坐标（中间点）
    split_point = width // 2
    # 得到右侧图像
    # right_image = img.crop((split_point, 0, width, height))
    # right_image = img[split_point, height]
    right_image = img[:, split_point:, :]

    return right_image

def main():
    testpath = "../身份识别任务数据集/训练集/"
    outpath = "../预处理数据/"
    clssifyHand(testpath,outpath)

if __name__ == '__main__':
    main()