# -*- coding:utf-8 -*-
# @author   :fairyCaichi
# @time     :2023/06/22
# @file     :classifyHand.py
# @contact  :17866548902@163.com
import  cv2
import  mediapipe as mp
import numpy as np
import os
import sys

def clssifyHand(img,imagename,classOrcrop = 0):
    #
    # :param rootpath: 图像根路径
    # :param classOrcrop: 0 返回单手掌图像；1 返回point，进行掌心区域划分
    # :return: 保存新路径
    #
    # 生成手部对象，注意的是,在后面处理的是RGB格式图像 所以使用 hands.process()处理的图像必须是RGB格式
    myHands = mp.solutions.hands
    hands = myHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    handsdata = []
    points = []
    # 必须是RGB格式 而得到的图像默认是BGR格式所以要转

    img_R = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_R)
    # 检测所有手的列表,对列表进行访问可以获得 手的位置信息
    if (result.multi_hand_landmarks):
        for handLms in result.multi_hand_landmarks:
            # 每一个标志位都有一个id 获取并将其显示
            handsdata.append([])
            points.append([])
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                # 获取界面中的坐标 ,这里经过测试是获取的小数需要*界面获取真正坐标
                cx, cy = int(lm.x * w), int(lm.y * h)
                handsdata[len(handsdata) - 1].append([id, cx, h - cy])
                points[len(points) - 1].append([id, cx, cy])
            # 绘图
                if classOrcrop == 1:
                    cv2.putText(img, str(int(id)), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    mpDraw.draw_landmarks(img, handLms, myHands.HAND_CONNECTIONS)
            # if classOrcrop == 1:
                    cv2.imshow("Hand Landmarks", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    # handsdata = np.array(handsdata)[:, :, 1:]
    handsdata = np.array(handsdata)
    points = np.array(points)
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
    if len(final_data) == 2:
        if int(imagename.split('.')[0][-1]) == 0:
            newimage = imageleftCut(img) #切割出左手
            # return newimage
        elif int(imagename.split('.')[0][-1]) == 1:
            newimage = imagerightCut(img)  # 切割出右手
            # return newimage
            # print("成功保存右手～")
    elif len(final_data) == 1:
        # 0是手心，1是手背

        if imagename[-1] == '1' and classOrcrop == 0:
            print("图中只有手背，无法进行掌纹识别")
            return "error"
        elif imagename[-1] == '0':
            newimage = img

    else:
        print(" 未检测到手")
        return "error"
    if classOrcrop == 0:
        return newimage
    elif classOrcrop == 1:
        return points

def imageleftCut(img):
    """
    :info :切割出左手手心向上
    :param imagepath: 有两只手
    :return:
    """
    # img = cv2.imread(imagepath)
    height,width,_=img.shape
    # 计算切割点的横坐标（中间点）
    split_point = width // 2
    # 得到左侧图像
    # left_image = img.crop((0, 0, split_point, height))
    # left_image = img[, height]
    left_image = img[:, :split_point, :]

    return left_image

def imagerightCut(img):
    """
    :info :切割出右手手心向上
    :param imagepath: 有两只手
    :return:
    """
    # img = cv2.imread(imagepath)
    height,width,_ = img.shape
    # 计算切割点的横坐标（中间点）
    split_point = width // 2
    # 得到右侧图像
    # right_image = img.crop((split_point, 0, width, height))
    # right_image = img[split_point, height]
    right_image = img[:, split_point:, :]

    return right_image

def cropPalmprint(img, points):
    """
    根据给定的坐标截取图片中的区域

    :param img: 输入图像
    :param points: 要截取区域的坐标列表，每个点的坐标为 (x, y)
    :return: 截取的区域图像
    """
    # img = cv2.imread(imgpath)
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

    return region

def main():
    imgpath = "/Users/fariy/Desktop/MUC/数字图像处理/期末/身份识别任务数据集/训练集/00009/2023-03-03-1.JPG"
    img = cv2.imread(imgpath)
    imagename = imgpath.split('/')[-1].split('.')[0]
    newimg = clssifyHand(img,imagename)
    if newimg == "error":
        print("输入图像有误，重新编辑运行")
        sys.exit()
    handsdata = clssifyHand(newimg,imagename,1)

    handsdata = np.array(handsdata)[:, :, 1:]
    handsdata = np.squeeze(handsdata)
    points = [tuple(sublist) for sublist in handsdata]
    # print(points)
    # 根据索引选取点
    selected_points = [points[i] for i in [0, 1, 5, 9, 13, 17]]
    # print(selected_points)
    region = cropPalmprint(newimg, selected_points) #掌心区域
    # 显示掌心区域
    # cv2.imshow("原始图像",img)
    # cv2.waitKey(0)
    # cv2.imshow("预处理后的图像", newimg)
    # cv2.waitKey(0)
    cv2.imshow("提取的掌心区域", region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()