import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.decomposition import PCA


# 是否在GPU上面跑
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义超参
batch_size = 16
epochs = 300

train_directory = os.path.join('../datasets/train')
valid_directory = os.path.join('../datasets/val')

# 数据增强
image_transforms = {
    'train': transforms.Compose([
        # 将图片转换为灰度图，out_channels为3
        transforms.Grayscale(num_output_channels=3),
        # 随机长宽比裁剪
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        # 随机旋转
        transforms.RandomRotation(degrees=15),
        # 中心裁剪
        transforms.CenterCrop(size=224),
        # 转换为tensor类型
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# 读取图片
data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

# 获取data_size
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])

# dataloader
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)


# 创建网络
resnet18 = models.resnet18()
# 添加自己的全连接层
fc_x = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(fc_x, 256),
    nn.Dropout(0.5, inplace=True),
    nn.ReLU(),
    nn.Linear(256, 32),
)
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

        for batch_idx, (x, label) in enumerate(train_data):
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

            for batch_idx, (x, label) in enumerate(valid_data):
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

        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_acc / valid_data_size

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
        if epoch >= 290:
            torch.save(model, model_path + '/' + 'A_B_model_' + str(epoch + 1) + '.pth')
    return model, history, best_acc, best_epoch

"""
start_time = time.time()

model, history, best_acc, best_epoch = train_and_valid(resnet18, criteon, optimizer, epochs)

end_time = time.time()
print("total_time:", end_time-start_time)

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
plt.savefig('Palmprint' + model_path + '_accuracy_curve.png')
plt.close()
"""



# 是否在GPU上面跑
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 模型路径
pt_path = '../datasets/A_B_model_493.pth'
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='../datasets/test',transform=test_transform,)

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False,)

resnet18 = models.resnet18()
resnet18 = torch.load(pt_path,map_location=torch.device('cpu'))
model = resnet18.to(device)

with torch.no_grad():
    model.eval()
    number = 0
    total = 0
    test_data_size = len(test_loader)
    for batch_idx, (x, label) in enumerate(test_loader):
        # 在gpu跑
        x = x.to(device)
        label = label.to(device)

        # 损失
        logits = model(x)

        # 计算acc
        pred = logits.argmax(dim=1)
        correct = torch.eq(pred, label).float().sum().item()
        number += correct
        total += x.size(0)
    avg_test_acc = number / total
    print('Test Accuracy of the model on the test images: {} %'.format(avg_test_acc))