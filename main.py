import os, sys, glob, shutil, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #申明环境变量
import cv2

from PIL import Image
import numpy as np

import torch
torch.manual_seed(0)  # 为cpu设置种子用于生成随机数
torch.backends.cudnn.deterministic = False  # cuda的随机数种子不固定
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset


use_cuda = True  # 使用GPU


class LabelSmoothing(nn.Module):
    """
    平滑标签
    """
    def __init__(self, smoothing=0.0):

        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        返回平滑标签计算的loss
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SVHN_Model1(nn.Module):
    """
    模型构建
    """
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        model_conv = models.resnet18(pretrained=True)  # 使用resnet18
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        return c1, c2, c3, c4  # 由于5、6个字符所占比例不大，只考虑四个字符的定长字符串


class SVHNDataset(Dataset):  # 自定义数据集
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (4 - len(lbl)) * [10]  # 将每个picture的label统一长度
        return img, torch.from_numpy(np.array(lbl[:4]))

    def __len__(self):
        return len(self.img_path)


def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()  # 启用BN层与DP层
    train_loss = []

    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
        target = target.long()
        c0, c1, c2, c3 = model(input)
        loss = criterion(c0, target[:, 0]) + \
               criterion(c1, target[:, 1]) + \
               criterion(c2, target[:, 2]) + \
               criterion(c3, target[:, 3])
               # criterion(c4, target[:, 4])

        optimizer.zero_grad()  # 清除梯度
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    return np.mean(train_loss)


def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 预测模型不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.long()
            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            c0, c1, c2, c3= model(input)
            loss = criterion(c0, target[:, 0]) + \
                   criterion(c1, target[:, 1]) + \
                   criterion(c2, target[:, 2]) + \
                   criterion(c3, target[:, 3])
                   # criterion(c4, target[:, 4])
            val_loss.append(loss.item())
    return np.mean(val_loss)


def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None

    # 数据增强次数
    for _ in range(tta):
        test_pred = []

        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()

                c0, c1, c2, c3= model(input)
                if use_cuda:
                    output = np.concatenate([
                        c0.data.cpu().numpy(),
                        c1.data.cpu().numpy(),
                        c2.data.cpu().numpy(),
                        c3.data.cpu().numpy()
                        ], axis=1)
                else:
                    output = np.concatenate([
                        c0.data.numpy(),
                        c1.data.numpy(),
                        c2.data.numpy(),
                        c3.data.numpy()
                        ], axis=1)

                test_pred.append(output)

        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred

    return test_pred_tta

def main():
    train_path = glob.glob('input/train/*.png')
    train_path.sort()  # 排序路径
    train_json = json.load(open('input/train.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    print(len(train_path), len(train_label))

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),  # 统一大小
                        transforms.RandomCrop((60, 120)),  # 随机裁剪
                        transforms.ColorJitter(0.3, 0.3, 0.2),  # 改变亮度
                        # transforms.RandomRotation(10),  # 随机旋转
                        transforms.ToTensor(),  # 转为tensor
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  #归一化
                    ])),
        batch_size = 40,
        shuffle=True,
        num_workers=10,
    )

    val_path = glob.glob('input/val/*.png')
    val_path.sort()
    val_json = json.load(open('input/val.json'))
    val_label = [val_json[x]['label'] for x in val_json]
    print(len(val_path), len(val_label))

    val_loader = torch.utils.data.DataLoader(
        SVHNDataset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )


    model = SVHN_Model1()
    criterion = LabelSmoothing()
    optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9)
    best_loss = 1000.0

    # 是否使用GPU
    if use_cuda:
        model = model.cuda()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    for epoch in range(10):
        continue

        train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion)
        scheduler.step()

        val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
        val_predict_label = predict(val_loader, model, 1)
        val_predict_label = np.vstack([
            val_predict_label[:, :11].argmax(1),
            val_predict_label[:, 11:22].argmax(1),
            val_predict_label[:, 22:33].argmax(1),
            val_predict_label[:, 33:44].argmax(1)
        ]).T
        val_label_pred = []
        for x in val_predict_label:
            val_label_pred.append(''.join(map(str, x[x != 10])))

        val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))

        print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
        print('Val Acc', val_char_acc)
        # 记录下验证集精度
        if val_loss < best_loss:
            best_loss = val_loss
            # print('Find better model in Epoch {0}, saving model.'.format(epoch))
            torch.save(model.state_dict(), './model.pt')

    test_path = glob.glob('input/test_a/*.png')
    test_path.sort()
    # test_json = json.load(open('input/test_a.json'))
    test_label = [[1]] * len(test_path)
    print(len(test_path), len(test_label))

    test_loader = torch.utils.data.DataLoader(
        SVHNDataset(test_path, test_label,
                    transforms.Compose([
                        transforms.Resize((70, 140)),
                        # transforms.RandomCrop((60, 120)),
                        # transforms.ColorJitter(0.3, 0.3, 0.2),
                        # transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=40,
        shuffle=False,
        num_workers=10,
    )

    # 加载保存的最优模型
    model.load_state_dict(torch.load('model.pt'))

    test_predict_label = predict(test_loader, model, 1)
    print(test_predict_label.shape)

    test_label = [''.join(map(str, x)) for x in test_loader.dataset.img_label]
    test_predict_label = np.vstack([
        test_predict_label[:, :11].argmax(1),
        test_predict_label[:, 11:22].argmax(1),
        test_predict_label[:, 22:33].argmax(1),
        test_predict_label[:, 33:44].argmax(1),
        test_predict_label[:, 44:55].argmax(1),
    ]).T

    test_label_pred = []
    for x in test_predict_label:
        test_label_pred.append(''.join(map(str, x[x != 10])))

    import pandas as pd
    df_submit = pd.read_csv('input/test_A_sample_submit.csv')
    df_submit['file_code'] = test_label_pred
    df_submit.to_csv('submit.csv', index=None)


if __name__ == '__main__':
    main()











