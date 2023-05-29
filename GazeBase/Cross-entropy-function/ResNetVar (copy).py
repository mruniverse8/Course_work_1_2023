from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import CustomDataSet
import sys

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.inference import CustomKNN

PATH_DATA = CustomDataSet.PATH_DATA
LEN_DATA = CustomDataSet.LEN_DATA
Test_path = CustomDataSet.Test_path
Train_path = CustomDataSet.Train_path
Val_path = CustomDataSet.Val_path

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2d = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1d(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2d(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=336):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1d = nn.Conv1d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet34(num_classes=336):
    return ResNet(BasicBlock,  [3, 4, 6, 3], num_classes=num_classes)

def train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        pred = model(data).to(device)
        sys.stdout.flush()
        loss = loss_func(pred.float(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        _, predicted = torch.max(pred.data, 1)
        num_correct += (predicted.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()
        num_total += labels.size(0)
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Running loss = {} ".format(
                    epoch, batch_idx, loss.item(), running_loss
                )
            )
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = num_correct / num_total
    print('Epoch {} - Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch+1, epoch_loss, epoch_acc))

def test(model, loss_func, device, train_loader):
    running_loss = 0.0
    num_correct = 0
    num_total = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            pred = model(data)
            loss = loss_func(pred.float(), labels)
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(pred.data, 1)
            num_correct += (predicted.argmax(dim=-1) == labels.argmax(dim=-1)).sum().item()
            num_total += labels.size(0)
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = num_correct / num_total
    print('Test Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))

device = torch.device("cuda")
transform = transforms.ToTensor()

path_train = open(Train_path, "r")
path_val = open(Val_path, "r")

CustomDataSet.Getmemory()
dataset1 = CustomDataSet.CustomDataset(path_train)
dataset2 = CustomDataSet.CustomDataset(path_val)
CustomDataSet.Getmemory()
batch_size = 32
num_epochs = 10
learning_rate = 0.08


train_loader = CustomDataSet.CustomDataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = CustomDataSet.CustomDataLoader(dataset2, batch_size=batch_size, shuffle=False)
CustomDataSet.Getmemory()
for batch_idx, (data, labels) in enumerate(train_loader):
    print(data.shape, labels.shape)
    break

model = resnet34().to(device)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    print('\n')
    train(model, loss_func, device, train_loader, optimizer, epoch)
    test(model, loss_func, device, test_loader)


