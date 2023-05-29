from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import CustomDataSet
import sys
import os

from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import MatchFinder
from pytorch_metric_learning.distances import SNRDistance, DotProductSimilarity
from pytorch_metric_learning.utils.inference import CustomKNN

PATH_DATA = CustomDataSet.PATH_DATA
LEN_DATA = CustomDataSet.LEN_DATA
Test_path = CustomDataSet.Test_path
Train_path = CustomDataSet.Train_path
Val_path = CustomDataSet.Val_path

import torch
import torch.nn as nn


class NormalCNN(nn.Module):
    def __init__(self, num_classes=336):
        super(NormalCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(3, 32, kernel_size=9, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        
        self.conv3 = nn.Conv1d(32, 128, kernel_size=5, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=1,padding=1)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv5 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(1024, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 256)
        self.relu8 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(256, 128)
        self.relu9 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(p=0.5) 
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool3(out)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.pool4(out)
        
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.pool5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.pool6(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu7(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu8(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.relu9(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        return out

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
        loss = loss_func(pred, labels)
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
            loss = loss_func(pred, labels)
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

model = NormalCNN().to(device)
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs + 1):
    train(model, loss_func, device, train_loader, optimizer, epoch)
    test(model, loss_func, device, test_loader)


