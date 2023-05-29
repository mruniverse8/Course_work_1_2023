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

def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 80 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                ),'\n'
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    #print(train_embeddings.shape, test_embeddings.shape)
    #reduce the size of the data!
    MINTAM = min(5000, len(train_embeddings))
    train_subset_indices = torch.randperm(len(train_embeddings))[:MINTAM]
    train_embeddings = train_embeddings[train_subset_indices]
    train_labels = train_labels[train_subset_indices]
    
    MINTAM = min(5000, len(test_embeddings))
    test_subset_indices = torch.randperm(len(test_embeddings))[:MINTAM]
    test_embeddings = test_embeddings[test_subset_indices]
    test_labels = test_labels[test_subset_indices]

    #print(train_embeddings.shape, train_labels.shape)
    #print(test_embeddings.shape, test_labels.shape)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print("mAP@10 = {:.4f}".format(accuracies["mean_average_precision_at_r"]))
    print("mAP@1 = {:.4f}".format(accuracies["mean_average_precision"]))


device = torch.device("cuda")

transform = transforms.ToTensor()
batch_size = 64
learning_rate = 0.8
Train_path= open(Train_path, "r")
Val_path = open(Val_path, "r")

dataset1 = CustomDataSet.CustomDataset(Train_path)
dataset2 = CustomDataSet.CustomDataset(Val_path)

train_loader = CustomDataSet.CustomDataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = CustomDataSet.CustomDataLoader(dataset2, batch_size=batch_size, shuffle=False)

print("Almost dataloaders done!!!!")
model = resnet34().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 20


### pytorch-metric-learning stuff ###
distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)

#using default knn_func request facebook server which can't connect
knn_func = CustomKNN(SNRDistance())

accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision", "mean_average_precision_at_r"), k=1, knn_func=knn_func)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    test(dataset1, dataset2, model, accuracy_calculator)

