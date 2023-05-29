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
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = ".format(
                    epoch, batch_idx, loss
                ),'\n'
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    print(tester.data_device," GERE")
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    #print(train_embeddings.shape, test_embeddings.shape)
    #reduce the size of the data!
    MINTAM = min(800, len(train_embeddings))
    train_subset_indices = torch.randperm(len(train_embeddings))[:MINTAM]
    train_embeddings = train_embeddings[train_subset_indices]
    train_labels = train_labels[train_subset_indices]
    
    MINTAM = min(800, len(test_embeddings))
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
#    print("mAP@10 = {:.4f}".format(accuracies["mean_average_precision_at_r"]))
    print("mAP@1 = {:.4f}".format(accuracies["mean_average_precision"]))


last_model_path = './checkpoints/model_weights1'
save_path = './checkpoints/model_weights0.pth'
cond_save=False
if sys.argv[1] == "save":
    cond_save = True
    print("We are going to save the weights")

device = torch.device("cuda")

transform = transforms.ToTensor()
batch_size = 32
learning_rate = 0.8
Train_path= open(Train_path, "r")
Val_path = open(Val_path, "r")

dataset1 = CustomDataSet.CustomDataset(Train_path)
dataset2 = CustomDataSet.CustomDataset(Val_path)

train_loader = CustomDataSet.CustomDataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = CustomDataSet.CustomDataLoader(dataset2, batch_size=batch_size, shuffle=False)

print("Almost dataloaders done!!!!")
model = NormalCNN().to(device)
if os.path.exists(last_model_path):
    print("last model exist and we take it?")
    model = torch.load(last_model_path).to(device)
else:
    print("Training for scratch no weights loaded")
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
num_epochs = 128


### pytorch-metric-learning stuff ###
distance = distances.SNRDistance()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.8, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.8, distance=distance, type_of_triplets="hard"
)
#using default knn_func request facebook server which can't connect
knn_func = CustomKNN(distances.SNRDistance())

accuracy_calculator = AccuracyCalculator(include=("precision_at_1", "mean_average_precision"), k=1,knn_func=knn_func)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    test(dataset1, dataset2, model, accuracy_calculator)
if cond_save:
    torch.save(model.cpu(), save_path)
    print("Saved")
else:
    print("Not Saved")
