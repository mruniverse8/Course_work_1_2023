from random import shuffle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import CustomDataSet
import sys

from torchvision.models import resnet50
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


class SeqEmbeddingNet(nn.Module):
    def __init__(self, input_channels=3, input_length=6000, embedding_size=32):
        super(SeqEmbeddingNet, self).__init__()

        # define a fully connected layer to project the input data to a lower-dimensional space
        self.fc1 = nn.Linear(input_channels * input_length, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(128, embedding_size)

    def forward(self, x):
        # flatten the input tensor into a 1D tensor
        x = x.view(x.size(0), -1)
        # pass the input tensor through the fully connected layers with ReLU activation functions
        out = self.fc1(x)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        
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
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
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
batch_size = 32
learning_rate = 0.8
Train_path= open(Train_path, "r")
Val_path = open(Val_path, "r")

dataset1 = CustomDataSet.CustomDataset(Train_path)
dataset2 = CustomDataSet.CustomDataset(Val_path)

train_loader = CustomDataSet.CustomDataLoader(dataset1, batch_size=batch_size, shuffle=True)
test_loader = CustomDataSet.CustomDataLoader(dataset2, batch_size=batch_size, shuffle=False)

print("Almost dataloaders done!!!!")
model = SeqEmbeddingNet().to(device)
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

