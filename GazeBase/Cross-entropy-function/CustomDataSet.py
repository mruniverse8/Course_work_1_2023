
#from operator import length_hint
import os
#from pandas.core.arrays.datetimelike import dtype_to_unit
import torch
import pandas as pd
#import numpy as np
import psutil
import random
#from torch.autograd.grad_mode import F
from torch.utils.data import Dataset, DataLoader

PATH_DATA = './MERGED_SPECIFIC_DATA'
DEVICE = "cuda"
device = "cuda"
LEN_DATA = 6000
Test_path = "./test_data.txt"
Train_path = "./train_data.txt"
Val_path = "./val_data.txt"
num_clases = 336

def Getmemory():
    pid = psutil.Process()
    memory_info = pid.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
class CustomDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.transform = transform
        self.data = []
        self.labels = []
        # Read and preprocess each CSV file in the data folder
        self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, label
    def get_labels(self, line):
        one_hot = [0]*num_clases
        pid = int(line[3:6], base=10)
        one_hot[pid] = 1
        return  one_hot
        return pid

    def fill_panda(self, data):
        colx = data.columns
        while len(data) < LEN_DATA:
            new_row = pd.Series([None] * len(data.columns), index=data.columns)
            data = pd.concat([data, pd.DataFrame([new_row], columns=data.columns)], ignore_index=True)
        if len(data) > LEN_DATA:
            start = random.randint(0,len(data) - LEN_DATA)
            data = data.iloc[start: start+LEN_DATA]
        return data

    def read_data(self):
        # Iterate over the CSV files in the data folder
        self.data = []
        self.labels = []
        c = 0
        print(self.data_folder)
        for filename in self.data_folder:
            c += 1
            if (c > 528):
                break
            filename = filename[:-1]
            if filename.endswith('.csv'):
                filepath = os.path.join(PATH_DATA, filename)
                data = pd.read_csv(filepath, usecols=lambda col: col != "n")
                #data = pd.read_csv(filepath, index_col=False)
                #data = data.drop(columns=["n"])
                # print(data.columns, len(data))
                if (len(data) < LEN_DATA):
                    del data
                    continue
                data = self.fill_panda(data)
                data = data.fillna(0)
                data_values = torch.tensor(data.to_numpy(), dtype=torch.float32)
                if (c == 1):
                    print(data.columns, len(data))
                del data
                label_values = self.get_labels(filename)
                # print(filepath, label_values)
                self.data.append(data_values)
                self.labels.append(label_values)
            else:
                assert(False)
        print("data loaded!")
        # Convert the data and labels to tensors
        self.data = torch.stack(self.data)
        self.labels = torch.as_tensor(self.labels, dtype=torch.float32)

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)

    def collate_fn(self, batch):
        samples = torch.stack([x[0] for x in batch])
        labels = torch.tensor([x[1] for x in batch])
        return samples, labels

def CheckMem():
    if torch.cuda.is_available():
        # Get the maximum amount of CUDA memory available
        max_memory_bytes = torch.cuda.max_memory_allocated()
        max_memory_gb = round(max_memory_bytes / (1024 ** 3), 2)
        print(f"Maximum CUDA memory available: {max_memory_gb} GB")
    else:
        print("CUDA is not available")
# path = './test_data.txt'
# path = open(path, "r")
# dataset = CustomDataset(path)
# dataloader = CustomDataLoader(dataset, batch_size=32, shuffle=True)
# for batch_idx, (data, labels) in enumerate(dataloader):
#     print(batch_idx, data.shape, labels.shape)
