from pandas import array
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import io
import pandas as pd

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

#train_dataloader = DataLoader(training_data, batch_size=64)
#test_dataloader = DataLoader(test_data, batch_size=64)
i = 0
images = []
labels = []
split = []
is_query = []
is_gallery = []
for (img, label) in training_data:
    i += 1
    image = img[0].numpy()
    #plt.imshow(image, cmap='gray')
    #if i == 1:
    #   plt.show()
    name_of_image = f"./minsti/image_{i} "
    name_of_image = name_of_image + ".jpg"
    images += [name_of_image]
    labels += [label]
    split += ["train"]
    is_query += [None]
    is_gallery += [None]
    #plt.imsave(name_of_image, image, cmap='gray')

for (img, label) in test_data:
    i += 1
    image = img[0].numpy()
    #plt.imshow(image, cmap='gray')
#    if i == 1:
       #plt.show()
    name_of_image = f"./minsti/val_image_{i} "
    name_of_image = name_of_image + ".jpg"
    images += [name_of_image]
    labels += [label]
    split += ["validation"]
    is_query += [1]
    is_gallery += [0]
  #  plt.imsave(name_of_image, image, cmap='gray')

images = pd.DataFrame({'path': images})
labels = pd.DataFrame({'label': labels})
split= pd.DataFrame({'split': split})
is_query = pd.DataFrame({'is_query': is_query})
is_gallery = pd.DataFrame({'is_gallery': is_gallery})
df = pd.concat([images, labels, split, is_query, is_gallery], axis=1)
df.to_csv('minst_dataframe.csv', index=False)
