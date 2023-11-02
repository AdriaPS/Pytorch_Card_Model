import torch
import torch.nn as nn  # Import the specific functions from torch related to the NN
import torch.optim as optim  # To define the optimizer layer
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms  # Torchvision will make working with image files easier
from torchvision.datasets import ImageFolder
import timm  # Library to load an architecture for image classification

import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # Pandas and NumPy for Data Exploration
import numpy as np

from PlayingCardDataset import PlayingCardDataset
from SimpleCardClassifier import SimpleCardClassifier
import utility as util

# The first thing we will do to train a PyTorch Model is to set the Dataset because that is how PyTorch will load the
# data as it trains and evaluates the model. To create this Dataset we will be creating a Python class.

dataset = PlayingCardDataset(data_dir=util.data_dir)
image, label = dataset[0]  # With this we can get both the image of the card and the class of the card from the Dataset

# This will create a dictionary associating each value with the names of the different folders

target_to_class = {v: k for k, v in ImageFolder(util.data_dir).class_to_idx.items()}

# The model needs the input to always be consistent, for that we will use the transforms library to resize all
# images to the same size.

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = PlayingCardDataset(data_dir=util.data_dir)

# The dataset is iterable, it is the main thing we will do to then wrapped in a DataLoader which will handle the
# processing of reading each of the images.
# for image, label in dataset:


# batch_size tells us how many of the examples we need to pull out of the Dataset each time we iterate through the
# DataLoader and shuffle allows us to tell the DataLoader that each time we load a new example of the Dataset if we
# want to pull it randomly or in order.
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for images, labels in dataloader:
    break

images.shape, labels.shape

# When the Dataset and DataLoader is created, we will create the PyTorch Model for the Dataset.
model = SimpleCardClassifier(num_classes=53)

# To test if the model accepts the data we want to put in it
example_out = model(images)
example_out.shape  # [batch_size, num_classes]

# After the model is created, we will train our model

