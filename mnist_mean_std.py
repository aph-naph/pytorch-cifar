'''Train MNIST with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from activations_autofn import *
from data_utils import get_data_loaders
from utils import progress_bar

from datetime import datetime

import logging
import random # only used for seeding
import numpy as np # only used for seeding

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print(len(dataloader))
    mean = 0
    std = 0
    print('==> Computing mean and std..')
    for i, (inputs, targets) in enumerate(dataloader):
        mean += inputs.mean()
        std += inputs.std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

batch_size = 128
val_split = 0.1
num_workers = 8

train_ds = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())
mean, std = get_mean_and_std(train_ds)
print(f"Mean: {mean}, Std: {std}")