# -*- coding: utf-8 -*-

"""
 @File    : ConvolutionalNeuralNetwork.py

 @Date    : 17-11-22

 @Author  : Dingbang Li (Dean)

 @Mail    : 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# hyper params
num_epochs = 5
batch_size = 100
learning_rate = 0.01

train_dataset = dsets.MNIST(root='../data/MNIST',
                            train=True,
                            transform=transforms.ToTensor,
                            download=False)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True,
                                           batch_size=batch_size,
                                           num_workers=2)

test_dataset = dsets.MNIST(root='../data/MNIST',
                           train=False,
                           transform=transforms.ToTensor,
                           download=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=True,
                                          batch_size=batch_size,
                                          num_workers=2)


# model 2 conv layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # A sequential container, Modules will be added to it in the order they are passed in the constructor.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # kernel size is 2*2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self):
        pass