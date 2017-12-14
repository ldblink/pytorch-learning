# -*- coding: utf-8 -*-

"""
 @File    : ConvolutionalNeuralNetwork.py

 @Date    : 17-11-22

 @Author  : Dean

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
                            transform=transforms.ToTensor(),
                            download=False)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           shuffle=True,
                                           batch_size=batch_size)

test_dataset = dsets.MNIST(root='../data/MNIST',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=False)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          shuffle=True,
                                          batch_size=batch_size)


# model 2 convolution layer
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
            # But some experiments show that the performance will be better putting the BN layer after the Activation.
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)  # 28*28 -> 14*14 -> 7*7

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

cnn = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)

        # f + b + o
        optimizer.zero_grad()
        labels_pred = cnn.forward(images).view(labels.size(0), -1)
        loss = criterion(labels_pred, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print('Epoch [%d/%d], Step [%d/%d], Loss : %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0]))

# test the model
cnn.eval()  # evaluation mode, only effect Dropout/BatchNorm.
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    labels = Variable(labels)
    output = cnn.forward(images)
    _, labels_pred = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (labels_pred == labels.data).sum()

print('Accuracy: %d %%' % (100 * correct / total))
