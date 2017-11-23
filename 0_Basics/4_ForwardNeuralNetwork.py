# -*- coding: utf-8 -*-

"""
 @File    : 4_ForwardNeuralNetwork.py

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
input_size = 784
output_size = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
hidden_size = 100

# MNIST data set
train_data_set = dsets.MNIST(root='../data/MNIST',
                             train=True,
                             download=False,
                             transform=transforms.ToTensor())

test_data_set = dsets.MNIST(root='../data/MNIST',
                            train=False,
                            download=False,
                            transform=transforms.ToTensor())

train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                shuffle=True,
                                                batch_size=batch_size)

test_data_loader = torch.utils.data.DataLoader(test_data_set,
                                               shuffle=True,
                                               batch_size=batch_size)


# model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        o = self.fc1(x)
        o = self.relu(o)
        o = self.fc2(o)
        return o

net = Net(input_size, hidden_size, output_size)

# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_data_loader):

        target = Variable(labels)

        # f + b + o
        optimizer.zero_grad()
        pred = net.forward(Variable(images).view(-1, 784))
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print('Epoch [%d/%d], Step [%d/%d], Loss : %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data_loader), loss.data[0]))

correct = 0
total = 0
for images, labels in test_data_loader:
    outputs = net.forward(Variable(images).view(-1, 784))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy: %d %%' % (100 * correct / total))