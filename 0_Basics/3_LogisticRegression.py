# -*- coding: utf-8 -*-

"""
 @File    : 3_LogisticRegression.py

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
input_size = 784
output_size = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST data set
train_date_set = dsets.MNIST(root='../MNIST',
                             train=True,
                             transform=transforms.ToTensor(),
                             download=True)
train_data_loader = torch.utils.data.DataLoader(dataset=train_date_set,
                                                shuffle=True,
                                                batch_size=batch_size)

test_date_set = dsets.MNIST(root='../MNIST',
                            train=False,
                            transform=transforms.ToTensor(),
                            download=True)
test_data_loader = torch.utils.data.DataLoader(dataset=test_date_set,
                                               shuffle=True,
                                               batch_size=batch_size)


# model
class LogisticRegression(nn.Module):

    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        output = self.linear(x)
        return output


model = LogisticRegression(input_size, output_size)

# loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# trin
for epoch in range(num_epochs):
    for i, data in enumerate(train_data_loader):
        images, labels = data
        images, labels = Variable(images).view(-1, 784), Variable(labels)

        # f + b + o
        optimizer.zero_grad()
        pred = model.forward(images)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print('Epoch [%d/%d], Step [%d/%d], Loss : %.4f'
                  % (epoch+1, num_epochs, i+1, len(train_data_loader), loss.data[0]))

correct = 0
total = 0
for images, labels in test_data_loader:
    pred = model.forward(Variable(images).view(-1, 784))
    _, predicted = torch.max(pred.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy: %d %%' % (100 * correct / total))
