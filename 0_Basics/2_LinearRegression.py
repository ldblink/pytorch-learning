# -*- coding: utf-8 -*-

"""
 @File    : LinearRegression.py

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
import matplotlib.pyplot as plt


# hyper params
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy Dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.liner = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.liner(x)
        return out

model = LinearRegression(input_size, output_size)

# loss
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# train
for epoch in range(num_epochs):
    inputs = Variable(torch.from_numpy(x_train))
    targets = Variable(torch.from_numpy(y_train))

    # forward + backward + optim
    optimizer.zero_grad()
    pred = model.forward(inputs)
    loss = criterion(pred, targets)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 4:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, num_epochs, loss.data[0]))

predicted = model.forward(Variable(torch.from_numpy(x_train)))

plt.plot(x_train, y_train, 'ro', label='Origin data')
plt.plot(x_train, predicted.data.numpy(), label='Fitted line')
plt.legend()
plt.show()

