# -*- coding: utf-8 -*-

"""
 @File    : OfficalTutorial.py

 @Date    : 17-11-22

 @Author  : Dean

 @Mail    : 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# x = torch.randn(3)
# x = Variable(x, requires_grad=True)
# print(x)
#
# y = x * 2
#
# for i in range(3):
#     y = y * 2
#
# print(y)
#
# y.backward(torch.FloatTensor([0.1, 1.0, 0.0001]))  # y is a non-scalar, so torch.FloatTensor is its gradicents
# print(x.grad)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channels, 5*5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y=w*x+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 16*5*5 input's size, 120 output's size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pool over a 2*2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if window is a square, 2 is equals to (2,2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
print('\n')

params = list(net.parameters())
print(len(params))
print(params[0].size())
print('\n')

input = Variable(torch.randn(1, 1, 32, 32))  # batch_size, channel_size, img_height, img_width,
output = net.forward(input)
print(output)

#####################################################
# output = net(input)
# target = Variable(torch.arange(1, 11))
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
#
# net.zero_grad()
#
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
#
# loss.backward()
#
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
######################################################

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
ouput = net(input)
target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()
optimizer.step()

