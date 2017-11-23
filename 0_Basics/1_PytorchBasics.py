# -*- coding: utf-8 -*-

"""
 @File    : PytorchBasics.py

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


# basic autograd example 1 ==================================
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

y = w * x + b  # y = 2 * x + 3

# compute gradients
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)


# basic autograd example 2 ===================================
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# linear layer
linear = nn.Linear(3, 2)
print('w: ', linear.weight)
print('b: ', linear.bias)

# loss & optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(linear.parameters(), lr=0.01)

# foreward
pred = linear(x)

# compute loss
loss = criterion(pred, y)
print('loss before SGD: ', loss.data[0])
print('\n')

# backward
loss.backward()

print('dL/dw: ', linear.weight.grad)
print('dL/db: ', linear.bias.grad)

# update parms
optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
print('loss after SGD: ', loss.data[0])


# bridge to numpy ===================================
a = np.array([[1,2], [3,4]])
b = torch.from_numpy(a)
c = b.numpy()


# implement the inpur pipline =======================
train_date_set = dsets.CIFAR10(root='../CIFAR10',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_date_set,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)

for images, labels in train_loader:
    # TODO
    # training code
    pass


# input pipline for custom data set =================================
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # Initialize file path or list of file names.
        pass

    def __getitem__(self, item):
        # TODO
        # Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # Preprocess the data (e.g. torchvision.Transform).
        # Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        length = 0
        # TODO
        # Change length to the total size of the dataset
        return length

custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)


# using pre-trained model ===============================================
# download and load pre-trained renet
resnet = torchvision.models.resnet18(pretrained=True)



# save & load model =====================================================
# entire model
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# only params
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))



