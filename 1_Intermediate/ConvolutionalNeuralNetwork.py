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
