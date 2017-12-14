# -*- coding: utf-8 -*-

"""
 @File    : RecurrentNeuralNetwork.py

 @Date    : 17-11-28

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
sequence_length = 28
input_size = 28
hidden_size = 128
num_layer = 2
num_class = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST data
train_set = dsets.MNIST(root='../data/MNIST',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=False)

train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

test_set = dsets.MNIST(root='../data/MNIST',
                       train=False,
                       transform=transforms.ToTensor(),
                       download=False)

test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size,
                                          shuffle=True)


# RNN
class RNN(nn.Module):
    """
    LSTM
    x: outside data
    W: weight
    b: bias
    c: cell state
    h: hidden layer data
    ---------------------------------------------
    forget gate:
        output 0~1 scalar(f_t) to decide whether to forget->0/remain->1

        f_t = sigmoid(W_f . [h_(t-1), x_t] + b_f)

        or

        f_t = sigmoid(W_if . x_t + b_if + W_hf . h_(t-1) + b_hf)

    input gate:
        output 0~1 scalar(i_t) to decide whether to forget->0/remain->1

        i_t = sigmoid(W_i . [h_(t-1), x_t] + b_i)

        or

        i_t = sigmoid(W_ii . x_t + b_ii + W_hi . h_(t-1) + b_hi)

    candidate cell state:
        this state vector can be added into cell state

        g_t = tanh(W_g . [h_(t-1), x_t] + b_g)

        or

        g_t = tanh(W_ig . x_t + b_ig + W_hc . h_(t-1) + b_hg)

    update cell state:
        forget pre-state, remain new stateNone

        c_t = f_t * c_(t-1) + i_t * g_t

    output gate:
        decide which need to be output

        o_t = sigmoid(W_io . x_t + b_io + W_ho . h_(t-1) + b_ho)

        according the cell state decide the final output

        h_t = o_t * tanh(c_t)

    PyTorch API: see the doc
    """
    def __init__(self, input_size, hidden_size, num_layer, num_class):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # the num of features in the hidden state h
        self.num_layer = num_layer  # num of recurrent layers
        # if batch_first=True, input/output tensors are provided as (batch, sequence, feature)
        # else provided as (sequence, batch, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x, hidden):
        # forward
        # output, (h_n, c_n) = lstm(x, (h0, c0))
        # output(seq_len, batch, hidden_size*num_directions)/(batch, seq_len, hidden_size*num_directions)
        # (h_n, c_n) hidden states
        out, hidden = self.lstm(x, hidden)

        # THIS STEP IS INTEGRANT !!! ----------------------------------
        hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
        # -------------------------------------------------------------

        # only take the last batch, meaning the last output is the predicted result
        out = self.fc(out[:, -1, :])    # (batch, seq_len = -1, hidden_size*num_directions)
        return out, hidden


rnn = RNN(input_size, hidden_size, num_layer, num_class)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

hidden = (Variable(torch.randn(num_layer, batch_size, hidden_size)),
          Variable(torch.randn(num_layer, batch_size, hidden_size)))

# train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, sequence_length, input_size))
        labels = Variable(labels)

        # f + b + o
        optimizer.zero_grad()
        outputs, hidden = rnn.forward(images, hidden)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 99:
            print('Epoch [%d/%d], Step [%d/%d], Loss : %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_loader), loss.data[0]))

# test the model
rnn.eval()
correct = 0.0
total = 0.0
for images, labels in test_loader:
    images = Variable(images.view(-1, sequence_length, input_size))
    labels = Variable(labels)
    output, hidden = rnn.forward(images, hidden)
    _, labels_pred = torch.max(output.data, 1)
    total += labels.size(0)
    correct += (labels_pred == labels.data).sum()

print('Accuracy: %d %%' % (100 * correct / total))
