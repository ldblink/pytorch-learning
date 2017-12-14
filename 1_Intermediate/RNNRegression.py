# -*- coding: utf-8 -*-

"""
 @File    : RNNRegression.py

 @Date    : 17-12-14

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


TIME_STEP = 10
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYER = 2
LEARNING_RATE = 0.02
USE_LSTM = True

steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target (cos)')
plt.plot(steps, x_np, 'b-', label='input (sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYER,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, hidden):
        rnn_out, hidden = self.rnn(x, hidden)
        # THIS STEP IS INTEGRANT !!! ----------------------------------
        hidden = Variable(hidden.data)
        # -------------------------------------------------------------
        outs = []
        for t in range(rnn_out.size(1)):
            outs.append(self.fc(rnn_out[:, t, :]))
        return torch.stack(outs, dim=1), hidden


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYER,
            batch_first=True
        )
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        # THIS STEP IS INTEGRANT !!! ----------------------------------
        hidden = (Variable(hidden[0].data), Variable(hidden[1].data))
        # -------------------------------------------------------------
        outs = []
        for t in range(out.size(1)):
            outs.append(self.fc(out[:, t, :]))
        return torch.stack(outs, dim=1), hidden


if USE_LSTM:
    rnn = LSTM()
    hidden = (Variable(torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE)),
              Variable(torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE)))
else:
    rnn = RNN()
    hidden = (Variable(torch.zeros(NUM_LAYER, 1, HIDDEN_SIZE)))

optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)  # optimize all cnn parameters
criterion = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(100):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    # prediction, h_state = rnn(x, h_state)   # rnn output
    prediction, hidden = rnn(x, hidden)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    # plt.plot(steps, y_np.flatten(), 'r-')
    # plt.plot(steps, x_np.flatten(), 'g-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()