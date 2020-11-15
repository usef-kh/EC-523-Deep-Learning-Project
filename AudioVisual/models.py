import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.svm import SVC

class SVM:
    def __init__(self):
        self.svm = SVC(kernel='rbf', C=1)

    def fit(self, X, y):
        self.svm.fit(X, y)


class ELM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super(ELM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes # 2 for first ELM, 5 for second
        self.device = device

        # just two layers; declare parameters: (maybe use xavier_uniform inits?)
        self.alpha = nn.init.uniform_(torch.empty(self.input_size, self.hidden_size, device=self.device), a=-1., b=1.)
        self.beta = nn.init.uniform_(torch.empty(self.hidden_size, self.output_size, device=self.device), a=-1., b=1.)

        self.bias = torch.zeros(self.hidden_size, device=self.device)

        self.activation = torch.nn.functional.gelu() # other activations? # they said 'gaussian kernel'

    def forward(self, x):
        h = self.activation(torch.add(x.mm(self.alpha), self.bias)) # forward used for training
        return h.mm(self.beta)

    def forwardToHidden(self, x):  # the output of this is what we feed to the next ELM AFTER THIS ONE IS TRAINED
        return self.activation(torch.add(x.mm(self.alpha), self.bias))


class CNN_2D(nn.Module):

    def __init__(self):
        super(CNN_2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(?, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 7)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.drop(x)

        x = F.elu(self.conv1(x))
        x = self.max_pool(x)

        x = F.elu(self.conv2(x))
        x = self.avg_pool(x)

        x = F.elu(self.conv3(x))
        x = self.avg_pool(x)

        x = F.elu(self.conv4(x))
        x = x.view(-1, ?)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        x = F.elu(self.fc3(x))

        return x