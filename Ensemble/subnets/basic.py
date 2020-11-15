import torch.nn as nn
import torch.nn.functional as F

from utils.metric_history import History


class Subnet1Features(nn.Module):
    def __init__(self):
        super(Subnet1Features, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(256 * 6 * 6, 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.history = History()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 6 * 6)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        return x


class Subnet1(Subnet1Features):
    def __init__(self):
        super(Subnet1, self).__init__()
        self.lin3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super(Subnet1, self).forward(x)
        x = self.lin3(x)
        return x


class Subnet2Features(nn.Module):
    def __init__(self):
        super(Subnet2Features, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(256 * 6 * 6, 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.history = History()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        return x


class Subnet2(Subnet2Features):
    def __init__(self):
        super(Subnet2, self).__init__()
        self.lin3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super(Subnet2, self).forward(x)
        x = self.lin3(x)

        return x


class Subnet3Features(nn.Module):
    def __init__(self):
        super(Subnet3Features, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.lin1 = nn.Linear(256 * 6 * 6, 4096)
        self.lin2 = nn.Linear(4096, 4096)

        self.history = History()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 6 * 6)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        return x


class Subnet3(Subnet3Features):
    def __init__(self):
        super(Subnet3, self).__init__()
        self.lin3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super(Subnet3, self).forward(x)
        x = self.lin3(x)

        return x
