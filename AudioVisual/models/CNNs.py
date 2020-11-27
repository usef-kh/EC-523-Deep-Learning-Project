import torch.nn as nn
import torch.nn.functional as F


class CNN_2DFeatures(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=2)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=2)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # print(x.shape)
        x = F.elu(self.conv1(x))
        # print(x.shape)
        x = self.max_pool(x)
        # print(x.shape)
        x = F.elu(self.conv2(x))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.conv3(x))
        x = self.avg_pool(x)
        # print(x.shape)
        x = F.elu(self.conv4(x))
        # print(x.shape)
        x = x.view(-1, 512 * 2 * 2)
        # print(x.shape)
        x = self.drop(x)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        return x


class CNN_2D(CNN_2DFeatures):

    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super().forward(x)
        x = F.elu(self.fc3(x))

        return x


class CNN_3DFeatures(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1a = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv5a = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)

        self.max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=2)
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=2)

        self.fc1 = nn.Linear(256 * 1 * 8 * 8, 4096)
        self.fc2 = nn.Linear(4096, 4096)

        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.elu(self.conv1a(x))
        x = self.max_pool(x)

        x = F.elu(self.conv2a(x))
        x = self.avg_pool(x)

        x = F.elu(self.conv3a(x))
        x = F.elu(self.conv3b(x))
        x = self.avg_pool(x)

        x = F.elu(self.conv4a(x))
        x = F.elu(self.conv4b(x))
        x = self.avg_pool(x)

        x = F.elu(self.conv5a(x))
        x = F.elu(self.conv5b(x))
        x = self.avg_pool(x)

        x = x.view(-1, 256 * 1 * 8 * 8)

        x = self.drop(x)

        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))

        return x


class CNN_3D(CNN_3DFeatures):

    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(4096, 7)

    def forward(self, x):
        x = super().forward(x)
        x = F.elu(self.fc3(x))

        return x
