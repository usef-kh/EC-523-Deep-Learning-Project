import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Vgg(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3a = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4a = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv4a = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv4b = nn.Conv2d(512, 512, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn3_2 = nn.BatchNorm2d(256)

        self.lin1 = nn.Linear(512 * 3 * 3, 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, 7)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.drop(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool(x)
        x = self.drop(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))

        return x


# class VggFeatures(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.convert = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
#
#         self.vgg = torchvision.models.vgg19(pretrained=True).features.eval().cuda()
#
#     def forward(self, x):
#         x = self.convert(x)
#         x = self.vgg(x)
#         x = x.view(-1, 512 * 1 * 1)
#         return x
#
#
# class Vgg(VggFeatures):
#     def __init__(self):
#         super().__init__()
#         self.lin = nn.Linear(512 * 1 * 1, 7)
#
#     def forward(self, x):
#         x = super().forward(x)
#         x = self.lin(x)
#         return x
