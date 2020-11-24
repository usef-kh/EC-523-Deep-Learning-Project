import torch
import torch.nn as nn
import torchvision

class Vgg(nn.Module):
    def __init__(self):
        super().__init__()
        self.convert = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.vgg = torchvision.models.vgg19(pretrained=True).features
        self.lin1 = nn.Linear(512 * 1 * 1, 7)

    def forward(self, x):
        x = self.convert(x)

        x = self.vgg(x)
        # print(x.shape)
        x = x.view(-1, 512 * 1 * 1)
        x = self.lin1(x)

        return x