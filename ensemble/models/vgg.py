import torch.nn as nn
import torchvision


class VggFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.convert = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        self.vgg = torchvision.models.vgg19(pretrained=True).features.eval().cuda()

    def forward(self, x):
        x = self.convert(x)
        x = self.vgg(x)
        x = x.view(-1, 512 * 1 * 1)
        return x


class Vgg(VggFeatures):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(512 * 1 * 1, 7)

    def forward(self, x):
        x = super().forward(x)
        x = self.lin(x)
        return x
