import torch
import torch.nn as nn


class Ensemble(nn.Module):
    def __init__(self, subnet1, subnet2, subnet3, vgg):
        super().__init__()
        self.sub1 = subnet1
        self.sub2 = subnet2
        self.sub3 = subnet3
        self.vgg = vgg

        self.final_lin = nn.Linear(4096 * 4, 7)

    def forward(self, x):
        x_1 = self.sub1(x)
        x_2 = self.sub2(x)
        x_3 = self.sub3(x)
        x_vgg = self.vgg(x)

        x_cat = torch.cat((x_1, x_2, x_3, x_vgg), dim=1)

        x = self.final_lin(x_cat)

        return x
