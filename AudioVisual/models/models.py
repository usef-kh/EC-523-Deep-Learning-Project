import torch
import torch.nn as nn

from models.ELM import ELM


class GenderFeatureExtractor(nn.Module):
    def __init__(self, num_classes, cnn2d, cnn3d):
        super().__init__()
        self.cnn2d = cnn2d
        self.cnn3d = cnn3d

        self.elm = ELM(8192, 100, num_classes)

    def forward(self, x):
        keyframes, specs = x

        x_vid = self.cnn3d(keyframes)
        x_aud = self.cnn2d(specs)

        x = torch.cat((x_aud, x_vid), dim=1)
        # x = x_aud + x_vid

        x = self.elm(x)

        return x

    def forwardToHidden(self, x):
        keyframes, specs = x

        x_vid = self.cnn3d(keyframes)
        x_aud = self.cnn2d(specs)

        x = torch.cat((x_aud, x_vid), dim=1)
        # x = x_aud + x_vid

        x = self.elm.forwardToHidden(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes, gfe):
        super().__init__()
        self.gfe = gfe

        self.elm = ELM(100, 100, num_classes)

    def forward(self, x):
        x = self.gfe.forwardToHidden(x)

        x = self.elm(x)

        return x

    def forwardToHidden(self, x):
        x = self.gfe.forwardToHidden(x)

        x = self.elm.forwardToHidden(x)

        return x
