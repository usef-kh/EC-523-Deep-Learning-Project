import torch
import torch.nn as nn


class ELM1Features(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cnn2d, cnn3d):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes  # 2 for first ELM, 5 for second

        self.cnn2d = cnn2d
        self.cnn3d = cnn3d
        self.alpha = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.alpha(x))  # forward used for training

        return h

    def passThroughPrev(self, keyframes, specs):
        x_vid = self.cnn3d(keyframes)
        x_aud = self.cnn2d(specs)
        x = torch.cat((x_aud, x_vid), dim=1)

        return x


class ELM1(ELM1Features):
    def __init__(self, input_size, hidden_size, num_classes, cnn2d, cnn3d):
        super().__init__(input_size, hidden_size, num_classes, cnn2d, cnn3d)

        self.beta = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        x = super().forward(x)
        x = self.beta(x)

        return x

    def forwardToHidden(self, x):  # the output of this is what we feed to the next ELM AFTER THIS ONE IS TRAINED
        return super().forward(x)


class ELM2Features(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cnn2d, cnn3d, elm1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes  # 2 for first ELM, 6 for second
        self.cnn2d = cnn2d
        self.cnn3d = cnn3d
        self.elm1 = elm1
        self.alpha = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.alpha(x))  # forward used for training

        return h

    def passThroughPrev(self, keyframes, specs):
        x_vid = self.cnn3d(keyframes)
        x_aud = self.cnn2d(specs)
        x = torch.cat((x_aud, x_vid), dim=1)
        x = self.elm1(x)

        return x


class ELM2(ELM2Features):
    def __init__(self, input_size, hidden_size, num_classes, cnn2d, cnn3d, elm1):
        super().__init__(input_size, hidden_size, num_classes, cnn2d, cnn3d, elm1)

        self.beta = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        x = super().forward(x)
        x = self.beta(x)

        return x

    def forwardToHidden(self, x):  # the output of this is what we feed to the next ELM AFTER THIS ONE IS TRAINED
        return super().forward(x)
