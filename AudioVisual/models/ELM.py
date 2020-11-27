import torch
import torch.nn as nn


class ELMFeatures(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = num_classes  # 2 for first ELM, 5 for second
        self.device = device

        self.alpha = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.alpha(x))  # forward used for training

        return h


class ELM(ELMFeatures):
    def __init__(self, input_size, hidden_size, num_classes, device=None):
        super().__init__(input_size, hidden_size, num_classes, device)

        self.beta = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        x = super().forward(x)
        x = self.beta(x)

        return x

    def forwardToHidden(self, x):  # the output of this is what we feed to the next ELM AFTER THIS ONE IS TRAINED
        return super().forward(x)
