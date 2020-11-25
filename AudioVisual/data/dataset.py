import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_keyframes, x_specs = self.X[idx]
        x_keyframes = Image.fromarray(np.array(x_keyframes))
        x_specs = Image.fromarray(np.array(x_specs))

        if self.transform:
            x_keyframes = self.transform(x_keyframes)
            x_specs = self.transform(x_specs)

        y = torch.tensor(self.Y[idx]).type(torch.long)
        sample = (x_keyframes, x_specs, y)

        return sample
