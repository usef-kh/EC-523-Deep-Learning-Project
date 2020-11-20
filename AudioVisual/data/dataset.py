import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

        # print(X.dtype, np.max(X),  np.min(X), Y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # x = Image.fromarray(np.array(self.X[idx]))
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)

        y = torch.tensor(self.Y[idx]).type(torch.long)
        sample = (x, y)

        return sample