import torch
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

        x_keyframes, x_specs = self.X[0][idx], self.X[1][idx]
        x_keyframes = torch.from_numpy(x_keyframes)
        x_specs = torch.from_numpy(x_specs)

        # if self.transform:
        #     x_specs = self.transform(x_specs)
        #
        #     n_frames = x_keyframes.shape[-1]
        #     for i in range(n_frames):
        #         x_frames

        y = torch.tensor(self.Y[idx]).type(torch.long)
        sample = (x_keyframes, x_specs, y)

        return sample
