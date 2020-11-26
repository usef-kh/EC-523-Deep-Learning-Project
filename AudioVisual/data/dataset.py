import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        # self.transform = transform            (frames, specs), (gender, labels)

    def __len__(self):
        return len(self.X[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_keyframes = torch.from_numpy(self.X[0][idx])
        x_specs = torch.from_numpy(self.X[1][idx])

        # if self.transform:
        #     x_specs = self.transform(x_specs)
        #
        #     n_frames = x_keyframes.shape[-1]
        #     for i in range(n_frames):
        #         x_frames

        y_gender = torch.tensor(self.Y[0][idx]).type(torch.long)
        y_emotion = torch.tensor(self.Y[1][idx]).type(torch.long)

        sample = (x_keyframes, x_specs, y_gender, y_emotion)

        return sample
