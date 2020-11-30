import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        # x_keyframes = [12, 277, 277], x_specs = [25, 101], y_gender = [1], y_emotion = [1].
        self.keyframes = X[0] # [n, 12, 277, 277]
        self.specs = X[1]     # [n, 25, 101]
        self.gender = Y[0]    # [n, 1]
        self.emotion = Y[1]   # [n, 1]
        # self.transform = transform            (frames, specs), (gender, labels)

    def __len__(self):
        # X is a tuple: (frames, specs), X[0] refers to frames, so len(X[0] get the total length of data set, which is n.
        return len(self.keyframes)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_keyframes = torch.from_numpy(self.keyframes[idx]).type(torch.float)
        x_specs = torch.from_numpy(self.specs[idx]).type(torch.float)

        # for data augmentation.
        # if self.transform:
        #     x_specs = self.transform(x_specs)
        #
        #     n_frames = x_keyframes.shape[-1]
        #     for i in range(n_frames):
        #         x_frames

        y_gender = torch.tensor(self.gender[idx]).type(torch.long)
        y_emotion = torch.tensor(self.emotion[idx]).type(torch.long)
        sample = (x_keyframes, x_specs, y_gender, y_emotion)

        return sample
