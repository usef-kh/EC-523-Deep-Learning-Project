import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data.dataset import CustomDataset
from data.processor import process_video, prepare_paths


def prepare_data(data_paths):
    X_arr, Y_arr = [], []

    for emotion, paths in data_paths.items():
        for path in paths:
            # print(path)
            x = process_video(path)
            if x is not None:
                X_arr.append(x)
                Y_arr.append(emotion)

    X = np.zeros((len(X_arr), 84, 277, 277))
    Y = np.zeros(len(X_arr), )

    i = 0
    for x, y in zip(X_arr, Y_arr):
        X[i] = x
        Y[i] = y
        i += 1

    return X, Y


def get_dataloaders(base_dir):
    train_paths, val_paths, test_paths = prepare_paths(base_dir)

    xtrain, ytrain = prepare_data(train_paths)
    xval, yval = prepare_data(val_paths)
    xtest, ytest = prepare_data(test_paths)

    print(xtrain.shape, ytrain.shape)
    print(xval.shape, yval.shape)
    print(xtest.shape, ytest.shape)

    print([(emotions, len(paths)) for emotions, paths in train_paths.items()])
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = CustomDataset(xtrain, ytrain, transform)
    val = CustomDataset(xval, yval, transform)
    test = CustomDataset(xtest, ytest, transform)

    trainloader = DataLoader(train, batch_size=32, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=32, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader


if __name__ == '__main__':
    bas_dir = r"..\..\datasets\enterface\original"

    get_dataloaders(bas_dir)
