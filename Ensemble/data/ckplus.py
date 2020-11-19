import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data.dataset import CustomDataset


def load_data(path='../datasets/ckplus'):
    ckplus = {}
    for dir, subdir, files in os.walk(path):
        if files:
            sections = os.path.split(dir)
            emotion = sections[-1]

            ckplus[emotion] = [os.path.join(dir, file) for file in files]

    emotion_mapping = {emotion: i for i, emotion in enumerate(ckplus.keys())}

    return ckplus, emotion_mapping


def prepare_data(data, emotion_mapping):
    n_images = sum(len(paths) for paths in data.values())

    image_array = np.zeros(shape=(n_images, 48, 48))
    image_label = np.zeros(n_images)

    i = 0
    for emotion, img_paths in data.items():
        for path in img_paths:
            img = np.array(Image.open(path))
            image_array[i] = img
            image_label[i] = emotion_mapping[emotion]
            i += 1

    return image_array, image_label


def split_data(X, y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=1)
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.25, random_state=1)  # 0.25 * 0.8 = 0.2

    return (xtrain, ytrain), (xval, yval), (xtest, ytest)


def get_dataloaders():
    ckplus, emotion_mapping = load_data()

    images, labels = prepare_data(ckplus, emotion_mapping)
    train, val, test = split_data(images, labels)

    xtrain, ytrain = train
    xval, yval = val
    xtest, ytest = test

    mu, st = 0, 1
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(48, scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mu,), std=(st,))
    ])

    train = CustomDataset(xtrain, ytrain, train_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=100, shuffle=True, num_workers=2)
    valloader = DataLoader(val, batch_size=100, shuffle=True, num_workers=2)
    testloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
