import collections
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


def load_data(path='../datasets/ckplus'):
    ckplus = {}
    for dir, subdir, files in os.walk(path):

        if files:

            sections = os.path.split(dir)
            emotion = sections[-1]

            temp = collections.defaultdict(list)
            for file in files:
                subject = file.split("_")[0]
                temp[subject].append(os.path.join(dir, file))

            ckplus[emotion] = temp

    return ckplus


def prepare_data(data):
    n_images = sum(len(paths) for paths in data.values())
    emotion_mapping = {emotion: i for i, emotion in enumerate(data.keys())}

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


def split_data(data):
    train = collections.defaultdict(list)
    test = collections.defaultdict(list)
    val = collections.defaultdict(list)

    for emotion, subjects in data.items():

        subjects_train, subjects_xtest = train_test_split(list(subjects.keys()), test_size=0.2, random_state=1,
                                                          shuffle=True)
        subjects_train, subjects_val = train_test_split(subjects_train, test_size=0.25, random_state=1,
                                                        shuffle=True)  # 0.25 * 0.8 = 0.2

        for subject in subjects_train:
            train[emotion].extend(subjects[subject])

        for subject in subjects_val:
            val[emotion].extend(subjects[subject])

        for subject in subjects_xtest:
            test[emotion].extend(subjects[subject])

    return train, val, test


def get_dataloaders():

    ckplus = load_data()
    train, val, test = split_data(ckplus)

    xtrain, ytrain = prepare_data(train)
    xval, yval = prepare_data(val)
    xtest, ytest = prepare_data(test)


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


if __name__ == '__main__':
    ckplus = load_data('../../datasets/ckplus')

    train, val, test = split_data(ckplus)

    # for k, v in train.items():
    #     print(k)
    #     print(*v, sep='\n')

    image_array, image_label = prepare_data(train)
    #
    #
    # train, val, test = split_data(image_array, image_label)
    #
    for i in range(12):
        print(image_label[i])
        plt.figure()
        plt.imshow(image_array[i])
        plt.show()

    # print(ckplus)
