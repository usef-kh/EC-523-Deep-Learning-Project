import collections
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from data.dataset import CustomDataset
from data.processor import process_video, process_audio


def prepare_gender():
    gender_mapping = {'m': 0, 'f': 1}

    gender = {}
    with open("../datasets/enterface/gender.txt") as txtfile:
        for line in txtfile:
            (subject_id, subject_gender) = line.split()
            gender[subject_id] = gender_mapping[subject_gender]

    return gender


def get_subject_id(path):

    i = path.find("subject ") + 8

    idx = ''
    while path[i].isdigit():
        idx += path[i]
        i += 1

    return idx


def split(dataset):
    train = collections.defaultdict(list)
    test = collections.defaultdict(list)
    val = collections.defaultdict(list)

    for emotion, paths in dataset.items():
        # 0.25 * 0.8 = 0.2
        train_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=1, shuffle=True)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=1, shuffle=True)

        train[emotion].extend(train_paths)
        val[emotion].extend(val_paths)
        test[emotion].extend(test_paths)

    return train, val, test


def prepare_paths(video_dir='../../datasets/enterface/original', audio_dir='../../datasets/enterface/wav'):
    paths = collections.defaultdict(list)

    possible_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    emotion_mapping = {emotion: i for i, emotion in enumerate(possible_emotions)}

    for curr_dir, sub_dir, files in os.walk(video_dir):
        if files:
            './../datasets/enterface/wav\subject 1\anger\garbage.wav'
            emotion = os.path.split(os.path.split(curr_dir)[0])[-1]

            # catch the exception in folder structure from subject 6
            if emotion not in emotion_mapping:
                emotion = os.path.split(curr_dir)[1]

            files = [os.path.join(curr_dir, file) for file in files if file[-2:] != 'db']

            emotion_id = emotion_mapping[emotion]
            paths[emotion_id].extend(files)

    path_tuples = collections.defaultdict(list)

    for emotion, avi_paths in paths.items():
        for avi_path in avi_paths:
            wav_file = avi_path[len(video_dir) + 1:][:-3] + 'wav'
            wav_path = os.path.join(audio_dir, wav_file)

            path_tuples[emotion].append((avi_path, wav_path))

    return split(path_tuples)


def prepare_data(data):  # dataold type will be dictionary,  emotion:  path.

    gender_mapping = prepare_gender()

    frames, specs, gender, labels = [], [], [], []
    for emotion_id, paths in data.items():
        for avi_path, wav_path in paths:

            key_frames = process_video(avi_path)
            spectrograms = process_audio(wav_path)

            assert (key_frames is None) == (spectrograms is None), "Processors must accept/reject the same paths"

            if (key_frames is not None) and (spectrograms is not None):
                if frames == []:
                    frames = key_frames
                    specs = spectrograms
                else:

                    assert len(key_frames) == len(spectrograms), "Processors must create the same number of samples"

                    frames = np.vstack((frames, key_frames))
                    specs = np.vstack((specs, spectrograms))

                subject_id = get_subject_id(wav_path)  # or avi path, its the same
                gender_id = gender_mapping[subject_id]

                labels += [emotion_id] * len(key_frames)
                gender += [gender_id] * len(key_frames)

    labels = np.array(labels)
    gender = np.array(gender)

    print("frame dims", frames.shape)
    print("specs dims", specs.shape)
    print("label dims", labels.shape)
    print("gender dims", gender.shape)

    return (frames, specs), (gender, labels)


def get_dataloaders(data_dir='/projectnb/ec523/ykh/project/datasets/enterface/processed', bs=32):
    xtrain, ytrain = torch.load(os.path.join(data_dir, 'train'))
    xval, yval = torch.load(os.path.join(data_dir, 'val'))
    xtest, ytest = torch.load(os.path.join(data_dir, 'test'))

    train = CustomDataset(xtrain, ytrain)
    val = CustomDataset(xval, yval)
    test = CustomDataset(xtest, ytest)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True)  # , num_workers=2)
    valloader = DataLoader(val, batch_size=bs, shuffle=True)  # , num_workers=2)
    testloader = DataLoader(test, batch_size=bs, shuffle=True)  # , num_workers=2)

    return trainloader, valloader, testloader
