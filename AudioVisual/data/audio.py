import collections
import os

import librosa
import numpy as np
import torchvision.transforms as transforms
from librosa.feature import melspectrogram
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


class AudioData:

    def __init__(self, audio_dir=r"..\datasets\enterface\wav"):
        possible_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        self.emotion_mapping = {emotion: i for i, emotion in enumerate(possible_emotions)}

        self.data_paths = self.prepare_paths(audio_dir)

    def prepare_paths(self, audio_dir):
        wav_paths = collections.defaultdict(list)

        for dir, sub_dir, files in os.walk(audio_dir):
            if files:
                emotion = os.path.split(os.path.split(dir)[0])[-1]

                # catch the exception in folder structure from subject 6
                if emotion not in self.emotion_mapping:
                    emotion = os.path.split(dir)[1]

                files = [os.path.join(dir, file) for file in files]

                wav_paths[emotion].extend(files)

        return wav_paths

    def split(self):
        train = collections.defaultdict(list)
        test = collections.defaultdict(list)
        val = collections.defaultdict(list)

        for emotion, paths in self.data_paths.items():
            # 0.25 * 0.8 = 0.2
            train_paths, test_paths = train_test_split(paths, test_size=0.2, random_state=1, shuffle=True)
            train_paths, val_paths = train_test_split(train_paths, test_size=0.25, random_state=1, shuffle=True)

            train[emotion].extend(train_paths)
            val[emotion].extend(val_paths)
            test[emotion].extend(test_paths)

        return train, val, test

    def process_audio(self, path):
        y, sr = librosa.load(path, sr=None)

        n_samples = len(y)
        chunk_len = int(2.02 * sr)  # do i ceil?
        n_chunks = int(n_samples // chunk_len)

        spectrograms = []
        for i in range(n_chunks):
            chunk = y[i * chunk_len: (i * chunk_len + chunk_len)]

            spectrogram = melspectrogram(
                y=chunk,
                sr=sr,
                win_length=int(sr / 1000) * 40,
                hop_length=int(sr / 1000) * 20,
                n_mels=25
            )

            spectrograms.append(spectrogram)

        if not spectrograms:
            return

        samples = np.zeros((n_chunks, *spectrograms[0].shape, 3))

        for i, spec in enumerate(spectrograms):
            spec_db = librosa.power_to_db(spec, ref=np.max)
            delta = librosa.feature.delta(spec_db, width=3)
            double_delta = librosa.feature.delta(delta, width=3)

            for j, feature in enumerate([spec_db, delta, double_delta]):
                samples[i, :, :, j] = feature

        return samples

    def prepare_data(self, data):
        n_samples, X_arr, Y_arr = 0, [], []

        for emotion, paths in data.items():

            for path in paths:
                samples = self.process_audio(path)

                if samples is not None:
                    X_arr.append(samples)
                    Y_arr.append(emotion)

                    n_samples += samples.shape[0]

        X = np.zeros((n_samples, *X_arr[0].shape[1:]))
        Y = np.zeros(n_samples)
        i = 0
        for x, y in zip(X_arr, Y_arr):
            n = x.shape[0]
            X[i: i + n] = x
            Y[i: i + n] = self.emotion_mapping[y]

            i += n

        X, Y = shuffle(X, Y)

        return X, Y

    def get_dataloaders(self):
        train, val, test = self.split()

        xtrain, ytrain = self.prepare_data(train)
        xval, yval = self.prepare_data(val)
        xtest, ytest = self.prepare_data(test)

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train = CustomDataset(xtrain, ytrain, transform)
        val = CustomDataset(xval, yval, transform)
        test = CustomDataset(xtest, ytest, transform)

        trainloader = DataLoader(train, batch_size=100, shuffle=True, num_workers=2)
        valloader = DataLoader(val, batch_size=100, shuffle=True, num_workers=2)
        testloader = DataLoader(test, batch_size=100, shuffle=True, num_workers=2)

        return trainloader, valloader, testloader


if __name__ == '__main__':
    dataset = AudioData(r"..\..\datasets\enterface\wav")
    print(dataset.data_paths)
    print('hi yousif')
    train, val, test = dataset.get_dataloaders()

    train = iter(train)
    X, Y = train.next()
    print(X.shape, Y.shape)
    # print(X[0])
