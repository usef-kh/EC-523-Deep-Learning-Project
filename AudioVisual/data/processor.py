import collections
import os

import cv2
import librosa
import librosa.display
import numpy as np
import torch
import torchvision.transforms as transforms
from librosa.feature import melspectrogram
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import CustomDataset


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


def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]  # theres only 1 face in our images
        face = frame[y:y + h, x:x + w]  # Extract face from frame
    else:
        face = frame

    resized_face = cv2.resize(face, (277, 277), interpolation=cv2.INTER_AREA)

    return resized_face


def process_video(path):
    video = cv2.VideoCapture(path)

    # Constants
    frame_shape = (576, 720)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    # if n_frames / fps > 7:
    #     return
    chunk_len = int(2 * fps)
    n_chunks = int(np.ceil(n_frames / chunk_len))
    n_keyframes = chunk_len // 4

    def get_keyframes(chunk, shift=4, window_len=7, n_keyframes=12):
        keyframes = np.zeros((n_keyframes, 277, 277))
        for i in range(n_keyframes):
            window = chunk[i * shift: (i * shift + window_len)]
            chi = []
            for w in window:
                hist = cv2.calcHist(w, [0], None, [256], [0, 256])
                chi.append(chisquare(hist))
            # get the minimum chi values, and retrive the idex of that value, which is the key frame index.
            idx = chi.index(min(chi))
            keyframe = window[idx]

            # do face detection and resize to 277 277
            keyframe = face_detection(keyframe)

            keyframes[i, :, :] = keyframe

        return keyframes

    check = True
    chunks = np.zeros((n_chunks, chunk_len, *frame_shape), dtype=np.uint8)
    i = 0
    while check and i < n_chunks:

        for j in range(chunk_len):
            check, frame = video.read()
            if check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = get_keyframes(gray)
                chunks[i][j] = gray
            else:
                break

        i += 1

    chunk_keys = np.zeros((n_chunks, n_keyframes, 277, 277), dtype=np.uint8)
    for i, chunk in enumerate(chunks):
        frames = get_keyframes(chunk)
        chunk_keys[i] = frames

    return chunk_keys


def process_audio(path):
    y, sr = librosa.load(path, sr=None)
    # get the total length of input audio
    n_samples = len(y)
    # if n_samples / sr > 7:
    #     return
    chunk_len = int(2 * sr)  # do i ceil?
    n_chunks = int(np.ceil(n_samples / chunk_len))

    spectrograms = []
    for i in range(n_chunks):
        chunk = np.zeros((chunk_len,))
        remaining_len = len(y[i * chunk_len: (i * chunk_len + chunk_len)])
        chunk[:remaining_len] = y[i * chunk_len: (i * chunk_len + chunk_len)]
        spec = melspectrogram(
            y=chunk,
            sr=sr,
            win_length=int(sr / 1000) * 40,
            hop_length=int(sr / 1000) * 20,
            n_mels=25
        )
        # expand one more dimension and stack vertically
        spec = np.expand_dims(spec, 0)

        if spectrograms == []:
            spectrograms = spec
        else:
            spectrograms = np.vstack((spectrograms, spec))

    if spectrograms == []:
        return

    features = np.zeros((n_chunks, 3, *spectrograms[0].shape))
    for i, spec in enumerate(spectrograms):
        spec_db = spec  # librosa.power_to_db(spec, ref=np.max)
        delta = librosa.feature.delta(spec_db, width=3)
        double_delta = librosa.feature.delta(delta, width=3)

        for j, feature in enumerate([spec_db, delta, double_delta]):
            features[i, j, :, :] = feature

    return features


def get_subject_id(path):
    i = path.find("subject ") + 8

    idx = ''
    while path[i] != '\\':
        idx += path[i]
        i += 1

    return idx


def prepare_gender():
    gender_mapping = {'m': 0, 'f': 1}

    gender = {}
    with open("../../datasets/gender.txt") as txtfile:
        for line in txtfile:
            (subject_id, subject_gender) = line.split()
            gender[subject_id] = gender_mapping[subject_gender]

    return gender


def prepare_data(data):  # data type will be dictionary,  emotion:  path.

    gender_mapping = prepare_gender()

    i = 0

    frames, specs, gender, labels = [], [], [], []
    for emotion_id, paths in data.items():
        for avi_path, wav_path in paths:
            key_frames = process_video(avi_path)
            spectrograms = process_audio(wav_path)
            if key_frames is not None and specs is not None:
                if frames == []:
                    frames = key_frames
                    specs = spectrograms
                else:
                    assert len(key_frames) == len(spectrograms)
                    frames = np.vstack((frames, key_frames))
                    specs = np.vstack((specs, spectrograms))

                subject_id = get_subject_id(wav_path)  # or avi path, its the same
                gender_id = gender_mapping[subject_id]

                labels += [emotion_id] * len(key_frames)
                gender += [gender_id] * len(key_frames)



            elif key_frames is None or specs is None:
                raise RuntimeError('frames or spectrograms is broken')

            i += 1
            if i > 2:
                break

        if i > 2:
            break

    labels = np.array(labels)
    gender = np.array(gender)

    print("frame dims", frames.shape)
    print("specs dims", specs.shape)
    print("label dims", labels.shape)
    print("gender dims", gender.shape)

    return (frames, specs), (gender, labels)


def get_dataloaders(video_dir=None, audio_dir=None):
    train_paths, val_paths, test_paths = prepare_paths(video_dir, audio_dir)

    xtrain, ytrain = prepare_data(train_paths)
    xval, yval = prepare_data(val_paths)
    xtest, ytest = prepare_data(test_paths)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = CustomDataset(xtrain, ytrain, transform)
    val = CustomDataset(xval, yval, transform)
    test = CustomDataset(xtest, ytest, transform)

    trainloader = DataLoader(train, batch_size=1, shuffle=True)  # , num_workers=2)
    valloader = DataLoader(val, batch_size=1, shuffle=True)  # , num_workers=2)
    testloader = DataLoader(test, batch_size=1, shuffle=True)  # , num_workers=2)

    return trainloader, valloader, testloader


video_dir = '../../datasets/enterface/original'
audio_dir = '../../datasets/enterface/wav'
train_paths, val_paths, test_paths = prepare_paths(video_dir, audio_dir)

print("Train")
xtrain, ytrain = prepare_data(train_paths)

print("Val")
xval, yval = prepare_data(val_paths)

print("Test")
xtest, ytest = prepare_data(test_paths)

torch.save((xtrain, ytrain), 'train')
torch.save((xval, yval), 'train')
torch.save((xtest, ytest), 'test')
