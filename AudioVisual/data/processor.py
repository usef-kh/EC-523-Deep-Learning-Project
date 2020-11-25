import collections
import os

import cv2
import librosa
import librosa.display
import numpy as np
from librosa.feature import melspectrogram
from scipy.stats import chisquare
from sklearn.model_selection import train_test_split


def process_audio(path):
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

    features = np.zeros((n_chunks, 3, *spectrograms[0].shape))

    for i, spec in enumerate(spectrograms):
        spec_db = librosa.power_to_db(spec, ref=np.max)
        delta = librosa.feature.delta(spec_db, width=3)
        double_delta = librosa.feature.delta(delta, width=3)

        for j, feature in enumerate([spec_db, delta, double_delta]):
            features[i][j] = feature

    return features


def process_video(path):
    video = cv2.VideoCapture(path)

    # Constants
    frame_shape = (576, 720)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    chunk_len = int(np.ceil(2.02 * fps))
    # n_chunks = int(np.ceil(n_frames / chunk_len))
    # print(n_frames, chunk_len)
    n_chunks = int(n_frames // chunk_len)
    n_keyframes = chunk_len // 4

    if n_frames/fps > 7:
        return

    def get_keyframes(chunk, shift=4, window_len=7):
        keyframes = np.zeros((n_keyframes, *(277, 277)))
        for i in range(n_keyframes):
            window = chunk[i * shift: (i * shift + window_len)]
            chi = []
            for w in window:
                hist = cv2.calcHist(w, [0], None, [256], [0, 256])
                chi.append(chisquare(hist))

            idx = chi.index(min(chi))
            keyframe = window[idx]

            # do face detection and resize to 277 277
            keyframe = face_detection(keyframe)

            keyframes[i] = keyframe

        return keyframes

    check = True
    chunks = np.zeros((n_chunks, chunk_len, *frame_shape), dtype=np.uint8)
    i = 0
    while check and i < n_chunks:

        for j in range(chunk_len):
            check, frame = video.read()
            if check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                chunks[i][j] = gray
            else:
                break

        i += 1

    key_frames = np.zeros((84, *(277, 277)), dtype=np.uint8)
    i = 0
    for chunk in chunks:
        for frame in get_keyframes(chunk):
            key_frames[i] = frame
            i += 1

    return key_frames


def process_video_old(path):
    video = cv2.VideoCapture(path)

    # Constants
    frame_shape = (576, 720)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    chunk_len = int(np.ceil(2.02 * fps))
    # n_chunks = int(np.ceil(n_frames / chunk_len))
    n_chunks = int(n_frames // chunk_len)
    n_keyframes = chunk_len // 4

    def get_keyframes(chunk, shift=4, window_len=7):
        keyframes = np.zeros((n_keyframes, *frame_shape))
        for i in range(n_keyframes):
            window = chunk[i * shift: (i * shift + window_len)]
            chi = []
            for w in window:
                hist = cv2.calcHist(w, [0], None, [256], [0, 256])
                chi.append(chisquare(hist))

            idx = chi.index(min(chi))
            keyframe = window[idx]

            # do face detection and resize to 277 277
            keyframe = face_detection(keyframe)

            keyframes[i] = keyframe

        return keyframes

    check = True
    chunks = np.zeros((n_chunks, chunk_len, *frame_shape), dtype=np.uint8)
    i = 0
    while check and i < n_chunks:

        for j in range(chunk_len):
            check, frame = video.read()
            if check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                chunks[i][j] = gray
            else:
                break

        i += 1

    chunk_keys = np.zeros((n_chunks, n_keyframes, *frame_shape), dtype=np.uint8)
    for i, chunk in enumerate(chunks):
        frames = get_keyframes(chunk)
        chunk_keys[i] = frames

    return chunk_keys


def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    if faces != []:
        x, y, w, h = faces[0]  # theres only 1 face in our images
        face = frame[y:y + h, x:x + w]  # Extract face from frame
    else:
        face = frame

    resized_face = cv2.resize(face, (277, 277), interpolation=cv2.INTER_AREA)

    return resized_face


def prepare_paths(base_dir):
    paths = collections.defaultdict(list)

    possible_emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    emotion_mapping = {emotion: i for i, emotion in enumerate(possible_emotions)}

    for curr_dir, sub_dir, files in os.walk(base_dir):
        if files:
            emotion = os.path.split(os.path.split(curr_dir)[0])[-1]

            # catch the exception in folder structure from subject 6
            if emotion not in emotion_mapping:
                emotion = os.path.split(curr_dir)[1]

            files = [os.path.join(curr_dir, file) for file in files if file[-2:] != 'db']

            emotion_id = emotion_mapping[emotion]
            paths[emotion_id].extend(files)

    return split(paths)


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

# audio_path = r'..\..\datasets\enterface\wav\subject 15\fear\sentence 1\s15_fe_1.wav'
# chunks = process_audio(audio_path)
# print(chunks.shape)
#
# for chunk in chunks:
#     fig, axes = plt.subplots(3, 1)
#     for feature, ax in zip(chunk, axes):
#         ax.imshow(feature)
#     plt.show()

#
#
# axes[0].imshow(spec)
# axes[1].imshow(delta)
# axes[2].imshow(double_delta)
# # for ax, spec in zip(axes, specs):
# #     print(spec.shape)
# #     ax.imshow(spec)
# plt.show()

# video_path = r'..\..\datasets\enterface\original\subject 15\fear\sentence 1\s15_fe_1.avi'
# key_frames = process_video(video_path)
#
# for frames in key_frames:
#
#     for frame in frames:
#         plt.figure()
#         plt.imshow(frame, cmap='gray')
#         plt.show()
#
#         face = face_detection(frame)
#
#         plt.figure()
#         plt.imshow(face, cmap='gray')
#         plt.show()
