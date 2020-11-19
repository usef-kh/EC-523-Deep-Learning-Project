import cv2
import librosa
import librosa.display
import numpy as np
from librosa.feature import melspectrogram
from matplotlib import pyplot as plt
from scipy.stats import chisquare


def process_audio_old(path):
    y, sr = librosa.load(path, sr=None)


    n_samples = len(y)
    chunk_len = int(2.02 * sr)  # do i ceil?
    n_chunks = int(n_samples // chunk_len)
    print(sr)

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

    spectrograms = [librosa.power_to_db(spec, ref=np.max) for spec in spectrograms]

    return spectrograms


def process_video(path):
    video = cv2.VideoCapture(path)

    # Constants
    frame_shape = (576, 720)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    chunk_len = int(np.ceil(2.02 * fps))
    # n_chunks = int(np.ceil(n_frames / chunk_len))
    n_chunks = int(n_frames // chunk_len)
    n_keyframes = chunk_len // 4

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

            keyframes[i] = keyframe

        return keyframes

    chunk_keys = np.zeros((n_chunks, n_keyframes, *frame_shape), dtype=np.uint8)
    for i, chunk in enumerate(chunks):
        frames = get_keyframes(chunk)

        chunk_keys[i] = frames

    print(chunk_keys.shape)
    return chunk_keys


def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    x, y, w, h = faces[0]  # theres only 1 face in our images

    face = frame[y:y + h, x:x + w]  # Extract face from frame
    resized_face = cv2.resize(face, (277, 277), interpolation=cv2.INTER_AREA)

    return resized_face


def process_audio(path):
    y, sr = librosa.load(path)

    n_samples = len(y)
    window_len = int(40 / 1000 * sr)
    shift = int(0.5 * window_len)
    n_chunks = n_samples // shift

    chunks = []
    for i in range(0, n_samples, shift):
        chunks.append(y[i:i + window_len])

    print(len(chunks), n_chunks)
    print(n_samples, sr, n_samples / sr)
    print(y)


audio_path = r'..\..\datasets\enterface\wav\subject 15\fear\sentence 1\s15_fe_1.wav'
specs = process_audio_old(audio_path)

fig, axes = plt.subplots(len(specs), 1)
for ax, spec in zip(axes, specs):
    ax.imshow(spec)
plt.show()

video_path = r'..\..\datasets\enterface\original\subject 15\fear\sentence 1\s15_fe_1.avi'
key_frames = process_video(video_path)

for frames in key_frames:

    for frame in frames:
        plt.figure()
        plt.imshow(frame, cmap='gray')
        plt.show()

        face = face_detection(frame)

        plt.figure()
        plt.imshow(face, cmap='gray')
        plt.show()
