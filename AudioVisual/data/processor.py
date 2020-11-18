import cv2
import numpy as np
from librosa.feature import melspectrogram
from matplotlib import pyplot as plt
from scipy.stats import chisquare


def process_audio(y, sr):
    spectogram = melspectrogram(
        y=y,
        sr=sr,
        win_length=int(sr / 1000) * 40,
        hop_length=int(sr / 1000) * 20,
        n_mels=25
    )

    return spectogram


def process_video(path):
    video = cv2.VideoCapture(path)

    # Constants
    frame_shape = (576, 720)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    chunk_len = int(np.ceil(2.02 * fps))
    n_chunks = int(np.ceil(n_frames / chunk_len))
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


path = r'..\..\datasets\enterface\original\subject 11\fear\sentence 3\s12_fe_3.avi'
# Load the cascade

frames = process_video(path)

def face_detection(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    x, y, w, h = faces[0]  # theres only 1 face in our images

    face = frame[y:y + h, x:x + w]  # Extract face from frame
    resized_face = cv2.resize(face, (277, 277), interpolation=cv2.INTER_AREA)

    return resized_face
