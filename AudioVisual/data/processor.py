import cv2
import librosa
import librosa.display
import numpy as np

from librosa.feature import melspectrogram
from scipy.stats import chisquare


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


def get_keyframes(chunk, shift=4, window_len=7):
    n_keyframes = len(chunk) // 4

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


def process_video(path):
    video = cv2.VideoCapture(path)

    # Constants
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_shape = (height, width)
    chunk_len = int(2 * fps)
    n_chunks = int(np.ceil(n_frames / chunk_len))
    n_keyframes = chunk_len // 4

    if n_frames / fps > 7:
        return

    # Load video into respective chunks
    chunks = np.zeros((n_chunks, chunk_len, *frame_shape), dtype=np.uint8)

    i = 0
    check = True
    while check and i < n_chunks:

        for j in range(chunk_len):
            check, frame = video.read()
            if check:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                chunks[i][j] = gray
            else:
                break

        i += 1

    # Retreive keyframes and resize them
    chunk_keys = np.zeros((n_chunks, n_keyframes, 277, 277), dtype=np.uint8)
    for i, chunk in enumerate(chunks):
        frames = get_keyframes(chunk)
        chunk_keys[i] = frames

    return chunk_keys


def process_audio(path):
    y, sr = librosa.load(path, sr=None)

    # Constants
    n_samples = len(y)
    chunk_len = int(2 * sr)  # do i ceil?
    n_chunks = int(np.ceil(n_samples / chunk_len))

    if n_samples / sr > 7:
        return

    spectrograms = []
    for i in range(n_chunks):

        # initialize a zero array of chunk length and put whatever is possible, this is equivalent to zero padding
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
