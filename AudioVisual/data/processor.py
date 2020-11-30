import cv2
import librosa
import librosa.display
import numpy as np

from librosa.feature import melspectrogram
from scipy.stats import chisquare


def face_detection(frame):
    """
    use open cv to perform face detection on a given frame and resize to (277, 277)
    if dont find a face, resize the frame only
    :param frame: frame
    :return: resized face
    """
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
    """
    scan the chunk and output the key frames in this chunk
    :param chunk: the frames in a 2 second length video.
    :param shift: window shift, defined by the paper, Hossain, M.S. and Muhammad, G., 2019. Emotion recognition using deep learning approach from audio–visual emotional big data. Information Fusion, 49, pp.69-78.
    :param window_len:   size of the window for getting the key frames,
    :return: key frames in this chunk (12, 277, 277)
    """
    n_keyframes = len(chunk) // 4

    keyframes = np.zeros((n_keyframes, 277, 277))
    for i in range(n_keyframes):
        # extract the ith window in the chunk.
        window = chunk[i * shift: (i * shift + window_len)]
        chi = []
        for w in window:
            # get the pixel gray values distribution of the image
            hist = cv2.calcHist(w, [0], None, [256], [0, 256])
            # calculate the chi square error of the gray values distribution
            chi.append(chisquare(hist))
        # get the minimum chi values, and retrieve the index of that value, which is the key frame index.
        idx = chi.index(min(chi))
        keyframe = window[idx]

        # do face detection and resize to 277 277
        keyframe = face_detection(keyframe)

        keyframes[i, :, :] = keyframe

    return keyframes


def process_video(path):
    """
    load the video and split into chunks of 2 seconds and retrieve the key frames in each chunk
    :param path: video path
    :return: key frames in this video. [num_chunks, 12, 277, 277]
    """
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
    check = True # frame is read correctly
    while check and i < n_chunks:

        for j in range(chunk_len):
            check, frame = video.read()
            if check:
                # convert to gray scale and put all the images into chunks, 4 D array
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                chunks[i][j] = gray
            else:
                break

        i += 1

    # Retrieve keyframes and resize them
    chunk_keys = np.zeros((n_chunks, n_keyframes, 277, 277), dtype=np.uint8)
    for i, chunk in enumerate(chunks):
        frames = get_keyframes(chunk)
        chunk_keys[i] = frames

    return chunk_keys


def process_audio(path):
    """
    load the audio and split into chunks of 2 seconds and retrieve the (spectrograms, delta, double_delta) in each chunk
    :param path: audio path
    :return: key frames in this video. [num_chunks, 3, 25, 101]
    """
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

        delta = librosa.feature.delta(spec, width=3)
        double_delta = librosa.feature.delta(delta, width=3)

        for j, feature in enumerate([spec, delta, double_delta]):
            features[i, j, :, :] = feature

    return features
