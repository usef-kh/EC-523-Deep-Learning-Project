import numpy as np
import pandas as pd
import torch

print("Loading Data")
fer2013 = pd.read_csv('fer2013.csv')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def reformat_data(X, Y):
    data = []

    for x, y in zip(torch.from_numpy(X), torch.from_numpy(Y)):
        x, y = x.type(torch.DoubleTensor), y.type(torch.long)
        data.append((x.unsqueeze(0), y))

    return data


print("Preparing Data")
xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])
xtest, ytest = prepare_data(fer2013[fer2013['Usage'] == 'PublicTest'])

print("Reformatting Data")
train = reformat_data(xtrain, ytrain)
val = reformat_data(xval, yval)
test = reformat_data(xtest, ytest)

print("Saving Data")

torch.save(train, 'train')
torch.save(val, 'val')
torch.save(test, 'test')

print("Done")
