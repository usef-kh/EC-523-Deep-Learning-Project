import torch

from data.data import prepare_paths, prepare_data

video_dir = '../datasets/enterface/original'
audio_dir = '../datasets/enterface/wav'
train_paths, val_paths, test_paths = prepare_paths(video_dir, audio_dir)

print("Train")
xtrain, ytrain = prepare_data(train_paths)

print("Val")
xval, yval = prepare_data(val_paths)

print("Test")
xtest, ytest = prepare_data(test_paths)

torch.save((xtrain, ytrain), 'train')
torch.save((xval, yval), 'val')
torch.save((xtest, ytest), 'test')