import os
import warnings

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.dataset import CustomDataset
from models.CNNs import CNN_2DFeatures, CNN_3DFeatures
from models.ELM import ELM, ELMFeatures
from models.pseudoInverse import pseudoInverse

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
warnings.filterwarnings("ignore")


def load_features(model, params):
    """ Load params into all layers of 'model'
        that are compatible, then freeze them"""
    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    # Load layers
    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    # Freeze layers
    for name, param in model.named_parameters():
        param.requires_grad = False


def train_ELM(model, optimizer, train_loader):
    model.train()
    correct = 0
    for batch_idx, sample in enumerate(train_loader):
        x_keyframes, x_specs, y_gender, y_emotion = sample

        data = passThroughCNNs(x_keyframes, x_specs)  # CNNs
        data = elm1(data)  # Gender ELM

        data, target = data.to(device), y_emotion.to(device)

        data, target = Variable(data, requires_grad=False, volatile=True), Variable(target, requires_grad=False,
                                                                                    volatile=True)

        hiddenOut = model.forwardToHidden(data)
        optimizer.train(inputs=hiddenOut, targets=target)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        temp = pred.eq(target.data).cpu().sum()
        correct += temp

        print(batch_idx, temp / len(data))

    print('Accuracy:{}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset),
                                              100. * correct / len(train_loader.dataset)))


def test(model, test_loader):
    model.train()
    correct = 0
    for batch_idx, sample in enumerate(test_loader):
        x_keyframes, x_specs, y_gender, y_emotion = sample

        data = passThroughCNNs(x_keyframes, x_specs)  # CNNs
        data = elm1(data)  # Gender ELM

        data, target = data.to(device), y_emotion.to(device)

        data, target = Variable(data, requires_grad=False, volatile=True), Variable(target, requires_grad=False,
                                                                                    volatile=True)

        output = model.forward(data)
        pred = output.data.max(1)[1]
        temp = pred.eq(target.data).cpu().sum()
        correct += temp

        print(batch_idx, temp / len(data))

    print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def passThroughCNNs(keyframes, specs):
    x_vid = cnn3d(keyframes.to(device))
    x_aud = cnn2d(specs.to(device))
    x = torch.cat((x_aud.to(device), x_vid), dim=1)

    return x


# Building CNNs
print("Building 2D CNN")
cnn2d_path = r"/projectnb/ec523/ykh/project/AudioVisual/checkpoints/cnn2d/epoch_300"
cnn2d_params = torch.load(cnn2d_path)['params']
cnn2d = CNN_2DFeatures().to(device)
load_features(cnn2d, cnn2d_params)

print("Building 3D CNN")
cnn3d_path = r"/projectnb/ec523/ykh/project/AudioVisual/checkpoints/cnn3d/epoch_300"
cnn3d_params = torch.load(cnn3d_path)['params']
cnn3d = CNN_3DFeatures().to(device)
load_features(cnn2d, cnn2d_params)

# Building Gender ELM
print("Building Gender ELM")
elm1_params = torch.load("elm1").state_dict()  # I accidentally saved the whole model rather than the state dict
elm1 = ELMFeatures(input_size=8192, hidden_size=100, num_classes=2, device=device).to(device)
load_features(elm1, elm1_params)

# Building dataset
print("Loading Dataset")
data_dir = r"/projectnb/ec523/ykh/project/datasets/enterface/processed"
xtrain, ytrain = torch.load(os.path.join(data_dir, 'train'))
xval, yval = torch.load(os.path.join(data_dir, 'val'))
xtest, ytest = torch.load(os.path.join(data_dir, 'test'))

print("Building Dataset & Loaders")
train_dataset = CustomDataset(xtrain, ytrain)
val_dataset = CustomDataset(xval, yval)
test_dataset = CustomDataset(xtest, ytest)

trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# ELM
print("Initializing ELM")
num_classes = 6  # 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'
input_size = 100  # output of prev elm
hidden_size = 100

model = ELM(input_size, hidden_size, num_classes, device=device).to(device)
optimizer = pseudoInverse(model.parameters(), C=0.001, L=0)

print("Training ELM")
train_ELM(model, optimizer, trainloader)

print("Saving Model")
torch.save(model, 'Final_ELM')

print("Evaluating Model")
# print("Val")
# test(model, valloader)

print("Test")
test(model, testloader)
