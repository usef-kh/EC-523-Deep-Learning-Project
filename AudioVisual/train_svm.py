import sys

import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as SVM

from data.enterface import get_dataloaders
from utils.hparams import setup_hparams
from utils.setup_network import setup_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def passThroughNetwork(dataloader):
    X, Y = [], []
    for batch_idx, sample in enumerate(dataloader):
        data = (sample[0].to(device), sample[1].to(device))
        target = sample[-1]

        output = net.forwardToHidden(data)
        Y.append(target.to("cpu"))
        X.append(output.to("cpu"))

    return torch.cat(X, dim=0).data.numpy(), torch.cat(Y, dim=0).data.numpy()


print("Creating Networks")
hps = setup_hparams(sys.argv[1:])
logger, net = setup_network(hps)
net = net.to(device)

print("loading Data")
trainloader, valloader, testloader = get_dataloaders(bs=4)

print("Passing training data through network")
xtrain, ytrain = passThroughNetwork(trainloader)
# print("Passing validation data through network")
# xval, yval = passThroughNetwork(valloader)
print("Passing test data through network")
xtest, ytest = passThroughNetwork(testloader)

print("Training SVM on CPU")
svm = SVM(kernel='rbf', C=1)
svm.fit(xtrain, ytrain)
pred = svm.predict(xtest)

acc = accuracy_score(ytest, pred)

print("Accuracy on test Set:", acc * 100)
