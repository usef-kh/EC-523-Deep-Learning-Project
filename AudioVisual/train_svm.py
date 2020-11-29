import sys

import numpy as np
import torch
from sklearn.svm import SVC as SVM
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from data.enterface import get_dataloaders
from utils.hparams import setup_hparams
from utils.setup_network import setup_network

device = "cpu"


def passThroughNetwork(dataloader):
    X, Y = [], []
    for batch_idx, sample in enumerate(dataloader):
        data = elm2.passThroughPrev(sample[0], sample[1])
        data, target = data.to(device), sample[-1].to(device)
        data, target = Variable(data, requires_grad=False, volatile=True), Variable(target, requires_grad=False,
                                                                                    volatile=True)
        output = elm2.forward(data)
        Y.append(target)
        X.append(output)

    return torch.cat(X, dim=0).data.numpy(), torch.cat(Y, dim=0).data.numpy()


print("Creating Networks")
hps = setup_hparams(sys.argv[1:])
logger, elm2 = setup_network(hps)

print("loading Data")
trainloader, valloader, testloader = get_dataloaders()

print("Passing data through network")
xtrain, ytrain = passThroughNetwork(trainloader)
# xval, yval = passThroughNetwork(valloader)
xtest, ytest = passThroughNetwork(testloader)

svm = SVM(kernel='rbf', C=1)
svm.fit(xtrain, ytrain)
pred = svm.predict(xtest)

acc = accuracy_score(ytest, pred)

print("Accuracy on test Set:", acc * 100)


# # plot decision boundary
# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     """Plot the decision function for a 2D SVC"""
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#
#     # create grid to evaluate model
#     x = np.linspace(xlim[0], xlim[1], 50)
#     y = np.linspace(ylim[0], ylim[1], 50)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     print(X.shape, Y.shape, xy.shape)
#     P = model.decision_function(xy)  # .reshape(X.shape)
#     print("P", P)  # score for each class
#     P_vals = np.argmax(P, axis=1)
#     print("P vals", P_vals, "\len P", P.shape)
#
#     # plot decision boundary and margins
#     ax.contour(X, Y, P_vals, colors=['r', 'b', 'y', 'g'],
#                levels=[0, 1, 2, 3], alpha=0.5,
#                linestyles=['--', '-', '--', '-'])
#
#     # plot support vectors
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none');
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
