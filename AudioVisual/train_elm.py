import sys
import warnings

import torch
from torch.autograd import Variable

from data.enterface import get_dataloaders
from models.ELM import pseudoInverse
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.setup_network import setup_network

device = "cpu"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore")


def train(model, optimizer, dataloader):
    model.train()
    correct = 0

    if hps['network'] == 'elm1':
        choice = 2
    else:
        choice = 3

    for batch_idx, sample in enumerate(dataloader):
        data = (sample[0].to(device), sample[1].to(device))
        target = sample[choice].to(device)

        hiddenOut = model.forwardToHidden(data)

        optimizer.train(inputs=hiddenOut, targets=target)
        output = model.forward(data)
        pred = output.data.max(1)[1]
        temp = pred.eq(target.data).cpu().sum()
        correct += temp

        print(batch_idx, temp / len(sample[0]))

    print('Accuracy:{}/{} ({:.2f}%)\n'.format(correct, len(dataloader.dataset),
                                              100. * correct / len(dataloader.dataset)))


def test(model, dataloader):
    model.train()
    correct = 0

    if hps['network'] == 'elm1':
        choice = 2
    else:
        choice = 3

    for batch_idx, sample in enumerate(dataloader):
        data = (sample[0].to(device), sample[1].to(device))
        target = sample[choice].to(device)
        target = Variable(target, requires_grad=False, volatile=True)

        output = model.forward(data)
        pred = output.data.max(1)[1]
        temp = pred.eq(target.data).cpu().sum()
        correct += temp

        print(batch_idx, temp / len(sample[0]))

    print('Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))


def run(net, logger, hps):
    # Create dataloaders
    print('start loading data')

    trainloader, valloader, testloader = get_dataloaders(bs=16)
    net = net.to(device)

    optimizer = pseudoInverse(net.parameters(), C=0.001, L=0)

    print("Training", hps['name'], "on", device)
    train(net, optimizer, trainloader)

    print("Saving Model")
    save(net, logger, hps, epoch=1)

    print("Evaluating", hps['name'])
    test(net, valloader)


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = setup_network(hps)
    for name, param in net.named_parameters():
        print(name, param.shape)

    run(net, logger, hps)
