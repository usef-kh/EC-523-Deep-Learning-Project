import sys
import warnings

import torch
import torch.nn as nn
from torch import optim

from data.fer2013 import get_dataloaders
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.setup_network import build_network

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataloader, criterion, optimizer):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss


def evaluate(net, dataloader, criterion):
    net = net.eval()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item()

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        n_samples += labels.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples

    return acc, loss


def run(net, logger, hps):
    print("Running simple training loop")
    # Create dataloaders
    trainloader, valloader, testloader = get_dataloaders()

    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Training", hps['name'], "on", device)
    for epoch in range(hps['start_epoch'], 100):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        if (epoch + 1) % 20 == 0:
            save(net, logger, hps, epoch, simple=True)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')

    # Reduce Learning Rate
    print("Reducing learning rate from 0.001 to 0.0001")
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001

    # Train for 20 extra epochs
    for epoch in range(epoch, 120):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        if (epoch + 1) % 20 == 0:
            save(net, logger, hps, epoch, simple=True)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')

    acc_test, loss_test = evaluate(net, testloader, criterion)
    print('Test Accuracy: %2.2f %%' % acc_test,
          'Test Loss: %2.6f %%' % loss_test,
          sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = build_network(hps)

    run(net, logger, hps)
