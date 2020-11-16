import sys
import warnings

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.fer2013 import load_dataset
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


def run_simple(net, logger, hps):
    print("Running simple training loop")
    # Create dataloaders
    trainloader, valloader, testloader = load_dataset()

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
              save(net, logger, hps, epoch)
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

        if (epoch + 1) % hps['save_freq'] == 0:
              save(net, logger, hps, epoch + 1)
              logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')


def run(net, logger, hps):
    # Create dataloaders
    trainloader, valloader, testloader = load_dataset()

    net = net.to(device)

    learning_rate = float(hps['lr'])
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = nn.CrossEntropyLoss()

    print("Training", hps['name'], "on", device)
    for epoch in range(hps['start_epoch'], hps['n_epochs']):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        # Update learning rate if plateau
        scheduler.step(acc_v)
        
        if (epoch + 1) % hps['save_freq'] == 0:
              save(net, logger, hps, epoch + 1)
              logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = build_network(hps)

    if hps['simple']:
        run_simple(net, logger, hps)
    else:
        run(net, logger, hps)
