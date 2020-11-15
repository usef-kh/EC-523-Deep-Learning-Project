import os
import sys
import warnings

import torch
import torch.nn as nn
from torch import optim

from subnets import basic, tuned
from data.fer2013 import load_dataset
from utils.checkpoint import save, restore
from utils.logger import Logger

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
networks = {
    'sub1_basic': basic.Subnet1,
    'sub2_basic': basic.Subnet2,
    'sub3_basic': basic.Subnet3,
    'sub1_tuned': tuned.Subnet1,
    'sub2_tuned': tuned.Subnet2,
    'sub3_tuned': tuned.Subnet3
}


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


def setup_hparams(args):
    hps = {
        'name': None,
        'n_epochs': 20,
        'model_save_dir': None,
        'restore_epoch': None,
        'start_epoch': 0
    }

    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise RuntimeError(key + ' is not a known hyper parameter')
        else:
            hps[key] = value

    if hps['name'] not in networks:
        raise RuntimeError("Name possibilities are: " + ' '.join(networks.keys()))

    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    if hps['restore_epoch']:
        hps['start_epoch'] = int(hps['restore_epoch'])

    hps['n_epochs'] = int(hps['n_epochs'])

    return hps


def run(args):
    # Important parameters
    hps = setup_hparams(args)

    # Create dataloaders
    trainloader, valloader, testloader = load_dataset()

    # Prepare network
    net = networks[hps['name']]()
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    print("Training on", device)
    for epoch in range(hps['start_epoch'], hps['n_epochs']):
        acc_tr, loss_tr = train(net, trainloader, criterion, optimizer)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)

        acc_v, loss_v = evaluate(net, valloader, criterion)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)

        save(net, logger, hps, epoch)
        logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.2f %%' % acc_tr,
              'Val Accuracy: %2.2f %%' % acc_v,
              sep='\t\t')


if __name__ == "__main__":
    run(sys.argv[1:])
