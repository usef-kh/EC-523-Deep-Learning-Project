import os
import torch


def save(net, hps, epoch):
    path = os.path.join(hps['model_save_dir'], 'epoch_' + str(epoch))
    torch.save(net.state_dict(), path)


def restore(net, hps):
    path = os.path.join(hps['model_save_dir'], 'epoch_' + hps['restore_epoch'])

    if os.path.exists(path):
        net.load_state_dict(torch.load(path))

    else:
        print("Restore point is not available. Training from scratch")
        hps['start_epoch'] = 0
