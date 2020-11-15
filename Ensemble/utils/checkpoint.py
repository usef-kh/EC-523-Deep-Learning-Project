import os

import torch


def save(net, logger, hps, epoch):
    path = os.path.join(hps['model_save_dir'], 'epoch_' + str(epoch))

    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    torch.save(checkpoint, path)


def restore(net, logger, hps):
    path = os.path.join(hps['model_save_dir'], 'epoch_' + hps['restore_epoch'])

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)

            logger.restore_logs(checkpoint['logs'])
            net.load_state_dict(checkpoint['params'])
            print("Network Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e.message)
            hps['start_epoch'] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hps['start_epoch'] = 0
