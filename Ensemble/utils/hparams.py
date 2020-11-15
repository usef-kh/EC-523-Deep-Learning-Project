import os

from subnets import basic, tuned

hps = {
    'name': None,
    'n_epochs': 120,
    'model_save_dir': None,
    'restore_epoch': None,
    'start_epoch': 0,
    'simple': False,
    'lr': 0.001
}

networks = {
    'sub1_basic': basic.Subnet1,
    'sub2_basic': basic.Subnet2,
    'sub3_basic': basic.Subnet3,
    'sub1_tuned': tuned.Subnet1,
    'sub2_tuned': tuned.Subnet2,
    'sub3_tuned': tuned.Subnet3
}


def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise RuntimeError(key + ' is not a known hyper parameter')
        else:
            hps[key] = value

    if hps['name'] not in networks:
        raise RuntimeError("Name possibilities are: " + ' '.join(networks.keys()))

    if hps['simple']:
        hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'] + '_simpletrain')
    else:
        hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    if hps['restore_epoch']:
        hps['start_epoch'] = int(hps['restore_epoch'])

    hps['n_epochs'] = int(hps['n_epochs'])

    hps['network'] = networks[hps['name']]

    return hps
