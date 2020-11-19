import os

hps = {
    'type': 'ensemble',
    'name': None,
    'subnet_type': 'tuned',
    'sub1_path': None,
    'sub1': None,
    'sub2_path': None,
    'sub2': None,
    'sub3_path': None,
    'sub3': None,
    'n_epochs': 300,
    'model_save_dir': None,
    'restore_epoch': None,
    'start_epoch': 0,
    'lr': 0.01,
    'save_freq': 20,
}


def setup_hparams(args):

    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints/ckplus', hps['name'])

    if hps['type'] == 'ensemble':
        if hps['subnet_type'] != 'tuned' and hps['subnet_type'] != 'basic':
            raise ValueError(hps['subnet_type'] + " is not a supported subnet type")
    elif hps['type'] != 'subnet':
        raise ValueError(hps['type'] + " is not a supported network")

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    if hps['restore_epoch']:
        hps['start_epoch'] = int(hps['restore_epoch'])

    hps['n_epochs'] = int(hps['n_epochs'])
    if hps['n_epochs'] < 20:
        hps['save_freq'] = min(5, hps['n_epochs'])

    return hps
