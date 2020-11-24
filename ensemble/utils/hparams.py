import os

hps = {
    'network': '',              # ensemble vs subnet
    'name': '',                 # network name
    'n_epochs': 300,
    'model_save_dir': None,     # where will checkpoints be stored (path created automatically using hps[name])
    'restore_epoch': None,      # continue training from a specific saved point
    'start_epoch': 0,
    'lr': 0.01,                 # starting learning rate
    'save_freq': 20,            # how often to create checkpoints

    # the following parameters are only applicable if network=ensemble
    'subnet_type': 'tuned',     # tuned vs. basic
    'sub1_path': None,          # where to load sub1 features
    'sub2_path': None,          # where to load sub2 features
    'sub3_path': None,          # where to load sub3 features
}

possible_nets = {
    'ensemble',
    'sub1_tuned', 'sub1_basic',
    'sub2_tuned', 'sub2_basic',
    'sub3_tuned', 'sub3_basic'
}


def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    # Invalid network check
    if hps['network'] not in possible_nets:
        raise ValueError("Invalid network.\nPossible ones include:" + '\n - '.join(possible_nets))

    if hps['subnet_type'] not in {'tuned', 'basic'}:
        raise ValueError("Invalid subnet type.\nPossible ones include:" + '\n - tuned\n - basic')

    # invalid parameter check
    try:
        hps['n_epochs'] = int(hps['n_epochs'])
        hps['start_epoch'] = int(hps['start_epoch'])
        hps['save_freq'] = int(hps['save_freq'])
        hps['lr'] = float(hps['lr'])

        if hps['restore_epoch']:
            hps['restore_epoch'] = int(hps['restore_epoch'])
            hps['start_epoch'] = int(hps['restore_epoch'])

        # make sure we can checkpoint regularly or at least once (at the end)
        if hps['n_epochs'] < 20:
            hps['save_freq'] = min(5, hps['n_epochs'])

    except Exception as e:
        raise ValueError("Invalid input parameters")

    # create checkpoint directory
    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    return hps
