# import os
#
# hps = {
#     'type': 'ensemble',         # ensemble vs subnet
#     'name': None,               # must be ensemble, sub*_basic, sub*_tuned (where * can be 1, 2, 3)
#     'subnet_type': 'tuned',     # tuned vs basic
#     'n_epochs': 300,
#     'model_save_dir': None,
#     'restore_epoch': None,      # if you want to continue training from a specific saved point
#     'start_epoch': 0,
#     'lr': 0.01,                 # starting learning rate
#     'save_freq': 20,            # how often to create checkpoints
#     'simple': False,            # simpletrain.py vs train.py
#
#     # the following parameters are only applicable if type=ensemble
#
#     'sub1': None,               # tuned vs basic
#     'sub2': None,               # tuned vs basic
#     'sub3': None,               # tuned vs basic
#     'sub1_path': None,          # where to load sub1 features from
#     'sub2_path': None,          # where to load sub2 features from
#     'sub3_path': None,          # where to load sub3 features from
# }
#
#
# def setup_hparams(args):
#     for arg in args:
#         key, value = arg.split('=')
#         if key not in hps:
#             raise ValueError(key + ' is not a valid hyper parameter')
#         else:
#             hps[key] = value
#
#     if hps['simple']:
#         hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'] + '_simple')
#     else:
#         hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])
#
#     if hps['type'] == 'ensemble':
#         if hps['subnet_type'] != 'tuned' and hps['subnet_type'] != 'basic':
#             raise ValueError(hps['subnet_type'] + " is not a supported subnet type")
#
#     elif hps['type'] != 'subnet':
#         raise ValueError(hps['type'] + " is not a supported network")
#
#     # create model save dir if not present
#     if not os.path.exists(hps['model_save_dir']):
#         os.makedirs(hps['model_save_dir'])
#
#     if hps['restore_epoch']:
#         hps['start_epoch'] = int(hps['restore_epoch'])
#
#     hps['n_epochs'] = int(hps['n_epochs'])
#     if hps['n_epochs'] < 20:
#         hps['save_freq'] = min(5, hps['n_epochs'])
#
#     return hps
#
#
#
