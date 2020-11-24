import torch

from models import basic_subnets, tuned_subnets, ensemble, vgg
from utils.checkpoint import load_features, restore
from utils.logger import Logger

nets = {
    'sub1_basic': basic_subnets.Subnet1,
    'sub2_basic': basic_subnets.Subnet2,
    'sub3_basic': basic_subnets.Subnet3,
    'sub1_tuned': tuned_subnets.Subnet1,
    'sub2_tuned': tuned_subnets.Subnet2,
    'sub3_tuned': tuned_subnets.Subnet3,
    'ensemble': ensemble.Ensemble,
    'vgg': vgg.Vgg
}


def setup_network(hps):
    if hps['network'] != 'ensemble':
        net = nets[hps['network']]()

    else:
        if hps['subnet_type'] == 'tuned':
            sub1 = tuned_subnets.Subnet1Features()
            sub2 = tuned_subnets.Subnet2Features()
            sub3 = tuned_subnets.Subnet3Features()
        else:
            sub1 = basic_subnets.Subnet1Features()
            sub2 = basic_subnets.Subnet2Features()
            sub3 = basic_subnets.Subnet3Features()

        try:
            sub1_params = torch.load(hps['sub1_path'])['params']
            sub2_params = torch.load(hps['sub2_path'])['params']
            sub3_params = torch.load(hps['sub3_path'])['params']

            load_features(sub1, sub1_params)
            load_features(sub2, sub2_params)
            load_features(sub3, sub3_params)

        except Exception as e:
            print("Ensemble Build Failure")
            raise RuntimeError(e)

        net = nets[hps['network']](sub1, sub2, sub3)

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
