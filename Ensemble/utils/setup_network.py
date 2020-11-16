import torch

from ensemble.ensemble import Ensemble
from subnets import basic, tuned
from utils.checkpoint import load_features, restore
from utils.logger import Logger

nets = {
    'sub1_basic': basic.Subnet1,
    'sub2_basic': basic.Subnet2,
    'sub3_basic': basic.Subnet3,
    'sub1_tuned': tuned.Subnet1,
    'sub2_tuned': tuned.Subnet2,
    'sub3_tuned': tuned.Subnet3
}


def build_network(hps):
    if hps['type'] == 'subnet':
        if hps['name'] not in nets:
            raise ValueError("Name possibilities are: " + ' '.join(nets.keys()))
        net = nets[hps['name']]()

    else:
        if hps['subnet_type'] == 'tuned':
            sub1 = tuned.Subnet1Features()
            sub2 = tuned.Subnet2Features()
            sub3 = tuned.Subnet3Features()
        else:
            sub1 = basic.Subnet1Features()
            sub2 = basic.Subnet2Features()
            sub3 = basic.Subnet3Features()

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

        net = Ensemble(sub1, sub2, sub3)

    # Prepare network
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
