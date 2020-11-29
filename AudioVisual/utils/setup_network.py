import torch

from models import CNNs, ELM
from utils.checkpoint import restore, load_features
from utils.logger import Logger

nets = {
    'cnn2d': CNNs.CNN_2D,
    'cnn3d': CNNs.CNN_3D,
    'elm1': ELM.ELM1,
    'elm2': ELM.ELM2,
}


def setup_network(hps):
    if hps['network'] in {'cnn2d', 'cnn3d'}:
        net = nets[hps['network']]()

    else:

        cnn2d = CNNs.CNN_2DFeatures()
        cnn3d = CNNs.CNN_3DFeatures()

        cnn2d_params = torch.load(hps['cnn2d_path'])['params']
        cnn3d_params = torch.load(hps['cnn3d_path'])['params']

        load_features(cnn2d, cnn2d_params)
        load_features(cnn3d, cnn3d_params)

        if hps['network'] == 'elm2':
            elm1 = ELM.ELM1Features(input_size=8192, hidden_size=100, num_classes=2)
            elm1_params = torch.load(hps['elm1_path'])['params']
            load_features(elm1, elm1_params)

            net = ELM.ELM2(input_size=100, hidden_size=100, num_classes=6, cnn2d=cnn2d, cnn3d=cnn3d, elm1=elm1)

        elif hps['network'] == 'elm1':
            net = ELM.ELM1(input_size=8192, hidden_size=100, num_classes=2, cnn2d=cnn2d, cnn3d=cnn3d)

        else:
            elm1 = ELM.ELM1Features(input_size=8192, hidden_size=100, num_classes=2)
            elm1_params = torch.load(hps['elm1_path'])['params']
            load_features(elm1, elm1_params)

            elm2 = ELM.ELM2Features(input_size=100, hidden_size=100, num_classes=6, cnn2d=cnn2d, cnn3d=cnn3d, elm1=elm1)
            elm2_params = torch.load(hps['elm2_path'])['params']
            load_features(elm2, elm2_params)

            net = elm2

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
