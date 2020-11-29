import torch

from models import CNNs, models
from utils.checkpoint import restore, load_features
from utils.logger import Logger

nets = {
    'cnn2d': CNNs.CNN_2D,
    'cnn3d': CNNs.CNN_3D,
    'elm1': models.GenderFeatureExtractor,
    'elm2': models.FeatureExtractor,
}


def setup_network(hps):
    if hps['network'] in {'cnn2d', 'cnn3d'}:
        net = nets[hps['network']]()

    else:

        cnn2d = CNNs.CNN_2DFeatures()
        cnn3d = CNNs.CNN_3DFeatures()

        if hps['network'] == 'elm1':
            cnn2d_params = torch.load(hps['cnn2d_path'])['params']
            cnn3d_params = torch.load(hps['cnn3d_path'])['params']

            load_features(cnn2d, cnn2d_params)
            load_features(cnn3d, cnn3d_params)

            net = models.GenderFeatureExtractor(2, cnn2d, cnn3d)

        elif hps['network'] == 'elm2':
            gfe = models.GenderFeatureExtractor(2, cnn2d, cnn3d)

            gfe_params = torch.load(hps['elm1_path'])['params']
            load_features(gfe, gfe_params)

            net = models.FeatureExtractor(6, gfe)

        elif hps['network'] == 'svm':
            gfe = models.GenderFeatureExtractor(2, cnn2d, cnn3d)
            net = models.FeatureExtractor(6, gfe)

            net_params = torch.load(hps['elm2_path'])['params']
            load_features(net, net_params)

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
