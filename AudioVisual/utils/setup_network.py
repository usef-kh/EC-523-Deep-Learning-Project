from models import CNNs
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'cnn2d': CNNs.CNN_2D,
    'cnn3d': CNNs.CNN_3D,
}


def setup_network(hps):
    net = nets[hps['network']]()

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
