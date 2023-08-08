import logging
from models.img_resnet import ResNet50, ResNet101, Part_Block

__factory = {
    'resnet50': ResNet50,
    'resnet101': ResNet101
}

def build_model(config):

    logger = logging.getLogger('reid.model')
    logger.info("Initializing model: {}".format(config.MODEL.NAME))
    if config.MODEL.NAME not in __factory.keys():
        raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
    else:
        model = __factory[config.MODEL.NAME](config)

    attention = Part_Block()

    return model, attention
