from nets import ResNet
import timm
from config import *


# initialize model
def get_model(model):
    if model == 'ResNet-18':
        return timm.create_model('resnet18', pretrained=True, num_classes=config['N_CLASSES'])
    elif model == 'ResNet-34':
        return timm.create_model('resnet34', pretrained=True, num_classes=config['N_CLASSES'])
    elif model == 'ResMLP-12':
        return timm.create_model('resmlp_12_224', pretrained=True, num_classes=config['N_CLASSES'])
    elif model == 'ResMLP-24':
        return timm.create_model('resmlp_24_224', pretrained=True, num_classes=config['N_CLASSES'])
    else:
        print('No required model')
        return None
