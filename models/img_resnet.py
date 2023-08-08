import torchvision
import torch
from torch import nn
from torch.nn import init
from models import pooling


class ResNet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet50.layer4[0].conv2.stride = (1, 1)
            resnet50.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        x = self.base(x)
        return x


class ResNet101(nn.Module):
    def __init__(self, config):
        super(ResNet101, self).__init__()
        resnet101 = torchvision.models.resnet101(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet101.layer4[0].conv2.stride = (1, 1)
            resnet101.layer4[0].downsample[0].stride = (1, 1)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])

    def forward(self, x):
        x = self.base(x)
        return x


class Part_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2048, out_channels=4, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x


class GAP_Classifier(nn.Module):

    def __init__(self, config, num_identities):
        super().__init__()
        self.bn = nn.BatchNorm2d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(config.MODEL.FEATURE_DIM, num_identities, kernel_size=1)
        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))
        self.classifier = nn.Linear(config.MODEL.FEATURE_DIM, num_identities)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):

        x = self.bn(x)
        x = self.conv(x)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)

        return x
