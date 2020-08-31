"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""

import torch, os
import numpy as np
import torch.nn as nn
import pretrainedmodels as ptm
from torch.autograd import Variable

import pretrainedmodels.utils as utils
import torchvision.models as models

def initialize_weights(model):
    for idx,module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(0,0.01)
            module.bias.data.zero_()

"""============================================================="""
class NetworkSuperClass(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass, self).__init__()
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__[opt.arch](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__[opt.arch](num_classes=1000, pretrained=None)
        self.input_space, self.input_range, self.mean, self.std = self.model.input_space, self.model.input_range, self.model.mean, self.model.std

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)

        self.shared_norm = opt.shared_norm

        if self.shared_norm:
            self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        else:
            self.model.last_linear_class = torch.nn.Linear(self.model.last_linear.in_features, opt.classembed)
            self.model.last_linear_res = torch.nn.Linear(self.model.last_linear.in_features, opt.intraclassembed)
            self.model.last_linear = None

    def forward(self, x, feat='embed'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        x_class = self.model.last_linear_class(x)
        x_class = torch.nn.functional.normalize(x_class, dim=-1)

        x_res = self.model.last_linear_res(x)
        x_res = torch.nn.functional.normalize(x_res, dim=-1)

        x = torch.cat([x_class, x_res], dim=1)

        return x


class NetworkSuperClass_baseline(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_baseline, self).__init__()
        if not opt.not_pretrained:
            print('Getting pretrained weights...')
            self.model = ptm.__dict__[opt.arch](num_classes=1000, pretrained='imagenet')
            print('Done.')
        else:
            print('Not utilizing pretrained weights!')
            self.model = ptm.__dict__[opt.arch](num_classes=1000, pretrained=None)
        self.input_space, self.input_range, self.mean, self.std = self.model.input_space, self.model.input_range, self.model.mean, self.model.std

        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

    def forward(self, x, feat='embed'):
        x = self.model(x)
        x = torch.nn.functional.normalize(x,dim=-1)

        return x