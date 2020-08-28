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


def mixup_process(out, mix_indices, lam):
    out = out * lam + out[mix_indices] * (1 - lam)

    return out


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam



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

        self.shared_norm = opt.shared_norm

        if self.shared_norm:
            self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)
        else:
            self.model.last_linear_class = torch.nn.Linear(self.model.last_linear.in_features, opt.classembed)
            self.model.last_linear_res = torch.nn.Linear(self.model.last_linear.in_features, opt.intraclassembed)
            self.model.last_linear = None

    def forward(self, x, feat='embed'):
        if not feat == 'embed':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x2 = x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x2= torch.nn.functional.avg_pool2d(x2, kernel_size=12, stride=6).view(x2.size(0),-1)

            if feat in ['lay2', 'lay2_norm']:
                x = x2
            elif feat in ['lay24', 'lay24_norm']:
                x = torch.cat([x, x2], dim=1)
            elif feat in ['lay4']:
                x = x

            if 'norm' in feat:
                x = torch.nn.functional.normalize(x, dim=-1)

        else:
            if self.shared_norm:
                x = self.model(x)
                x = torch.nn.functional.normalize(x,dim=-1)
            else:
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
        if not feat == 'embed':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x2 = x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x2= torch.nn.functional.avg_pool2d(x2, kernel_size=12, stride=6).view(x2.size(0),-1)

            if feat in ['lay2', 'lay2_norm']:
                x = x2
            elif feat in ['lay24', 'lay24_norm']:
                x = torch.cat([x, x2], dim=1)
            elif feat in ['lay4']:
                x = x

            if 'norm' in feat:
                x = torch.nn.functional.normalize(x, dim=-1)

        else:
            x = self.model(x)
            x = torch.nn.functional.normalize(x,dim=-1)

        return x


class NetworkSuperClass_mixup(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_mixup, self).__init__()
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

        self.mixup_alpha = opt.mixup_alpha
        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

    def forward(self, x, feat='embed', mixup_hidden=None, mix_indices=None):
        if not feat == 'embed':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x2 = x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x2= torch.nn.functional.avg_pool2d(x2, kernel_size=12, stride=6).view(x2.size(0),-1)

            if feat in ['lay2', 'lay2_norm']:
                x = x2
            elif feat in ['lay24', 'lay24_norm']:
                x = torch.cat([x, x2], dim=1)
            elif feat in ['lay4']:
                x = x

            if 'norm' in feat:
                x = torch.nn.functional.normalize(x, dim=-1)

        else:

            layer_mix = -1
            if mixup_hidden  is not None:
                # choose mixing layer
                layer_mix = np.random.choice(mixup_hidden, size=1)

                # sample mixing paramter
                lam = get_lambda(self.mixup_alpha)
                lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
                lam = Variable(lam)

            if layer_mix == 0: # mix layer 0
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)

            if layer_mix == 1: # mix layer 1
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer2(x)

            if layer_mix == 2: # mix layer 2
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer3(x)

            if layer_mix == 3: # mix layer 3
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer4(x)

            if layer_mix == 4: # mix layer 4
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.model.last_linear(x)
            x = torch.nn.functional.normalize(x,dim=-1)

        if mixup_hidden is not None:
            return x, lam
        else:
            return x


class NetworkSuperClass_mixup_twohead(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_mixup_twohead, self).__init__()
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

        self.mixup_alpha = opt.mixup_alpha
        self.model.last_linear_class = torch.nn.Linear(self.model.last_linear.in_features, opt.classembed)
        self.model.last_linear_res = torch.nn.Linear(self.model.last_linear.in_features, opt.intraclassembed)
        self.model.last_linear = None

    def forward(self, x, feat='embed', mixup_hidden=None, mix_indices=None):
        if not feat == 'embed':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x2 = x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            x2= torch.nn.functional.avg_pool2d(x2, kernel_size=12, stride=6).view(x2.size(0),-1)

            if feat in ['lay2', 'lay2_norm']:
                x = x2
            elif feat in ['lay24', 'lay24_norm']:
                x = torch.cat([x, x2], dim=1)
            elif feat in ['lay4']:
                x = x

            if 'norm' in feat:
                x = torch.nn.functional.normalize(x, dim=-1)

        else:

            layer_mix = -1
            if mixup_hidden  is not None:
                # choose mixing layer
                layer_mix = np.random.choice(mixup_hidden, size=1)

                # sample mixing paramter
                lam = get_lambda(self.mixup_alpha)
                lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
                lam = Variable(lam)

            if layer_mix == 0: # mix layer 0
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)

            if layer_mix == 1: # mix layer 1
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer2(x)

            if layer_mix == 2: # mix layer 2
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer3(x)

            if layer_mix == 3: # mix layer 3
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.layer4(x)

            if layer_mix == 4: # mix layer 4
                x = mixup_process(x, mix_indices, lam=lam)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)

            x_class = self.model.last_linear_class(x)
            x_class = torch.nn.functional.normalize(x_class, dim=-1)

            x_res = self.model.last_linear_res(x)
            x_res = torch.nn.functional.normalize(x_res, dim=-1)

            x = torch.cat([x_class, x_res], dim=1)

        if mixup_hidden is not None:
            return x, lam
        else:
            return x


class NetworkSuperClass_rotation(nn.Module):
    def __init__(self, opt):
        super(NetworkSuperClass_rotation, self).__init__()
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

        self.model.last_linear_dml = torch.nn.Linear(self.model.last_linear.in_features, opt.classembed)
        self.model.last_linear_ss = torch.nn.Linear(self.model.last_linear.in_features, opt.n_rots)
        self.model.last_linear = None

    def forward(self, x, feat='embed'):
        if feat is not 'embed':
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

            x = self.model.last_linear_ss(x)

        else:
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

            x = self.model.last_linear_dml(x)
            x = torch.nn.functional.normalize(x, dim=-1)

        return x