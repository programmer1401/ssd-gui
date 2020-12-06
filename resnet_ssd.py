import os

import torch
from torch.autograd import Variable
from torchvision.models.resnet import Bottleneck

from data import coco, voc
from layers import Detect, PriorBox, L2Norm
import torch.nn as nn
import torch.nn.functional as F


class Resnet_SSD(nn.Module):

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(Resnet_SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 2]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.resnet = base
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.Sequential(*extras)

        self.loc = nn.Sequential(*head[0])
        self.conf = nn.Sequential(*head[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply resnet
        # 首先是 7x7的卷积
        for k in range(4):
            x = self.resnet[k](x)

        # 然后是3个resblock
        for k in range(4, 6):
            x = self.resnet[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        for k in range(6, len(self.resnet)):
            x = self.resnet[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))

            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def _make_layer(inplanes, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

def resnet(block, layers):
    inplanes = 64
    conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    bn1 = nn.BatchNorm2d(64)
    relu = nn.ReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    layer1 = _make_layer(inplanes, block, 64, layers[0])
    layer2 = _make_layer(inplanes*4, block, 128, layers[1], stride=2)
    layer3 = _make_layer(inplanes*8, block, 256, layers[2], stride=2)

    resnet = nn.Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3)
    return resnet

def add_extras(cfg, i):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                     kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return nn.Sequential(*layers)


def multibox(resnet, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    resnet_source = [5, 6]

    for k, v in enumerate(resnet_source):
        loc_layers += [nn.Conv2d(resnet[v][3].conv3.out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(resnet[v][3].conv3.out_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return resnet, extra_layers, (loc_layers, conf_layers)


extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_resnet_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    base_, extras_, head_ = multibox(resnet(Bottleneck, [3, 4, 6]),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)

    return Resnet_SSD(phase, size, base_, extras_, head_, num_classes)
