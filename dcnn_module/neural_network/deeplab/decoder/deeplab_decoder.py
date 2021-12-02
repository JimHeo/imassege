import torch
import torch.nn as nn
import torch.nn.functional as F
from dcnn_module.neural_network.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from collections import deque

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, multi_loss):
        super(Decoder, self).__init__()
        self.multi_loss = multi_loss
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1))
        self.head1 = nn.Conv2d(304, num_classes, kernel_size=1, stride=1, padding="same")
        self.head2 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding="same")
        self._init_weight()


    def forward(self, x, low_level_feat):
        out = deque()
        low_level_features = low_level_feat[-2]
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        if self.multi_loss: out.appendleft(self.head1(x))
        x = self.last_conv(x)
        x = self.head2(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        if self.multi_loss:
            out.appendleft(x)
            return out
        else:
            return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()