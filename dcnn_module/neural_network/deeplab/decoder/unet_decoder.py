import torch
import torch.nn as nn
from dcnn_module.neural_network.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from collections import deque

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, multi_loss):
        super(Decoder, self).__init__()
        self.multi_loss = multi_loss
        if backbone == 'resnet':
            low_level_inplanes = [64, 64, 256, 512]
        else:
            raise NotImplementedError
        
        self.bilinear1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, low_level_inplanes[-1], 1, 1, padding="same", bias=False),
            BatchNorm(low_level_inplanes[-1]),
            nn.ReLU()
        )
        self.block1 = Block(2 * low_level_inplanes[-1], low_level_inplanes[-1])
        self.bilinear2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(low_level_inplanes[-1], low_level_inplanes[-2], 1, 1, padding="same", bias=False),
            BatchNorm(low_level_inplanes[-2]),
            nn.ReLU()
        )
        self.block2 = Block(2 * low_level_inplanes[-2], low_level_inplanes[-2])
        self.bilinear3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(low_level_inplanes[-2], low_level_inplanes[-3], 1, 1, padding="same", bias=False),
            BatchNorm(low_level_inplanes[-3]),
            nn.ReLU()
        )
        self.block3 = Block(2 * low_level_inplanes[-3], low_level_inplanes[-3])
        self.bilinear4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(low_level_inplanes[-3], low_level_inplanes[-4], 1, 1, padding="same", bias=False),
            BatchNorm(low_level_inplanes[-4]),
            nn.ReLU()
        )
        self.block4 = Block(2 * low_level_inplanes[-4], low_level_inplanes[-4])
        self.head1 = nn.Conv2d(low_level_inplanes[-1], num_classes, 1, padding="same")
        self.head2 = nn.Conv2d(low_level_inplanes[-2], num_classes, 1, padding="same")
        self.head3 = nn.Conv2d(low_level_inplanes[-3], num_classes, 1, padding="same")
        self.head4 = nn.Conv2d(low_level_inplanes[-4], num_classes, 1, padding="same")

        self._init_weight()

    def forward(self, x, low_level_feat):
        out = deque()
        x = self.bilinear1(x)
        x = torch.cat((x, low_level_feat[-1]), dim=1)
        x = self.block1(x)
        if self.multi_loss: out.appendleft(self.head1(x))
        x = self.bilinear2(x)
        x = torch.cat((x, low_level_feat[-2]), dim=1)
        x = self.block2(x)
        if self.multi_loss: out.appendleft(self.head2(x))
        x = self.bilinear3(x)
        x = torch.cat((x, low_level_feat[-3]), dim=1)
        x = self.block3(x)
        if self.multi_loss: out.appendleft(self.head3(x))
        x = self.bilinear4(x)
        x = torch.cat((x, low_level_feat[-4]), dim=1)
        x = self.block4(x)
        x = self.head4(x)
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
                
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        # store the convolution and RELU layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels + out_channels, out_channels, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.pointwise = nn.Conv2d(2 * (in_channels + out_channels), out_channels, 1, padding="same")
  
    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.relu1(conv1)
        conv1 = torch.cat([x, conv1], dim=1)
        conv2 = self.conv2(conv1)
        conv2 = self.bn2(conv2)
        conv2 = self.relu2(conv2)
        x = torch.cat([x, conv1, conv2], dim=1)
        x = self.pointwise(x)
        return x