import torch
import torch.nn as nn
from dcnn_module.neural_network.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from dcnn_module.neural_network.deeplab.aspp import build_aspp
from dcnn_module.neural_network.deeplab.decoder import build_decoder
from dcnn_module.neural_network.deeplab.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='xception', decoder='deeplab',output_stride=16, input_channels=3, num_classes=1, sync_bn=False, freeze_bn=False, pretrained=False, multi_loss=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone_str = backbone
        self.backbone = build_backbone(input_channels, backbone, output_stride, BatchNorm, pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder_str = decoder
        self.decoder = build_decoder(num_classes, decoder, backbone, BatchNorm, multi_loss)

        self.freeze_bn = freeze_bn
        
    def __str__(self):
        return "DeepLab_" + self.backbone_str + "_" + self.decoder_str

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16, input_channels=3)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


