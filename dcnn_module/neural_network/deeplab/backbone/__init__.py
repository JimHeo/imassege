from dcnn_module.neural_network.deeplab.backbone import resnet, xception, drn, mobilenet

def build_backbone(input_channels, backbone, output_stride, BatchNorm, pretrained):
    if backbone == 'resnet':
        return resnet.ResNet101(input_channels, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'xception':
        return xception.AlignedXception(input_channels, output_stride, BatchNorm, pretrained=pretrained)
    elif backbone == 'drn':
        return drn.drn_d_54(input_channels, BatchNorm, pretrained=pretrained)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(input_channels, output_stride, BatchNorm, pretrained=pretrained)
    else:
        raise NotImplementedError
