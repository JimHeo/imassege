from dcnn_module.neural_network.deeplab.decoder.deeplab_decoder import Decoder as DeepLabDecoder
from dcnn_module.neural_network.deeplab.decoder.unet_decoder import Decoder as UNetDecoder

def build_decoder(num_classes, decoder, backbone, BatchNorm, multi_loss):
    if decoder == 'deeplab':
        return DeepLabDecoder(num_classes, backbone, BatchNorm, multi_loss)
    elif decoder == 'unet':
        return UNetDecoder(num_classes, backbone, BatchNorm, multi_loss)
    else:
        raise NotImplementedError
