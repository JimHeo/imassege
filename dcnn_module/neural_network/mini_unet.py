# import the necessary packages
from torch import nn
import torch

class UNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=1, feature_map=(8, 16)):
        super().__init__()
        # initialize the encoder and decoder
        self.enc_feature_map = feature_map
        self.dec_feature_map = feature_map[::-1]
        self.num_classes = num_classes
        self.encoder = Encoder(input_channel, self.enc_feature_map)
        self.decoder = Decoder(self.dec_feature_map)
        # initialize the regression head and store the class variables
        self.head = nn.Conv2d(self.dec_feature_map[-1], self.num_classes, 1)
        
    def forward(self, x):
        # grab the features from the encoder
        enc_features = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        seg_map = self.head(dec_features)
        # return the segmentation map
        return seg_map


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        # store the convolution and RELU layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, input_channel, feature_map):
        super().__init__()
        channels = (input_channel, *feature_map)
        # store the encoder blocks and maxpooling layer
        self.enc_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
            )
        self.pool = nn.MaxPool2d(2)
	
    def forward(self, x):
        # initialize an empty list to store the intermediate outputs
        block_outputs = []
        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass the inputs through the current encoder block, store
            # the outputs, and then apply maxpooling on the output
            x = block(x)
            block_outputs.append(x)
            x = self.pool(x)
        # return the list containing the intermediate outputs
        return block_outputs
    
    
class Decoder(nn.Module):
    def __init__(self, feature_map):
        super().__init__()
        # initialize the number of channels, upsampler blocks, and
        # decoder blocks
        self.feature_map = feature_map
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(feature_map[i], feature_map[i + 1], 2, 2) for i in range(len(feature_map) - 1)]
            )
        self.dec_blocks = nn.ModuleList(
            [Block(feature_map[i], feature_map[i + 1]) for i in range(len(feature_map) - 1)])
        
    def forward(self, x, enc_features):
        # loop through the number of channels
        for i in range(len(self.feature_map) - 1):
            # pass the inputs through the upsampler blocks
            x = self.upconvs[i](x)
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current
            # decoder block
            x = torch.cat([x, enc_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        # return the final decoder output
        return x