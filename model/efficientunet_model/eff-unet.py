from .efficientnet import *
from .efficientunet import *
import torch.nn as nn

class Eff_unet(nn.Module):
    def __init__(self, out_channels=3, concat_input=True, pretrained=True):
        super().__init__()
        self.encoder = EfficientNet.encoder('efficientnet-b4', pretrained=pretrained)
        self.model = EfficientUnet(self.encoder, out_channels=out_channels, concat_input=concat_input)

    def forward(self, x):
        return self.model(x)
        