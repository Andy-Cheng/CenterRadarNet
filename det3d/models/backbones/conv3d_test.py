
import numpy as np
import torch
from torch import nn

# from ..registry import BACKBONES
from unet3d.model import AbstractUNet
from unet3d.buildingblocks import ResNetBlock
    
# @BACKBONES.register_module
class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels=1, out_channels=8, f_maps=64, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_padding=1, name="ResidualUNet3D", **kwargs):
        super(ResidualUNet3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             basic_module=ResNetBlock,
                                             f_maps=f_maps,  # 16
                                             layer_order=layer_order, # gcr
                                             num_groups=num_groups, # 8
                                             num_levels=num_levels, # 2
                                             conv_padding=conv_padding, # 1
                                             is3d=True, 
                                             name=name)
        
    def forward(self, x):
        ret, decoder_features = super().forward(x)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        
        return ret
    
    
# @BACKBONES.register_module
class HRNet3D(nn.Module):
    """
    HRNet3D model
    """
    def __init__(self, **kwargs):
        super(HRNet3D, self).__init__()
        
    def forward(self, x):
        ret = super().forward(x)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        
        return ret, None
    
    
class HighResolutionModule(nn.Module):
    def __init__(self, ):
        super(HighResolutionModule, self).__init__()


    def forward(self, x):
        pass

class HRNet3D(nn.Module):
    def __init__(self, in_channels):
        super(HRNet3D, self).__init__()



    def forward(self, x):
        pass
    
if __name__ == '__main__':
    # test models with dummy data
    hrnet3d = HRNet3D(1)
    data = torch.rand(1, 1, 24, 32, 176)
    out = hrnet3d(data)
    print(out.shape)
