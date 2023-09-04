import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import BACKBONES
from .hr_util import *
from .. import builder

@BACKBONES.register_module
class HRNet3D(nn.Module):

    def __init__(self, backbone_cfg='hrnet3D', feat_transform=None, **kwargs):
        super(HRNet3D, self).__init__()
        self.backbone = HRNetBackbone(backbone_cfg)()
        if kwargs['final_conv_in'] == kwargs['final_conv_out']:
            self.final_conv = nn.Identity()
        else:
            self.final_conv = nn.Conv3d(kwargs['final_conv_in'], kwargs['final_conv_out'], kernel_size=1)
        
        self.final_fuse = kwargs['final_fuse']
        self.backbone_cfg = backbone_cfg
        self.with_feat_transform = False
        if feat_transform is not None:
            self.feat_transform = builder.build_feat_transform(feat_transform)
            self.with_feat_transform = True

    def forward(self, x_):
        # 2D HRNet
        if self.backbone_cfg == 'hrnet':
            x = self.backbone(x_)
            if self.with_feat_transform:
                x = self.feat_transform(x)
            _, _, h, w = x[0].size()
            feats = x[0] # pick the highest resolution feature
            return feats
        
        # 3D HRNet
        x = self.backbone(x_)
        _, _, h, w, l = x[0].size() # z y x size of a driving sene
        feat1 = x[0]
        if self.final_fuse == 'top':
            feats = feat1
            feats = self.final_conv(feats)
        else:
            feat2 = F.interpolate(x[1], size=(h, w, l), mode="trilinear", align_corners=True)
            feat3 = F.interpolate(x[2], size=(h, w, l), mode="trilinear", align_corners=True)
            feats = torch.cat([feat1, feat2, feat3], 1)
            if self.final_fuse == 'conat_conv':
                feats = self.final_conv(feats)
        # Feature transform. 2D BEV or 3D
        if self.with_feat_transform:
            if self.feat_transform.transform_dim == '2':
                N, C, D, H, W = feats.shape
                feats = feats.view(N, C * D, H, W)
                feats = self.feat_transform(feats)
            else:
                feats = self.feat_transform(feats)
                N, C, D, H, W = feats.shape
                feats = feats.view(N, C * D, H, W)
        else:
            N, C, D, H, W = feats.shape
            feats = feats.view(N, C * D, H, W)
        return feats
    
if __name__ == '__main__':
    # test models with dummy data
    hrnet3d = HRNet3D()
    dummy_radar = torch.rand(1 ,1, 24, 32, 176)
    out = hrnet3d(dummy_radar)
    print(out.shape)
