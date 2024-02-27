import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import BACKBONES
from .. import builder

@BACKBONES.register_module
class RODNet(nn.Module):

    def __init__(self, **kwargs):
        super(RODNet, self).__init__()
        self.backbone = RadarStackedHourglass(256, 128, stacked_num=1)


    def forward(self, x_):
        feats = self.backbone(x_)
        
        return feats


class RadarStackedHourglass(nn.Module):

    def __init__(self, in_channels, out_channels, stacked_num=1, conv_op=None):
        super(RadarStackedHourglass, self).__init__()
        self.stacked_num = stacked_num
        if conv_op is None:
            self.conv1a = nn.Conv2d(in_channels=in_channels, out_channels=128,
                                    kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        else:
            self.conv1a = conv_op(in_channels=in_channels, out_channels=32,
                                  kernel_size=(5, 3, 3), stride=(1, 1, 1), padding=(2, 1, 1))

        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))

        self.hourglass = []
        for i in range(stacked_num):
            self.hourglass.append(nn.ModuleList([RODEncode(), RODDecode(),
                                                 nn.Conv2d(in_channels=128, out_channels=out_channels,
                                                           kernel_size=(5, 5), stride=(1, 1),
                                                           padding=(2, 2)),
                                                 nn.Conv2d(in_channels=out_channels, out_channels=128,
                                                           kernel_size=(5, 5), stride=(1, 1),
                                                           padding=(2, 2))]))
        self.hourglass = nn.ModuleList(self.hourglass)
        self.relu = nn.ReLU()
        self.bn1a = nn.BatchNorm2d(num_features=128)
        self.bn1b = nn.BatchNorm2d(num_features=128)

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))

        out = []
        for i in range(self.stacked_num):
            x, x1, x2, x3 = self.hourglass[i][0](x)
            x = self.hourglass[i][1](x, x1, x2, x3)
            featmap = self.hourglass[i][2](x)
            out.append(featmap)
            if i < self.stacked_num - 1:
                featmap_ = self.hourglass[i][3](featmap)
                x = x + featmap_
        return out[-1] # assume only one hourglass or we take the featmap from the last hourglass


class RODEncode(nn.Module):

    def __init__(self):
        super(RODEncode, self).__init__()
        self.conv1a = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256,
                                kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512,
                                kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv3b = nn.Conv2d(in_channels=512, out_channels=512,
                                kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        self.skipconv1a = nn.Conv2d(in_channels=128, out_channels=128,
                                    kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.skipconv1b = nn.Conv2d(in_channels=128, out_channels=128,
                                    kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.skipconv2a = nn.Conv2d(in_channels=128, out_channels=256,
                                    kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.skipconv2b = nn.Conv2d(in_channels=256, out_channels=256,
                                    kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.skipconv3a = nn.Conv2d(in_channels=256, out_channels=512,
                                    kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.skipconv3b = nn.Conv2d(in_channels=512, out_channels=512,
                                    kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.bn1a = nn.BatchNorm2d(num_features=128)
        self.bn1b = nn.BatchNorm2d(num_features=128)
        self.bn2a = nn.BatchNorm2d(num_features=256)
        self.bn2b = nn.BatchNorm2d(num_features=256)
        self.bn3a = nn.BatchNorm2d(num_features=512)
        self.bn3b = nn.BatchNorm2d(num_features=512)

        self.skipbn1a = nn.BatchNorm2d(num_features=128)
        self.skipbn1b = nn.BatchNorm2d(num_features=128)
        self.skipbn2a = nn.BatchNorm2d(num_features=256)
        self.skipbn2b = nn.BatchNorm2d(num_features=256)
        self.skipbn3a = nn.BatchNorm2d(num_features=512)
        self.skipbn3b = nn.BatchNorm2d(num_features=512)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.skipbn1a(self.skipconv1a(x)))
        x1 = self.relu(self.skipbn1b(self.skipconv1b(x1)))
        x = self.relu(self.bn1a(self.conv1a(x)))  # (B, 2, W, 128, 128) -> (B, 64, W, 128, 128)
        x = self.relu(self.bn1b(self.conv1b(x)))  # (B, 64, W, 128, 128) -> (B, 64, W/2, 64, 64)

        x2 = self.relu(self.skipbn2a(self.skipconv2a(x)))
        x2 = self.relu(self.skipbn2b(self.skipconv2b(x2)))
        x = self.relu(self.bn2a(self.conv2a(x)))  # (B, 64, W/2, 64, 64) -> (B, 128, W/2, 64, 64)
        x = self.relu(self.bn2b(self.conv2b(x)))  # (B, 128, W/2, 64, 64) -> (B, 128, W/4, 32, 32)

        x3 = self.relu(self.skipbn3a(self.skipconv3a(x)))
        x3 = self.relu(self.skipbn3b(self.skipconv3b(x3)))
        x = self.relu(self.bn3a(self.conv3a(x)))  # (B, 128, W/4, 32, 32) -> (B, 256, W/4, 32, 32)
        x = self.relu(self.bn3b(self.conv3b(x)))  # (B, 256, W/4, 32, 32) -> (B, 256, W/4, 16, 16)

        return x, x1, x2, x3


class RODDecode(nn.Module):

    def __init__(self):
        super(RODDecode, self).__init__()
        self.convt1 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.convt3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                         kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))
        self.prelu = nn.PReLU()

    def forward(self, x, x1, x2, x3):
        x = self.prelu(self.convt1(x + x3))  # (B, 256, W/4, 16, 16) -> (B, 128, W/2, 32, 32)
        x = self.prelu(self.convt2(x + x2))  # (B, 128, W/2, 32, 32) -> (B, 64, W, 64, 64)
        x = self.convt3(x + x1)  # (B, 64, W, 64, 64) -> (B, 3, W, 128, 128)
        return x
