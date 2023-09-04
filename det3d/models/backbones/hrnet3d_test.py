import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..registry import BACKBONES


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(
        inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(
            planes, momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
        inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(
            planes, momentum=bn_momentum
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, bn_type=None, bn_momentum=0.1
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(
            planes, momentum=bn_momentum
        )
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(
            planes, momentum=bn_momentum
        )
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(
            planes * 4, momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=False)
        self.relu_in = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_in(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            bn_type=bn_type,
            bn_momentum=bn_momentum,
        )
        self.fuse_layers = self._make_fuse_layers(
            bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.relu = nn.ReLU(inplace=False)

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            print(error_msg) # TODO: change to logger
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            print(error_msg) # TODO: change to logger
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            print(error_msg) # TODO: change to logger
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        stride=1,
        bn_type=None,
        bn_momentum=0.1,
    ):
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(bn_type=bn_type)(
                    num_channels[branch_index] * block.expansion, momentum=bn_momentum
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self, num_branches, block, num_blocks, num_channels, bn_type, bn_momentum=0.1
    ):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    bn_type=bn_type,
                    bn_momentum=bn_momentum,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self, bn_type, bn_momentum=0.1):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv3d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                num_inchannels[i], momentum=bn_momentum
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm3d(
                                        num_outchannels_conv3x3, momentum=bn_momentum
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv3d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm3d(
                                        num_outchannels_conv3x3, momentum=bn_momentum
                                    ),
                                    nn.ReLU(inplace=False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="trilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolution3DNet(nn.Module):
    def __init__(self, cfg, bn_type, bn_momentum, **kwargs):
        super(HighResolution3DNet, self).__init__()
        self.full_res_stem = False
        if kwargs['full_res_stem']:
            print("using full-resolution stem with stride=1") # TODO: change to logging
            stem_stride = 1
            self.conv1 = nn.Conv3d(
                1, 16, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm3d(
                16, momentum=bn_momentum
            )
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(
                Bottleneck, 16, 16, 4, bn_type=bn_type, bn_momentum=bn_momentum
            )
            self.full_res_stem = True
        else:
            stem_stride = 2
            self.conv1 = nn.Conv3d(
                3, 16, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm3d(
                16, momentum=bn_momentum
            )
            self.conv2 = nn.Conv3d(
                16, 16, kernel_size=3, stride=stem_stride, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm3d(
                16, momentum=bn_momentum
            )
            self.relu = nn.ReLU(inplace=False)
            self.layer1 = self._make_layer(
                Bottleneck, 16, 16, 4, bn_type=bn_type, bn_momentum=bn_momentum
            )

        self.stage2_cfg = cfg["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]

        self.transition1 = self._make_transition_layer(
            [64], num_channels, bn_type=bn_type, bn_momentum=bn_momentum # TODO: 64 is hardcoded
        )

        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )

        self.stage3_cfg = cfg["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )

        self.stage4_cfg = cfg["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, bn_type=bn_type, bn_momentum=bn_momentum
        )

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=True,
            bn_type=bn_type,
            bn_momentum=bn_momentum,
        )

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer, bn_type, bn_momentum
    ):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv3d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm3d(
                                num_channels_cur_layer[i], momentum=bn_momentum
                            ),
                            nn.ReLU(inplace=False),
                        )
                    )
                else:3
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv3d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm3d(
                                outchannels, momentum=bn_momentum
                            ),
                            nn.ReLU(inplace=False),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self, block, inplanes, planes, blocks, stride=1, bn_type=None, bn_momentum=0.1
    ):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(
                    planes * block.expansion, momentum=bn_momentum
                ),
            )

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample,
                bn_type=bn_type,
                bn_momentum=bn_momentum,
            )
        )

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(inplanes, planes, bn_type=bn_type, bn_momentum=bn_momentum)
            )

        return nn.Sequential(*layers)

    def _make_stage(
        self,
        layer_config,
        num_inchannels,
        multi_scale_output=True,
        bn_type=None,
        bn_momentum=0.1,
    ):
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                    bn_type,
                    bn_momentum,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):

        if self.full_res_stem:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

        x = self.layer1(x)
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

class HRNet3DBackbone(object):
    def __init__(self, backbone_cfg):
        self.backbone_cfg = backbone_cfg

    def __call__(self):
        from hrnet3D_config import MODEL_CONFIGS
        if self.backbone_cfg == "hrnet3D":
            arch_net = HighResolution3DNet(
                MODEL_CONFIGS["hrnet3D"], bn_type="torchsyncbn", bn_momentum=0.1, full_res_stem=True
            )
        else:
            raise Exception("Architecture undefined!")

        return arch_net


# @BACKBONES.register_module
class HRNet3D(nn.Module):

    def __init__(self, backbone_cfg='hrnet3D', **kwargs):
        super(HRNet3D, self).__init__()
        self.backbone = HRNet3DBackbone(backbone_cfg)()

        # TODO: implement the seg layer for pretraining on occupancy task
        # self.num_classes = self.configer.get("data", "num_classes")


    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w, l = x[0].size() # z y x size of a driving sene

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w, l), mode="trilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w, l), mode="trilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w, l), mode="trilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        # TODO: implement the seg layer for pretraining on occupancy task
        N, C, D, H, W = feats.shape
        feats = feats.view(N, C * D, H, W)
        return feats
    
if __name__ == '__main__':
    # test models with dummy data
    hrnet3d = HRNet3D()
    dummy_radar = torch.rand(1 ,1, 24, 32, 176)
    out = hrnet3d(dummy_radar)
    print(out.shape)
