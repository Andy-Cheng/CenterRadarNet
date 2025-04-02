from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
from .. import builder
from det3d.models.losses.jde_loss import JDELoss
from det3d.models.bbox_heads.center_head import SepHead
from torch import nn
from det3d.models import build_detector


@DETECTORS.register_module
class MultiModal(nn.modules):
    def __init__(
        self,
        config
    ):
        super(MultiModal, self).__init__()
        self.model_1 = build_detector(config['models'][0].model, config['models'][0].train_cfg, config['models'][0].test_cfg)
        self.model_2 = build_detector(config['models'][1].model, config['models'][1].train_cfg, config['models'][1].test_cfg)

    def distillation_loss(self, feat_set_1, feat_set_2):
        pass

    def forward(self, example, **kwargs):
        feat_set_1 = self.model_1(example, return_feat=True, return_loss=False)
        feat_set_2 = self.model_1(example, return_feat=True, return_loss=True)



        # TODO: in the future implementation. Make a base class to subclass fusion and distillation

            
    

        