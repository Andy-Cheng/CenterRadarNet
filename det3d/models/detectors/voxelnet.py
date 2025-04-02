from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
from det3d.models.losses.jde_loss import JDELoss
from det3d.models.bbox_heads.center_head import SepHead
from torch import nn
from det3d.utils.vis_util import draw_bev

vis_bev = True

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        jde_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        if jde_cfg and jde_cfg.pop('enable', False):
            self.jde_loss = JDELoss(**jde_cfg)
            if 'weight_cfg' in jde_cfg:
                self.jde_weight = jde_cfg.weight_cfg.initial_weight
                self.jde_weight_steps = jde_cfg.weight_cfg.steps
                self.jde_weight_rate = jde_cfg.weight_cfg.rate
            else:
                self.jde_weight = jde_cfg.weight
                self.jde_weight_steps = []
                self.jde_weight_rate = 1.0

            emb_head_cfg = jde_cfg.emb_head_cfg
            self.num_classes = [len(t["class_names"]) for t in emb_head_cfg.tasks]
            self.class_names = [t["class_names"] for t in emb_head_cfg.tasks]
            self.tasks = nn.ModuleList()
            for _ in self.num_classes:
                self.tasks.append(SepHead(emb_head_cfg.share_conv_channel, emb_head_cfg.head, bn=True, final_kernel=3))
        else:
            self.jde_loss = None
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_feature = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        x, _ = self.extract_feat(example)
        if vis_bev:
            draw_bev(x[0], f"./bev_vis_lidar/{example['meta'][0]['seq']}_{example['meta'][0]['frame']}.png")
        preds, _ = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 