import itertools
import logging
from munch import DefaultMunch
from det3d.utils.config_tool import get_downsample_factor
from math import ceil

tasks = [
    dict(num_class=1, class_names=["Car"]),
    dict(num_class=1, class_names=["BusorTruck"])
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

BATCH_SIZE=1

JDE=dict(
  enable=False,
  max_frame_length=5,
  repeat_frames=True,
  embedding_dim=32,
  weight=1.0,
  distance_cfg=dict(
    type='LpDistance',
    p=2, 
    power=1,
  ),
  loss_fcn_cfg=dict(
    type='TripletMarginLoss',
    margin=0.05,
    swap=False,
    smooth_loss=False,
    triplets_per_anchor="all"
  ),
  miner_cfg=dict(
    type='BatchEasyHardMiner',
    pos_strategy='all',
    neg_strategy='all'
  ),
  reducer_cfg=dict(
    type='AvgNonZeroReducer'
  ),
  emb_head_cfg=dict(
    head={'emb': (32, 2)}, # emb_feat_size, num of conv,
    share_conv_channel=64,
    tasks=tasks
  )
)

DATASET = dict(
  DIR=dict(
    DATA_ROOT='/mnt/ssd3/CRUW3D/seqs',
    RAD_DIR='rad_cart_mag',
    RDR_CALIB='/mnt/ssd3/CRUW3D/calib/calib_radar_lidar.json',
    LABEL_FILE='/mnt/ssd3/CRUW3D/labels/CRUW3DCarTruck.json',
    START_END_FILE=''
  ),
  TYPE_COORD= 1, # 1: Radar, 2: Lidar, 3: Camera
  LABEL= dict(
    IS_CONSIDER_ROI=True,
    ROI_TYPE='roi1',
    ROI_DEFAULT=[0,29,-10,10,-5,10], # x_min_max, y_min_max, z_min_max / Dim: [m]
    IS_CHECK_VALID_WITH_AZIMUTH=True,
    MAX_AZIMUTH_DEGREE=[-60, 60],
    CONSIDER_RADAR_VISIBILITY=False,
  ),
  ROI = dict( # used for radar cube cropping and feature map size
    roi1 = {'y': [-10.65, 10.95], 'x': [0, 30.00]}
  ),
  LABEL_ROI=dict(
    roi1=[0, -10,  -5, 25, 10, 10] # [xyz_min, xyz_max]
  ),
  RDR_CUBE = dict(
      IS_CONSIDER_ROI=True,
      GRID_SIZE=0.15, # [m],
      NORMALIZING_VALUE=1e+7 # 'fixed' # todo: try other normalization strategies
  ),
  DEAR = dict(
    REDUCE_TYPE='none', # 'none', 'avg', 'max'
    NORMALIZING_VALUE=1e+18,
    IS_CONSIDER_ROI=True,
    ROI={'r': [0., 118.03710938], 'a': [-52, 52], 'e': [-17, 19]}
  ),
  CLASS_INFO=dict(
    # If containing cls, make the ID as number
    # In this case, we consider ['Sedan', 'Bus or Truck'] as Sedan (Car)
    CLASS_ID={
      'Car': 1,
      'BusorTruck':2,
      'Motorcycle': -1,
      'Bicycle': -1,
      'BicycleGroup': -1,
      'Pedestrian': -1,
      'PedestrianGroup': -1,
      'Background': 0,
    },
    NUM_CLS= None,# automatically consider, just make this blank (not including background)
    ),
  
    # List of items to be returned by the dataloader
    GET_ITEM= {
      'rdr_sparse_cube'   : False,
      'rdr_tesseract'     : False,
      'rdr_cube'          : True,
      'rdr_cube_doppler'  : False,
      'ldr_pc_64'         : False,
      'cam_front_img'     : False,
    },
    LIST_BAD = [], # seq not used in training,
    LIST_TARGET=[],
    # DEAR_BUFFER_SIZE=6*BATCH_SIZE # dear loading buffer size per worker process
  )

hr_final_conv_out = 128

# model settings
model = dict(
    type="RadarNetSingleStage",
    jde_cfg=JDE,
    pretrained=None,
    reader=dict(
        type='RadarFeatureNet',
    ),
    backbone=dict(
        type="HRNet3D",
        backbone_cfg='hrnet2d_1',
        final_conv_in = 256,
        final_conv_out = hr_final_conv_out,
        final_fuse = 'top',
        ds_factor=1,
    ),

    # todo: modify neck and head config 
    neck=None,
    bbox_head=dict(
        type="CenterHead",
        in_channels=hr_final_conv_out,
        tasks=tasks,
        dataset='kradar',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # weight of loss from common_heads
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (num output feat maps, )
        share_conv_channel=64,
        dcn_head=False
    ),
)

# todo: modify gussian map params
assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=1, # TODO: check this
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=10,
    min_radius=2,
    consider_radar_visibility=DATASET['LABEL']['CONSIDER_RADAR_VISIBILITY'],
    radar_visibility_cfg=dict(bin=[20, 60, 100], mod_coeff=[0.7, 0.8, 0.9, 1.0]), # bin means num of poitns
    expand_channel_dim=False # expand an axis if no doppler axis is in the radar tensor
)


train_cfg = dict(assigner=assigner)


test_cfg_range = DATASET['ROI'][DATASET['LABEL']['ROI_TYPE']]
test_cfg_range_label = DATASET['LABEL_ROI'][DATASET['LABEL']['ROI_TYPE']]
test_cfg = dict(
    post_center_limit_range=test_cfg_range_label, # [x_min, -y, -z, x_max, y, z] RoI
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=100, # select first nms_pre_max_size numnber of bbox to do nms
        nms_post_max_size=25, # select nms_post_max_size bbox after nms
        nms_iou_threshold=0.0,
    ),
    score_threshold=0.1,
    pc_range=[test_cfg_range['x'][0], test_cfg_range['y'][0]],
    out_size_factor=1.,
    voxel_size=[0.15, 0.15],
    input_type='rdr_tensor',
    app_emb_save_path='' # inside are seq folders
)


# dataset settings
dataset_type = "CRUWDataset"


train_pipeline = [
    # dict(type="LoadRadarData"),  
    dict(type="AssignLabelRadar", cfg=train_cfg["assigner"], flip_y_prob=0.0),
]

# val_preprocessor = dict(
# )
test_pipeline = [
    # dict(type="LoadRadarData"),  
    dict(type="AssignLabelRadar", cfg=train_cfg["assigner"]),
]

jde_data_cfg = dict(enable=JDE['enable'], max_frame_length=JDE['max_frame_length'], \
                    repeat_frames=JDE['repeat_frames'])

data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET, JDE=jde_data_cfg),
        split='train',
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    test=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET, JDE=jde_data_cfg),
        split='test', # todo: change
        class_names=class_names,
        pipeline=test_pipeline,
        mode='test'
    ),
    val=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET, JDE=jde_data_cfg),
        split='train',
        class_names=class_names,
        pipeline=test_pipeline,
        mode='val'
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

cuda_device = '0'
