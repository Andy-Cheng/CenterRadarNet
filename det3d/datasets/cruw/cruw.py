import os
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from tqdm import tqdm
import numpy as np
from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from munch import DefaultMunch
import collections
import json
from cruw import CRUW
import math
# from det3d.datasets.cruw import CRUW3DEvaluator


@DATASETS.register_module
class CRUWDataset(Dataset):
    def __init__(self, cfg, split, class_names=None, pipeline=None, mode='train'):
        super().__init__()
        cfg = DefaultMunch.fromDict(cfg)
        self.debug = False
        self.class_names = class_names
        self.cfg = cfg
        self.enable_jde = getattr(cfg.JDE, 'enable', False)
        self.cfg.update(class_names=class_names)
        self.split = split
        if self.enable_jde:
            self.enable_jde = True if mode == 'train' else False
        
        ### Class info ###
        self.dict_cls_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.cfg.DATASET.CLASS_INFO.NUM_CLS = len(list(set(list(self.dict_cls_id.values())).difference(set([0,-1])))) # background: 0, ignore: -1
        ### Class info ###
        if not self.cfg.DATASET.DIR.START_END_FILE is None and not self.cfg.DATASET.DIR.START_END_FILE == '':
            with open(self.cfg.DATASET.DIR.START_END_FILE, 'r') as f:
                self.seq_start_end = json.load(f)[split]

        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###
        self.type_coord = self.cfg.DATASET.TYPE_COORD # 1: Radar, 2: Lidar, 3: Camera
        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###

        ### Radar Tesseract ###
        if self.cfg.DATASET.GET_ITEM.rdr_tesseract:
            # load physical values
            self.arr_r, self.arr_a, self.arr_e = self.load_physical_values()
            # consider roi
            self.is_consider_roi_rdr = cfg.DATASET.DEAR.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr:
                self.consider_roi_tesseract(cfg.DATASET.DEAR.ROI)
        ### Radar Tesseract ###

        ### Radar Cube ###
        if self.cfg.DATASET.GET_ITEM.rdr_cube:
            # Default ROI for CB
            polar_xz_grid = np.load('/mnt/ssd3/CRUW3D/polar_xz_grid.npz') # x and z are under camera's coordinate
            # self.arr_z_cb = np.arange(-30, 30, 0.4) # CURW2022 does not have vertical resolution
            self.arr_y_cb = np.sort(-polar_xz_grid['xline'])
            self.arr_x_cb = polar_xz_grid['zline']

            self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(cfg.DATASET.ROI[cfg.DATASET.LABEL['ROI_TYPE']])
        ### Radar Cube  ###

        ### Label ###
        self.roi_label = self.cfg.DATASET.LABEL.ROI_DEFAULT
        if self.cfg.DATASET.LABEL.IS_CONSIDER_ROI:
            roi_type = self.cfg.DATASET.LABEL.ROI_TYPE
            self.roi_label = cfg.DATASET.LABEL_ROI[roi_type]
        else:
            raise AttributeError('* Exception error (Dataset): check config file DATASET.LABEL.ROI_TYPE')
        self.is_roi_check_with_azimuth = self.cfg.DATASET.LABEL.IS_CHECK_VALID_WITH_AZIMUTH
        self.max_azimtuth_rad = self.cfg.DATASET.LABEL.MAX_AZIMUTH_DEGREE
        self.max_azimtuth_rad = [self.max_azimtuth_rad[0]*np.pi/180., self.max_azimtuth_rad[1]*np.pi/180.]
        self.get_calib()
        self.load_samples(cfg.DATASET.LIST_TARGET)

        ### Label ###
        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()
        self.cruw3d_evaluator = CRUW3DEvaluator(self.samples, [0.3, 0.5, 0.7])

    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)
        
    def load_samples(self, list_target):
        TYPE_COORD = {1: 'Radar', 2: 'Lidar', 3: 'Camera'}
        new_file_name = os.path.basename(self.cfg.DATASET.DIR.LABEL_FILE).split('.')[0]
        new_file_name += f'_{self.split}_{TYPE_COORD[self.cfg.DATASET.TYPE_COORD]}_{self.cfg.DATASET.LABEL.ROI_TYPE}'
        for classname in self.cfg.class_names:
            new_file_name += f'_{classname}'
        if self.debug:
            new_file_name += '_seq_['
            new_file_name += '-'.join(list_target)
            new_file_name += ']'
        new_file_name += '.json'
        new_file_path = os.path.join(os.path.dirname(self.cfg.DATASET.DIR.LABEL_FILE), new_file_name)
        label_file = self.cfg.DATASET.DIR.LABEL_FILE
        pre_process_label = True
        if os.path.exists(new_file_path):
            pre_process_label = False
            label_file = new_file_path
            self.label_file = label_file
        with open(label_file, 'r') as f:
            samples = json.load(f)[self.split]
        self.samples = []
        if pre_process_label:
            print('Preprocessing label data')
            # coordinate transformation, filter out labels out of ROI and not in our interest class
            if self.cfg.DATASET.TYPE_COORD == 1:
                # assume only translational offset between LiDAR and radar
                tr_rl_translation = np.array(self.calib['tr_rl']).reshape((4, 4))[:3, 3]
            for sample in samples:
                seq = sample['seq']
                rdr_frame_id = sample['rdr_frame']
                path_rdr_tensor = os.path.join(self.cfg.DATASET.DIR.DATA_ROOT, seq, self.cfg.DATASET.DIR.RAD_DIR, f'{rdr_frame_id}.npy')
                if not os.path.exists(path_rdr_tensor):
                    print(f'Missing {path_rdr_tensor}')
                    continue
                objs = []
                for obj in sample['objs']:
                    if obj['obj_type'] not in self.cfg.class_names:
                        continue
                    obj_xyz = (np.array(obj['xyz']) + tr_rl_translation).tolist()
                    if self.check_to_add_obj(obj_xyz):
                        obj['xyz'] = obj_xyz
                        objs.append(obj)
                sample['objs'] = objs
                if self.debug:
                    if seq not in list_target:
                        continue
                self.samples.append(sample)
            new_labels = {}
            new_labels[self.split] = self.samples
            with open(new_file_path, 'w') as f:
                json.dump(new_labels, f, indent=2)
        else:
            self.samples = samples
            
    # physical valuses of each DEAR tensors' cell
    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat(os.path.join(self.cfg.DATASET.DIR.DATA_ROOT, 'resources', 'info_arr.mat'))
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat(os.path.join(self.cfg.DATASET.DIR.DATA_ROOT, 'resources', 'arr_doppler.mat'))['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

    def consider_roi_tesseract(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = []
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi
        for k, v in roi_polar.items():
            if v is not None:
                min_max = v if k == 'r' else (np.array(v) * deg2rad).tolist() 
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx.append(idx_min)
                self.list_roi_idx.append(idx_max)
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  v_new if k == 'r' else (np.array(v_new) * rad2deg).tolist()
                    self.cfg.DATASET.DEAR.ROI[k] = v_new


    def consider_roi_cube(self, roi_cart):
        # to get indices
        self.list_roi_idx_cb = [
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]
        idx_attr = 0
        for k, v in roi_cart.items():
            if v is not None:
                min_max = np.array(v).tolist()
                # print(min_max)
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        if max_val > arr[-1]:
            return arr[idx_min:idx_max+1], idx_min, idx_max
        return arr[idx_min:idx_max], idx_min, idx_max-1

    ### Loading values from txt ###
    def get_calib(self, ):
        '''
        * return: [X, Y, Z]
        * if you want to get frame difference, get list_calib[0]
        '''
        calib = {}
        with open(self.cfg.DATASET.DIR.RDR_CALIB, 'r') as f:
            calib.update(json.load(f))
        self.calib = calib

    def check_to_add_obj(self, object_xyz):
        x, y, z = object_xyz
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if self.is_roi_check_with_azimuth:
            min_azi, max_azi = self.max_azimtuth_rad
            azimuth_center = np.arctan2(y, x)
            if (azimuth_center < min_azi) or (azimuth_center > max_azi)\
                or (x < x_min) or (y < y_min) or (z < z_min)\
                or (x > x_max) or (y > y_max) or (z > z_max):
                return False
        return True

    def get_spcube(self, seq, frame):
        path_spcube = os.path.join(self.cfg.DATASET.DIR.RDR_PC_DIR, seq, 'radar', self.cfg.DATASET.DIR.RDR_PC_TYPE, f'{frame}.npy')
        points = np.load(path_spcube)
        points[:, 3] /= self.cfg.DATASET.RDR_SP_CUBE.NORMALIZING_VALUE
        # points = np.asarray(o3d.io.read_point_cloud(path_spcube).points)
        return points

    def get_tesseract(self, seq, rdr_frame_id):
        path_tesseract = os.path.join(self.cfg.DATASET.DIR.DEAR_DIR, seq, 'radar_DEAR_D_downsampled_2', f'tesseract_{rdr_frame_id}.npy')
        arr_dear = np.flip(np.load(path_tesseract), axis=1) # elevation-axis is flipped
        arr_dear[arr_dear < 0.] = 0.
        ### considering ROI ###
        if self.is_consider_roi_rdr:
            idx_r_0, idx_r_1, idx_a_0, idx_a_1, \
                idx_e_0, idx_e_1 = self.list_roi_idx
            arr_dear = arr_dear[:, idx_e_0:idx_e_1+1,\
                idx_a_0:idx_a_1+1, idx_r_0:idx_r_1+1]
        ### considering ROI ###
        if self.cfg.DATASET.DEAR.REDUCE_TYPE == 'avg':
            arr_dear = arr_dear.mean(axis=0)
        elif self.cfg.DATASET.DEAR.REDUCE_TYPE == 'max':
            arr_dear = arr_dear.max(axis=0)
        # self.dear_buffer[self.counter] = arr_dear.copy()
        # worker_info = torch.utils.data.get_worker_info()
        # print(f'worker id: {worker_info.id}, counter: {self.counter}')
        # del arr_dear
        # self.counter = (self.counter + 1) % len(self.dear_buffer)
        # return self.dear_buffer[self.counter]
        return arr_dear

    def get_cube(self, seq, rdr_frame_id):
        # Flip the x axis for CRUW rad cube, since CRUW stores rad cube following camera's coordinate and make it to DYX
        arr_cube = np.transpose(np.flip(np.load(os.path.join(self.cfg.DATASET.DIR.DATA_ROOT, seq, self.cfg.DATASET.DIR.RAD_DIR, f'{rdr_frame_id}.npy')), axis=1), (2, 1, 0))
        norm_val = float(self.cfg.DATASET.RDR_CUBE.NORMALIZING_VALUE)
        # RoI selection
        idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        arr_cube = arr_cube[:,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        arr_cube[arr_cube < 0.] = 0.
        # normalize
        arr_cube = arr_cube / norm_val
        return arr_cube
    

    def get_pc_lidar(self, seq, frame_id, calib_info=None):
        path_lidar = os.path.join(self.cfg.DATASET.DIR.LIDAR_PC_DIR, seq, 'os2-64', f'os2-64_{frame_id}.pcd')
        pc_lidar = []
        with open(path_lidar, 'r') as f:
            lines = [line.rstrip('\n') for line in f][13:]
            pc_lidar = [point.split() for point in lines]
        pc_lidar = np.array(pc_lidar, dtype=np.float32).reshape(-1, 9)[:, :4]
        # only get the front view
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > 0.01)].reshape(-1, 4)
        if self.type_coord == 1: # Rdr coordinate
            tr_rl_translation = np.array(self.calib['tr_rl']).reshape((4, 4))[:3, :]
            tr_rl_translation[3] = 0.
            pc_lidar = np.array(list(map(lambda x: \
                [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
                pc_lidar.tolist())))
            pc_lidar = pc_lidar + tr_rl_translation[None, :]
            # only get PC visible to radar
            x_min, x_max, y_min, y_max, z_min, z_max = self.cfg.DATASET.LABEL.ROI_DEFAULT
            pc_roi_check = np.all([pc_lidar[:, 0] > x_min, pc_lidar[:, 0] < x_max,\
                                   pc_lidar[:, 1] > y_min, pc_lidar[:, 1] < y_max,\
                                   pc_lidar[:, 2] > z_min, pc_lidar[:, 2] < z_max], axis=0)
            pc_lidar = pc_lidar[pc_roi_check].reshape(-1, 4)
        return pc_lidar
    
    def get_description(self, seq):
        try:
            path_desc = os.path.join(self.cfg.DATASET.DIR.LIDAR_PC_DIR, seq, 'description.txt')
            with open(path_desc, 'r') as f:
                line = f.readline()
            road_type, capture_time, climate = line.split(',')
            dict_desc = {
                'capture_time': capture_time,
                'road_type': road_type,
                'climate': climate,
            }
        except:
            raise FileNotFoundError(f'* Exception error (Dataset): check description {path_desc}')
        
        return dict_desc


    def __len__(self):
        return len(self.samples)

    def get_data_indices(self, path_label):
        f = open(path_label, 'r')
        line = f.readlines()[0]
        f.close()
        seq_id = path_label.split('/')[-3]
        rdr_idx, ldr_idx, camf_idx, _, _ = line.split(',')[0].split('=')[1].split('_')
        return seq_id, rdr_idx, ldr_idx, camf_idx

    def get_second_sample_idx(self, idx, seq):
        start_end = self.seq_start_end[seq]
        target_bin = None
        for bin in start_end:
            if bin[0] <= idx <= bin[1]:
                target_bin = bin
                break
        left_edge = max(idx-self.cfg.JDE.max_frame_length, target_bin[0])
        right_edge = min(idx+self.cfg.JDE.max_frame_length, target_bin[1])
        idx_2 = np.random.choice([*range(left_edge, idx), *range(idx+1, right_edge+1)])
        return idx_2

    def get_sample_by_idx(self, idx):
        sample = self.samples[idx]
        dict_item = {}
        dict_item['meta'] = {'seq': sample['seq'], 'frame': sample['frame'], 'rdr_frame': sample['rdr_frame']}
        dict_item['objs'] = sample['objs']
        # dict_item['meta']['desc'] = self.get_description(dict_item['seq'])
        if self.cfg.DATASET.GET_ITEM['rdr_sparse_cube']:
            dict_item['rdr_sparse_cube'] = self.get_spcube(sample['seq'], sample['frame'])
        if self.cfg.DATASET.GET_ITEM['rdr_tesseract']:
            dict_item['rdr_tesseract'] = self.get_tesseract(sample['seq'], sample['rdr_frame'])
        if self.cfg.DATASET.GET_ITEM['rdr_cube']:
            rdr_cube = self.get_cube(sample['seq'], sample['rdr_frame'])
            dict_item['rdr_cube'] = rdr_cube
        if self.cfg.DATASET.GET_ITEM['ldr_pc_64']:
            dict_item['ldr_pc_64'] = self.get_pc_lidar(sample['seq'], sample['frame'])
        dict_item.update(mode=self.split)
        dict_item, _ = self.pipeline(dict_item, info=self.cfg)
        return dict_item

    def __getitem__(self, idx):
        dict_item = self.get_sample_by_idx(idx)
        if self.enable_jde:
            idx_2 = self.get_second_sample_idx(idx, dict_item['meta']['seq'])
            return [dict_item, self.get_sample_by_idx(idx_2)]
        else:
            return dict_item
        
    def evaluation(self, detections, output_dir=None, testset=False):
        # TODO: integrate the evaluation code to here
        self.cruw3d_evaluator.reset()
        self.cruw3d_evaluator.process(detections)
        res = self.cruw3d_evaluator.evaluate()
        return res, None
    
    @staticmethod
    def collate_fn(batch_list):
        if isinstance(batch_list[0], list):
            new_batch_list = []
            for batch in batch_list:
                new_batch_list += batch
            batch_list = new_batch_list
        if None in batch_list:
            print('* Exception error (Dataset): collate_fn')
            return None
        example_merged = collections.defaultdict(list)
        for example in batch_list:
            if type(example) is list:
                for subexample in example:
                    for k, v in subexample.items():
                        example_merged[k].append(v)
            else:
                for k, v in example.items():
                    example_merged[k].append(v)
        ret = {}
        for key, elems in example_merged.items():
            if key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                        "ind", "mask", "cat", "obj_id"]:
                ret[key] = collections.defaultdict(list)
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        ret[key][str(idx)].append(torch.tensor(ele))
                for kk, vv in ret[key].items():
                    res.append(torch.stack(vv))
                ret[key] = res  # [task], task: (batch, num_class_in_task, feat_shape_h, feat_shape_w)
            elif key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels",
                   "cyv_voxels", "cyv_num_points", "cyv_num_voxels"]:
                ret[key] = torch.tensor(np.concatenate(elems, axis=0))
            elif key == "points":
                ret[key] = [torch.tensor(elem) for elem in elems]
            elif key in ["coordinates", "cyv_coordinates"]:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                    )
                    coors.append(coor_pad)
                ret[key] = torch.tensor(np.concatenate(coors, axis=0))
            elif key in ['gt_boxes_and_cls']:
                ret[key] = torch.tensor(np.stack(elems, axis=0))
            elif key in ['rdr_tensor']:
                elems = np.stack(elems, axis=0)
                ret[key] = torch.tensor(elems)
            elif key in ['meta', 'calib_kradar']:
                ret[key] = elems
            else:
                ret[key] = np.stack(elems, axis=0)

        return ret
    
'''
New label format example:

data = {'train': [frame_info], 'test'; [frame_info]}
frame_info = 
    {
        "seq",
        "frame",
        "rdr_frame",
        "objs":[obj]
    }
obj = 
    {
        "obj_id": "0",
        "obj_type": "Sedan",
        "euler": [
        0,
        0,
        0.011563076580653683
        ],
        "xyz": [
        21.533344860569517,
        -3.5592597474205974,
        0.3055397143112053
        ],
        "lwh": [
        3.4579122180458848,
        1.7079880922772295,
        1.3824741692813867
        ]
    }
obj pose in lidar coordinate
'''

import os 
from collections import OrderedDict
import pandas as pd
import numpy as np
from det3d.core.bbox import box_np_ops
from functools import partial
from .rotate_iou import rotate_iou_gpu_eval, d3_box_overlap_kernel
import numba

class CRUW3DEvaluator():
    def __init__(
        self,
        dataset_dicts,
        iou_thresholds,
        class_names = ["Car", "Cyclist", "Pedestrian"],
    ):

        self._dataset_dicts = {}
        for dikt in dataset_dicts:
            # seq, frame, rdr_frame = dikt['seq'], dikt['frame'], dikt['rdr_frame']
            seq, frame, rdr_frame = dikt['seq_name'], dikt['frame_name'], dikt['frame_name']
            dict_key = f'{seq}/{frame}/{rdr_frame}'
            self._dataset_dicts[dict_key] = dikt
        self._id_to_name = {i: name for i, name in enumerate(class_names)}
        self._class_names = class_names
        self._iou_thresholds = iou_thresholds

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        # List[Dict], each key'ed by category (str) + vectorized 3D box (10) + 2D box (4) + score (1) + file name (str)
        self._predictions_kitti_format = []
        self._groundtruth_kitti_format = []


    def process(self, preds):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for key, val in preds.items():
            if key not in self._dataset_dicts:
                continue # to do remove in the release version
            predictions_kitti = []
            for box3d, score, label_preds in zip(val['box3d'], val['scores'], val['label_preds']):
                if int(label_preds)>0: # only eval car
                    continue
                class_name = self._class_names[int(label_preds)]
                # x, y, z, L, W, H, theta =  box3d.tolist()
                x, y, z, L, W, H, pitch, yaw, roll =  box3d.tolist()
                # predictions_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, -y, -z, x, -theta, float(score)])
                predictions_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, x, y, z, 0., float(score)])
            self._predictions_kitti_format.append(pd.DataFrame(predictions_kitti))
            # groundtruths
            gt_dataset_dict = self._dataset_dicts[key]['objs']
            groundtruth_kitti = []
            for anno in gt_dataset_dict:
                class_name = anno['obj_type']
                if not class_name == 'Car':
                    continue
                # groundtruth in KITTI format.
                # L, W, H = anno['lwh']
                # x, y, z = anno['xyz']
                # L, H, W = anno['scale']
                L, W, H = anno['scale']
                x, y, z = anno['position']
                theta = box_np_ops.limit_period(anno['euler'][2], offset=0.5, period=np.pi * 2)
                # theta = box_np_ops.limit_period(anno['ry'], offset=0.5, period=np.pi * 2)

                # groundtruth_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, -y, -z, x, -theta])
                groundtruth_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, x, y, z, 0.])
            self._groundtruth_kitti_format.append(pd.DataFrame(groundtruth_kitti))


    def process_2(self, preds):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        for key, val in preds.items():
            predictions_kitti = []
            for box3d, score, label_preds in zip(val['box3d'], val['scores'], val['label_preds']):
                class_name = self._class_names[int(label_preds)]
                if not class_name == 'Car':
                    continue
                # x, y, z, L, W, H, theta =  box3d.tolist()
                x, y, z, L, W, H, pitch, yaw, roll =  box3d.tolist()
                # predictions_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, -y, -z, x, -theta, float(score)])
                predictions_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, x, y, z, 0., float(score)])
            self._predictions_kitti_format.append(pd.DataFrame(predictions_kitti))
            # groundtruths
            seq, frame, _ = key.split('/')
            frame = int(frame)
            frame = f'{frame:06}'
            gt = self._dataset_dicts[seq][frame]
            groundtruth_kitti = []
            for anno in gt['annotations']:

                class_name = anno['obj_type']
                if not class_name == 'Car':
                    continue
                # groundtruth in KITTI format.
                # L, W, H = anno['lwh']
                # x, y, z = anno['xyz']
                # L, H, W = anno['scale']
                L, W, H = anno['scale']
                x, y, z = anno['position']
                # theta = box_np_ops.limit_period(anno['euler'][2], offset=0.5, period=np.pi * 2)
                theta = box_np_ops.limit_period(anno['ry'], offset=0.5, period=np.pi * 2)

                # groundtruth_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, -y, -z, x, -theta])
                groundtruth_kitti.append([class_name, -1, -1, 0, 0, 0, 0, 0, H, W, L, x, y, z, 0.])
            self._groundtruth_kitti_format.append(pd.DataFrame(groundtruth_kitti))


    def evaluate(self):
        predictions_kitti_format = self._predictions_kitti_format
        groundtruth_kitti_format = self._groundtruth_kitti_format

        assert len(predictions_kitti_format) == len(groundtruth_kitti_format)
        formatted_predictions = [
            KITTIEvaluationEngine._format(idx, x, True) for idx, x in enumerate(predictions_kitti_format)
        ]
        formatted_groundtruth = [
            KITTIEvaluationEngine._format(idx, x, False) for idx, x in enumerate(groundtruth_kitti_format)
        ]

        engine = KITTIEvaluationEngine(id_to_name=self._id_to_name)
        results = engine.evaluate(formatted_groundtruth, formatted_predictions, overlap_thresholds=self._iou_thresholds)

        results = OrderedDict({k: 100. * v for k, v in results.items()})

        return results



class KITTIEvaluationEngine():

    _DEFAULT_KITTI_LEVEL_TO_PARAMETER = {
        "levels": ("all", ),
        "max_occlusion": (10, ), # dummy large number
        "max_truncation": (1.0, ), # dummy large number
        "min_height": (-1, ) # set the min height to zero since 2D bbox are set to zero manually 
    }

    def __init__(self, id_to_name, num_shards=50, sample_points=41):
        self.id_to_name = id_to_name
        self.sample_points = sample_points
        self.num_shards = num_shards
        self.filter_data_fn = partial(
            clean_kitti_data, difficulty_level_to_params=self._DEFAULT_KITTI_LEVEL_TO_PARAMETER
        )

    @staticmethod
    def _format(idx, kitti_format, is_prediction):
        if len(kitti_format) == 0:
            annotations = dict(
                id=f'{idx:06d}',
                name=[],
                truncated=np.array([]),
                occluded=np.array([]),
                alpha=np.array([]),
                bbox=np.empty((0, 4)),
                dimensions=np.empty((0, 3)),
                location=np.empty((0, 3)),
                rotation_y=np.array([]),
                score=np.array([])
            )
            return annotations

        data = np.array(kitti_format)
        annotations = dict(
            id=f'{idx:06d}',
            name=data[:, 0],
            truncated=data[:, 1].astype(np.float64),
            occluded=data[:, 2].astype(np.int64),
            alpha=data[:, 3].astype(np.float64),
            bbox=data[:, 4:8].astype(np.float64),
            dimensions=data[:, 8:11][:, [2, 0, 1]].astype(np.float64),
            location=data[:, 11:14].astype(np.float64),
            rotation_y=data[:, 14].astype(np.float64),
        )

        if is_prediction:
            annotations.update({'score': data[:, 15].astype(np.float64)})
        else:
            annotations.update({'score': np.zeros([len(annotations['bbox'])])})
        return annotations

    def get_shards(self, num, num_shards):
        """Shard number into evenly sized parts. `Remaining` values are put into the last shard.

        Parameters
        ----------
        num: int
            Number to shard

        num_shards: int
            Number of shards

        Returns
        -------
        List of length (num_shards or num_shards +1), depending on whether num is perfectly divisible by num_shards
        """
        assert num_shards > 0, "Invalid number of shards"
        num_per_shard = num // num_shards
        remaining_num = num % num_shards
        full_shards = num_shards * (num_per_shard > 0)
        if remaining_num == 0:
            return [num_per_shard] * full_shards
        else:
            return [num_per_shard] * full_shards + [remaining_num]

    def evaluate(self, gt_annos, dt_annos, overlap_thresholds):
        # pr_curves = self.eval_metric(gt_annos, dt_annos, metric, overlap_thresholds)
        gt_annos, dt_annos = self.validate_anno_format(gt_annos, dt_annos)

        box3d_pr_curves = self.eval_metric(gt_annos, dt_annos, 'BOX3D_AP', overlap_thresholds)
        mAP_3d = self.get_mAP(box3d_pr_curves["precision"], box3d_pr_curves["recall"])

        bev_pr_curves = self.eval_metric(gt_annos, dt_annos, 'BEV_AP', overlap_thresholds)
        mAP_bev = self.get_mAP(bev_pr_curves["precision"], bev_pr_curves["recall"])

        results = OrderedDict()
        for class_i, class_name in self.id_to_name.items():
            for diff_i, diff in enumerate(["All"]):
                for thresh_i, thresh in enumerate(overlap_thresholds):
                    results['kitti_box3d_r40/{}_{}_{}'.format(class_name, diff, thresh)] = \
                        mAP_3d[class_i, diff_i, thresh_i]
        for class_i, class_name in self.id_to_name.items():
            for diff_i, diff in enumerate(["All"]):
                for thresh_i, thresh in enumerate(overlap_thresholds):
                    results['kitti_bev_r40/{}_{}_{}'.format(class_name, diff, thresh)] = \
                        mAP_bev[class_i, diff_i, thresh_i]
        return results

    def get_mAP(self, precision, recall):
        """ Get mAP from precision.
        Parameters
        ----------
        precision: np.ndarray
            Numpy array of precision curves at different recalls, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        recall: np.ndarray
            Numpy array of recall values corresponding to each precision, of shape
            [num_classes, num_difficulties, num_overlap_thresholds,self.sample_points]

        Returns
        -------
        ap: np.ndarray
            Numpy array of mean AP evaluated at different points along PR curve.
            Shape [num_classes, num_difficulties, num_overlap_thresholds]
        """
        precisions, recall_spacing = self.get_sampled_precision_recall(precision, recall)
        ap = sum(precisions) / len(recall_spacing)
        return ap

    def get_sampled_precision_recall(self, precision, recall):
        """Given an array of precision, recall values, sample evenly along the recall range, and interpolate the precision
        based on AP from section 6 from https://research.mapillary.com/img/publications/MonoDIS.pdf

        Parameters
        ----------
        precision: np.ndarray
            Numpy array of precision curves at different recalls, of shape
            [num_classes, num_difficulties, num_overlap_thresholds, self.sample_points]

        recall: np.ndarray
            Numpy array of recall values corresponding to each precision, of shape
            [num_classes, num_difficulties, num_overlap_thresholds, self.sample_points]

        Returns
            sampled_precision: list of np.ndarrays, of shape (num_classes, num_difficulties, num_overlap_thresholds)
                The maximum precision values corresponding to the sampled recall.
            sampled_recall: list
                Recall values evenly spaced along the recall range.
        """
        # recall_range = self.recall_range
        recall_range = (0.0, 1.0)
        precisions = []
        # Don't count recall at 0
        recall_spacing = [1. / (self.sample_points - 1) * i for i in range(1, self.sample_points)]
        recall_spacing = list(filter(lambda recall: recall_range[0] <= recall <= recall_range[1], recall_spacing))
        for r in recall_spacing:
            precisions_above_recall = (recall >= r) * precision
            precisions.append(precisions_above_recall.max(axis=3))

        return precisions, recall_spacing

    @staticmethod
    def validate_anno_format(gt_annos, dt_annos):
        """Verify that the format/dimensions for the annotations are correct.
        Keys correspond to defintions here:
        https://github.com/NVIDIA/DIGITS/blob/v4.0.0-rc.3/digits/extensions/data/objectDetection/README.md
        """
        necessary_keys = ['name', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score']
        for i, (gt_anno, dt_anno) in enumerate(zip(gt_annos, dt_annos)):
            for key in necessary_keys:
                assert key in gt_anno, "{} not present in GT {}".format(key, i)
                assert key in dt_anno, "{} not present in prediction {}".format(key, i)
                if key in ['bbox', 'dimensions', 'location']:
                    # make sure these fields are 2D numpy array
                    assert len(gt_anno[key].shape) == 2, key
                    assert len(dt_anno[key].shape) == 2, key

            for key in ['truncated', 'occluded', 'alpha', 'rotation_y', 'score']:
                if len(gt_anno[key].shape) == 2:
                    gt_anno[key] = np.squeeze(gt_anno[key], axis=0)
                if len(dt_anno[key].shape) == 2:
                    dt_anno[key] = np.squeeze(dt_anno[key], axis=0)
        return gt_annos, dt_annos

    def eval_metric(self, gt_annos, dt_annos, metric, overlap_thresholds):
        assert len(gt_annos) == len(dt_annos), "Must provide a prediction for every ground truth sample"
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, self.num_shards)

        overlaps, overlaps_by_shard, total_gt_num, total_dt_num = \
            self.calculate_match_degree_sharded(gt_annos, dt_annos, metric)
        # all_thresholds = -1.0 * dist_thresholds[metric, :, :, :] if metric == Metrics.BBOX_3D_NU_AP else \
        #     overlap_thresholds[metric, :, :, :]

        num_minoverlap = len(overlap_thresholds)
        num_classes = len(self.id_to_name)
        num_difficulties = 1

        precision = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        recall = np.zeros([num_classes, num_difficulties, num_minoverlap, self.sample_points])
        instances_count = np.zeros([num_classes, num_difficulties])

        for class_idx in range(num_classes):
            for difficulty_idx in range(num_difficulties):
                gt_data_list, dt_data_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, \
                total_num_valid_gt = self.prepare_data(gt_annos, dt_annos, class_idx, difficulty_idx)
                instances_count[class_idx, difficulty_idx] = total_num_valid_gt

                for thresh_idx, min_overlap in enumerate(overlap_thresholds):
                    thresholds_list = []
                    for i in range(len(gt_annos)):
                        threshold = compute_threshold_jit(
                            overlaps[i],
                            gt_data_list[i],
                            dt_data_list[i],
                            ignored_gts[i],
                            ignored_dets[i],
                            min_overlap=min_overlap,
                            compute_fp=False
                        )
                        thresholds_list += threshold.tolist()
                    thresholds = np.array(
                        get_thresholds(np.array(thresholds_list), total_num_valid_gt, self.sample_points)
                    )
                    # TODO: Refactor hard coded numbers and strings
                    # [num_threshold, num_fields], fields: tp, fp, fn, aoe, aos, iou/dist error, -log(Probability,
                    # bev iou error)
                    pr = np.zeros([len(thresholds), 8])

                    idx = 0
                    for shard_idx, num_samples_per_shard in enumerate(shards):
                        gt_datas_part = np.concatenate(gt_data_list[idx:idx + num_samples_per_shard], 0)
                        dt_datas_part = np.concatenate(dt_data_list[idx:idx + num_samples_per_shard], 0)
                        dc_datas_part = np.concatenate(dontcares[idx:idx + num_samples_per_shard], 0)
                        ignored_dets_part = np.concatenate(ignored_dets[idx:idx + num_samples_per_shard], 0)
                        ignored_gts_part = np.concatenate(ignored_gts[idx:idx + num_samples_per_shard], 0)
                        fused_compute_statistics(
                            overlaps_by_shard[shard_idx],
                            pr,
                            total_gt_num[idx:idx + num_samples_per_shard],
                            total_dt_num[idx:idx + num_samples_per_shard],
                            ignores_per_sample[idx:idx + num_samples_per_shard],
                            gt_datas_part,
                            dt_datas_part,
                            dc_datas_part,
                            ignored_gts_part,
                            ignored_dets_part,
                            min_overlap=min_overlap,
                            thresholds=thresholds,
                            compute_angular_metrics=True
                        )
                        idx += num_samples_per_shard

                    for i in range(len(thresholds)):
                        recall[class_idx, difficulty_idx, thresh_idx, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                        precision[class_idx, difficulty_idx, thresh_idx, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])

        return {
            "recall": recall,
            "precision": precision,
        }

    def prepare_data(self, gt_annos, dt_annos, class_idx, difficulty_idx):
        """Wrapper function for cleaning data before computing metrics.
        """
        gt_list = []
        dt_list = []
        ignores_per_sample = []
        ignored_gts, ignored_dets, dontcares = [], [], []
        total_num_valid_gt = 0

        for gt_anno, dt_anno in zip(gt_annos, dt_annos):
            num_valid_gt, ignored_gt, ignored_det, ignored_bboxes = self.filter_data_fn(
                gt_anno, dt_anno, class_idx, difficulty_idx, self.id_to_name
            )
            ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
            ignored_dets.append(np.array(ignored_det, dtype=np.int64))

            if len(ignored_bboxes) == 0:
                ignored_bboxes = np.zeros((0, 4)).astype(np.float64)
            else:
                ignored_bboxes = np.stack(ignored_bboxes, 0).astype(np.float64)

            ignores_per_sample.append(ignored_bboxes.shape[0])
            dontcares.append(ignored_bboxes)
            total_num_valid_gt += num_valid_gt
            gt_list.append(
                np.concatenate([
                    gt_anno["bbox"], gt_anno["rotation_y"][..., np.newaxis], gt_anno["alpha"][..., np.newaxis],
                    gt_anno["dimensions"]
                ], 1)
            )

            dt_list.append(
                np.concatenate([
                    dt_anno["bbox"], dt_anno["rotation_y"][..., np.newaxis], dt_anno["alpha"][..., np.newaxis],
                    dt_anno["dimensions"], dt_anno["score"][..., np.newaxis]
                ], 1)
            )

        ignores_per_sample = np.stack(ignores_per_sample, axis=0)
        return gt_list, dt_list, ignored_gts, ignored_dets, dontcares, ignores_per_sample, total_num_valid_gt

    def calculate_match_degree_sharded(self, gt_annos, dt_annos, metric):
        assert len(gt_annos) == len(dt_annos)
        total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)
        total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)

        overlaps_by_shard = []
        sample_idx = 0
        num_ground_truths = len(gt_annos)
        shards = self.get_shards(num_ground_truths, self.num_shards)

        for num_samples_per_shard in shards:
            gt_annos_part = gt_annos[sample_idx:sample_idx + num_samples_per_shard]
            dt_annos_part = dt_annos[sample_idx:sample_idx + num_samples_per_shard]

            if metric == 'BEV_AP':
                loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.bev_box_overlap(dt_boxes, gt_boxes).astype(np.float64)
            elif metric == "BOX3D_AP":
                loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
                gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
                dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
                rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
                dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
                shard_match = self.box_3d_overlap(dt_boxes, gt_boxes).astype(np.float64)
            else:
                raise ValueError("Unknown metric")

            # On each shard, we compute an IoU between all N predicted boxes and K GT boxes.
            # Shard overlap is a (N X K) array
            overlaps_by_shard.append(shard_match)

            sample_idx += num_samples_per_shard

        # Flatten into unsharded list
        overlaps = []
        sample_idx = 0
        for j, num_samples_per_shard in enumerate(shards):
            gt_num_idx, dt_num_idx = 0, 0
            for i in range(num_samples_per_shard):
                gt_box_num = total_gt_num[sample_idx + i]
                dt_box_num = total_dt_num[sample_idx + i]
                overlaps.append(
                    overlaps_by_shard[j][dt_num_idx:dt_num_idx + dt_box_num, gt_num_idx:gt_num_idx + gt_box_num, ]
                )
                gt_num_idx += gt_box_num
                dt_num_idx += dt_box_num
            sample_idx += num_samples_per_shard
        return overlaps, overlaps_by_shard, total_gt_num, total_dt_num

    def bev_box_overlap(self, boxes, qboxes, criterion=-1):
        """Compute overlap in BEV"""
        riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
        return riou

    def box_3d_overlap(self, boxes, qboxes, criterion=-1):
        """Compute 3D box IoU"""
        # For scale cuboid: use x, y to calculate bev iou, for kitti, use x, z to calculate bev iou
        rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
        d3_box_overlap_kernel(boxes, qboxes, rinc, criterion, True)
        return rinc


def clean_kitti_data(gt_anno, dt_anno, current_class, difficulty, id_to_name, difficulty_level_to_params=None):
    """Function for filtering KITTI data by difficulty and class.

    We filter with the following heuristics:
        If a ground truth matches the current class AND it falls below the difficulty
        threshold, we count it as a valid gt (append 0 in `ignored_gt` list).

        If a ground truth matches the current class but NOT the difficulty, OR it matches
        a class that is semantically too close to penalize (i.e. Van <-> Car),
        we ignore it (append 1 in `ignored_gt` list)

        If a ground truth doesn't belong to the current class, we ignore it (append -1 in `ignored_gt`)

        If a ground truth corresponds to a "DontCare" box, we append that box to the `ignored_bboxes` list.

        If a prediction matches the current class AND is above the minimum height threshold, we count it
        as a valid detection (append 0 in `ignored_dt`)

        If a prediction matches the current class AND it is too small, we ignore it (append 1 in `ignored_dt`)

        If a prediction doesn't belong to the class, we ignore it (append -1 in `ignored_dt`)

    Parameters
    ----------
    gt_anno: dict
        KITTI format ground truth. Please refer to note at the top for details on format.

    dt_anno: dict
        KITTI format prediction.  Please refer to note at the top for details on format.

    current_class: int
        Class ID, as int

    difficulty: int
        Difficulty: easy=0, moderate=1, difficult=2

    id_to_name: dict
        Mapping from class ID (int) to string name

    difficulty_level_to_params: dict default= None

    Returns
    -------
    num_valid_gt: int
        Number of valid ground truths

    ignored_gt: list[int]
        List of length num GTs. Populated as described above.

    ignored_dt: list[int]
        List of length num detections. Populated as described above.

    ignored_bboxes: list[np.ndarray]
        List of np.ndarray corresponding to boxes that are to be ignored
    """
    ignored_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = id_to_name[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0

    for i in range(num_gt):
        bbox = gt_anno["bbox"][i]
        gt_name = gt_anno["name"][i].lower()
        height = bbox[3] - bbox[1]
        valid_class = -1

        # For KITTI, Van does not penalize car detections and person sitting does not penalize pedestrian
        if gt_name == current_cls_name:
            valid_class = 1
        elif current_cls_name == "Pedestrian".lower() and "Person_sitting".lower() == gt_name:
            valid_class = 0
        elif current_cls_name == "Car".lower() and "Van".lower() == gt_name:
            valid_class = 0
        else:
            valid_class = -1

        # Filter by occlusion/truncation
        ignore_for_truncation_occlusion = False
        if ((gt_anno["occluded"][i] > difficulty_level_to_params["max_occlusion"][difficulty])
            or (gt_anno["truncated"][i] > difficulty_level_to_params["max_truncation"][difficulty])
            or (height <= difficulty_level_to_params["min_height"][difficulty])):
            ignore_for_truncation_occlusion = True

        if valid_class == 1 and not ignore_for_truncation_occlusion:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore_for_truncation_occlusion and (valid_class == 1)):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)

        # Track boxes are in "dontcare" areas
        if gt_name == "dontcare":
            ignored_bboxes.append(bbox)

    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1
        height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])

        # If a box is too small, ignore it
        if height < difficulty_level_to_params["min_height"][difficulty]:
            ignored_dt.append(1)
        elif valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, ignored_bboxes


@numba.jit(nopython=True, fastmath=True)
def compute_threshold_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    min_overlap,
    compute_fp=False,
):
    """Compute TP/FP statistics.
    Modified from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * det_size

    NO_DETECTION = np.finfo(np.float32).min
    tp, fp, fn = 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            overlap = overlaps[j, i]
            dt_score = dt_scores[j]

            # Not hit during TP/FP computation
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                assert not compute_fp, "For sanity, compute_fp shoudl be False if we are here"
                det_idx = j
                valid_detection = dt_score

        # No matched prediction found, valid GT
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # Matched prediction, but NO valid GT or matched prediction is too small so we ignore it (NOT BECAUSE THE
        # CLASS IS WRONG)
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # Matched prediction
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            assigned_detection[det_idx] = True

    return thresholds[:thresh_idx]


@numba.jit(nopython=True, fastmath=True)
def get_thresholds(scores, num_gt, num_sample_pts=41):
    """Get thresholds from a set of scores, up to num sample points

    Parameters
    ----------
    score: np.ndarray
        Numpy array of scores for predictions

    num_gt: int
        Number of ground truths

    num_sample_pts: int, default: 41
        Max number of thresholds on PR curve

    Returns
    -------
    threshold: np.ndarray
        Array of length 41, containing recall thresholds
    """
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (((r_recall - current_recall) < (current_recall - l_recall)) and (i < (len(scores) - 1))):
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pts - 1.0)
    return thresholds


@numba.jit(nopython=True, fastmath=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    min_overlap,
    thresholds,
    compute_angular_metrics=True,
):
    """Compute TP/FP statistics.
    Taken from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    without changes to avoid introducing errors"""

    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, thresh in enumerate(thresholds):
            # The key line that determines the ordering of the IoU matrix
            overlap = overlaps[dt_num:dt_num + dt_nums[i], gt_num:gt_num + gt_nums[i]]

            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_det = ignored_dets[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, error_yaw, similarity, _, match_degree, confidence_error, scale_error = \
                compute_statistics_jit(
                    overlap,
                    gt_data,
                    dt_data,
                    ignored_gt,
                    ignored_det,
                    dontcare,
                    min_overlap=min_overlap,
                    thresh=thresh,
                    compute_fp=True,
                    compute_angular_metrics=compute_angular_metrics)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            pr[t, 5] += match_degree
            pr[t, 6] += confidence_error
            pr[t, 7] += scale_error
            if error_yaw != -1:
                pr[t, 3] += error_yaw
            if similarity != -1:
                pr[t, 4] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


@numba.jit(nopython=True, fastmath=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    ignored_bboxes,
    min_overlap,
    thresh=0.0,
    compute_fp=False,
    compute_angular_metrics=False
):
    """Compute TP/FP statistics.
    Modified from https://github.com/sshaoehuai/PointRCNN/blob/master/tools/kitti_object_eval_python/eval.py
    """
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_yaws = dt_datas[:, 4]
    gt_yaws = gt_datas[:, 4]
    dt_alphas = dt_datas[:, 5]
    gt_alphas = gt_datas[:, 5]
    dt_bboxes = dt_datas[:, :4]
    gt_dimensions = gt_datas[:, 6:9]
    dt_dimensions = dt_datas[:, 6:9]

    assigned_detection = [False] * det_size
    ignored_threshold = [False] * det_size
    if compute_fp:
        for i in range(det_size):
            if (dt_scores[i] < thresh):
                ignored_threshold[i] = True

    NO_DETECTION = np.finfo(np.float32).min
    tp, fp, fn, error_yaw, similarity, match_degree, scale_error, confidence_error = 0, 0, 0, 0, 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta_yaw = np.zeros((gt_size, ))
    delta_alpha = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION

        max_overlap = np.finfo(np.float32).min
        target_scale_iou = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            if (ignored_threshold[j]):
                continue
            overlap = overlaps[j, i]
            scale_iou = compute_scale_error(gt_dimensions[i, :], dt_dimensions[j, :])
            dt_score = dt_scores[j]

            # Not hit during TP/FP computation
            if (not compute_fp and (overlap > min_overlap) and dt_score > valid_detection):
                assert not compute_fp, "For sanity, compute_fp shoudl be False if we are here"
                det_idx = j
                valid_detection = dt_score
            elif (
                compute_fp and (overlap > min_overlap) and (overlap > max_overlap or assigned_ignored_det)
                and ignored_det[j] == 0
            ):
                max_overlap = overlap
                target_scale_iou = scale_iou
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False

            elif (compute_fp and (overlap > min_overlap) and (valid_detection == NO_DETECTION) and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        # No matched prediction found, valid GT
        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            fn += 1

        # Matched prediction, but NO valid GT or matched prediction is too small so we ignore it (NOT BECAUSE THE
        # CLASS IS WRONG)
        elif ((valid_detection != NO_DETECTION) and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True

        # Matched prediction
        elif valid_detection != NO_DETECTION:
            tp += 1
            match_degree += abs(max_overlap)
            scale_error += 1.0 - abs(target_scale_iou)
            confidence_error += -math.log(dt_scores[det_idx])
            # Build a big list of all thresholds associated to true positives
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1

            if compute_angular_metrics:
                delta_yaw[delta_idx] = abs(angle_diff(float(gt_yaws[i]), float(dt_yaws[det_idx]), 2 * np.pi))
                delta_alpha[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1

            assigned_detection[det_idx] = True

    if compute_fp:
        for i in range(det_size):
            if (not (assigned_detection[i] or ignored_det[i] == -1 or ignored_det[i] == 1 or ignored_threshold[i])):
                fp += 1
        nstuff = 0
        fp -= nstuff
        if compute_angular_metrics:
            tmp_yaw = np.zeros((fp + delta_idx, ))
            tmp_alpha = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp_yaw[i + fp] = delta_yaw[i]
                tmp_alpha[i + fp] = (1.0 + np.cos(delta_alpha[i])) / 2.0

            if tp > 0 or fp > 0:
                error_yaw = np.sum(tmp_yaw)
                similarity = np.sum(tmp_alpha)
            else:
                error_yaw = -1
                similarity = -1

    return tp, fp, fn, error_yaw, similarity, thresholds[:thresh_idx], match_degree, confidence_error, scale_error


@numba.jit(nopython=True)
def angle_diff(x, y, period):
    """Get the smallest angle difference between 2 angles: the angle from y to x.

    Parameters
    ----------
    x: float
        To angle.
    y: float
        From angle.
    period: float
        Periodicity in radians for assessing angle difference.

    Returns:
    ----------
    diff: float
        Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


@numba.jit(nopython=True, fastmath=True)
def compute_scale_error(gt_dimension, dt_dimension):
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.

    Parameters
    ----------
    gt_dimension: List[float]
        GT annotation sample.
    dt_dimension: List[float]
        Predicted sample.

    Returns: float
    ----------
        Scale IOU.
    """

    # Compute IOU.
    min_wlh = [
        min(gt_dimension[0], dt_dimension[0]),
        min(gt_dimension[1], dt_dimension[1]),
        min(gt_dimension[2], dt_dimension[2])
    ]
    volume_gt = gt_dimension[0] * gt_dimension[1] * gt_dimension[2]
    volume_dt = dt_dimension[0] * dt_dimension[1] * dt_dimension[2]
    intersection = min_wlh[0] * min_wlh[1] * min_wlh[2]
    union = volume_gt + volume_dt - intersection
    iou = intersection / union

    return iou


