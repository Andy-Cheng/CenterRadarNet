import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm
import numpy as np
from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from munch import DefaultMunch
import collections
import json
from collections import defaultdict

@DATASETS.register_module
class KRadarDataset(Dataset):
    def __init__(self, cfg, split, class_names=None, pipeline=None):
        super().__init__()
        cfg = DefaultMunch.fromDict(cfg) # if cfg is dict
        self.class_names = class_names
        self.cfg = cfg
        self.cfg.update(class_names=class_names)
        self.split = split
        self.debug = True if len(cfg.DATASET.LIST_TARGET) > 0 else False
        if split == 'train':
            self.LIST_BAD = self.cfg.DATASET.LIST_BAD
        else:
            self.LIST_BAD = []
        ### Class info ###
        self.dict_cls_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.cfg.DATASET.CLASS_INFO.NUM_CLS = len(list(set(list(self.dict_cls_id.values())).difference(set([0,-1])))) # background: 0, ignore: -1
        ### Class info ###

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
        self.is_get_cube_dop = False 
        if self.cfg.DATASET.GET_ITEM.rdr_cube:
            # Default ROI for CB (When generating CB from matlab applying interpolation)
            self.arr_bev_none_minus_1 = None
            self.arr_z_cb = np.arange(-30, 30, 0.4)
            self.arr_y_cb = np.arange(-80, 80, 0.4)
            self.arr_x_cb = np.arange(0, 100, 0.4)

            self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(cfg.DATASET.ROI[cfg.DATASET.LABEL['ROI_TYPE']])
            # self.is_get_cube_dop = cfg.DATASET.GET_ITEM['rdr_cube_doppler']
            # self.offset_doppler = cfg.DATASET.RDR_CUBE.DOPPLER.OFFSET
            # self.is_dop_another_dir = cfg.DATASET.RDR_CUBE.DOPPLER.IS_ANOTHER_DIR
            # self.dir_dop = cfg.DATASET.DIR.DIR_DOPPLER_CB
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
        # self.initialize_dear_shared_buffer()
        # self.init_dear_buffer()

        # debug memory usage
        # debug_path = '/mnt/ssd1/kradar_dataset/radar_tensor/4/radar_DEAR_D_downsampled_2/tesseract_00116.npy'
        # self.arr_dear = np.maximum(np.flip(np.load(debug_path), axis=1), 0.) # elevation-axis is flipped
        # idx_r_0, idx_r_1, idx_a_0, idx_a_1, \
        #     idx_e_0, idx_e_1 = self.list_roi_idx
        # self.arr_dear = self.arr_dear[:, idx_e_0:idx_e_1+1,\
        #     idx_a_0:idx_a_1+1, idx_r_0:idx_r_1+1]
        # self.arr_dear = torch.from_numpy(self.arr_dear)
        # self.arr_dear = torch.empty((32, 37, 107, 256))
        # debug memory usage

    # def initialize_dear_shared_buffer(self):
    #     # Shared memory buffer
    #     buffer_size = 200
    #     array_shape = (32, 37, 107, 256)
    #     self.buffer = Array('f', buffer_size * np.prod((32, 37, 107, 256)))
    #     self.buffer = np.frombuffer(self.buffer.get_obj()).reshape((buffer_size,) + array_shape)
        
    #     # Synchronization
    #     self.lock = Lock()
    #     # Atomic counter for buffer index
    #     self.counter = Value('i', 0)

    # def init_dear_buffer(self):
    #     idx_r_0, idx_r_1, idx_a_0, idx_a_1, idx_e_0, idx_e_1 = self.list_roi_idx
    #     self.dear_buffer = np.empty((self.cfg.DATASET.DEAR_BUFFER_SIZE, 32, idx_e_1-idx_e_0+1, idx_a_1-idx_a_0+1, idx_r_1-idx_r_0+1), dtype=np.float32)
    #     self.counter = 0


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
                path_rdr_tensor = os.path.join(self.cfg.DATASET.DIR.DEAR_DIR, seq, 'radar_DEAR_D_downsampled_2', f'tesseract_{rdr_frame_id}.npy')\
                    if self.cfg.DATASET.GET_ITEM['rdr_tesseract'] else \
                    os.path.join(self.cfg.DATASET.DIR.RDR_CUBE_DIR, seq, f'cube_{rdr_frame_id}.npy')
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
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
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
    ### General functions ###

    ### Loading values from txt ###
    def get_calib(self, ):
        '''
        * return: [X, Y, Z]
        * if you want to get frame difference, get list_calib[0]
        '''
        calib = {}
        with open(self.cfg.DATASET.DIR.RDR_CALIB, 'r') as f:
            calib.update(json.load(f))
        with open(self.cfg.DATASET.DIR.CAM_CALIB, 'r') as f:
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

    def get_spcube(self, path_spcube):
        return np.load(path_spcube)

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
        arr_cube = np.flip(np.load(os.path.join(self.cfg.DATASET.DIR.RDR_CUBE_DIR, seq, f'cube_{rdr_frame_id}.npy')), axis=0).astype(np.float32) # z-axis is flipped
        norm_val = float(self.cfg.DATASET.RDR_CUBE.NORMALIZING_VALUE)
        # RoI selection
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        arr_cube[arr_cube < 0.] = 0.
        # normalize
        arr_cube = arr_cube / norm_val
        return arr_cube
        
    def get_cube_direct(self, path_cube):
        '''
        get the preprocessed cube and return it.
        '''
        arr_cube = np.load(path_cube).astype(np.float32)
        return arr_cube



    def get_cube_doppler(self, path_cube_doppler, dummy_value=100.):
        arr_cube = np.flip(loadmat(path_cube_doppler)['arr_zyx'], axis=0)
        # print(np.count_nonzero(arr_cube==-1.)) # no value -1. in doppler cube

        ### Null value is -10. for Doppler & -1. for pw (from matlab) ###
        arr_cube[np.where(arr_cube==-10.)] = dummy_value

        if self.is_consider_roi_rdr_cb:
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]

        arr_cube = arr_cube + self.offset_doppler # to send negative value as tensor

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
            path_desc = os.path.join(self.cfg.DATASET.DIR.DATA_ROOT, seq, 'description.txt')
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

    def get_path_data_from_path_label(self, path_label):
        seq_id, radar_idx, lidar_idx, camf_idx = self.get_data_indices(path_label)
        path_header = path_label.split('/')[:-2]

        ### Sparse tensor
        path_radar_sparse_cube = None
        if self.is_get_sparse_cube:
            if self.is_sp_another_dir:
                path_radar_sparse_cube = os.path.join(self.dir_sp, path_header[-1], self.name_sp_cube, 'spcube_'+radar_idx+'.npy')
            else:
                path_radar_sparse_cube = '/'+os.path.join(*path_header, self.name_sp_cube, 'spcube_'+radar_idx+'.npy')

        path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract_DEAR_npy', 'tesseract_'+radar_idx+'.npy')
        # path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
        # path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube_npy_f32', 'cube_'+radar_idx+'.npy')
        path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube_npy_roi', 'cube_'+radar_idx+'.npy')

        path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
        path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
        path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
        path_desc = '/'+os.path.join(*path_header, 'description.txt')

        ### In different folder
        path_radar_cube_doppler = None
        if self.is_get_cube_dop:
            if self.is_dop_another_dir:
                path_radar_cube_doppler = os.path.join(self.dir_dop, path_header[-1], 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')
            else:
                path_radar_cube_doppler = '/'+os.path.join(*path_header, 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')

        ### Currently not used
        # path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
        # path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
        # path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')

        dict_path = {
            'rdr_sparse_cube'   : path_radar_sparse_cube,
            'rdr_tesseract'     : path_radar_tesseract,
            'rdr_cube'          : path_radar_cube,
            'rdr_cube_doppler'  : path_radar_cube_doppler,
            'ldr_pc_64'         : path_lidar_pc_64,
            'cam_front_img'     : path_cam_front,
            'path_calib'        : path_calib,
            'path_desc'         : path_desc,
            'path_label'        : path_label,
        }

        return dict_path
    
    def __getitem__(self, idx):
        # try:
        sample = self.samples[idx]
        dict_item = {}
        dict_item['meta'] = {'seq': sample['seq'], 'frame': sample['frame'], 'rdr_frame': sample['rdr_frame']}
        dict_item['objs'] = sample['objs']
        # dict_item[f'meta']['desc'] = self.get_description(dict_item['seq'])
        # try:
            ### Get only required data ###
        if self.cfg.DATASET.GET_ITEM['rdr_sparse_cube']:
            # rdr_sparse_cube = 
            # dict_item['rdr_sparse_cube'] = self.get_spcube(rdr_sparse_cube)
            pass
        if self.cfg.DATASET.GET_ITEM['rdr_tesseract']:
            dict_item['rdr_tesseract'] = self.get_tesseract(sample['seq'], sample['rdr_frame'])
        if self.cfg.DATASET.GET_ITEM['rdr_cube']:
            rdr_cube = self.get_cube(sample['seq'], sample['rdr_frame'])
            # rdr_cube = self.get_cube_direct(dict_path['rdr_cube'])
            dict_item['rdr_cube'] = rdr_cube
        # if self.cfg.DATASET.GET_ITEM['rdr_cube_doppler']:
            # dict_item['rdr_cube_doppler'] = self.get_cube_doppler(dict_path['rdr_cube_doppler'])
        if self.cfg.DATASET.GET_ITEM['ldr_pc_64']:
            dict_item['ldr_pc_64'] = self.get_pc_lidar(sample['seq'], sample['frame'])
        ### Get only required data ###
        dict_item.update(mode=self.split)
        dict_item, _ = self.pipeline(dict_item, info=self.cfg)
        return dict_item
        # except:
        #     print(f'Exception error (Dataset): __getitem__ error: {sample["seq"]}/{sample["frame"]}/{sample["rdr_frame"]}')
        #     return None
        
    def evaluation(self, detections, output_dir=None, testset=False):
        # TODO: implement the evaluation


        res = None

        return res, None
    
    @staticmethod
    def collate_fn(batch_list):
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
                        "ind", "mask", "cat"]:
                ret[key] = collections.defaultdict(list)
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        ret[key][str(idx)].append(torch.tensor(ele))
                for kk, vv in ret[key].items():
                    res.append(torch.stack(vv))
                ret[key] = res  # [task], task: (batch, num_class_in_task, feat_shape_h, feat_shape_w)
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

data = [frame_info]
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
        0.028713687634213336,
        -1.5706896719911638,
        1.5620672579302894
        ],
        "xyz": [
        5.095513326251193,
        0.2371579400743387,
        25.564737699514197
        ],
        "lwh": [
        3.280036683179418,
        1.9069884147384841,
        1.4570745923842519
        ]
    }
obj pose in lidar coordinate

'''
