from det3d.datasets import build_dataset
import argparse
import os
import pickle
from det3d.torchie import Config
from det3d.torchie.apis import (
    get_root_logger,
)
from tqdm import tqdm
from utils.viz_kradar_funcs import *
from pathlib import Path
import torch
from collections import defaultdict

# scenario_frame_dict = {
#     '7': '00182_00149',
# 	'21': '00506_00502',
# 	'22': '00492_00490',
# 	'26': '00252_00247',
# 	'42': '00294_00285',
# 	'44': '00575_00566',
# 	'53': '00479_00475',
# 	'57': '00294_00289',
#     '34': '00171_00168',
# }


seq_to_viz = ["7", "54", "49", "41", "40" ] # '11', "25", "24", "23", "21", "20"

# scenario_frame_dict = {
#     '11': '00337_00304',
#     '53': '00479_00475',
# 	'57': '00294_00289',
# 	'42': '00294_00285',
#     '34': '00171_00168',
#     # '12': '00465_00429',
#     '22': '00538_00536',
# }

# scenario_frame_dict = {
#     '3': '00180_00150',
#     '12': '00310_00274',
# 	'27': '00302_00298',
# 	'28': '00242_00239',
# 	'48': '00192_00239',
#     '2': '00195_00165',
#     '12': '00300_00264',
#     '12': '00316_00280',
#     '12': '00817_00781',

# }

scenario_frame_dict = {
    '3': '00180_00150',
    '12': '00310_00274',
	'27': '00302_00298',
	'28': '00242_00239',
	'48': '00192_00239',
    '2': '00195_00165',
    '12': '00300_00264',
    '12': '00316_00280',
    '12': '00817_00781',

}




def parse_args():
    parser = argparse.ArgumentParser(description="Kradar detection visualization")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--pred_path", help="prediction pickle file")
    parser.add_argument("--checkpoint_name", help="checkpoint file generating the prediction")
    parser.add_argument("--viz_conf_thres", help="visualization confidence threshold")
    args = parser.parse_args()
    return args

# group prediction with sequence
def load_pred(pred_file):
    with open (pred_file, 'rb') as f:
        pred = pickle.load(f)
    if len(pred.items()) < 59:
        return pred
    print('Transforming prediction to dict indexed by sequence...')
    new_pred = defaultdict(dict)
    for seq_frame, v in tqdm(pred.items()):
        seq, frame = seq_frame.split('/')
        new_pred[seq].update({frame: v})
    with open(pred_file, "wb") as f:
        pickle.dump(new_pred, f)
    return new_pred


def viz_frame(dataset, pred_frame, save_dir_name, conf_thres, seq):
    if isinstance(pred_frame['box3d'], torch.Tensor):
        pred_frame['box3d'] = pred_frame['box3d'].numpy() 
        pred_frame['scores'] = pred_frame['scores'].numpy() 
        pred_frame['label_preds'] = pred_frame['label_preds'].numpy() 
    func_show_radar_cube_bev(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
        pred_frame['scores'], pred_frame['label_preds']), save_dir_name, conf_thres, seq)
    # func_show_pointcloud(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
    # pred_frame['scores'], pred_frame['label_preds']), save_dir_name, conf_thres, seq)

'''
    save_dir_name: name of the model used for inference
'''

def viz_seq(kradar_dataset, pred_seq, checkpoint_name, conf_thres):
    pred_frame_tmp = list(pred_seq.values())[0]
    seq_root_seg = Path(pred_frame_tmp['metadata']['path']['path_calib']).parts[:-2]
    save_dir = os.path.join(*seq_root_seg, 'inference_viz', f'{checkpoint_name}_{conf_thres}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get_label_bboxes()
    conf_thres = float(conf_thres)
    for k, pred_frame in tqdm(sorted(pred_seq.items(), key=lambda x: x[0])):
        path_label_parts = pred_frame['metadata']['path']['path_label'].split('/')
        seq = path_label_parts[-3]
        frame = path_label_parts[-1].split('.')[0]
        # if (seq, frame) in scenario_frame_dict:
        if seq in seq_to_viz:
        # if seq == '11':
            # if draw_idx == 0:
            #     o3d_vis.run()
            viz_frame(kradar_dataset, pred_frame, save_dir, conf_thres, seq)
    # o3d_vis.destroy_window()
    

# Currently support single dataset visualization
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"Start visualization")
    dataset = build_dataset(cfg.data.test)
    pred = load_pred(args.pred_path)
    for seq, pred_seq in sorted(pred.items()):
        # if seq in ['7', '8', '10']:
        logger.info(f'Now visualizing sequence {seq}')
        viz_seq(dataset, pred_seq, args.checkpoint_name, args.viz_conf_thres)



def load_tracking_file(path):
    with open(path, 'r') as f:
        tracklet_file = f.readlines()
    tracklets = [x.strip() for x in tracklet_file]
    frame_to_dets = defaultdict(list)
    for tracklet in tracklets:
        frame_id, tracking_id, x, y, xl, yl, rotation, score, _, _ ,_ = tracklet.split(',')
        frame_to_dets[frame_id].append((tracking_id, x, y, xl, yl, rotation, score))

    return frame_to_dets



seq_to_offset = {
    '7': 33,
    '10': 26,
    '11': 33,
    '21': 4,
    '23': 2,
    '24': 3,
    '25': -1,
    '19': 37,
    '54': 6
}

import matplotlib.patches as patches


class RotatingRectangle(patches.Rectangle):
    def __init__(self, xy, width, height, rel_point_of_rot, **kwargs):
        super().__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = rel_point_of_rot
        self.xy_center = self.get_xy()
        self.set_angle(self.angle)

    def _apply_rotation(self):
        angle_rad = self.angle * np.pi / 180
        m_trans = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                            [np.sin(angle_rad), np.cos(angle_rad)]])
        shift = -m_trans @ self.rel_point_of_rot
        self.set_xy(self.xy_center + shift)

    def set_angle(self, angle):
        self.angle = angle
        self._apply_rotation()

    def set_rel_point_of_rot(self, rel_point_of_rot):
        self.rel_point_of_rot = rel_point_of_rot
        self._apply_rotation()

    def set_xy_center(self, xy):
        self.xy_center = xy
        self._apply_rotation()

# Get the jet colormap
jet = plt.get_cmap('jet')

# Create an array with the colors of the jet colormap
colors = jet(np.arange(jet.N))

# Set the RGBA value for zero values to be white (or any other color)
colors[0, :] = np.array([1, 1, 1, 1])  # RGBA value
import matplotlib.colors as mcolors
# Create a new colormap from those colors
new_cmap = mcolors.LinearSegmentedColormap.from_list('new_jet', colors)



def viz_radar_cube_bev_tracking(dataset, seq, frame_id, rdr_frame_id, gt_objs, viz_dir, tracking_id_to_color, used_colors, magnifying=1.):
    # if frame_id != '00470':
    #     return tracking_id_to_color, used_colors
    rdr_cube_bev = np.log10(dataset.get_cube(seq, rdr_frame_id)).max(axis=0) # rdr_cube in Y-X\

    rdr_cube_bev = np.transpose(rdr_cube_bev)

    ### Jet map visualization ###
    # rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    # show corresponding image
    ax.axis('off')

    ax.imshow(rdr_cube_bev[::-1, :],cmap=new_cmap, vmin=10.0,vmax=20)
    # ax.set_xlim([0,rdr_cube_bev.shape[1]])
    # ax.set_ylim([rdr_cube_bev.shape[0],0])

    
    for gt_obj in gt_objs: # (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        cls_name, obj_id, [x, y, z, theta, xl, yl, zl] = gt_obj
        if obj_id in tracking_id_to_color:
            clr = tracking_id_to_color[obj_id]
        else:
            clr = rand_colors()
            while clr in used_colors:
                clr = rand_colors()
            tracking_id_to_color[obj_id] = clr
        
        xl, yl, zl = magnifying * xl, magnifying * yl, magnifying * zl# TODO: set magnifying for better viz.
        # if cls_name not in ['Car', 'Pedestrian']  or x > 29 or y > 10 or y < -10 or z > 7.6 or z < -2:
        #     continue
        bbx_length = xl / 0.4
        bbx_width = yl /0.4

        # with rotation
        # rot  = -theta + 180
        rot  = -theta * 180 / np.pi

        bbx_cx=(80-0- x) / 0.4
        bbx_cy=abs((y-30)/0.4)

        rect1 = RotatingRectangle((bbx_cy,bbx_cx), width=bbx_width, height=bbx_length, 
                        rel_point_of_rot=np.array([bbx_width/2, bbx_length/2]),
                        angle=rot,fill=None,edgecolor=tracking_id_to_color[obj_id],linewidth=4)
        
        ax.add_patch(rect1)
        
    fig.savefig(os.path.join(viz_dir, f'{frame_id}.png'))
    plt.close()
    return tracking_id_to_color, used_colors


def viz_tracking():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    logger = get_root_logger(cfg.log_level)
    dataset = build_dataset(cfg.data.test)

    # arrange gt keyed by sequence
    new_gt_file = dataset.label_file.replace('.json', '_viz.json')
    if os.path.exists(new_gt_file):
        print('Using cached gt viz file')
        with open(new_gt_file, 'r') as f:
            gt = json.load(f)
    else:
        print('Processing gt file for viz...')
        gt = defaultdict(dict)
        for sample in dataset.samples:
            seq = sample['seq']
            frame = sample['frame']
            rdr_frame = sample['rdr_frame']
            new_objs = []
            for obj in sample['objs']:
                new_objs.append([obj['obj_type'], obj['obj_id'], [*obj['xyz'], obj['euler'][2], *obj['lwh']]])
            gt[seq].update({f'{frame}_{rdr_frame}': new_objs})
        with open(new_gt_file, 'w') as f:
            json.dump(gt, f, indent=2)

    target_seq = '5'
    viz_dir = 'tmp_Radar'
    tracking_id_to_color, used_colors = {}, []
    for frame, gt_objs in tqdm(gt[target_seq].items()):
        frame_id, rdr_frame_id = frame.split('_')
        tracking_id_to_color, used_colors = viz_radar_cube_bev_tracking(dataset, target_seq, frame_id, rdr_frame_id, gt_objs, viz_dir,  tracking_id_to_color, used_colors)
    with open('color_table.json', 'w') as f:
        json.dump(tracking_id_to_color, f, indent=2)



if __name__ == '__main__':
    # scenario_frame_dict = [(k, v) for k, v in scenario_frame_dict.items()]
    # scenario_frame_dict = [('12', '00310_00274'), ('12', '00300_00264'), ('12', '00316_00280'), ('12', '00817_00781')]
    # scenario_frame_dict = [[(k, v) for k, v in scenario_frame_dict.items()][0]]
    # main()
    viz_tracking()
