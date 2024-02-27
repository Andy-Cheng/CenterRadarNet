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


seq_to_viz = []
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
    parser.add_argument("--is_test", action='store_true', help="whether to visualize test set")
    parser.add_argument("--pred_path", help="prediction pickle file")
    parser.add_argument("--viz_conf_thres", help="visualization confidence threshold")
    args = parser.parse_args()
    return args

# group prediction with sequence
def load_pred(pred_file):
    with open (pred_file, 'rb') as f:
        pred = pickle.load(f)
    print('Transforming prediction to dict indexed by sequence...')
    new_pred = defaultdict(dict)
    for seq_frame, v in tqdm(pred.items()):
        seq, frame, rdr_frame = seq_frame.split('/')
        new_pred[seq].update({frame: v})
    # with open(pred_file, "wb") as f:
    #     pickle.dump(new_pred, f)
    return new_pred


def viz_frame(dataset, pred_frame, save_dir, conf_thres, seq, gt_seq):
    if isinstance(pred_frame['box3d'], torch.Tensor):
        pred_frame['box3d'] = pred_frame['box3d'].numpy() 
        pred_frame['scores'] = pred_frame['scores'].numpy() 
        pred_frame['label_preds'] = pred_frame['label_preds'].numpy() 
    func_show_radar_cube_bev_cruw(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
        pred_frame['scores'], pred_frame['label_preds']), save_dir, conf_thres, seq, gt_seq)
    # func_show_pointcloud(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
    # pred_frame['scores'], pred_frame['label_preds']), save_dir_name, conf_thres, seq)

'''
    save_dir_name: name of the model used for inference
'''

def viz_seq(viz_root, dataset, pred_seq, gt_seq, conf_thres, seq):
    save_dir = os.path.join(viz_root, 'inference_viz', seq, f'conf_{conf_thres}')
    os.makedirs(save_dir, exist_ok=True)
    # get_label_bboxes()
    conf_thres = float(conf_thres)
    for k, pred_frame in tqdm(sorted(pred_seq.items(), key=lambda x: x[0])):
        # frame = pred_frame['metadata']['frame']
        # if (seq, frame) in scenario_frame_dict:
        # if seq in seq_to_viz:
        # if seq == '11':
            # if draw_idx == 0:
            #     o3d_vis.run()
        viz_frame(dataset, pred_frame, save_dir, conf_thres, seq, gt_seq)
    # o3d_vis.destroy_window()
    

# Currently support single dataset visualization
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"Start visualization")
    dataset = build_dataset(cfg.data.test if args.is_test else cfg.data.train)
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
            new_objs = []
            for obj in sample['objs']:
                new_objs.append([obj['obj_type'], obj['obj_id'], [*obj['xyz'], obj['euler'][2], *obj['lwh']]])
            gt[seq].update({frame: new_objs})
        with open(new_gt_file, 'w') as f:
            json.dump(gt, f, indent=2)

    pred = load_pred(args.pred_path)
    viz_root = os.path.dirname(args.pred_path)
    for seq, pred_seq in sorted(pred.items()):
        # if seq in ['7', '8', '10']:
        logger.info(f'Now visualizing sequence {seq}')
        viz_seq(viz_root, dataset, pred_seq, gt[seq], args.viz_conf_thres, seq)



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



        


if __name__ == '__main__':
    # scenario_frame_dict = [(k, v) for k, v in scenario_frame_dict.items()]
    # scenario_frame_dict = [('12', '00310_00274'), ('12', '00300_00264'), ('12', '00316_00280'), ('12', '00817_00781')]
    # scenario_frame_dict = [[(k, v) for k, v in scenario_frame_dict.items()][0]]
    main()
    # viz_tracking()
