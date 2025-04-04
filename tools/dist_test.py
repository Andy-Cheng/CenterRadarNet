import argparse
import copy
import json
import os
import sys

try:
    import apex
except:
    print("No APEX!")
import numpy as np
import torch
import yaml
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle 
import time
from det3d.torchie.utils import count_parameters 
from pathlib import Path
from collections import defaultdict


'''
viz format example:
 each frame: [{
    "obj_id": "0",
    "obj_type": "Sedan",
    "psr": {
      "position": {
        "x": 26.323807590819317,
        "y": -4.672016669736319,
        "z": -0.2779390447352759
      },
      "rotation": {
        "x": 0,
        "y": 0,
        "z": -0.016496852089850397
      },
      "scale": {
        "x": 3.280036683179418,
        "y": 1.9069884147384841,
        "z": 1.4570745923842519
      }
    }
  }]
'''

def save_pred(type_coord, class_names, pred, root, checkpoint_name, dataset_split):
    tr_LR_tvec = np.array([2.54, -0.3, -0.7]).reshape(1, 3)
    # TODO: make pred_new sorted by seq_name
    pred_new = {}
    pred_viz_format = defaultdict(dict)
    for k, v in pred.items():
        v_new = {}
        for item_key, item_val in v.items():
            v_new.update({item_key:item_val.to('cpu').numpy() if isinstance(item_val, torch.Tensor) else item_val})
        pred_new.update({k: v_new})
        seq, frame, rdr_frame = k.split('/')
        frame_objs, viz_frame_objs = [], []
        if type_coord == '1':
            v_new['box3d'][:, 0:3] += tr_LR_tvec
        for i in range(len(v_new['box3d'])):
            frame_obj = {}
            frame_obj['obj_id'] = ''
            frame_obj['obj_type'] = class_names[int(v_new['label_preds'][i])]
            frame_obj['score'] = float(v_new['scores'][i])
            frame_obj['psr'] = {}
            x, y, z, l, w, h, theta = v_new['box3d'][i].tolist()
            frame_obj['psr']['position'] = [x, y, z]
            frame_obj['psr']['rotation'] = [0., 0., theta]
            frame_obj['psr']['scale'] = [l, w, h]
            frame_objs.append(frame_obj)
            frame_obj['psr']['position'] = {'x': x + tr_LR_tvec[0, 0], 'y': y + tr_LR_tvec[0, 1], 'z': z + tr_LR_tvec[0, 2]}
            frame_obj['psr']['rotation'] = {'x': 0, 'y': 0, 'z': theta}
            frame_obj['psr']['scale'] = {'x': l, 'y': w, 'z': h}
            viz_frame_objs.append(frame_obj)
        pred_viz_format[seq][frame] = {'objs': viz_frame_objs, 'split': dataset_split}
        
    save_pred_dir = os.path.join(root, f"{checkpoint_name}")
    os.makedirs(save_pred_dir, exist_ok=True)
    with open(os.path.join(save_pred_dir, f"{dataset_split}_prediction.pkl"), "wb") as f:
        pickle.dump(pred_new, f)
    with open(os.path.join(save_pred_dir, f"{dataset_split}_prediction_viz_format.json"), "w") as f:
        json.dump(pred_viz_format, f, indent=2)
    return pred_new






def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")
    parser.add_argument("--pred_file", default="none")


    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():

    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info("Distributed testing: {}".format(distributed))
    logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    logger.info(f'Model parameter count: {count_parameters(model)}')
    

    cfg['DATASET']['MODE'] = 'test'
    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not args.speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    # put model on gpus
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        # model = fuse_bn_recursively(model)
        model = model.cuda()

    model.eval()
    mode = "val"

    logger.info(f"work dir: {args.work_dir}")
    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 /3)

    time_start = 0 
    time_end = 0 
    checkpoint_file_name = args.checkpoint.split('/')[-1].split('.')[0]

    if args.pred_file != 'none':
        with open(args.pred_file, 'rb') as f:
            pred_np_array = pickle.load(f)

    else:
        for i, data_batch in enumerate(data_loader):
            if i == start:
                torch.cuda.synchronize()
                time_start = time.time()

            if i == end:
                torch.cuda.synchronize()
                time_end = time.time()

            with torch.no_grad():
                outputs = batch_processor(
                    model, data_batch, train_mode=False, local_rank=args.local_rank,
                )
            for output in outputs:
                if 'token' in output['metadata']:
                    token = output["metadata"]["token"]
                    for k, v in output.items():
                        if k not in [
                            "metadata",
                        ]:
                            output[k] = v.to(cpu_device)
                    detections.update(
                        {token: output,}
                    )
                elif 'path' in output['metadata']:
                    label_path_parts = Path(output['metadata']['path']['path_label']).parts
                    seq_name = label_path_parts[-3]
                    sample_idx = label_path_parts[-1].split('.')[0]
                    detections.update(
                        {f'{seq_name}/{sample_idx}': output,}
                    )
                else:
                    seq_name = output['metadata']['seq']
                    frame_name = output['metadata']['frame']
                    rdr_frame_name = output['metadata']['rdr_frame']
                    # if 'app_emb' in output:
                    #     app_emb = output.pop('app_emb')
                    #     save_path = os.path.join(cfg.test_cfg.app_emb_save_path, seq_name)
                    #     os.makedirs(save_path, exist_ok=True)
                    #     np.save(os.path.join(save_path, f'{frame_name}.npy'), app_emb)
                    detections.update(
                        {f'{seq_name}/{frame_name}/{rdr_frame_name}': output,}
                    )
                    
                if args.local_rank == 0:
                    prog_bar.update()
                

        synchronize()

        all_predictions = all_gather(detections)

        try:
            print("\n Total time per frame: ", (time_end -  time_start) / (end - start)) # TODO: fix bug
        except:
            pass

        if args.local_rank != 0:
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)

        pred_np_array = save_pred(cfg.DATASET.TYPE_COORD, cfg.class_names, predictions, args.work_dir, checkpoint_file_name, 'test' if args.testset else 'train')

    result_dict, _ = dataset.evaluation(pred_np_array, output_dir=args.work_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict.items():
            print(f"Evaluation {k}: {v}")
    data_split = 'test' if args.testset else 'train'
    with open(os.path.join(args.work_dir, checkpoint_file_name, f"eval_result_{data_split}.json"), "w") as f:
        json.dump(result_dict, f, indent=2)

    if args.txt_result:
        assert False, "No longer support kitti"

if __name__ == "__main__":
    main()
