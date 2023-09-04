import json
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="combine train test viz file")
    parser.add_argument("train_viz_file", help="path for json visualization file with prediction on train set")
    parser.add_argument("test_viz_file", help="path for json visualization file with prediction on test set")
    args = parser.parse_args()
    return args


def combine_file(train_viz_path, test_viz_path):
    with open(train_viz_path, 'r') as f:
        combine_pred = json.load(f)
    with open(test_viz_path, 'r') as f:
        test_pred = json.load(f)
    for seq, frames in test_pred.items():
        combine_pred[seq].update(frames)
    return combine_pred



if __name__ == '__main__':
    args = parse_args()
    combine_pred = combine_file(args.train_viz_file, args.test_viz_file)
    save_path = os.path.join(os.path.dirname(args.train_viz_file), 'train_test_prediction_viz_format.json')
    with open(save_path, 'w') as f:
        json.dump(combine_pred, f, indent=2)