import argparse
import os
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(description="dataset IO to hugging face hub")
    parser.add_argument("seq_to_upload", help="sequence to upload and their IDs")
    parser.add_argument("sub_dirs", help="subdirs to upload")
    parser.add_argument("--seq_root", default="/mnt/ssd3/CRUW3D/seqs", help="local sequence root")
    parser.add_argument("--remote_seq_root", default="sequences", help="remote sequence root")
    parser.add_argument("--version", default="main", help="remote branch to upload")
    parser.add_argument("--pre_dir", default=None, help="prefix directory")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    hf_api = HfApi()
    with open(args.seq_to_upload, 'r') as f:
        seq_to_upload = f.readlines()
    seq_to_upload = [line.strip().split(',') for line in seq_to_upload]
    with open(args.sub_dirs, 'r') as f:
        sub_dirs = f.readlines()
    sub_dirs = [line.strip() for line in sub_dirs]
    for seq in seq_to_upload:
        seq_id = seq[0]
        seq_name = seq[1]
        seq_dir = os.path.join(args.seq_root, seq_name)
        if not os.path.exists(seq_dir):
            print(f"Sequence {seq_name} not found")
            continue
        for sub_dir in sub_dirs:
            full_path = os.path.join(seq_dir, sub_dir)
            path_in_repo = f'{args.remote_seq_root}/{seq_id}/{args.pre_dir}/{sub_dir}' if args.pre_dir else f'{args.remote_seq_root}/{seq_id}/{sub_dir}'
            if os.path.isdir(full_path):
                hf_api.upload_folder(
                    folder_path=full_path,
                    path_in_repo=path_in_repo,
                    repo_id="uwipl/CRUW3D",
                    repo_type="dataset",
                    revision=args.version,
                )
            elif os.path.isfile(full_path):
                hf_api.upload_file(
                    path_or_fileobj=full_path,
                    path_in_repo=path_in_repo,
                    repo_id="uwipl/CRUW3D",
                    repo_type="dataset",
                    revision=args.version,
                )