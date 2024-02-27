import os
import shutil
from tqdm import tqdm

source_parent_dir = '/mnt/nas_cruw/CRUW_2022'
destination_parent_dir = '/mnt/ssd3/CRUW3D/seqs'

# Iterate over each directory in the destination parent directory
for directory in os.listdir(destination_parent_dir):
    print('Processing directory: {}'.format(directory))
    destination_dir = os.path.join(destination_parent_dir, directory, 'ra_cart')
    source_dir = os.path.join(source_parent_dir, directory, 'radar/npy/ra_cart')

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Copy each file from the source directory to the destination directory
    for filename in tqdm(os.listdir(source_dir)):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        shutil.copy2(source_file, destination_file)
