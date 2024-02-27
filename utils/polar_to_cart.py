import numpy as np
import scipy.io as sio
import cruw
from cruw.mapping.coor_transform import cart2pol_ramap
from cruw.mapping.ops import ra2idx_interpolate, bilinear_interpolate
import os
from tqdm import tqdm
import concurrent.futures

def rad2cart(rad, range_grid, angle_grid, xz_grid):
    """
    Convert RAD tenosr to cart coordinates
    :param rad: range azimuth doppler tensor
    :param range_grid:
    :param angle_grid:
    :param dim: output cart RF image dimension
    :param zrange: largest range in z axis (in meter)
    :param magnitude_only: convert magnitude map only
    """
    xline, zline = xz_grid
    dim = (len(zline), len(xline))
    rf_cart = np.zeros([dim[0], dim[1], rad.shape[-1]]).astype(rad.dtype)
    for zi in range(dim[0]):
        for xi in range(dim[1]):
            x, z = xline[xi], zline[zi]
            rng, agl = cart2pol_ramap(x, z)
            # rid, aid = ra2idx(rng, agl, range_grid, angle_grid)
            rid_inter, aid_inter = ra2idx_interpolate(rng, agl, range_grid, angle_grid)
            rf_cart[zi, xi] = bilinear_interpolate(rad, aid_inter, rid_inter)

    return rf_cart, (xline, zline)


cruw3d_root = '/mnt/ssd3/CRUW3D/seqs'
target_dir = 'rad_cart'


def process_seq(seq_id):
    print(f'Now processing: {seq_id}')
    os.makedirs(os.path.join(cruw3d_root, seq_id, target_dir), exist_ok=True)
    seq_dir = os.path.join(cruw3d_root, seq_id)
    for mat_file in sorted(os.listdir(os.path.join(seq_dir, 'rad'))):
        if mat_file.endswith('.mat'):
            mat = sio.loadmat(os.path.join(seq_dir, 'rad', mat_file))
            # radar_rad = np.abs(mat['Dopdata_crop']).astype(np.float32)
            radar_rad = mat['Dopdata_crop']
            radar_rad_cart, (xline, zline) = rad2cart(radar_rad, cruw.range_grid, cruw.angle_grid, cruw.xz_grid)
            radar_rad_cart_stacked = np.stack([radar_rad_cart.real, radar_rad_cart.imag], axis=0).astype(np.float32)
            with open(os.path.join(seq_dir, target_dir, mat_file.replace('.mat', '.npy')), 'wb') as f:
                np.save(f, radar_rad_cart_stacked)


multi_process = True
 
if __name__ == '__main__':
    cruw = cruw.CRUW(data_root='./', sensor_config_name='cruw-devkit/cruw/dataset_configs/sensor_config_cruw2022_3ddet.json')
    # seqs_to_pass = ['2021_1120_1616', '2022_0203_1428', '2021_1120_1632']
    # targted_seqs = ['2021_1120_1616', '2022_0203_1428', '2021_1120_1632']
    if multi_process:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            seq_ids = sorted(os.listdir(cruw3d_root))
            executor.map(process_seq, seq_ids)
    else:
        for seq_id in sorted(os.listdir(cruw3d_root)):
            process_seq(seq_id)