import numpy as np
import os

cruw3d_root = '/mnt/ssd3/CRUW3D/seqs'
target_seq = ['2021_1120_1616', '2021_1120_1618', '2022_0203_1428', '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0203_1512', '2022_0217_1232', '2022_0217_1251', '2022_0217_1307', '2022_0217_1322', '2021_1120_1632', '2021_1120_1634']
mins, maxs = [], []
for seq in target_seq:
    rad_paths = os.listdir(os.path.join(cruw3d_root, seq, 'rad_cart_mag'))
    rad_path = np.random.choice(rad_paths)
    rad_path = os.path.join(cruw3d_root, seq, 'rad_cart_mag', rad_path)
    rad = np.load(rad_path)
    print(rad.min(), rad.max())
    mins.append(rad.min())
    maxs.append(rad.max())
print(min(mins), max(maxs))
    