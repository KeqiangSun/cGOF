import cv2
import os
import numpy as np
from PIL import Image


def resize_dir(inp_dir, size=(128, 128)):
    h, w = size
    inp_dir = os.path.dirname(inp_dir+'/')
    out_dir = inp_dir + f'_{h}'
    for r, ds, fs in os.walk(inp_dir):
        for f in fs:
            img_path = os.path.join(r, f)
            img = cv2.imread(img_path)
            out = cv2.resize(img, size)
            save_path = img_path.replace(inp_dir, out_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, out)


def resize_dir_pil(inp_dir, size=(128, 128)):
    h, w = size
    inp_dir = os.path.dirname(inp_dir+'/')
    out_dir = inp_dir + f'_{h}'
    for r, ds, fs in os.walk(inp_dir):
        for f in fs:
            img_path = os.path.join(r, f)
            img = Image.open(img_path)
            out = img.resize(size, Image.ANTIALIAS)
            save_path = img_path.replace(inp_dir, out_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out.save(save_path)


def split_img(inp_path, grid=(1, 5)):
    img = cv2.imread(inp_path)*0.5
    save_dir = inp_path[:-4]
    os.makedirs(save_dir, exist_ok=True)
    h, w, c = img.shape
    row, col = grid
    h_split = int(h/row)
    w_split = int(w/col)
    for i in range(row):
        for j in range(col):
            img_split = img[i*h_split:(i+1)*h_split, j*w_split:(j+1)*w_split]
            save_path = os.path.join(save_dir, f"{i}_{j}.png")
            cv2.imwrite(save_path, img_split)


def flip_dir(inp_dir, flipcode=1):
    inp_dir = os.path.dirname(inp_dir+'/')
    out_dir = inp_dir + "_flip"
    for r, ds, fs in os.walk(inp_dir):
        for f in fs:
            img_path = os.path.join(r, f)
            img = cv2.imread(img_path)
            out = cv2.flip(img, flipcode)
            save_path = img_path.replace(inp_dir, out_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, out)


def alpha_blend_dir(hr_dir, lr_dir):
    out_dir = os.path.dirname(hr_dir+'/') + f'_alphablend/'
    os.makedirs(out_dir, exist_ok=True)
    for r, ds, fs in os.walk(hr_dir):
        for f in fs:
            if f[-4:] != '.png':
                continue
            hr_path = os.path.join(r, f)
            hr_img = cv2.imread(hr_path)

            lr_path = hr_path.replace(hr_dir, lr_dir).replace('img', 'face').replace('_00.png', '.png')
            print(lr_path)
            lr_img = cv2.imread(lr_path, flags=cv2.IMREAD_UNCHANGED)

            h, w, c = lr_img.shape
            corner = hr_img[-h:, -w:]
            hr_img[-h:, -w:] = corner*(1-lr_img[..., -1:]/255) + lr_img[..., :-1]*(lr_img[..., -1:]/255)
            cv2.imwrite(hr_path.replace(hr_dir, out_dir), hr_img)


def cat_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/id/pigan_recon4_snm_depr10000/splits",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        seeds_rows="s19i4,s57i0,s23i0,s65i0,s70i3",
        cols="0,1,6,8,11"
        ):
    seeds_rows = seeds_rows.split(',')
    cols = cols.split(',')
    output_img = []
    for s_i in seeds_rows:
        seed, i = s_i.strip('s').split('i')
        img_line = []
        for j in cols:
            img_name = f"grid_seed{seed}/img_seed{seed}_i{i}_j{j}.png"
            img_path = os.path.join(data_root, img_name)
            img = cv2.imread(img_path)
            img_line.append(img)

            pickup_dir = os.path.join(data_root, "../pickup") + '/'
            os.makedirs(pickup_dir, exist_ok=True)
            dest_path = os.path.join(pickup_dir, os.path.basename(img_name))
            os.system(f"cp {img_path} {dest_path}")

        img_line = np.concatenate(img_line, axis=1)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=0)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img


def cat_exp_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300/splits/imgs",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        # seeds="439,166,181,373,236,247,673,390,472,544",
        seeds="31,373,236,439,166,160,183,302,374,551,673,247,544,191,231",
        rows_cols="i8j0,i6j8,i10j11,i1j7,i1j2,i1j3,i9j3,i6j6,i2j11,i7j4,i10j6",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    pickup_dir = os.path.join(data_root, "../pickup") + '/'
    os.makedirs(pickup_dir, exist_ok=True)
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            img_name = f"grid_seed{seed}/row{row}/{data_type}_seed{seed}_i{row}_j{col}.png"
            img_path = os.path.join(data_root, img_name)
            img = cv2.imread(img_path)
            img_line.append(img)

            dest_path = os.path.join(pickup_dir, os.path.basename(img_name))
            os.system(f"cp {img_path} {dest_path}")

        # img_line.insert(1, np.ones((img_line[0].shape[0],5,3))*255)
        img_line = np.concatenate(img_line, axis=1)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=0)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img
# cat_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/imgs",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="31,373,236,439,166,160,183,302,374,551,673,247,544,191,231",
#         rows_cols="i10j10,i0j0,i6j11,i1j3",
#         data_type="img",
#         )
# cat_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/imgs",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i9j3,i6j6,i6j0,i0j4,i8j7,i10j5,i4j3,i7j4,i10j6",
#         data_type="img",
#         )
# cat_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/imgs",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i8j0",
#         data_type="img",
#         )
# cat_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.2/splits/imgs",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="31,373,236,439,166,160,183,302,374,551,673,247,544,191,231",
#         rows_cols="i6j8,i10j11,i6j9,i2j4",
#         data_type="img",
#         )
# cat_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.2/splits/imgs",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i9j3,i6j6,i6j0,i0j4,i8j7,i10j5,i4j3,i7j4,i10j6",
#         data_type="img",
#         )

def cat_sr_exp_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300/splits/imgs/pickup_aug_sr",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        # seeds="439,166,181,373,236,247,673,390,472,544",
        seeds="31,373,236,439,166,160,183,302,374,551,673,247,544,191,231",
        rows_cols="i8j0,i6j8,i10j11,i1j7,i1j2,i1j3,i9j3,i6j6,i2j11,i7j4,i10j6",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            img_name = f"img_seed{seed}_i{row}_j{col}_00.png"
            img_path = os.path.join(data_root, img_name)
            img = cv2.imread(img_path)
            img_line.append(img)

        img_line = np.concatenate(img_line, axis=1)
        output_img.append(img_line)
        output_img.append(np.ones((25,img_line.shape[1],3),dtype=img_line.dtype)*255)
    output_img = np.concatenate(output_img, axis=0)
    cv2.imwrite(os.path.join(data_root, 'concat_img.png'), output_img)
    return output_img
# cat_sr_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/pickup_wild_aug_sr/restored_faces",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i8j0",
#         data_type="img",
#         )
# cat_sr_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/pickup_wild_aug_sr/restored_faces",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i10j10,i0j0,i6j11,i1j3",
#         data_type="img",
#         )
# cat_sr_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.2/splits/pickup_mild_aug_sr/restored_faces",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i6j8,i10j11,i6j9,i2j4",
#         data_type="img",
#         )

# cat_sr_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/pickup_aug_sr/restored_faces",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i9j3,i6j6,i6j0,i8j7,i4j3,i7j4,i10j6,i0j4",
#         data_type="img",
#         )
# cat_sr_exp_imgs(
#         data_root="/home/kqsun/Tasks/pigan/imgs/exp/exp/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_1.3/splits/pickup_aug_sr/restored_faces",
#         # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
#         # seeds="439,166,181,373,236,247,673,390,472,544",
#         seeds="373,236,439,166,160,183,302,374,551,247,544,231",
#         rows_cols="i8j0",
#         data_type="img",
#         )

def cat_pose_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/poses/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_anglemultiplier3.0_15x15/splits/imgs",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        seeds="544,443,191,318,258,181,247,231,673,031,479,389,252",
        # seeds="544,443,191,318,258",
        rows_cols="i4j4,i6j5,i7j5,i7j7,i7j9,i6j9,i4j10",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    pickup_dir = os.path.join(data_root, "../pickup") + '/'
    os.makedirs(pickup_dir, exist_ok=True)
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            seed = int(seed)
            row = int(row)
            col = int(col)
            img_name = f"grid_seed{seed}/{data_type}_{seed:03}_{row:02}_{col:02}.png"
            img_path = os.path.join(data_root, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img_line.append(img)

            dest_path = os.path.join(pickup_dir, os.path.basename(img_name))
            os.system(f"cp {img_path} {dest_path}")

        img_line = np.concatenate(img_line, axis=1)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=0)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img


def cat_sr_pose_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/poses/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_anglemultiplier3.0_15x15/splits/pickup_aug_sr/restored_faces/",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        # seeds="439,166,181,373,236,247,673,390,472,544",
        # seeds="544,443,191,231,247",
        # rows_cols="i4j4,i8j5,i7j7,i8j9,i4j10",
        # rows_cols="i4j4,i6j5,i7j5,i7j7,i7j9,i6j9,i4j10",
        # seeds="544,443,191,318,258",
        # rows_cols="i4j4,i6j5,i7j7,i6j9,i4j10",
        # seeds="544,443,191,318,258,181,247,673,252",
        seeds="544,258,318,181,673",
        rows_cols="i4j4,i6j5,i7j7,i6j9,i4j10",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    pickup_dir = os.path.join(data_root, "../pickup") + '/'
    os.makedirs(pickup_dir, exist_ok=True)
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            seed = int(seed)
            row = int(row)
            col = int(col)
            img_name = f"{data_type}_{seed:03}_{row:02}_{col:02}_00.png"
            img_path = os.path.join(data_root, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img_line.append(img)


        img_line = np.concatenate(img_line, axis=1)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=0)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img


def cat_id_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/id/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300/splits/imgs",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        # seeds="439,166,181,373,236,247,673,390,472,544",
        #          1,  2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42
        # seeds="183,551,1003,1004,1073,1125,1165,1178,1183,1199,1234,1275,1305,1416,1486,1522,1542,1555,1608,1635,1642,1643,1649,1651,1666,1676,1677,1712,1717,1718,1732,1752,1761,1782,1840,1857,1863,1893,1898,1923,1929,1990",
        # seeds="183,1003,1073,1125,1165,1183,1199,1486,1555,1651,1857",
        # rows_cols="i1j7,i6j9,i5j3,i1j3,i5j4,i8j8",
        # rows_cols="i5j3,i5j4,i3j0,i1j2,i0j1,i0j5,i1j6,i2j1,i2j4,i7j3",
        seeds="302,551,1003,1165,1486,1555",
        rows_cols="i5j3,i5j4,i1j2,i0j1,i0j5,i2j4,i2j9",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    pickup_dir = os.path.join(data_root, "../pickup") + '/'
    os.makedirs(pickup_dir, exist_ok=True)
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            seed = int(seed)
            row = int(row)
            col = int(col)
            img_name = f"grid_seed{seed}/row{row}/{data_type}_seed{seed:03}_i{row}_j{col}.png"
            img_path = os.path.join(data_root, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img_line.append(img)

            dest_path = os.path.join(pickup_dir, os.path.basename(img_name))
            os.system(f"cp {img_path} {dest_path}")

        img_line = np.concatenate(img_line, axis=0)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=1)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img


def cat_sr_id_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/id/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300/splits/pickup_aug_sr/restored_faces/",
        # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
        # seeds="439,166,181,373,236,247,673,390,472,544",
        # seeds="160,183,302,374,551",
        # # rows_cols="i5j3,i1j3,i5j4,i1j7,i8j8",
        # rows_cols="i5j3,i2j11,i5j4,i1j11,i8j8",
        # seeds="160,374,183,302,551,288",
        # rows_cols="i5j3,i2j11,i5j4,i3j0,i1j2,i0j1,i0j5,i1j6,i2j1,i2j4,i7j3",
        # seeds="302,551,1003,1165,1486,1555",
        # rows_cols="i5j3,i5j4,i1j2,i0j1,i0j5,i2j4,i2j9",
        seeds="302,551,1003,1165,1486",
        rows_cols="i5j3,i5j4,i1j2,i0j1,i2j9",
        data_type="img",
        ):
    rows_cols = rows_cols.split(',')
    seeds = seeds.split(',')
    output_img = []
    pickup_dir = os.path.join(data_root, "../pickup") + '/'
    os.makedirs(pickup_dir, exist_ok=True)
    for seed in seeds:
        img_line = []
        for row_col in rows_cols:
            row, col = row_col.strip('i').split('j')
            seed = int(seed)
            row = int(row)
            col = int(col)
            img_name = f"{data_type}_seed{seed:03}_i{row}_j{col}_00.png"
            img_path = os.path.join(data_root, img_name)
            print(img_path)
            img = cv2.imread(img_path)
            img_line.append(img)


        img_line = np.concatenate(img_line, axis=0)
        output_img.append(img_line)
    output_img = np.concatenate(output_img, axis=1)
    cv2.imwrite(os.path.join(pickup_dir, 'concat_img.png'), output_img)
    return output_img


if __name__ == "__main__":
    # cat_imgs(
    #     data_root="/home/kqsun/Tasks/pigan/imgs/id/pigan_recon4_snm_depr10000/splits",
    #     # seeds_rows="s1i0,s4i2,s13i0,s19i4,s23i0,s36i3,s38i0,s42i0,s46i2,s48i4,s49i3,s54i0,s55i0,s57i0,s62i3,s62i1,s65i0,s69i2,s70i3,s78i3,s80i2,s83i1,s106i2",
    #     seeds_rows="s19i4,s57i0,s23i0,s65i0,s70i3",
    #     cols="0,1,6,8,11"
    # )
    cat_imgs(
        data_root="/home/kqsun/Tasks/pigan/imgs/exp/pigan_recon4_snm_depr10000/splits",
        seeds_rows="s7i4,s91i4,s91i2,s109i3,s124i3",
        cols="4,1,8,0,9"
    )