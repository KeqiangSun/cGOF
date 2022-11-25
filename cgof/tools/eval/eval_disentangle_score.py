import scipy.io
import numpy as np
from glob import glob
import os
import argparse


def calculate_id_std(coeff):
    mean_coeff = np.mean(coeff, axis=1, keepdims=True)
    mean_coeff = mean_coeff/np.linalg.norm(mean_coeff, axis=2, keepdims=True)
    # var_coeff = np.mean(np.sum((coeff - mean_coeff)**2,axis = 2),axis = 1)
    var_coeff = np.mean(np.arccos(np.sum(mean_coeff*coeff, axis=2))**2, axis=1)
    std_coeff = np.mean(np.sqrt(var_coeff))
    return std_coeff


def calculate_std(coeff):
    mean_coeff = np.mean(coeff, axis=0, keepdims=True)
    var_coeff = np.mean(np.sum((coeff - mean_coeff)**2, axis=1), axis=0)
    std_coeff = np.mean(np.sqrt(var_coeff))
    return std_coeff


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--factor', type=int, default=0, help='factor variation mode. 0 = id, 1 = expression, 2 = lighting, 3 = pose, 4 = all.')
    parser.add_argument('--subject', type=int, default=1000, help='how many subjects to generate.')
    parser.add_argument('--variation', type=int, default=10, help='how many images to generate per subject.')
    parser.add_argument('--recon_data_root', type=str, default=None, help='where to find the reconstructed results.')

    args = parser.parse_args()
    return args


def main():
    mean_id_std = 0.5771492
    mean_exp_std = 0.2850049
    mean_angle_std = 0.15251215

    args = parse()
    factor = args.factor
    subject = args.subject
    variation = args.variation

    data_root = args.recon_data_root

    # load all data
    data_dict = {}
    for r, ds, fs in os.walk(data_root):
        for f in fs:
            if f.endswith('.mat'):
                name = f.split('.')[0]
                data_path = os.path.join(r, f)
                mat = scipy.io.loadmat(data_path)
                data_dict[name] = mat

    factor2key = {0: 'id', 1: 'exp', 3: 'angle'}

    std_ids = []
    std_exps = []
    std_angles = []
    for s in range(subject):
        ids = []
        exps = []
        angles = []
        for v in range(variation):
            # name = f'{s:03}_{v:02}'
            name = f'{s:04}_{v:02}'
            data = data_dict[name]
            ids.append(data['id'])
            exps.append(data['exp'])
            angles.append(data['angle'])
        std_id = calculate_std(ids) / mean_id_std
        std_exp = calculate_std(exps) / mean_exp_std
        std_angle = calculate_std(angles) / mean_angle_std
        std_ids.append(std_id)
        std_exps.append(std_exp)
        std_angles.append(std_angle)
    factor2std = {
        0: np.mean(std_ids),
        1: np.mean(std_exps),
        3: np.mean(std_angles)
    }

    factors_num = len(factor2key)
    score = 1.0
    for k, v in factor2key.items():
        if k == factor:
            score *= factor2std[k]**(factors_num-1)
        else:
            score /= (factor2std[k]+1e-8)

    print(f"processing folder: factor{factor}_subject{subject}_variation{variation}")
    print(f"average id std: {factor2std[0]}")
    print(f"average exp std: {factor2std[1]}")
    print(f"average angle std: {factor2std[3]}")
    print(f"score: {score}")


if __name__ == "__main__":
    main()
