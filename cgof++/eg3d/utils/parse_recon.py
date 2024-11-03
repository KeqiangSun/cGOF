from scipy.io import loadmat
import glob
import numpy as np
import os

def parse(recon_results):
    distribution = {}
    for key, value in recon_results.items():
        value = np.array(value)
        shp = value.shape
        value = value.reshape([shp[0], -1])
        distribution[key] = {}
        distribution[key]['mean'] = np.mean(value, axis=0).reshape(shp[1:])
        distribution[key]['std'] = np.std(value, axis=0).reshape(shp[1:])
        L = np.linalg.cholesky(np.cov(value.T))
        distribution[key]['L'] = L
        distribution[key]['L_'] = np.linalg.inv(L)
        print(key)
        print(f" - mean: {distribution[key]['mean']},")
        print(f" - std: {distribution[key]['std']}")
    return distribution

def parse_coeff():
    file_list = glob.glob('*.mat')
    recon_results = {
        "id": [], "exp": [], "tex": [], "angle": [],
        "gamma": [], "trans": [], "lm68": [],
    }
    for p in file_list:
        a = loadmat(p)
        for key, value in recon_results.items():
            value.append(a[key])

    return parse(recon_results)

def load_results(path):
    file_list = glob.glob(os.path.join(path,'*.mat'))
    recon_results = {
        "id": [], "exp": [], "tex": [], "angle": [],
        "gamma": [], "trans": [], "lm68": [],
    }
    for p in file_list:
        a = loadmat(p)
        for key, value in recon_results.items():
            value.append(a[key])
    return recon_results

def parse_gamma():
    rr = parse_results('./epoch_20_000000/')
    gammas = rr['gamma']
    g = np.array(gammas)[:,0,:]
    g = g.T
    cov = np.cov(g)
    # np.linalg.det(cov)
    l = np.linalg.cholesky(cov)
    return g.mean(), l

def vis_gamma(g):
    import matplotlib.pyplot as plt
    for ind in range(27):
        x = g[:,ind]

        plt.subplot(3,9,ind+1)
        plt.hist(x,50,density=True,alpha=0.75)
        plt.xlim(-1,1)
        plt.ylim(0,5)
        plt.grid(True)
    plt.savefig(f'dist_g.png')

def check_shape(distribution):
    for k, v in distribution.items():
        print(f"{k}: meanshape:{v['mean'].shape}, \
            stdshape:{v['std'].shape}, covshape:{v['L'].shape}")

def main():
    input_path = "/home/kqsun/Data/ffhq/face_recon/mat/"
    output_path = "/home/kqsun/Data/ffhq/face_recon/coeff_distribution_L.npz"
    recon_results = load_results(input_path)
    distribution = parse(recon_results)
    check_shape(distribution)
    np.savez(output_path, **distribution)

if __name__ == "__main__":
    main()