"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import MyVisualizer
from util.preprocess import align_img
from PIL import Image
import numpy as np
from util.load_mats import load_lm3d
import torch 
from data.flist_dataset import default_flist_reader
from scipy.io import loadmat, savemat

from utils.utils import ensure_dir

def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def get_img_path(root='examples'):

    # im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    im_path = []
    for r, ds, fs in os.walk(root):
        for f in fs:
            if f.endswith('png') or f.endswith('jpg'):
                p = os.path.join(r, f)
                im_path.append(p)
    return im_path

def read_img(im_path, to_tensor=True):
    # to RGB 
    im = Image.open(im_path).convert('RGB')
    W,H = im.size
    im = im.resize([224,224])
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.set_coeff_static()
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path = get_img_path(name)

    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        
        # im_tensor, lm_tensor = read_data(im_path[i], lm_path[i], lm3d_std)
        im_tensor = read_img(im_path[i])
        data = {
            'imgs': im_tensor
        }
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        dataset = os.path.dirname(im_path[i]).split(name)[-1]

        if i % 100 == 0:
            visuals = model.get_current_visuals()  # get image results
            visualizer.display_current_results(
                visuals, 0, opt.epoch, dataset=dataset,
                save_results=True, count=i, name=img_name, add_image=False)

        # model.save_mesh(os.path.join(visualizer.img_dir, dataset, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        save_path = os.path.join(visualizer.img_dir, dataset, 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')
        ensure_dir(save_path)
        model.save_coeff(save_path) # save predicted coefficients

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    main(0, opt, opt.img_folder)

# python test_wo_ldmk.py --img_folder imgs/discofacegan/ --epoch 20 --name=face_recon_feat0.2_augment
# import scipy.io
# import numpy as np
# from glob import glob
# import os
# distribution = dict(np.load("/home/kqsun/Tasks/pigan/Deep3DFaceRecon_pytorch/coeff_distribution.npz", allow_pickle=True))
# mean = distribution['exp'].item()['mean']
# std = distribution['exp'].item()['std']

# exps = []
# data_root = "datasets/debug_exp_224/face_recon/epoch_20_000000"
# for f in glob(os.path.join(data_root, "*.mat")):
#     mat = scipy.io.loadmat(f)
#     exp = mat['exp']
#     exp_norm = (exp - mean) / std
#     exps.append(exp_norm)
# exps_cat = np.concatenate(exps, axis=0)
# np.save('exps.npy', exps_cat)


# import numpy as np
# from glob import glob
# import os

# ids = []
# bg_geos = []
# texs = []
# bg_texs = []
# data_root = "Deep3DFaceRecon_pytorch/ids/"
# for f in glob(os.path.join(data_root, "*id*.npz")):
#     data = dict(np.load(f, allow_pickle=True))
#     id = data['z_id'][0:1]
#     bg_geo = data['z_bg_geo'][0:1]
#     tex = data['z_tex'][0:1]
#     bg_tex = data['z_bg_tex'][0:1]
#     ids.append(id)
#     bg_geos.append(bg_geo)
#     texs.append(tex)
#     bg_texs.append(bg_tex)
# ids_ = np.concatenate(ids, axis=0)
# bg_geos_ = np.concatenate(bg_geos, axis=0)
# texs_ = np.concatenate(texs, axis=0)
# bg_texs_ = np.concatenate(bg_texs, axis=0)
# np.savez('ids.npz', ids=ids_, bg_geos=bg_geos_, texs=texs_, bg_texs=bg_texs_)