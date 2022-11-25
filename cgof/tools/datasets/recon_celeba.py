"""This script is the test script for Deep3DFaceRecon_pytorch
"""
import sys
sys.path.insert(0, 'Deep3DFaceRecon_pytorch')

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
# from IPython import embed; embed()
from torch import nn

def get_data_path(root='examples'):
    
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1],''),'detections',i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    # to RGB 
    ori_im = Image.open(im_path).convert('RGB')
    W,H = ori_im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    trans_params, im, lm, mask_new = align_img(ori_im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        ori_im = torch.tensor(np.array(ori_im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return trans_params, im, ori_im, lm, mask_new

def resize_n_crop_img(img, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    return img

def crop_batch_tensor(img, roi_box):
    """crop a pytorch tensor, given bounding box.

    Args:
        img ([type]): pytorch tensor in shape of B, C, H, W
        roi_box ([type]): [sx, sy, ex, ey]
                      or  [left, up, right, down]

    Returns:
        [type]: cropped tensor
    """
    # b, c, h, w = img.shape
    h, w = img.shape[-2:]
    b = len(img)
    roi_box = torch.Tensor(roi_box).to(img.device)
    # roi_box[0::2].clamp_(min=)
    sx, sy, ex, ey = [torch.round(_).to(torch.int16) for _ in roi_box]
    
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 4:
        res = torch.zeros((b, img.shape[1], dh, dw), dtype=img.dtype).to(img.device)
        # res = torch.zeros((dh, dw, 3), dtype=torch.uint8).to(img.device)
    elif len(img.sahpe) == 3:
        res = torch.zeros((b, dh, dw), dtype=img.dtype).to(img.device)
    
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[:, :, dsy:dey, dsx:dex] = img[:, :, sy:ey, sx:ex]
    # res.requires_grad=img.requires_grad
    # res.requires_grad_
    
    return res


def resize_n_crop_tensor(tensor, t, s, target_size=224., mask=None):
    
    s = np.array(s) if not isinstance(s, np.ndarray) else s
    t = np.array(t) if not isinstance(t, np.ndarray) else t
        
    b, c, h0, w0 = tensor.size()
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    tensor = nn.functional.interpolate(tensor, (int(h), int(w)), mode='bicubic')
    tensor = crop_batch_tensor(tensor, [left, up, right, below])

    return tensor


def read_img(im_path, to_tensor=True):
    # to RGB 
    ori_im = Image.open(im_path).convert('RGB')
    s = np.array(0.8975788298782894)
    t = np.array([128.04371143331298, 89.95939537909564])
    im = resize_n_crop_img(ori_im, t, s)
    
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        ori_im = torch.tensor(np.array(ori_im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im, ori_im,

def read_pigan(im_path, to_tensor=True):
    # to RGB 
    ori_im = Image.open(im_path).convert('RGB')
    if to_tensor:
        # im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        ori_im = torch.tensor(np.array(ori_im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    s = [0.8975788298782894]
    t = [128.04371143331298, 89.95939537909564]

    im = resize_n_crop_tensor(ori_im, t, s)
    
    return im, ori_im,

def main(rank, opt, name='examples'):
    device = torch.device(rank)
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()
    visualizer = MyVisualizer(opt)

    im_path, lm_path = get_data_path(name)
    lm3d_std = load_lm3d(opt.bfm_folder) 
    # sys.path.remove('Deep3DFaceRecon_pytorch')
    # print(f"sys.path:{sys.path}")
    for i in range(len(im_path)):
        print(i, im_path[i])
        img_name = im_path[i].split(os.path.sep)[-1].replace('.png','').replace('.jpg','')
        im_tensor, ori_im = read_pigan(im_path[i])
        data = {
            'imgs': im_tensor,
            'ori_im': ori_im,
            'trans_params': np.array([256, 256, 0.8975788298782894, 128.04371143331298, 89.95939537909564])
        }
        model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference
        # model.test_ori()           # run inference
        # model.test_with_grad()           # run inference
        model.forward()
        # visuals = model.get_current_visuals()  # get image results
        # visualizer.display_current_results(visuals, 0, opt.epoch, dataset=name.split(os.path.sep)[-1], 
        #     save_results=True, count=i, name=img_name, add_image=False)
        
        # model.save_mesh(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.obj')) # save reconstruction meshes
        model.save_coeff(os.path.join(visualizer.img_dir, name.split(os.path.sep)[-1], 'epoch_%s_%06d'%(opt.epoch, 0),img_name+'.mat')) # save predicted coefficients

if __name__ == '__main__':
    deep3dfacerecon_opt = TestOptions().parse()  # get test options

    # from deep3dfacerecon_opt import deep3dfacerecon_opt 

    print("opt:",deep3dfacerecon_opt)
    main(0, deep3dfacerecon_opt, deep3dfacerecon_opt.img_folder)
    if 'Deep3DFaceRecon_pytorch' in sys.path:
        sys.path.remove('Deep3DFaceRecon_pytorch')

