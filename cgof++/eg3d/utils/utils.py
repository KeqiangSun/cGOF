import argparse
import os
import numpy as np
import math
import cv2
import time
import collections

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch import nn

import logging

import pynvml
from torch.cuda._utils import _get_device_index
from torchvision.utils import save_image, make_grid
import copy
from utils.mesh_io import save_obj_vertex_color

from scipy.interpolate import CubicSpline


def depth2norm(d_im):
    if d_im.shape[0] == 1:
        d_im = d_im.squeeze(0)
    g = np.gradient(d_im)
    zy = g[-2]
    zx = g[-1]

    normal = np.dstack((zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    normal += 1
    normal /= 2
    normal *= 255
    return normal

def arctan(z, x):
    ret = torch.zeros_like(z)
    ret[x>0] = torch.atan(z[x>0]/x[x>0])
    ret[torch.logical_and(x<0, z>=0)] = (torch.atan(z[torch.logical_and(x<0, z>=0)]/x[torch.logical_and(x<0, z>=0)]) + torch.pi)
    ret[torch.logical_and(x<0, z<0)] = (torch.atan(z[torch.logical_and(x<0, z<0)]/x[torch.logical_and(x<0, z<0)]) - torch.pi)
    ret[torch.logical_and(x==0, z>0)] = (torch.ones_like(ret)[torch.logical_and(x==0, z>0)] * math.pi/2)
    ret[torch.logical_and(x==0, z<0)] = (-torch.ones_like(ret)[torch.logical_and(x==0, z<0)] * math.pi/2)
    ret[torch.logical_and(x==0, z==0)] = torch.zeros_like(z)[torch.logical_and(x==0, z==0)]
    return ret

def dist_depr_fn(weights_crop, z_diff_crop, mesh_thick):
    w = torch.nn.functional.relu(weights_crop)
    z = torch.nn.functional.relu(
        torch.abs(z_diff_crop)-mesh_thick/2)
    dist_depr = (torch.exp(5.0*z) - 1) * w
    return dist_depr


# importing the required module
import matplotlib.pyplot as plt
def plot_curve(y, x=None, name='curve.png'):
    plt.cla()
    if not x:
        x = np.arange(len(y))
    # plotting the points 
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('graph!')
    # function to show the plot
    # plt.show()
    plt.savefig(name)

'''
    curve:
    Given points of a line, interpolate it to a smooth line using Cubic Spline Interpolation.
'''

def get_distance(a,b):
    [a_x, a_y] = a
    [b_x, b_y] = b
    return np.sqrt((b_x - a_x)*(b_x - a_x) + (b_y - a_y)*(b_y - a_y))

def remove_duplicate(points,distances):
    points_clear = points[:]
    distances_clear = distances[:]
    i = 0
    while i<len(points_clear)-1:
        if distances_clear[i+1]-distances_clear[i]<0.0001:
            if i==0:
                points_clear=np.concatenate([np.array([points_clear[0]]),points_clear[2:]],axis = 0)
                distances_clear = [distances_clear[0]]+distances_clear[2:]
            elif i == len(points_clear)-2:
                points_clear=points_clear[:i+1]
                distances_clear = distances_clear[:i+1]
            else:
                points_clear=np.concatenate([points_clear[:i],points_clear[i+1:]],axis = 0)
                distances_clear = distances_clear[:i]+distances_clear[i+1:]
        else:
            i+=1
    #print(points_clear,distances_clear)     
    return points_clear,distances_clear
    
# points = np.array([[0,0],[2,2],[3,1],[2,0]])
def curve(points,samples,bc_type='not-a-knot'):
    points = np.array(points)
    if points.shape[-1] != 2:
        points = points.reshape((-1,2))
    distances = [0]
    has_distance_zero = False
    for i in range(len(points)-1):
        a = points[i]
        b = points[i+1]
        dist = get_distance(a,b)
        distances.append(distances[-1]+get_distance(a,b))
        if dist == 0:
            has_distance_zero = True
    if has_distance_zero:
        points,distances = remove_duplicate(points,distances)
    x = points[:,0]
    y = points[:,1]
    x_cs = CubicSpline(distances, x, bc_type=bc_type)
    y_cs = CubicSpline(distances, y, bc_type=bc_type)
    # t = np.arange(0,distances[-1]+0.1,0.1)
    # pre_x = x_cs(t)
    # pre_y = y_cs(t)
    # plt.plot(pre_x, pre_y, label="S")
    # plt.legend(loc='lower left', ncol=2)
    # plt.show()

    raw_curve = []
    # samples = 10

    #simple interpolate method from beier
    #s_interpolated = np.arange(0, distances[-1], (distances[-1]/(samples*100)))

    #x_interpolated = x_cs(s_interpolated)
    #y_interpolated = y_cs(s_interpolated)

    #curve = []
    #for x,y in zip(x_interpolated, y_interpolated):
        #raw_curve.append([x,y])

    samples_per_segment = int(samples*100/len(distances[1:]))+1
    for i in range(len(distances[1:])):
        step = (distances[i+1]-distances[i])/samples_per_segment
        for j in range(samples_per_segment):
            t = distances[i]+step*j
            # print(t)
            raw_curve.append([x_cs(t),y_cs(t)])
    raw_curve = np.array(raw_curve)
    # x = raw_curve[:,0]
    # y = raw_curve[:,1]
    # plt.plot(x, y, label="S")
    # plt.legend(loc='lower left', ncol=2)
    # plt.show()

    length = []
    for i in range(len(raw_curve)-1):
        length.append(get_distance(raw_curve[i],raw_curve[i+1]))
    accu_length = [0]
    for i in range(1,len(raw_curve)):
        accu_length.append(accu_length[-1]+length[i-1])
    dst = [raw_curve[0]]
    pre_raw = 0
    step_interp=accu_length[-1]/(samples-1)
    for i in range(1,samples-1):
        covered_interp=step_interp*i
        while covered_interp>accu_length[pre_raw+1]:
            pre_raw += 1
        dx=(covered_interp-accu_length[pre_raw])/length[pre_raw]
        dst.append(raw_curve[pre_raw]*(1.0-dx)+raw_curve[pre_raw+1]*dx)
    dst.append(points[-1]) #change from raw_curve[-1]
    dst = np.array(dst)
    return dst


def FitCurve(points, sample_ratio):

    point_num = points.shape[0]

    Mx = np.zeros((1, point_num))
    My = np.zeros((1, point_num))
    A = np.zeros((1, point_num-2))
    B = np.zeros((1, point_num-2))
    C = np.zeros((1, point_num-2))
    Dx = np.zeros((1, point_num-2))
    Dy = np.zeros((1, point_num-2))
    functions = np.zeros((point_num-1, 9))

    if point_num == 2:
        functions[0, 0] = np.linalg.norm(points[0, :] - points[1, 0])
        functions[0, 1] = points[0, 0]
        functions[0, 2] = (points[1, 0] - points[0, 0]) / functions[0, 0]
        functions[0, 3] = 0
        functions[0, 4] = 0
        functions[0, 5] = points[0, 1]
        functions[0, 6] = (points[1, 1] - points[0, 1]) / functions[0, 0]
        functions[0, 7] = 0
        functions[0, 8] = 0
    else:
        for i in range(functions.shape[0]):
            functions[i, 0] = np.linalg.norm(points[i, :] - points[i+1, :])
        
        for i in range(A.shape[1]):
            # print i
            A[0,i] = functions[i, 0]
            B[0,i] = 2*(functions[i, 0]+functions[i+1, 0])
            C[0,i] = functions[i+1, 0]
            Dx[0,i] = 6*((points[i+2, 0]-points[i+1, 0])/functions[i+1, 0]-(points[i+1, 0] - points[i, 0])/functions[i, 0])
            Dy[0,i] = 6*((points[i+2, 1]-points[i+1, 1])/functions[i+1, 0]-(points[i+1, 1] - points[i, 1])/functions[i, 0])
        
        C[0,0] = C[0,0] / B[0,0]
        Dx[0,0] = Dx[0,0] / B[0,0]
        Dy[0,0] = Dy[0,0] / B[0,0]
        for i in range(1,A.shape[1]):
            tmp = B[0,i] - A[0,i]*C[0,i-1]
            C[0,i] = C[0,i] / tmp
            Dx[0,i] = (Dx[0,i]-A[0,i]*Dx[0,i-1])/tmp
            Dy[0,i] = (Dy[0,i]-A[0,i]*Dy[0,i-1])/tmp
        
        Mx[0,point_num-2] = Dx[0,point_num-3]
        My[0,point_num-2] = Dy[0,point_num-3]
        for i in range(point_num-4,-1,-1):
            Mx[0,i+1]=Dx[0,i]-C[0,i]*Mx[0,i+2]
            My[0,i+1]=Dy[0,i]-C[0,i]*My[0,i+2]
        
        Mx[0,0] = 0
        Mx[0,point_num-1] = 0
        My[0,0] = 0
        My[0,point_num-1] = 0
        for i in range(functions.shape[0]):
            functions[i, 1] = points[i, 0]
            functions[i, 2] = (points[i+1, 0]-points[i, 0]) / functions[i, 0] - (2*functions[i, 0]*Mx[0,i]+functions[i, 0]*Mx[0,i+1])/6
            functions[i, 3] = Mx[0,i] / 2
            functions[i, 4] = (Mx[0,i+1]-Mx[0,i]) / (6*functions[i, 0])
            functions[i, 5] = points[i, 1]
            functions[i, 6] = (points[i+1, 1]-points[i, 1])/float(functions[i, 0]) - (2*functions[i, 0]*My[0,i]+functions[i, 0]*My[0,i+1])/6.0
            functions[i, 7] = My[0,i] / 2
            functions[i, 8] = (My[0,i+1]-My[0,i])/(6.0*functions[i, 0])
        
    samples_per_segment = 20
    rawcurve=[]
    for i in range(functions.shape[0]):
        step = functions[i, 0]/samples_per_segment
        for j in range(samples_per_segment):
            t = step*(j)
            sample_point = [functions[i, 1]+functions[i, 2]*t+functions[i, 3]*t*t+functions[i, 4]*t*t*t, functions[i, 5] + functions[i,6]*t + functions[i,7]*t*t+functions[i,8]*t*t*t]
            rawcurve.append(sample_point)
    rawcurve.append(list(points[point_num-1, :]))
    rawcurve = np.array(rawcurve)

    clength = np.zeros((1, rawcurve.shape[0]-1))
    for i in range(rawcurve.shape[0]-1):
        clength[0,i] = np.linalg.norm(rawcurve[i, :] - rawcurve[i+1, :])

    acc_length = np.zeros((1, rawcurve.shape[0]))
    total_length = 0
    for i in range(1,clength.shape[1]+1):
        acc_length[0,i] = acc_length[0,i-1]+clength[0,i-1]
        total_length = total_length + clength[0,i-1]
    acc_length = acc_length / total_length
    clength = clength / total_length

    curr_ratio = 0
    pre_raw = 0
    samplecurve = np.zeros((sample_ratio.shape[1], 2))
    for i in range(sample_ratio.shape[1]):
        curr_ratio = curr_ratio+sample_ratio[0,i]
        while(curr_ratio > acc_length[0,pre_raw+1]):
            if pre_raw >= clength.shape[1]-1:
                break
            pre_raw = pre_raw+1
        ratio = (curr_ratio-acc_length[0,pre_raw])/clength[0,pre_raw]
        samplecurve[i,:] = rawcurve[pre_raw,:] * (1-ratio) + rawcurve[pre_raw+1,:] * ratio
    return samplecurve


def optical_flow_warping(x, flo, pad_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    """
    B, C, H, W = x.size()
    device = x.device
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid.to(device) + flo.to(device)  # warp后，新图每个像素对应原图的位置

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

    mask = torch.ones(x.size(), device=device)
    mask = F.grid_sample(mask, vgrid, padding_mode=pad_mode)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


def cvt_depth_msra2pigan(d):
    depth = d.clone()
    mask = depth > 0
    # depth[mask] = depth[mask] - 1
    depth[torch.logical_not(mask)] = depth[mask].median()
    return depth


def get_intersection_depth(depth0, depth1):
    mask = torch.logical_and(depth0 > 0, depth1 > 0)
    return depth0*mask, depth1*mask


def vis_tensor_as_vert_color(color, position, path='z_diff.obj',):
    color = torch.abs(color)
    color = (color - color.min())/(color.max() - color.min())
    c = torch.cat(
        [position, color],
        dim=-1)
    save_obj_vertex_color(path, c)


def save_normalized_images(path="tmp.png", img=None, min_value=None, max_value=None):
    if min_value is None:
        min_value = img.min()
    if max_value is None:
        max_value = img.max()
    normalized = min_max_norm(img, min_value, max_value)
    save_image(normalized, path)


def resize(d, size):
    return nn.functional.interpolate(
        d, size=(size, size), mode='nearest')


def img2depth(face_recon, gen_imgs, trans_params, t, s):
    gen_imgs = (gen_imgs+1)/2
    gen_imgs_256 = nn.functional.interpolate(
        gen_imgs, size=(256, 256), mode='bilinear')
    gen_imgs_224 = resize_n_crop_tensor(gen_imgs_256, t, s)
    data = {
        'imgs': gen_imgs_224,
        'ori_im': gen_imgs_256,
        'trans_params': np.array(trans_params)
    }
    face_recon.set_input(data)
    face_recon.forward()
    d_recon = face_recon.depth
    m_recon = face_recon.pred_mask
    return d_recon, m_recon


def z2depth(face_recon, z):
    d_input, m_input = face_recon.render(z)
    return d_input, m_input


def z2pigandepth(face_recon, z, yaw, pitch):
    d_input, m_input, f_input = face_recon.render_pigan(z, yaw, pitch)
    return d_input, m_input, f_input


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
    roi_box = torch.Tensor(np.array(roi_box)).to(img.device)
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


class GradHooker():
    def __init__(self):
        self.grads = {}

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
        return hook


def update_param(opt, metadata, update_lr=False):
    for param_group in opt.param_groups:
        if param_group['betas'] != metadata['betas'] \
                or param_group['weight_decay'] != metadata['weight_decay']:
            param_group['betas'] = metadata['betas']
            param_group['weight_decay'] = metadata['weight_decay']
        if update_lr and param_group['lr'] != metadata['disc_lr']:
            param_group['lr'] = metadata['disc_lr']


def imwrite(save_path, *imgs, norm=False):
    if norm:
        o = []
        for img in imgs:
            img = (img-img.min())*255/(img.max()-img.min())
            o.append(img)
        o = np.concatenate(o, axis=1)
    else:
        o = np.concatenate(imgs, axis=1)
    cv2.imwrite(save_path, o)


def min_max_norm(img, min_value=None, max_value=None):
    if min_value is None:
        min_value = img.min()
    if max_value is None:
        max_value = img.max()
    return (img - min_value) / (max_value - min_value)

# def create_cmp_zs(z=None, dim_t=153, dim_h=103, dim_s=40, dim_e=10):

#     z_init = z[0:1]
#     device = z.device

#     zs = z_init.repeat(5,1)
#     zs[1, :dim_t] = torch.randn(dim_t).to(device)
#     zs[2, dim_t : (dim_t + dim_h)] = torch.randn(dim_h).to(device)
#     zs[3, (dim_t + dim_h) : (dim_t + dim_h + dim_s)] = torch.randn(dim_s).to(device)
#     zs[4, (dim_t + dim_h + dim_s):(dim_t + dim_h + dim_s + dim_e)] = torch.randn(dim_e).to(device)

#     return zs


def create_cmp_zs(z=None, *dims):
    idx = np.cumsum([0] + list(dims))

    z_init = z[0:1]
    device = z.device

    zs = z_init.repeat(len(idx), 1)
    for i in range(len(idx)-1):
        zs[i+1, idx[i]:idx[i+1]] = torch.randn(dims[i]).to(device)

    return zs


def img2tb(tb_writer, img_name, tensor, global_step, **kwargs):
    grid = make_grid(tensor, **kwargs)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    tb_writer.add_image(img_name, ndarr, global_step=global_step, dataformats="HWC")


def check_bp(a):
    try:
        a.sum().backward(retain_graph=True)
        return True
    except:
        return False

class Timer():
    def __init__(self):
        self.timestamps = collections.OrderedDict()

    def __call__(self, name=None):
        self.timestamps[name if name is not None else str(len(self.timestamps))] = time.time()

    def report(self):
        ret = []
        if len(self.timestamps) <= 1:
            ret.append('No period found.')
            return ret
        data = list(self.timestamps.items())
        for i in range(len(data)-1):
            n1, t1 = data[i]
            n2, t2 = data[i+1]
            ret.append(f'It takes {t2-t1} seconds from {n1} to {n2}.')
        return ret

    def print_report(self):
        print('*** timmer ***')
        ret = self.report()
        for line in ret:
            print(line)

    def clear(self):
        self.timestamps = collections.OrderedDict()

def divide_pred(pred):
    # the prediction contains the intermediate outputs of multiscale GAN,
    # so it's usually a list
    if type(pred) == list:
        fake = []
        real = []
        for p in pred:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    else:
        fake = pred[:pred.size(0) // 2]
        real = pred[pred.size(0) // 2:]

    return fake, real

def check_gpu_memory(device=None):
    pynvml.nvmlInit()
    device = _get_device_index(device, optional=True)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    mems = 0
    if len(procs) == 0:
        print("no processes are running")
    for p in procs:
        mem = p.usedGpuMemory / (1024 * 1024)
        mems += mem
    print('GPU Memory Used:', mems, 'process num:', len(procs))

class LOGGER():
    def __init__(self, local_rank=0, print_level='info'):
        self.local_rank = local_rank
        l = getattr(logging,print_level.upper(),logging.INFO)
        logging.basicConfig(
                format='[%(asctime)s] [%(levelname)s] %(message)s',
                level=l,
            )
    def __call__(self, *msg, info_level='info'):
        infos = list(map(str,msg))
        if self.local_rank == 0:
            for info in infos:
                getattr(logging,info_level,logging.debug)(info)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_pretrained_model(model, path, device):
    if not hasattr(model, "load_state_dict"):
        return
    model_statedict = model.state_dict() if hasattr(model, "state_dict") else model
    if os.path.isfile(path):
        # pretrained_model = torch.load(path, map_location=lambda storage, loc: storage), map_location=device
        pretrained_model = torch.load(path, map_location=device)
        pretrained_model_statedict = pretrained_model.state_dict() if hasattr(pretrained_model, "state_dict") else pretrained_model
        aligned_model = {}
        for k, v in pretrained_model_statedict.items():
            if k in model_statedict.keys() and v.size() == model_statedict[k].size():
                aligned_model[k] = v
            else:
                print(f'ckpt param {k} not found in the defined model.')
        model.load_state_dict(aligned_model, strict=False)
        print(f"load from {path} successfully.")
    else:
        print(f'{path} not found! Loading pretrain model failed!')

def load_pretrained_optimizer(model, path):
    if not hasattr(model, "load_state_dict"):
        return
    try:
        model_statedict = model.state_dict() if hasattr(model, "state_dict") else model
        if os.path.isfile(path):
            pretrained_model = torch.load(path, map_location=lambda storage, loc: storage)
            pretrained_model_statedict = pretrained_model.state_dict() if hasattr(pretrained_model, "state_dict") else pretrained_model
            aligned_model = {}
            for k, v in pretrained_model_statedict.items():
                if k in model_statedict.keys():
                    aligned_model[k] = v
                else:
                    print(f'ckpt param {k} not found in the defined model.')
            model.load_state_dict(aligned_model)
    except:
        print(f'loading {path} failed!')

def transform_mesh(meshes_world, cameras, **kwargs):
    """
    Args:
        meshes_world: a Meshes object representing a batch of meshes with
            vertex coordinates in world space.

    Returns:
        meshes_proj: a Meshes object with the vertex positions projected
        in NDC space

    NOTE: keeping this as a separate function for readability but it could
    be moved into forward.
    """
    # cameras = kwargs.get("cameras", self.cameras)
    if cameras is None:
        msg = "Cameras must be specified either at initialization \
            or in the forward pass of MeshRasterizer"
        raise ValueError(msg)

    n_cameras = len(cameras)
    if n_cameras != 1 and n_cameras != len(meshes_world):
        msg = "Wrong number (%r) of cameras for %r meshes"
        raise ValueError(msg % (n_cameras, len(meshes_world)))

    verts_world = meshes_world.verts_padded()

    # NOTE: Retaining view space z coordinate for now.
    # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
    # [0, 1] range.
    eps = kwargs.get("eps", None)
    verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
        verts_world, eps=eps
    )
    # view to NDC transform
    to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
    projection_transform = cameras.get_projection_transform(**kwargs).compose(
        to_ndc_transform
    )
    verts_ndc = projection_transform.transform_points(verts_view, eps=eps)

    verts_ndc[..., 2] = verts_view[..., 2]
    return verts_ndc

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
            [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)
#%%

class SoftWingLoss(object):
    def __init__(self, thres1=2, thres2=20, curvature=0.5):
        self.thres1 = thres1
        self.thres2 = thres2
        self.curvature = curvature

    def _cal_soft_wing_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        soft_wing =
            1. |x|                           , if |x| < thres1
            2. thres2*ln(1+|x|/curvature) + B, if |x| >= thres2
        """
        loss = (target - prediction).abs()

        idx_small = loss < self.thres1
        idx_big = loss >= self.thres1

        B = self.thres1 - self.thres2 * math.log(1 + self.thres1 / self.curvature)
        loss[idx_big] = self.thres2 * torch.log(1 + loss[idx_big] / self.curvature) + B

        loss = loss.mean()

        return loss


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


def save_img_with_landmarks(image, save_path, kpts_pred=None, kpts_target=None, radius=1, color=(0,0,255), color2=(0,255,0)):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().squeeze()
    alpha = image[..., -1:] if image.shape[-1] == 4 else None

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if kpts_pred is not None:
        if isinstance(kpts_pred, torch.Tensor):
            kpts_pred = kpts_pred.detach().cpu().numpy().squeeze()
        kpts_pred = kpts_pred.reshape(-1, 2)
        num_kpts = kpts_pred.shape[0]
        for i in range(num_kpts):
            pred_pt = (round(float(kpts_pred[i, 0])),
                       round(float(kpts_pred[i, 1])))
            image = cv2.circle(image, pred_pt, 1, color, radius)

    if kpts_target is not None:
        if isinstance(kpts_target, torch.Tensor):
            kpts_target = kpts_target.detach().cpu().numpy().squeeze()
        kpts_target = kpts_target.reshape(-1, 2)
        num_kpts_gt = kpts_target.shape[0]
        for i in range(num_kpts_gt):
            pred_pt = (round(float(kpts_target[i, 0])),
                       round(float(kpts_target[i, 1])))
            image = cv2.circle(image, pred_pt, 1, color2, radius)

    ensure_dir(save_path)
    if alpha is not None:
        image = np.dstack([image, alpha])
    cv2.imwrite(save_path, image)

    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def draw_landmarks(image, kpts_pred=None, color=(255,0,0)):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().squeeze()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if kpts_pred is not None:
        if isinstance(kpts_pred, torch.Tensor):
            kpts_pred = kpts_pred.detach().cpu().numpy().squeeze()
        kpts_pred = kpts_pred.reshape(-1, 2)
        num_kpts = kpts_pred.shape[0]
        for i in range(num_kpts):
            pred_pt = (round(float(kpts_pred[i, 0])),
                       round(float(kpts_pred[i, 1])))
            image = cv2.circle(image, pred_pt, 1, color, 1)

    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def ensure_dir(path):
    dir_path=os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def read_image_as_168_rgb(path):
    ori_img = cv2.imread(path)
    img_rgb = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    img_168 = cv2.resize(img_rgb,(168,168))
    return img_168

def read_image_as_512_rgb(path):
    ori_img = cv2.imread(path)
    img_rgb = cv2.cvtColor(ori_img,cv2.COLOR_BGR2RGB)
    img_512 = cv2.resize(img_rgb,(512,512))
    return img_512

def cvt_168_to_112(img_168):
    img_112 = img_168[28:140,28:140]
    return img_112

def cvt_128_to_112(img_128):
    img_112 = img_128[8:120,8:120]
    return img_112

# def cvt_rgb_to_gray(img_rgb):
#     img_gray = img_rgb[...,0]*0.299+img_rgb[...,1]*0.587+img_rgb[...,2]*0.114
#     return img_gray

# def normalize_img(img):
#     img_normalized = (img-img.mean())/img.std()
#     return img_normalized

def cvt_rgb_to_gray(img_rgb):
    img_gray = img_rgb[:,0,:,:]*0.299+img_rgb[:,1,:,:]*0.587+img_rgb[:,2,:,:]*0.114
    return img_gray.unsqueeze(1)
def normalize_img(img):
    img_normalized = (img-img.mean())/img.std()
    return img_normalized

def vis_landmarks(img, landmarks, save_path):
    lmks = landmarks.detach().cpu().numpy().reshape((-1,2))
    img_lmks = img.copy()
    for point in lmks:
        img_lmks = cv2.circle(img_lmks,(point[0],point[1]),1,(0,0,255))
    cv_img_lmks = cv2.cvtColor(img_lmks,cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path,cv_img_lmks)

from functools import wraps
def test_time(Run_Test_Time=False):
    if Run_Test_Time:
        def test_time_(func):
            @wraps(func)
            def with_logging(*args, **kwargs):
                start = time.time()
                output = func(*args, **kwargs)
                end = time.time()
                print(f'{func.__name__} takes {end-start} seconds.')
                return output
            return with_logging
    else:
        def test_time_(func):
            @wraps(func)
            def with_logging(*args, **kwargs):
                return func(*args, **kwargs)
            return with_logging
    return test_time_

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

#%%

def test_ldmk_prior(samples, i, epoch, config):
    image_input = samples['image']

    img_transformed = image_input[0,:,:,:].squeeze().numpy()
    heatmap = img_transformed[3]
    img_transformed = img_transformed[:3]

    if not os.path.exists('image/'):
        os.mkdir('image/')

    save_path = './image/image_input_{0}_{1}.png'.format(epoch, i)
    save_path_heatmap = './image/heatmap_{0}_{1}.png'.format(epoch, i)
    img_transformed = (img_transformed + 0.5) * 255
    img_transformed = img_transformed.transpose([1, 2, 0])
    cv2.imwrite(save_path, img_transformed)

    heatmap = heatmap*255
    cv2.imwrite(save_path_heatmap, heatmap)


def combine_image_landmark(image, landmark):
    heatmap = landmark_to_heatmap(image, landmark)
    image_out = torch.cat([image, heatmap], 0)

    return image_out


def landmark_to_heatmap(image, landmark):
    _, h, w, = image.shape
    heatmap = torch.zeros((1, h, w))

    for i in range(landmark.shape[0]):
        pts = landmark[i, :]
        if pts[0] > 0 and pts[1] > 0 and pts[0] < w and pts[1] < h:
            heatmap[0, int(pts[1]), int(pts[0])] = 1

    return heatmap


def get_loss_info(with_pose, is_semi_supervise, use_shift_loss, has_adv, has_kpts_dis,
    losses_pose, losses_ldmk, losses_shift, losses_adv, losses_disc, losses_kpts_dis):
    loss_info = ''
    if is_semi_supervise:
        loss_info += f'Loss_ldmk {losses_ldmk.val:.4f} ({losses_ldmk.avg:.4f})\t'
    if use_shift_loss:
        loss_info += f'Loss_shift {losses_shift.val:.4f} ({losses_shift.avg:.4f})\t'
    if with_pose:
        loss_info += f'Loss_pose {losses_pose.val:.4f} ({losses_pose.avg:.4f})\t'
    if has_adv:
        loss_info += f'Loss_adv {losses_adv.val:.4f} ({losses_adv.avg:.4f})\t'
        loss_info += f'Loss_disc {losses_disc.val:.4f} ({losses_disc.avg:.4f})\t'
    if has_kpts_dis:
        loss_info += f'Loss_kpts_dis {losses_kpts_dis.val:.4f} ({losses_kpts_dis.avg:.4f})\t'
    return loss_info


def get_checkpoints(root):
    files = [file for file in os.listdir(root) if file.endswith('.pth.tar') and file.startswith('snapshot_checkpoint_epoch_')]
    files.sort(key=lambda x: int(x[len("snapshot_checkpoint_epoch_"):-len(".pth.tar")]))
    checkpoint_list = [os.path.join(root, file) for file in files]

    return checkpoint_list

def parse_state_name(path):
    r"""
    Parse a file name with the given pattern:
    pattern = ($model_name)_s($scale)_i($iteration).pt

    Returns: None if the path doesn't fulfill the pattern
    """
    path = os.path.splitext(os.path.basename(path))[0]

    data = path.split('_')

    if len(data) < 3:
        return None

    # Iteration
    if data[-1][0] == "i" and data[-1][1:].isdigit():
        iteration = int(data[-1][1:])
    else:
        return None

    if data[-2][0] == "s" and data[-2][1:].isdigit():
        scale = int(data[-2][1:])
    else:
        return None

    name = "_".join(data[:-2])

    return name, scale, iteration

def getLastCheckPoint(dir, name, scale=None, iter=None):
    r"""
    Get the last checkpoint of the model with name @param name detected in the
    directory (@param dir)

    Returns:
    trainConfig, pathModel, pathTmpData

    trainConfig: path to the training configuration (.json)
    pathModel: path to the model's weight data (.pt)
    pathTmpData: path to the temporary configuration (.json)
    """
    trainConfig = os.path.join(dir, name + "_train_config.json")
    if not os.path.isfile(trainConfig):
        return None
    listFiles = [f for f in os.listdir(dir) if (
        os.path.splitext(f)[1] == ".pt" and
        parse_state_name(f) is not None and
        parse_state_name(f)[0] == name)]
    if scale is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[1] == scale]

    if iter is not None:
        listFiles = [f for f in listFiles if parse_state_name(f)[2] == iter]
    listFiles.sort(reverse=True, key=lambda x: (
        parse_state_name(x)[1], parse_state_name(x)[2]))
    if len(listFiles) == 0:
        return None

    pathModel = os.path.join(dir, listFiles[0])
    pathTmpData = os.path.splitext(pathModel)[0] + "_tmp_config.json"
    if not os.path.isfile(pathTmpData):
        return None
    return trainConfig, pathModel, pathTmpData

def _process_image_for_landmark(img):
    img_112 = cvt_128_to_112(img)
    img_112_gray = cvt_rgb_to_gray(img_112)
    img_112_gray_norm = normalize_img(img_112_gray)
    img_112_gray_norm = img_112_gray_norm[None,None,...]
    if isinstance(img_112_gray_norm, np.ndarray):
        img_112_gray_norm_th = torch.from_numpy(img_112_gray_norm.astype(np.float32)).cuda()
    elif isinstance(img_112_gray_norm, torch.Tensor):
        img_112_gray_norm_th = img_112_gray_norm.cuda()
    else:
        print('input image should be either np.ndarray or torch.Tensor!')
    return img_112_gray_norm_th
