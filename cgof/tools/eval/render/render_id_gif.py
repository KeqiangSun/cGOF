import os
import sys
add_path = os.path.realpath('Deep3DFaceRecon_pytorch')
print(add_path)
sys.path.insert(0, add_path)

import argparse
import math
import glob
import numpy as np
# import sys
# import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums

from utils.utils import *
from utils.mesh_io import *
from losses.contrastive_id_loss import Z_Manager

import cv2

from models import create_model
from deep3dfacerecon_opt import deep3dfacerecon_opt

from generators import generators
from siren import siren
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()


def init_face_recon(device=torch.device('cuda')):
    face_recon = create_model(deep3dfacerecon_opt)
    face_recon.setup(deep3dfacerecon_opt)
    face_recon.device = device
    face_recon.set_coeff_static()
    face_recon.parallelize()
    face_recon.eval()
    return face_recon


def generate_img(g, z, **kwargs):

    with torch.no_grad():
        img, depth_map = g.staged_forward(z, **kwargs)

        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    parser.add_argument('--subject', type=int, default=20, help='how many subjects to generate.')
    #parser.add_argument('--subject_list', nargs='+', default=None)
    parser.add_argument('--subject_list', nargs='+', default=[
        183,
        31, 302, 551, 1003, 1486, 544, 443, 191, 318, 183
        # 264, 288, 289, 302,
        # 318, 333, 369, 374, 389, 390,
        # 479, 534, 551, 673,
        # 1003, 1004, 1037, 1073, 1125, , 1168, 1178, 1183, 1199,
        # 1234, 1275, 1305, 1346, 1350, 1415, 1416, 1486, 1522, 1542,
        # 1555, 1600, 1608, 1635, 1642, 1643, 1649, 1651, 1666, 1676,
        # 1677, 1712, 1717, 1718, 1732, 1752, 1761, 1782, 1840, 1857,
        # 1863, 1893, 1898, 1923, 1929, 1990,
    ])

    parser.add_argument('--output_dir', type=str, default='vids/id_gif/')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--fix_sub', type=int, default=None)
    parser.add_argument('--fix_exp', type=int, default=None)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--white_back', action='store_true')
    parser.add_argument('--last_back', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300')
    parser.add_argument('--no_snm', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_depth', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--split', type=bool, default=True)
    parser.add_argument('--angle_multiplier', type=float, default=0.0)
    parser.add_argument('--port', type=str, default='22479')

    parser.add_argument('--rt_norm', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--rt_sampled_points', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--rt_face_recon', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--num_frames', type=int, default=200)

    opt = parser.parse_args()
    return opt


def main(rank, world_size, opt):
    torch.manual_seed(rank)
    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    # curriculum = getattr(curriculums, opt.curriculum)
    curriculum = curriculums.get(opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.9
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['white_back'] = opt.white_back
    curriculum['last_back'] = opt.last_back
    curriculum['nerf_noise'] = 0
    curriculum = {
        key: value for key, value in curriculum.items()
        if type(key) is str
    }
    if opt.no_snm:
        curriculum['sample_near_mesh'] = False
    rt_norm = opt.rt_norm

    save_path = os.path.join(
        opt.output_dir,
        f"id/{opt.curriculum}",)
    if opt.angle_multiplier > 1:
        save_path = os.path.join(
            opt.output_dir,
            f"id/{opt.curriculum}_anglemultiplier{opt.angle_multiplier}",)
    os.makedirs(save_path, exist_ok=True)

    SIREN = getattr(siren, curriculum['model'])
    GENERATOR = getattr(generators, curriculum['generator'])
    generator = GENERATOR(
        SIREN, curriculum['latent_dim'], **curriculum).to(device)
    load_pretrained_model(generator, opt.path, device=device)
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file, map_location=device)
    ema.copy_to([p for p in generator.parameters() if p.requires_grad])

    generator.set_device(device)
    generator_ddp = DDP(
        generator, device_ids=[rank], find_unused_parameters=True)
    generator = generator_ddp.module

    face_recon = init_face_recon(device)

    # generator.siren.mapping_network.transformer.s = torch.from_numpy(np.array([0.1]).astype(np.float32)).to(device)
    # generator.siren.mapping_network.transformer.t = torch.from_numpy(np.array([[0.0,-0.042,-0.1]]).astype(np.float32)).T.to(device)
    # generator.siren.mapping_network.transformer.p_y_r = torch.rand(3).to(device)*1e-9

    generator.set_device(device)
    generator_ddp.eval()

    yaw_angles = [0.]
    # face_angles = [-0.5, 0., 0.5]
    yaw_angles = [a + curriculum['h_mean'] for a in yaw_angles]

    pitch_angles = [0.]
    # pitch_angles = [-0.5, 0., 0.5]
    pitch_angles = [a + curriculum['v_mean'] for a in pitch_angles]

    dim_id = 80
    dim_exp = 64
    dim_bg_geo = 80
    dim_tex = 80
    dim_gamma = 27
    dim_bg_tex = 80

    ind = np.cumsum(
        [0, dim_id, dim_exp, dim_bg_geo, dim_tex, dim_gamma, dim_bg_tex])
    length = ind[-1]

    bs = 1

    zs = []
    for s in opt.subject_list:
        torch.manual_seed(s)
        z = torch.randn(length, device=device)
        zs.append(z)
    zs = torch.stack(zs).T.unsqueeze(0)
    zs = torch.nn.functional.interpolate(
        zs, size=opt.num_frames,
        mode='linear', align_corners=True)[0].T

    subject_per_gpu = math.ceil(opt.num_frames/world_size)
    subject_range = range(
        subject_per_gpu*rank, subject_per_gpu*(rank+1)
    )

    torch.manual_seed(31)
    z_exp = torch.randn((12, 12, dim_exp), device=device)
    z_exp = z_exp[6, 9]

    for i in subject_range:
        if i >= opt.num_frames:
            break
        images = []
        depths = []
        z = zs[i].unsqueeze(0)
        z[0, dim_id:(dim_id+dim_exp)] = z_exp

        yaw_noise = (np.random.uniform()-0.5)*opt.angle_multiplier
        pitch_noise = (np.random.uniform()-0.5)*opt.angle_multiplier

        yaw = yaw_angles[0]
        curriculum['h_mean'] = yaw + yaw_noise
        pitch = pitch_angles[0]
        curriculum['v_mean'] = pitch + pitch_noise

        fetch_data = {}
        img, tensor_img, depth_map = generate_img(
            generator_ddp.module, z, face_recon=face_recon,
            gt_depths=opt.save_depth, rt_norm=rt_norm,
            zy_data=fetch_data, **curriculum)

        images.append(tensor_img)
        depths.append(depth_map)

        if opt.split:
            split_path = os.path.join(save_path, f'splits/imgs/grid_seed{s}/')
            os.makedirs(split_path, exist_ok=True)
            img_path = os.path.join(split_path, f'img_seed{s}_i{i:04d}.png')
            img_vis = tensor_img[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
            cv2.imwrite(img_path, img_vis*255)
            if opt.save_depth:
                d_root = os.path.join(save_path, f'splits/depths/grid_seed{s}/')
                os.makedirs(d_root, exist_ok=True)
                d_path = os.path.join(d_root, f'depth_seed{s}_i{i:04d}.png')
                dep_vis = depth_map[0].numpy()
                dep_vis = (dep_vis-dep_vis.min())/(dep_vis.max() - dep_vis.min())
                dep_vis = cv2.cvtColor(dep_vis, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(d_path, dep_vis*255)
            if rt_norm:
                n_root = os.path.join(save_path, f'splits/normals/grid_seed{s}/')
                os.makedirs(n_root, exist_ok=True)
                n_path = os.path.join(n_root, f'normal_seed{s}_i{i:04d}.png')
                norm = generator_ddp.module.pred_normals.reshape(
                    opt.image_size, opt.image_size, 3).detach().cpu().numpy()
                norm = (norm + 1) / 2
                norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)*255
                cv2.imwrite(n_path, norm)
            if opt.rt_sampled_points:
                p_root = os.path.join(save_path, f'splits/sampled_points/grid_seed{s}/')
                os.makedirs(p_root, exist_ok=True)
                c_path = os.path.join(p_root, f'c_seed{s}_i{i:04d}.obj')
                f_path = os.path.join(p_root, f'f_seed{s}_i{i:04d}.obj')
                f2_path = os.path.join(p_root, f'f2_seed{s}_i{i:04d}.obj')
                save_obj_vertex(c_path, fetch_data['transformed_points'].reshape(-1, opt.image_size, opt.image_size, curriculum['num_steps'], 3)[:,::8,::8,::4,:].reshape(bs, -1, 3))
                fp = fetch_data['fine_points'].reshape(-1, opt.image_size, opt.image_size, curriculum['num_steps'], 3)[:,:,:,::4,:].reshape(bs, -1, 3)
                fpm = torch.logical_not(torch.logical_and(fp[..., 0].abs() > fp[..., 0].abs().max() * 0.5, fp[..., 2] > fp[..., 2].min()*0.1))
                save_obj_vertex(f_path, fp[fpm])
                fp2 = fetch_data['fine_points2'].reshape(-1, opt.image_size, opt.image_size, curriculum['num_steps'], 3)[:,:,:,::4,:].reshape(bs, -1, 3)
                fp2m = torch.logical_not(torch.logical_and(fp2[..., 0].abs() > fp2[..., 0].abs().max() * 0.5, fp2[..., 2] > fp2[..., 2].min()*0.1))
                save_obj_vertex(f2_path, fp2[fp2m])
            if opt.rt_face_recon:
                f_root = os.path.join(save_path, f'splits/face_recon/grid_seed{s}/')
                os.makedirs(f_root, exist_ok=True)

                d_path = os.path.join(f_root, f'depth_seed{s}_i{i:04d}.png')
                d = face_recon.d_pigan_input
                d[torch.logical_not(face_recon.m_pigan_input)] = face_recon.d_pigan_input[face_recon.m_pigan_input].mean()
                d = (d-d.min())/(d.max()-d.min())
                d = d.detach().cpu().numpy()[0, 0]
                d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
                d = np.dstack([d,np.uint8(face_recon.m_pigan_input[0].permute(1,2,0).detach().cpu().numpy() * 1.0)])
                cv2.imwrite(d_path, d*255)

                f_path = os.path.join(f_root, f'face_seed{s}_i{i:04d}.png')
                f = face_recon.f_pigan_input[0].permute(1,2,0)
                f[torch.logical_not(face_recon.m_pigan_input[0,0])] = face_recon.f_pigan_input[0].permute(1,2,0)[face_recon.m_pigan_input[0,0]].mean(0)
                f = (f-f.min())/(f.max()-f.min())
                f = f.detach().cpu().numpy()
                f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                f = np.dstack([f,np.uint8(face_recon.m_pigan_input[0].permute(1,2,0).detach().cpu().numpy() * 1.0)])
                cv2.imwrite(f_path, f*255) 

        depths = torch.cat(depths, dim=-1)[0].numpy()
        depths = (depths-depths.min())/(depths.max()-depths.min())
        depths = cv2.cvtColor(depths, cv2.COLOR_GRAY2BGR)
        images = torch.cat(images, dim=-1)[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
        images = (images-images.min())/(images.max()-images.min())

        img_path = os.path.join(save_path, f'grid_seed{s}.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)

if __name__ == '__main__':
    opt = parse()
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    if num_gpus > 1:
        mp.spawn(main, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    else:
        main(0, num_gpus, opt)


# python tools/eval/render/render_exp.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10/generator.pth --output_dir imgs/exp/ --curriculum pigan_recon4_snm_depr10000 --range 0 10 1 --image_size 128 --split False --save_depth False --rows 10 --exp_num 12 #--pre_exp
# import torch
# length = 80+64+80+80+27+80
# exp_num = 12
# dim_exp = 64
# torch.manual_seed(31)
# zs = torch.randn((1, 1, length)).repeat(rows, exp_num, 1)
# z_exp = torch.randn((rows, exp_num, dim_exp))