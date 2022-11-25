import argparse
import math
import glob
import numpy as np
import sys
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm

import curriculums

from utils.utils import *
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

times = []
def generate_img(g, z, **kwargs):

    with torch.no_grad():
        a = time.time()
        img, depth_map = g.staged_forward(z, **kwargs)
        b = time.time()
        times.append(b-a)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map


def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)

    parser.add_argument('--factor', type=int, default=0, help='factor variation mode. 0 = all, 1 = expression, 2 = lighting, 3 = pose.')
    parser.add_argument('--subject', type=int, default=20, help='how many subjects to generate.')
    parser.add_argument('--variation', type=int, default=5, help='how many images to generate per subject.')
    # --factor 1 --subject 10 --variation 10
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    parser.add_argument('--no_snm', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_depth', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--cmp', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--split', type=bool, default=True)
    parser.add_argument('--angle_multiplier', type=float, default=1.0)
    parser.add_argument('--psi', type=float, default=1.0)
    parser.add_argument('--port', type=str, default='22479')
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
    curriculum['psi'] = opt.psi
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {
        key: value for key, value in curriculum.items()
        if type(key) is str
    }
    if opt.no_snm:
        curriculum['sample_near_mesh'] = False

    save_path = os.path.join(
        opt.output_dir,
        f"disentangle_score/pigan/{opt.curriculum}",
        f"factor{opt.factor}_subject{opt.subject}_variation{opt.variation}_angle{opt.angle_multiplier}_psi{opt.psi}")
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

    generator.set_device(device)
    generator_ddp.eval()

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

    h_mean = curriculum['h_mean']
    v_mean = curriculum['v_mean']

    subject_per_gpu = math.ceil(opt.subject/world_size)
    for i in range(subject_per_gpu*rank, subject_per_gpu*(rank+1)):
        if i >= opt.subject:
            break
        # print(i)
        lats1 = torch.randn((bs, length), device=device)

        # seed = 31
        # torch.manual_seed(seed)
        # zs = torch.randn(
        #     (10, 1, length), device=device).repeat(1, 12, 1)
        # lats1 = zs[9, :bs]
        # z_exp = torch.randn(
        #     (1, 12, dim_exp), device=device
        # )[0, 5].unsqueeze(0).reapeat(bs, 1)
        # lats1[:, dim_id:dim_id+dim_exp] = z_exp

        yaw_noise = (np.random.uniform()-0.5)
        pitch_noise = (np.random.uniform()-0.5)

        for j in range(opt.variation):
            lats2 = torch.randn((bs, length), device=device)
            if opt.factor == 0:  # change id only
                lats = torch.cat(
                    [
                        lats2[:, ind[0]:ind[1]],  # id
                        lats1[:, ind[1]:ind[2]],  # exp
                        lats2[:, ind[2]:ind[3]],  # bg_geo
                        lats2[:, ind[3]:ind[4]],  # tex
                        lats1[:, ind[4]:ind[5]],  # gamma
                        lats2[:, ind[5]:ind[6]],  # bg_tex
                    ], dim=1)
            elif opt.factor == 1:  # change expression only
                lats = torch.cat(
                    [
                        lats1[:, ind[0]:ind[1]],  # id
                        lats2[:, ind[1]:ind[2]],  # exp
                        lats1[:, ind[2]:ind[3]],  # bg_geo
                        lats1[:, ind[3]:ind[4]],  # tex
                        lats1[:, ind[4]:ind[5]],  # gamma
                        lats1[:, ind[5]:ind[6]],  # bg_tex
                    ], dim=1)
            elif opt.factor == 2:  # change lighting only
                lats = torch.cat([
                        lats1[:, ind[0]:ind[1]],  # id
                        lats1[:, ind[1]:ind[2]],  # exp
                        lats1[:, ind[2]:ind[3]],  # bg_geo
                        lats1[:, ind[3]:ind[4]],  # tex
                        lats2[:, ind[4]:ind[5]],  # gamma
                        lats1[:, ind[5]:ind[6]],  # bg_tex
                    ], axis=1)
            elif opt.factor == 3:  # change pose only
                lats = torch.cat([
                        lats1[:, ind[0]:ind[1]],  # id
                        lats1[:, ind[1]:ind[2]],  # exp
                        lats1[:, ind[2]:ind[3]],  # bg_geo
                        lats1[:, ind[3]:ind[4]],  # tex
                        lats1[:, ind[4]:ind[5]],  # gamma
                        lats1[:, ind[5]:ind[6]],  # bg_tex
                    ], axis=1)
                yaw_noise = (np.random.uniform()-0.5) * opt.angle_multiplier
                pitch_noise = (np.random.uniform()-0.5) * opt.angle_multiplier
            elif opt.factor == 4:  # change all factors
                lats = torch.cat([
                        lats1[:, ind[0]:ind[1]],  # id
                        lats2[:, ind[1]:ind[2]],  # exp
                        lats1[:, ind[2]:ind[3]],  # bg_geo
                        lats1[:, ind[3]:ind[4]],  # tex
                        lats2[:, ind[4]:ind[5]],  # gamma
                        lats1[:, ind[5]:ind[6]],  # bg_tex
                    ], axis=1)
                yaw_noise = (np.random.uniform()-0.5) * opt.angle_multiplier
                pitch_noise = (np.random.uniform()-0.5) * opt.angle_multiplier
            curriculum['h_mean'] = h_mean + yaw_noise
            curriculum['v_mean'] = v_mean + pitch_noise
            img, tensor_img, depth_map = generate_img(
                generator_ddp.module, lats, face_recon=face_recon,
                gt_depths=opt.save_depth, **curriculum)

            images = tensor_img[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
            images = (images-images.min())/(images.max()-images.min())
            cv2.imwrite(
                os.path.join(save_path, f'{i:04}_{j:02}.png'), images*255)

        print(np.mean(times))

if __name__ == '__main__':
    opt = parse()
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    if num_gpus > 1:
        mp.spawn(main, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    else:
        main(0, num_gpus, opt)


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/eval/render/pigan_disentangle_score_ddp.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg20/generator.pth --image_size 128 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg20 --split True --save_depth False --factor 1 --subject 1000 --variation 10