# CUDA_VISIBLE_DEVICES=1 \
# python tools/eval/render/get_img_and_mesh.py \
# outputs/pigan_finetune/generator.pth \
# --range 0 1 1 --curriculum pigan  --save_depth \
# --output_dir imgs
import argparse
import math
import glob
import numpy as np
import sys
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/kqsun/Tasks/pigan')

import curriculums

from utils.utils import *
from losses.contrastive_id_loss import Z_Manager

import cv2

import mcubes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show(tensor_img):
    if len(tensor_img.shape) > 3:
        tensor_img = tensor_img.squeeze(0)
    tensor_img = tensor_img.permute(1, 2, 0).squeeze().cpu().numpy()
    plt.imshow(tensor_img)
    plt.show()


def generate_img(gen, z, **kwargs):

    with torch.no_grad():
        img, depth_map, gt_depth, gt_ldmks, gt_wets = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map, gt_depth


def generate_img_wo_gt_depth(gen, z, **kwargs):

    with torch.no_grad():
        img, depth_map = generator.staged_forward(z, **kwargs)
        tensor_img = img.detach()

        img_min = img.min()
        img_max = img.max()
        img = (img - img_min)/(img_max-img_min)
        img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
    return img, tensor_img, depth_map



def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]  # z
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]  # y
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]  # x

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


# def sample_generator(generator, z, max_batch=100000, voxel_resolution=256, voxel_origin=[0,0,0], cube_length=2.0, psi=0.5):
#     head = 0
#     samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
#     samples = samples.to(z.device)

#     sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)

#     transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
#     transformed_ray_directions_expanded[..., -1] = -1

#     print('getting truncated frequencies and phase shifts...')
#     generator.generate_avg_frequencies()
#     with torch.no_grad():
#         raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)
#         truncated_frequencies = generator.avg_frequencies + psi * (raw_frequencies - generator.avg_frequencies)
#         truncated_phase_shifts = generator.avg_phase_shifts + psi * (raw_phase_shifts - generator.avg_phase_shifts)

#     print('getting coarse_output...')
#     with torch.no_grad():
#         while head < samples.shape[1]:
#             print(f'head/samples.shape[1]:{head}/{samples.shape[1]}')
#             coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, 4)
#             sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
#             head += max_batch

#     sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()

#     return sigmas


def sample_generator(
    generator, z, max_batch=100000, voxel_resolution=256,
    voxel_origin=[0, 0, 0], cube_length=2.0, psi=0.5,
        mask=None, fov=12, cam_distance=1):

    head = 0
    samples, voxel_origin, voxel_size = create_samples(voxel_resolution, voxel_origin, cube_length)
    samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)

    if mask is not None:
        samples_bk = samples.clone()
        sigmas_bk = sigmas.clone()
        mask_size = mask.shape[1]
        f = (mask_size/2)/np.tan((fov/2)/180*np.pi)
        ratio = (cam_distance - samples[0][:, 2])/f
        mask_sampler = -samples[0][:, :2] / ratio.unsqueeze(1)
        mask_sampler += mask_size/2
        mask_sampler = mask_sampler.round().long()
        # mask_sampler = np.clip(mask_sampler, 0, mask_size-1)
        mask_sampler[torch.logical_or(
            mask_sampler < 0, mask_sampler >= 512
        )] = 0
        mask = torch.from_numpy(mask).to(device)
        m = mask[mask_sampler[:, 1], mask_sampler[:, 0]]
        # v = m.reshape((256,256,256)).cpu().numpy()
        # cv2.imwrite('vis_mask.png',v[...,0]*255)
        # print("")
        mask_indices = torch.where(torch.logical_and(
            m > 0.5,
            torch.logical_and(
                samples[0][:, 2] > -0.12,
                samples[0][:, 2] < 0.12
            )
        ))[0]
        samples = samples[:, mask_indices]
        sigmas = sigmas[:, mask_indices]

    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
    transformed_ray_directions_expanded[..., -1] = -1

    print('getting truncated frequencies and phase shifts...')
    generator.generate_avg_frequencies()
    with torch.no_grad():
        raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)
        truncated_frequencies = generator.avg_frequencies + psi * (raw_frequencies - generator.avg_frequencies)
        truncated_phase_shifts = generator.avg_phase_shifts + psi * (raw_phase_shifts - generator.avg_phase_shifts)

    print('getting coarse_output...')
    with torch.no_grad():
        while head < samples.shape[1]:
            print(f'head/samples.shape[1]:{head}/{samples.shape[1]}')
            coarse_output = generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, 4)
            sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
            head += max_batch

    if mask is not None:
        sigmas_bk[0][mask_indices] = sigmas
        sigmas = sigmas_bk

    # sigmas = sigmas[0] * m.unsqueeze(1)
    # sigmas = sigmas[0]
    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
    # sigmas = m.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()*90

    print("")
    return sigmas


def get_mesh(generator, z, opt, max_batch=500000):
    print('generator sampling...')
    mask = cv2.imread('face-parsing.PyTorch/res/test_res/img_0_mask.png', cv2.IMREAD_GRAYSCALE)
    mask = mask/255
    # mask_256 = cv2.resize(mask, (256, 256), cv2.INTER_CUBIC)
    # voxel_grid = voxel_grid.transpose(1, 2, 0)
    # voxel_grid *= mask_256
    # voxel_grid = voxel_grid.transpose(2, 0, 1)

    voxel_grid = sample_generator(
        generator, z, max_batch=max_batch, cube_length=opt.cube_size,
        voxel_resolution=opt.voxel_resolution, mask=mask)

    # os.makedirs(opt.output_dir, exist_ok=True)
    ensure_dir(opt.output_dir)

    voxel_grid = np.maximum(voxel_grid, 0)

    vertices, triangles = mcubes.marching_cubes(
        voxel_grid, opt.sigma_threshold)  # 0.59s

    mcubes.export_obj(
        vertices, triangles,
        os.path.join(opt.output_dir, f'{seed}.obj')) # 2.49s

    return vertices, triangles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    # parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--curriculum', type=str, default='CelebA')
    parser.add_argument('--save_depth', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--range', default=None, nargs=3, metavar=('start', 'end', 'step'),
                        type=int, help='specify a range')
    parser.add_argument('--split', type=bool, default=True)

    parser.add_argument('--cube_size', type=float, default=0.3)
    parser.add_argument('--voxel_resolution', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='shapes')
    parser.add_argument('--sigma_threshold',
                        type=float,
                        default=20.0,
                        help='threshold to consider a location is occupied')

    opt = parser.parse_args()

    # curriculum = getattr(curriculums, opt.curriculum)
    curriculum = curriculums.get(opt.curriculum)
    curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
    curriculum['img_size'] = opt.image_size
    curriculum['psi'] = 0.7
    curriculum['v_stddev'] = 0
    curriculum['h_stddev'] = 0
    curriculum['lock_view_dependence'] = opt.lock_view_dependence
    curriculum['last_back'] = curriculum.get('eval_last_back', False)
    curriculum['nerf_noise'] = 0
    curriculum = {key: value for key, value in curriculum.items() if type(key) is str}

    # os.makedirs(opt.output_dir, exist_ok=True)
    ensure_dir(opt.output_dir)

    generator = torch.load(opt.path, map_location=torch.device(device))
    ema_file = opt.path.split('generator')[0] + 'ema.pth'
    ema = torch.load(ema_file)
    ema.copy_to([p for p in generator.parameters() if p.requires_grad])

    # generator.siren.mapping_network.transformer.s = torch.from_numpy(np.array([0.1]).astype(np.float32)).to(device)
    # generator.siren.mapping_network.transformer.t = torch.from_numpy(np.array([[0.0,-0.042,-0.1]]).astype(np.float32)).T.to(device)
    # generator.siren.mapping_network.transformer.p_y_r = torch.rand(3).to(device)*1e-9

    generator.set_device(device)
    generator.eval()

    face_angles = [0.]
    # face_angles = [-0.5, -0.25, 0., 0.25, 0.5]

    face_angles = [a + curriculum['h_mean'] for a in face_angles]

    dim_id = 80
    dim_exp = 64
    dim_bg_geo = 80
    dim_tex = 80
    dim_gamma = 27
    dim_bg_tex = 80
    length = 80+64+80+80+27+80
    z_manager = Z_Manager(length, device)
    # self.dim_id = 0
    # self.dim_exp = 80
    # self.dim_bg_geo = 144
    # self.dim_tex = 224
    # self.dim_gamma = 304
    # self.dim_bg_tex = 331

    if opt.range is not None:
        seeds = list(range(*opt.range))
    else:
        seeds = list(opt.seeds)
    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        z_ori = torch.randn((1, length), device=device)

        zs = [z_ori]

        img_lines = []
        depth_lines = []
        for z_num, z in enumerate(zs):
            images = []
            depths = []
            gt_depths = []
            for yaw_num, yaw in enumerate(face_angles):
                curriculum['h_mean'] = yaw
                img, tensor_img, depth_map = generate_img_wo_gt_depth(
                    generator, z, gt_depths=opt.save_depth, **curriculum)
                images.append(tensor_img)
                depths.append(depth_map)

            print(f'z: {z}')
            vs, ts = get_mesh(generator, z, opt, max_batch=200000)

            depths = torch.cat(depths, dim=-1)[0].numpy()
            depths = (depths-depths.min())/(depths.max()-depths.min())
            depths = cv2.cvtColor(depths, cv2.COLOR_GRAY2BGR)
            images = torch.cat(images, dim=-1)
            images = images[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
            images = (images-images.min())/(images.max()-images.min())

            img_lines.append(images*255)
            depth_lines.append(depths*255)

        img_path = os.path.join(
            opt.output_dir, opt.curriculum, f'grid_seed{seed}.png')
        # os.makedirs(os.path.dirname(img_path), exist_ok=True)
        ensure_dir(img_path)
        # from IPython import embed; embed()
        catimg_lines = np.concatenate(img_lines, axis=0)
        catdepth_lines = np.concatenate(depth_lines, axis=0)
        output = np.concatenate([catdepth_lines, catimg_lines], axis=1)
        cv2.imwrite(img_path, output)

        if opt.split:
            split_path = os.path.join(
                opt.output_dir, opt.curriculum, f'splits/grid_seed{seed}')
            # os.makedirs(split_path, exist_ok=True)
            ensure_dir(split_path)
            for idx in range(len(img_lines)):
                img_path = os.path.join(split_path, f'img_{idx}.png')
                d_path = os.path.join(split_path, f'depth_{idx}.png')
                cv2.imwrite(img_path, img_lines[idx])
                cv2.imwrite(d_path, depth_lines[idx])
