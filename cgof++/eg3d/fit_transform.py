# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile


import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator

from utils.mesh_io import save_obj_vertex
from IPython import embed
from utils.utils import ensure_dir

from Deep3DFaceRecon_pytorch import init_face_recon
#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 0] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 2] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[0]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[2]
    samples[:, 2] *= -1

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--split', help='save split image and external parameters', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    split: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    python gen_samples.py --outdir=imgs/rebalance_pivot000 --trunc=0.7 --seeds=0-1 --shapes=True --network=outputs/00005-ffhq-FFHQ_512-gpus8-batch8-gamma1/network-snapshot-010000.pkl --shape-format=.ply --split=True
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    face_recon, _ = init_face_recon(device)
    transformer_param_names = [name for name, _ in face_recon.transformer.named_parameters()]
    transformer_parameters = [p for n, p in face_recon.transformer.named_parameters() if n in transformer_param_names]
    opt = torch.optim.Adam([{'params': transformer_parameters, 'name': 'transformer'}],
                        lr=0.000001, betas=(0,0.9), weight_decay=0)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim).astype(np.float32)).to(device)

        imgs = []
        angle_ys = [-0.5, -0.25, 0, 0.25, 0.5]
        angle_ps = [-0.5, -0.25, 0, 0.25, 0.5]
        for j, angle_p in enumerate(angle_ps):
            img_line = []
            for i, angle_y in enumerate(angle_ys):
                # cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_pivot = torch.tensor([0, 0, 0.2], device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, camera_params, face_recon=face_recon)['image']

                lm_input = G.lm_input.clip(0, 255)
                from utils.utils import img2depth
                d_recon, m_recon = img2depth(
                    face_recon, img,
                    face_recon.eg3d_trans_params,
                    face_recon.eg3d_trans_params[3:5],
                    face_recon.eg3d_trans_params[2:3])
                lm_pred = face_recon.inv_affine_ldmks_torch_(
                    face_recon.pred_lm).clip(0, 255)
                lm_loss = torch.nn.L1Loss()(lm_input, lm_pred.detach())
                print(f'lm loss: {lm_loss.data:.3f}')
                opt.zero_grad()
                lm_loss.backward()
                opt.step()

                from utils.utils import draw_landmarks
                import cv2
                lm_input_vis = lm_input.clone()
                lm_pred_vis = lm_pred.clone()
                lm_input_vis[...,1] = 255 - lm_input_vis[...,1]
                lm_pred_vis[...,1] = 255 - lm_pred_vis[...,1]
                img_256 = torch.nn.functional.interpolate(img.to(torch.float32), size=(256,256), mode='bilinear')
                img_256 = (img_256 + 1) / 2
                img_256_vis = img_256[0].permute(1,2,0).detach().cpu().numpy()[...,[2,1,0]]*255
                img_vis = draw_landmarks(img_256_vis, lm_input_vis, (0, 255, 0))
                img_vis = draw_landmarks(img_vis, lm_pred_vis, (0, 0, 255))
                ldmk_save_path = os.path.join(outdir, f'gen_ldmk_{seed:04d}.png')
                cv2.imwrite(ldmk_save_path, img_vis)

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                img_line.append(img)


                if split:
                    split_img_path = f'{outdir}/split/{seed:04d}/grid_{seed:04d}_{i:02d}_{j:02d}.png'
                    split_npy_path = f'{outdir}/split/{seed:04d}/grid_{seed:04d}_{i:02d}_{j:02d}.npz'
                    ensure_dir(split_img_path)
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(split_img_path)
                    np.savez(split_npy_path, yaw=np.pi/2+angle_y, pitch=np.pi/2+angle_p)
            img_line = torch.cat(img_line, dim=2)
            imgs.append(img_line)
        img = torch.cat(imgs, dim=1)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
        print(f's:{face_recon.transformer.s_msra2pigan}')
        print(f'R:{face_recon.transformer.R_msra2pigan}')
        print(f'T:{face_recon.transformer.T_msra2pigan}')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
