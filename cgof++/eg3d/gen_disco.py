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
""" Usage:
# python gen_disco.py --outdir=imgs --trunc=0.7 --shapes=false --network=/home/kqsun/Tasks/eg3d/eg3d/outputs/00005-ffhq-FFHQ_512-gpus8-batch8-gamma1/network-snapshot-001200.pkl  --factor=1
python gen_disco.py --outdir=imgs/recon3_lm2_warp30 --trunc=1.0 --shapes=false --subject=1000 --variation=10 --network=outputs/eg3d_128_recon3_lm2_warp30/00000-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-001000.pkl --factor=1 --angle_multiplier=0.0 --reload_modules=True
python gen_disco.py --outdir=imgs/eg3d_128_iter4800_recon4_snm_depr100_ldmk6_warp30 --trunc=1.0 --shapes=false --subject=1000 --variation=10 --network=outputs/eg3d_128_iter4800_recon4_snm_depr100_ldmk6_warp30/00007-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000800.pkl --factor=0 --angle_multiplier=0.0 --reload_modules=True
"""
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
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--factor', help='factor variation mode. 0 = all, 1 = expression, 2 = lighting, 3 = pose.', type=int, required=True, default=0, show_default=True)
@click.option('--subject', help='how many subjects to generate.', type=int, default=20, show_default=True)
@click.option('--variation', help='how many images to generate per subject.', type=int, default=20, show_default=True)
@click.option('--angle_multiplier', help='how wild is the head pose.', type=float, default=0.0, show_default=True)
@click.option('--exp_multiplier', help='how wild is the head pose.', type=float, default=1.5, show_default=True)
def generate_images(
    network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    factor: int,
    subject: int,
    variation: int,
    angle_multiplier: float,
    exp_multiplier: float,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    print(f'exp_multiplier: {exp_multiplier}.')
    device = torch.device('cuda')
    # from IPython import embed; embed()
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    face_recon, _ = init_face_recon(device)

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    save_path = os.path.join(
        outdir,
        f"factor{factor}_subject{subject}_variation{variation}")
    os.makedirs(save_path, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    dim_id = 80
    dim_exp = 64
    dim_bg_geo = 80
    dim_tex = 80
    dim_gamma = 27
    dim_bg_tex = 80
    length = dim_id + dim_exp + dim_bg_geo + dim_tex + dim_gamma + dim_bg_tex

    bs = 1

    # Generate images.
    for i in range(subject):

        print(f'Generating image for subject {i}')
        lats1 = torch.from_numpy(np.random.RandomState(i).randn(bs, G.z_dim)).to(device)
        yaw_noise = (np.random.uniform()-0.5) * angle_multiplier
        pitch_noise = (np.random.uniform()-0.5) * angle_multiplier

        for j in range(variation):
            lats2 = torch.from_numpy(np.random.randn(bs, G.z_dim)).to(device)

            if factor == 0:  # change id only
                lats = torch.cat(
                    [
                        lats2[:, :dim_id],
                        lats1[:, dim_id:dim_id+dim_exp],
                        lats2[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats2[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats2[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], dim=1)
            elif factor == 1:  # change expression only
                lats = torch.cat(
                    [
                        lats1[:, :dim_id],
                        lats2[:, dim_id:dim_id+dim_exp]*exp_multiplier,
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], dim=1)
            elif factor == 2:  # change lighting only
                lats = torch.cat([
                        lats1[:, :dim_id],
                        lats1[:, dim_id:dim_id+dim_exp],
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats2[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
            elif factor == 3:  # change pose only
                lats = torch.cat([
                        lats1[:, :dim_id],
                        lats1[:, dim_id:dim_id+dim_exp],
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
                yaw_noise = (np.random.uniform()-0.5) * angle_multiplier
                pitch_noise = (np.random.uniform()-0.5) * angle_multiplier
            elif factor == 4:  # change all factors
                lats = torch.cat([
                        lats1[:, :dim_id],
                        lats2[:, dim_id:dim_id+dim_exp]*exp_multiplier,
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats2[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)

            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw_noise, np.pi/2 + pitch_noise, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(lats.to(torch.float32), conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, camera_params, face_recon=face_recon)['image']

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{save_path}/{i:03}_{j:02}.png')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
