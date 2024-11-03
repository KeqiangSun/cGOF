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
from torchvision.utils import make_grid
from utils.mesh_io import *
import math
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


# def ellipse(num, x_max, y_max):
#     ids = np.arange(num)
#     xs = np.min((x_max*ids*2/(num-1), x_max*2-x_max*ids*2/(num-1)), axis=0)
#     ys = np.min((y_max*ids*2/(num-1), y_max*2-x_max*ids*2/(num-1)), axis=0)
#     theta = ids / (num-1) * math.pi * 2
#     yaw = np.cos(theta) * xs# + math.pi / 2
#     pitch = np.sin(theta) * ys# * 0.8# + math.pi / 2

#     from IPython import embed; embed()
#     return np.stack([yaw,pitch],axis=1)

def ellipse(num, x_max, y_max):
    # from IPython import embed; embed()
    ids = np.arange(num)
    yaw = x_max * np.sin(2 * 3.14 * ids / (num))
    pitch = -0.05 + y_max * np.cos(2 * 3.14 * ids / (num))
    # from IPython import embed; embed()
    return np.stack([yaw, pitch],axis=1)

def save_video(out_path, frames, cycle=False):
    out_fold = os.path.dirname(out_path)
    os.makedirs(out_fold, exist_ok=True)
    # frames = frames.detach().cpu().numpy().transpose(0,2,3,1)  # TxCxHxW -> TxHxWxC
    if cycle:
        if isinstance(frames, list):
            frames += frames[::-1]
        elif isinstance(frames, np.ndarray):
            frames = np.concatenate([frames, frames[::-1]], 0)
        else:
            print("cycle failed!")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    vid = cv2.VideoWriter(out_path, fourcc, 25, (frames[0].shape[1], frames[0].shape[0]))
    [vid.write(np.uint8(f[...,::-1])) for f in frames]
    vid.release()
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
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.ply')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--factor', help='factor variation mode. 0 = id, 1 = expression, 2 = lighting, 3 = pose, 4 = exp + pose, 5 = rotate_pose, 6 = interpolate_exp.', type=int, required=True, default=0, show_default=True)
@click.option('--subject', help='how many subjects to generate.', type=int, default=20, show_default=True)
@click.option('--variation', help='how many images to generate per subject.', type=int, default=20, show_default=True)
@click.option('--get_raw', help='save raw image?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--get_norm', help='save normal map?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--get_input', help='save inputs?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--angle_multiplier', help='how wild is the head pose.', type=float, default=0.0, show_default=True)
@click.option('--merge', help='merge variations?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--get_video', help='write results to video?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--nrow', help='Number of images displayed in each row of the grid.', type=int, default=5, show_default=True)
@click.option('--neural_rendering_resolution', help='NeRF resolution.', type=int, default=None, show_default=True)
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
    get_raw: bool,
    get_norm: bool,
    get_input: bool,
    angle_multiplier: float,
    merge: bool,
    get_video: bool,
    nrow: int,
    neural_rendering_resolution: int,
):
    bs = 1
    
    dim_id = 80
    dim_exp = 64
    dim_bg_geo = 80
    dim_tex = 80
    dim_gamma = 27
    dim_bg_tex = 80
    length = dim_id + dim_exp + dim_bg_geo + dim_tex + dim_gamma + dim_bg_tex

    device = torch.device('cuda')
    # subject_seeds = [
    #     1, 6, 17, 19, 26, 35, 50, 61, 63, 65, 72, 75, 84, 90, 91, 92,
    #     104, 107, 109, 111, 112, 117, 131, 135, 136, 140, 141, 142, 143, 146, 148, 150, 162, 169, 170, 179, 181, 182, 183, 188, 189, 190, 194, 196, 197, 198,
    #     208, 213, 214, 215, 223, 230, 232, 235, 243, 248, 249, 251, 255, 264, 269, 273, 280, 283, 284, 286, 289, 294,
    #     300, 302, 303, 307, 308, 313,  315, 316, 318, 320, 321, 325, 328, 329, 331, 332, 333, 334, 337, 343, 346, 354, 358, 360, 369, 370, 371, 381, 385, 387, 388, 
    #     400, 409, 413, 416, 417, 418, 420, 426, 430, 435, 437, 439, 442, 451, 453, 455, 460, 466, 471, 473, 479, 482, 483, 488, 490, 491, 494, 495, 497, 499, 
    #     500, 505, 506, 507, 508, 509, 512, 514, 517, 520, 521, 523, 524, 527, 528, 531, 533, 536, 537, 540, 541, 542, 545, 551, 552, 553, 554, 556, 559, 561, 572, 575, 576, 577, 578, 579, 583, 593, 594, 598,
    #     604, 606, 614, 616, 618, 612, 627, 630, 634, 636, 643, 645, 646, 647, 656, 666, 667, 669, 676, 677, 684, 686, 687, 695, 699,
    #     700, 703, 704, 711, 712, 
    #     715, 724, 725, 728, 730, 731, 734, 741, 742, 744, 745, 747, 752, 754, 756, 759, 760, 762, 763, 764, 765, 769, 770, 772, 773, 775, 777, 778, 780, 786, 790, 793, 
    #     806,
    #     807, 808, 809, 810, 813, 817, 820, 824, 828, 833, 835, 836, 837, 847, 849, 860, 861, 866, 867, 868, 869, 871, 872, 883, 884, 885, 891, 894,
    #     900, 903, 907, 920, 923, 925, 927, 938, 939, 942, 944, 947, 948, 950, 960, 962, 964, 971, 972, 973, 975, 979, 998,
    # ]
    # subject_seeds = [6,63,65,75,91,104,141,170,182,214,255,316,325,328,329,416,430,439,466,471,553,559,575,578,598,604,606,643,645,646,666,700,712] # exp
    # [19,90,182,196,232,255,316,430,490,509,512,520,545,551,553,554,559,575,583,669,686,695,703,728,730,731,742,744,759,760,763,765,772,777,793,806,808,809,810,813,837,860,861,867,871,872,883,891,920,944,947,948,950,972] # pose
    
    # set pose
    if factor in [3, 4]:
        pitch_noise = np.linspace(0.5,-0.5,variation) * angle_multiplier
        yaw_noise = np.linspace(0.5,-0.5,variation) * angle_multiplier
        poses = np.array(np.meshgrid(yaw_noise, pitch_noise))
        poses = poses.reshape([2,-1]).T
        variation = variation ** 2
    elif factor in [5]:
        poses = ellipse(variation, 0.5*angle_multiplier, 0.5*angle_multiplier)

    # set subject
    if factor in [1, 6]:
        subject_seeds = [6,63,65,75,91,104,141,170,182,214,255,316,325,328,329,416,430,439,466,471,553,559,575,578,598,604,606,643,645,646,666,700,712] #res 512
        # subject_seeds = [135,170,230,232,242,273,294,320,321,322,346,353,417,460,559,633,686,730,789,836,861,871] #res 128
        # subject_seeds = [63,91,182,559,575]
        # subject_seeds = range(1000, 1000+subject, 1)
        # subject_seeds = [1056,1118,1312,1344,1419,1466]
        # subject_seeds = range(subject)
    elif factor in [3, 5]:
        subject_seeds = [19,90,182,196,232,255,316,430,490,509,512,520,545,551,553,554,559,575,583,669,686,695,703,728,730,731,742,744,759,760,763,765,772,777,793,806,808,809,810,813,837,860,861,867,871,872,883,891,920,944,947,948,950,972] # pose
        # subject_seeds = range(subject)
    elif factor in [4]:
        # subject_seeds = [6,63,65,75,91,104,141,170,182,214,255,316,325,328,329,416,430,439,466,471,553,559,575,578,598,604,606,643,645,646,666,700,712]
        # subject_seeds = [1056,1118,1312,1344,1419,1466]
        # subject_seeds = [182,255,316,430,553,559,575]
        # subject_seeds = [1056,1419]
        subject_seeds = [170,255,104,325,416,430,471,643,646,700,1056,1419]
    elif factor in [7]:
        subject_seeds = [104,170,471,646,700,1419]
        # subject_seeds = [170,255,104,416,430,471,646,700,1056,1419]
        # subject_seeds = range(subject)
    else:
        subject_seeds = range(subject)
    subject_seeds = subject_seeds[:subject]

    # set expression
    if factor in [1]:
        # exp_seeds = [23, 9, 10, 11, 2, 142, 55, 60, 83, 29]
        exp_seeds = [5,24,29,35,38,39,41,50,51,55,60,83,93,97,98,100,101,103,109,117,124,130,135,142,147,175,197,201,206,212,266,377]
    elif factor in [4]:
        exp_seeds = [5,24,29,35,38,39,41,50,51,55,60,83,93,97,98,100,101,103,109,117,124,130,135,142,147,175,197,201,206,212,266,377]
        # exp_seeds = [23, 9, 10, 11, 2]
        # exp_seeds = [142, 55, 60, 83, 29]
        # exp_seeds = [83]
        rep = variation // len(exp_seeds)
        els = variation % len(exp_seeds)
        exp_seeds = exp_seeds * rep + exp_seeds[:els]
    elif factor in [6]:
        # exp_seeds = [23, 9, 10, 11, 2, 23]
        exp_seeds = [142, 55, 60, 83, 29, 142]
        exp_lats = []
        for s in exp_seeds:
            np.random.seed(s)
            l = torch.from_numpy(np.random.randn(bs, length)).to(device)
            exp_lats.append(l)
        exp_lats = torch.cat(exp_lats, dim=0).T.unsqueeze(0)
        # from IPython import embed; embed()
        exp_lats = torch.nn.functional.interpolate(exp_lats, variation, mode='linear')[0].T
    else:
        exp_seeds = range(variation)

    # print(f"seeds:{seeds}")
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
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
    
    G.rendering_kwargs['get_norm'] = get_norm

    save_path = os.path.join(
        outdir,
        f"factor{factor}_subject{subject}_variation{variation}")
    img_split_save_path = os.path.join(save_path, 'split/img')
    os.makedirs(img_split_save_path, exist_ok=True)
    
    if get_norm:
        norm_split_save_path = os.path.join(save_path, 'split/norm')
        os.makedirs(norm_split_save_path, exist_ok=True)

    if get_input:
        input_save_path = os.path.join(save_path, 'input')
        mesh_save_path = os.path.join(input_save_path, 'mesh')
        face_save_path = os.path.join(input_save_path, 'face')
        os.makedirs(mesh_save_path, exist_ok=True)
        os.makedirs(face_save_path, exist_ok=True)

    if merge:
        merge_save_path = os.path.join(save_path, 'merge')
        os.makedirs(merge_save_path, exist_ok=True)

    if get_video:
        video_save_path = os.path.join(save_path, 'video')
        os.makedirs(video_save_path, exist_ok=True)

    if shapes:
        shape_save_path = os.path.join(save_path, 'shape')
        os.makedirs(shape_save_path, exist_ok=True)
        
    if get_raw:
        raw_save_path = os.path.join(save_path, 'split/raw')
        os.makedirs(raw_save_path, exist_ok=True)
        

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)


    # Generate images.
    for i, s in enumerate(subject_seeds):

        print(f'Generating image for subject {s}')
        np.random.seed(s)
        lats1 = torch.from_numpy(np.random.randn(bs, G.z_dim)).to(device)
        yaw_noise = (np.random.uniform()-0.5) * angle_multiplier
        pitch_noise = (np.random.uniform()-0.5) * angle_multiplier

        if factor == 7:
            subject_seeds2 = (subject_seeds*2)[i:i+1+len(subject_seeds)]
            subj_lats = []
            for s in subject_seeds2:
                np.random.seed(s)
                l = torch.from_numpy(np.random.randn(bs, length)).to(device)
                subj_lats.append(l)
            subj_lats = torch.cat(subj_lats, dim=0).T.unsqueeze(0)
            # from IPython import embed; embed()
            subj_lats = torch.nn.functional.interpolate(subj_lats, variation, mode='linear')[0].T
        if merge or get_video:
            outputs = []

        for j in tqdm(range(variation)):
            if factor in [0,1,2,3,4,5]:
                np.random.seed(exp_seeds[j])
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
                        lats2[:, dim_id:dim_id+dim_exp],
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
            elif factor in [3, 5]:  # change pose only
                lats = torch.cat([
                        lats1[:, :dim_id],
                        lats1[:, dim_id:dim_id+dim_exp],
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
                yaw_noise, pitch_noise = poses[j]
                # print(f"yaw: {yaw_noise}, pitch: {pitch_noise}.")
            elif factor == 4:  # change all factors
                lats = torch.cat([
                        lats1[:, :dim_id],
                        lats2[:, dim_id:dim_id+dim_exp],
                        lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                        lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                        lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
                yaw_noise, pitch_noise = poses[j]
            elif factor == 6:
                lats2 = exp_lats[j:j+1]
                # from IPython import embed; embed()
                lats = torch.cat([
                    lats1[:, :dim_id],
                    lats2[:, dim_id:dim_id+dim_exp],
                    lats1[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                    lats1[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                    lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                    lats1[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
            elif factor == 7:
                np.random.seed(83)
                exp_lat = torch.from_numpy(np.random.randn(bs, G.z_dim)).to(device)
                subj_lat = subj_lats[j:j+1]
                lats = torch.cat([
                    subj_lat[:, :dim_id],
                    exp_lat[:, dim_id:dim_id+dim_exp],
                    subj_lat[:, dim_id+dim_exp:dim_id+dim_exp+dim_bg_geo],
                    subj_lat[:, dim_id+dim_exp+dim_bg_geo:dim_id+dim_exp+dim_bg_geo+dim_tex],
                    subj_lat[:, dim_id+dim_exp+dim_bg_geo+dim_tex:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma],
                    subj_lat[:, dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma:dim_id+dim_exp+dim_bg_geo+dim_tex+dim_gamma+dim_bg_tex],
                    ], axis=1)
            # cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0.2]), device=device)
            cam_pivot = torch.tensor([0, 0, 0.27], device=device)
            cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + yaw_noise, np.pi/2 + pitch_noise, cam_pivot, radius=cam_radius, device=device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

            ws = G.mapping(lats.to(torch.float32), conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            outs = G.synthesis(ws, camera_params, face_recon=face_recon, neural_rendering_resolution=neural_rendering_resolution)
            img = outs['image']

            if merge or get_video:
                outputs.append(img[0].detach())

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{img_split_save_path}/{s:03}_{j:03}.png')

            if get_raw:
                raw = outs['image_raw']
                raw = (raw.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(raw[0].cpu().numpy(), 'RGB').save(f'{raw_save_path}/{s:03}_{j:03}.png')

            if get_norm:
                normal = G.renderer.normal
                PIL.Image.fromarray(normal.astype(np.uint8), 'RGB').save(f'{norm_split_save_path}/{s:03}_{j:03}.png')

            if get_input:
                pigan_depth, pigan_mask, pigan_face, pigan_mesh = \
                    face_recon.render_mesh(
                        lats.to(torch.float32), G.theta, G.phi, G.center,
                    )
                mask = (pigan_mask[0].permute(1,2,0).detach().cpu().numpy()>0)*255.0
                # mesh0 = pigan_mesh[0].permute(1,2,0).clip(0,1).cpu().numpy()*255
                # face0 = pigan_face[0].permute(1,2,0).clip(0,1).cpu().numpy()*255
                # mask0 = pigan_mask[0].permute(1,2,0).repeat(1,1,3).clip(0,1).to(torch.uint8).cpu().numpy()*255
                # depth0 = pigan_depth[0].permute(1,2,0).repeat(1,1,3).clip(0,1).to(torch.uint8).cpu().numpy()*255
                # PIL.Image.fromarray(mesh0, 'RGB').save(f'{input_save_path}/mesh_{s:03}_{j:03}.png')
                # PIL.Image.fromarray(face0, 'RGB').save(f'{input_save_path}/face_{s:03}_{j:03}.png')
                # PIL.Image.fromarray(mask0, 'RGB').save(f'{input_save_path}/mask_{s:03}_{j:03}.png')
                # PIL.Image.fromarray(depth0, 'RGB').save(f'{input_save_path}/depth_{s:03}_{j:03}.png')
                mesh0 = pigan_mesh[0].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
                mesh0 = np.concatenate([mesh0, mask], axis=-1)
                cv2.imwrite(
                    os.path.join(mesh_save_path,f'mesh_{s:03}_{j:03}.png'), mesh0)
                face0 = pigan_face[0].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
                face0 = np.concatenate([face0, mask], axis=-1)
                cv2.imwrite(
                    os.path.join(face_save_path,f'face_{s:03}_{j:03}.png'), face0)
        if merge:
            import torch.nn as nn
            output_grid = make_grid(outputs, nrow, normalize=False)
            _,h,w = output_grid.shape
            mx = max(h,w)
            scale = 512*4 / mx
            h = int(h * scale)
            w = int(w * scale)
            output_grid = nn.functional.interpolate(output_grid.unsqueeze(0), [h,w])[0]
            output_grid = (output_grid.permute(1,2,0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(output_grid.cpu().numpy(), 'RGB').save(f'{merge_save_path}/{s:03}.png')

        if shapes:
            z = lats
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            # save the sampled points.
            save_obj_vertex(
                os.path.join(
                    shape_save_path,
                    f'sample_{s:04d}.obj'),
                    samples.reshape((shape_res, shape_res, shape_res, 3))[::32, ::32, ::32])

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(
                    np.transpose(sigmas, (2, 1, 0)),
                    [0, 0, 0],
                    # [-shape_res/2, -shape_res/2, -shape_res/2],
                    # [-0.5, -0.5, -0.5],
                    1,
                    os.path.join(shape_save_path, f'eg3d_{s:04d}.obj'),
                    # scale=shape_res,
                    level=10)
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(shape_save_path, f'eg3d_{s:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas

        if get_video:
            frames = [(f * 127.5 + 128).clamp(0, 255).cpu().numpy().transpose(1,2,0) for f in outputs]
            save_video(f'{video_save_path}/{s:03}.mp4', frames)
#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
