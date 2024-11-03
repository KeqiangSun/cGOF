# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# python gen_videos_from_given_latent_code.py --outdir=out --trunc=0.7 --w_path ./projector_out/00018_w/00018_w.npy   --network=./networks/ffhqrebalanced512-64.pkl --sample_mult=2
"""Generate lerp videos using pretrained network pickle."""
"""python edit_latent_code.py \
--network ./inversion/outputs/embeddings/leonardo_dicaprio/PTI/leonardo_dicaprio/model_leonardo_dicaprio.pt \
--w_path ./inversion/outputs/embeddings/leonardo_dicaprio/PTI/leonardo_dicaprio/optimized_noise_dict.pickle \
--c_path ./proj_data/input/crop_1024/leonardo_dicaprio_c.npy \
--z_path ./proj_data/input/crop_1024/proj_data/input/crop_1024/epoch_20_000000/leonardo_dicaprio.mat \
--outdir ./edit --sample_mult 2 --use_face_recon True"""
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc
from training.triplane import TriPlaneGenerator

from Deep3DFaceRecon_pytorch import init_face_recon
from torchvision.utils import save_image

from facenet_pytorch.facenet import FaceNet
# ----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
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
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


# ----------------------------------------------------------------------------

def gen_interp_video(G, face_recon, latent, mp4: str,  w_frames=60 * 4, kind='cubic', grid_dims=(1, 1),
                     num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image',
                     gen_shapes=False, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    name = mp4[:-4]
    if num_keyframes is None:

        num_keyframes = 1 // (grid_w * grid_h)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0],
                                                                                                      device=device)

    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(latent.shape[0], 1)
    ws = latent # 1, 14, 512

    if ws.shape[1] != G.backbone.mapping.num_ws:
        ws = ws.repeat([1,G.backbone.mapping.num_ws, 1])

    _ = G.synthesis(ws[:1], c[:1], face_recon=face_recon)  # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1]) # (5, 14, 512)
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)
    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}/'.format(name)
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    camera_lookat_point, radius=2.7, device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                # c = np.load("/home/kqsun/Tasks/EG3D-projector/eg3d/projector_test_data/1.npy")
                # c = np.reshape(c,(1,25))
                # c = torch.FloatTensor(c).cuda()

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                # latent  = np.load("/home/kqsun/Tasks/EG3D-projector/eg3d/projector_out/1_w/1_w.npy")
                # latent = torch.FloatTensor(latent).cuda()
                # img = G.synthesis(ws=latent, c=c[0:1], noise_mode='const')[image_mode][0]
                img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], face_recon=face_recon, noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))

                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0],
                                                                       cube_length=G.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total=samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                                       transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                                       w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head + max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        from shape_utils import convert_sdf_samples_to_ply
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                                   os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)
                    else:  # output mrc
                        with mrcfile.new_mmap(outdir + f'{frame_idx:04d}_shape.mrc', overwrite=True, shape=sigmas.shape,
                                              mrc_mode=2) as mrc:
                            mrc.data[:] = sigmas

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()
    all_poses = np.stack(all_poses)

    if gen_shapes:
        print(all_poses.shape)
        with open(mp4.replace('.mp4', '_trajectory.npy'), 'wb') as f:
            np.save(f, all_poses)


# ----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')


# ----------------------------------------------------------------------------

def run_G(z, w, c, G, face_recon, image_mode):
    G.z = z
    w.requires_grad = True
    frame = G.synthesis(
        w, c, face_recon=face_recon, noise_mode='const')[image_mode]
    return frame
def run_face_recog(frame, face_recog):
    low = float(frame.min())
    high = float(frame.max())
    frame = frame.clamp(min=low, max=high)
    frame = frame.sub(low).div(max(high - low, 1e-5))
    recog_input = frame[0].mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0)
    recog_embedding = face_recog.run_tensor(recog_input)
    return recog_embedding
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--w_path', 'w_path', help='Network pickle filename', required=True)
@click.option('--c_path', 'c_path', help='camera param path', required=True)
@click.option('--z_path', 'z_path', help='recon results path', required=True)
@click.option('--num-keyframes', type=int,
              help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.',
              default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'Cats']), required=False, metavar='STR',
              default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']),
              required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--use_face_recon', help='use face_recon?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
# python edit_latent_code.py --network ./inversion/outputs/embeddings/bieber/PTI/bieber/model_bieber.pt --w_path ./inversion/outputs/embeddings/bieber/PTI/bieber/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/bieber.mat --outdir ./edit_results/bieber/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
def generate_images(
        network_pkl: str,
        w_path:str,
        c_path:str,
        z_path:str,
        truncation_psi: float,
        truncation_cutoff: int,
        num_keyframes: Optional[int],
        w_frames: int,
        outdir: str,
        cfg: str,
        image_mode: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        reload_modules:bool,
        use_face_recon:bool,
):
    """Render a latent vector interpolation video.
    Examples:
    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
    Animation length and seed keyframes:
    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.
    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.
    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    G = torch.load(network_pkl).to(device)
    
    face_recog = FaceNet(save_root="/home/kqsun/Tasks/eg3d/eg3d", device=device)
    # face_recog.set_devices(device)

    if use_face_recon:
        face_recon, _ = init_face_recon(device)
        G.rendering_kwargs['sample_near_mesh'] = True
        G.rendering_kwargs['using_dist_depr'] = True

        import scipy.io
        mat = scipy.io.loadmat(z_path)
        mat = np.concatenate(
            [
                mat['id'], mat['exp'], np.zeros_like(mat['id']),
                mat['tex'], mat['gamma'], np.zeros_like(mat['id'])
            ], axis=1)
        mat = torch.from_numpy(mat).to(device)
        z = face_recon.split_z(mat)
        z = face_recon.norm_coeff(z)
        z['bg_geo'] = torch.zeros_like(z['id'])
        z['bg_tex'] = torch.zeros_like(z['id'])
        z = torch.cat([z['id'], z['exp'], z['bg_geo'], z['tex'], z['gamma'], z['bg_tex']], dim=1)
        # G.z = z
    else:
        face_recon = None
        G.rendering_kwargs['sample_near_mesh'] = False
        G.rendering_kwargs['using_dist_depr'] = False


    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0  # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14  # no truncation so doesn't matter where we cutoff

    grid = (1,1)
    # latent  = np.load(w_path)
    import pickle
    with open(w_path, 'rb') as handle:
        latent = pickle.load(handle)['projected_w']
    latent = torch.FloatTensor(latent).cuda()

    # ------------------- start editing -----------------
    # c = np.load(c_path)
    # c = np.reshape(c,(1,25))
    # c = torch.FloatTensor(c).cuda()
    
    if os.path.basename(c_path).split(".")[1] == "json":
        import json
        img_name = os.path.basename(z_path).split(".")[0]
        with open(c_path) as f:
            target_pose = np.asarray(json.load(f)[img_name]['pose']).astype(np.float32)
        o = target_pose[0:3, 3]
        o = 2.7 * o / np.linalg.norm(o)
        target_pose[0:3, 3] = o
        target_pose = np.reshape(target_pose, -1)    
    else:
        target_pose = np.load(c_path).astype(np.float32)
        target_pose = np.reshape(target_pose, -1)
    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    c = torch.tensor(target_pose, device=device).unsqueeze(0)

    with torch.no_grad():
        G.z = z
        frame = G.synthesis(latent, c, face_recon=face_recon, noise_mode='const')[image_mode][0]
    save_image(frame, os.path.join(outdir, f"fit_pic.png"), normalize=True)

    torch.manual_seed(31)
    z_exps = torch.randn((12, 12, 64), device=device)
    # rows_cols = "i8j0,i6j8,i10j11,i8j0"
    rows_cols = "i8j0,i6j8,i10j11,i8j0,i6j9,i0j0,i6j11,i1j3,i8j0"
    rows_cols = rows_cols.split(',')
    exp_num = len(rows_cols)
    picked = []
    for rc in rows_cols:
        row, col = rc[1:].split('j')
        picked.append(z_exps[int(row), int(col)])
    z_exp = torch.stack(picked)

    z = z.repeat([exp_num+1, 1])
    z[1:, 80:144] = z_exp
    with torch.no_grad():
        w = G.mapping(
            z, c.repeat([z.shape[0], 1]), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    delta_w = w[1:] - w[0:1]

    edit_f = latent.repeat([delta_w.shape[0], 1, 1])
    edit_f += delta_w * 0.5
    # edit_f.requires_grad = True

    frames = []
    # frame0 = run_G(z[0:1], latent, c, G, face_recon, image_mode)
    # recog_embedding0 = run_face_recog(frame0, face_recog).detach()
    recog_embedding0 = face_recog.run_file("/home/kqsun/Tasks/eg3d/eg3d/proj_data/input2/tom.png").detach()
    for i in range(edit_f.shape[0]):
        print('====================')
        z_i = z[i+1:i+2]
        edit_f_i = edit_f[i:i+1]
        edit_f_i.requires_grad = True
        optimizer = torch.optim.Adam([edit_f_i], lr=0.0001)
        for j in range(100):
            frame = run_G(z_i, edit_f_i, c, G, face_recon, image_mode)
            recog_embedding = run_face_recog(frame, face_recog)
            loss = torch.nn.CosineEmbeddingLoss()(recog_embedding.flatten(),recog_embedding0.flatten(),torch.tensor(1).to(device))
            optimizer.zero_grad()
            print(f'j: {j}, loss: {loss.mean().item()}')
            loss.mean().backward()
            optimizer.step()
        frames.append(frame)
    frames = torch.cat(frames, dim=0)

    for i in range(frames.shape[0]):
        save_image(
            frames[i:i+1], os.path.join(outdir, f"edit_pic_{i}.png"), normalize=True)

    # ------------------------------------

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------