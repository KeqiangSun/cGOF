
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# python run_projector.py --outdir=./projector_out --latent_space_type w_plus  --network=outputs/eg3d_128_iter4800_recon4_snm_depr100_ldmk3/00000-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000600.pkl --sample_mult=2  --image_path ./projector_test_data/1.png --c_path ./projector_test_data/1.npy --num_steps 2000 --reload_modules True
"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import torch
import legacy
from torchvision.transforms import transforms
from projector import w_projector, w_plus_projector
from PIL import Image

from torch_utils import misc
from training.triplane import TriPlaneGenerator

from Deep3DFaceRecon_pytorch import init_face_recon
import torch.nn as nn

# CUDA_VISIBLE_DEVICES=1 python run_projector.py --outdir=./projector_out --latent_space_type w_plus  --network=outputs/eg3d_128_iter4800_recon4_snm_depr100_ldmk6/00000-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000800.pkl --sample_mult=2  --image_path ./projector_test_data/trump.png --c_path ./projector_test_data/trump.npy --num_steps 2000 --reload_modules True --use_face_recon True
# ----------------------------------------------------------------------------

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

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--latent_space_type', help='latent_space_type', type=click.Choice(['w', 'w_plus']), required=False, metavar='STR',
              default='w', show_default=True)
@click.option('--image_path', help='image_path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--c_path', help='camera parameters path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--z_path', help='recon results path', type=str, required=True, metavar='STR', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float,
              help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--num_steps', 'num_steps', type=int,
              help='Multiplier for depth sampling in volume rendering', default=500, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--use_face_recon', help='use face_recon?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def run(
        network_pkl: str,
        outdir: str,
        sampling_multiplier: float,
        nrr: Optional[int],
        latent_space_type:str,
        image_path:str,
        c_path:str,
        z_path:str,
        num_steps:int,
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

    os.makedirs(outdir, exist_ok=True)

    # print('Loading networks from "%s"...' % network_pkl)
    # device = torch.device('cuda')
    # with dnnlib.util.open_url(network_pkl) as f:
    #     G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    if use_face_recon:
        face_recon, visualizer = init_face_recon(device)
        import scipy.io
        mat = scipy.io.loadmat(z_path)
        mat = np.concatenate(
            [
                mat['id'], mat['exp'], np.zeros_like(mat['id']),
                mat['tex'], mat['gamma'], np.zeros_like(mat['id'])
            ], axis=1)
        mat = torch.from_numpy(mat.astype(np.float32)).to(device)
        z = face_recon.split_z(mat)
        z = face_recon.norm_coeff(z)
        z['bg_geo'] = torch.zeros_like(z['id'])
        z['bg_tex'] = torch.zeros_like(z['id'])
        z = torch.cat([z['id'], z['exp'], z['bg_geo'], z['tex'], z['gamma'], z['bg_tex']], dim=1)
        # from IPython import embed; embed()
    else:
        face_recon = None
        G.rendering_kwargs['sample_near_mesh'] = False
        G.rendering_kwargs['using_dist_depr'] = False
        z=None

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(
        G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr


    image = Image.open(image_path).convert('RGB')
    image_name = os.path.basename(image_path)[:-4]
    c = np.load(c_path)
    c = np.reshape(c,(1,25))
    c = torch.FloatTensor(c).cuda()


    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize((G.img_resolution, G.img_resolution))
    ])
    from_im = trans(image).cuda()
    id_image = torch.squeeze((from_im.cuda() + 1) / 2) * 255

    if latent_space_type == 'w':
        w = w_projector.project(G, face_recon, c, outdir,id_image, device=torch.device('cuda'), w_avg_samples=600,num_steps = num_steps,
                                w_name=image_name)
    else:
        w = w_plus_projector.project(G, face_recon, c,outdir, id_image, z=z, device=torch.device('cuda'), w_avg_samples=600, w_name=image_name,num_steps = num_steps )
        pass

    w = w.detach().cpu().numpy()
    np.save(f'{outdir}/{image_name}_{latent_space_type}/{image_name}_{latent_space_type}.npy', w)

    PTI_embedding_dir = f'./projector/PTI/embeddings/{image_name}'
    os.makedirs(PTI_embedding_dir,exist_ok=True)

    # np.save(f'./projector/PTI/embeddings/{image_name}/{image_name}_{latent_space_type}.npy', w)


    ori_im = image.resize((256, 256), Image.ANTIALIAS)
    ori_im = torch.tensor(np.array(ori_im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    s = [0.9215189874]
    t = [128, 109.5384615424]
    im_tensor = resize_n_crop_tensor(ori_im, t, s)
    data = {
        'imgs': im_tensor,
        'ori_im': ori_im,
        'trans_params': np.array([256, 256]+s+t)
    }
    face_recon.set_input(data)  # unpack data from data loader
    face_recon.test()           # run inference
    face_recon.save_coeff(f'./projector/PTI/embeddings/{image_name}/{image_name}_coeff.mat') # save predicted coefficients
    visuals = face_recon.get_current_visuals()  # get image results
    visualizer.display_current_results(
        visuals, 0, 20, dataset="debug",
        save_results=True, count=0, name=image_name, add_image=False)

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------



