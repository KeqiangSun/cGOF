  
"""Script to render a video using a trained pi-GAN  model."""

import os
import sys
add_path = os.path.realpath('Deep3DFaceRecon_pytorch')
print(add_path)
sys.path.insert(0, add_path)


import argparse
import math
# import os

from torchvision.utils import save_image

import torch
from tqdm import tqdm
import numpy as np
import curriculums
from generators import generators
from siren import siren

from models import create_model
from deep3dfacerecon_opt import deep3dfacerecon_opt
from utils.utils import FitCurve, curve, str2bool
import cv2


def init_face_recon(device=torch.device('cuda')):
    face_recon = create_model(deep3dfacerecon_opt)
    face_recon.setup(deep3dfacerecon_opt)
    face_recon.device = device
    face_recon.set_coeff_static()
    face_recon.parallelize()
    face_recon.eval()
    return face_recon

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
parser.add_argument('--seed', type=int, default=31)
parser.add_argument('--output_dir', type=str, default='vids/poses_gif/')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_batch_size', type=int, default=2400000)
parser.add_argument('--lock_view_dependence', action='store_true')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--ray_step_multiplier', type=int, default=2)
parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--pose_range', type=float, default=1.2)
parser.add_argument('--curriculum', type=str, default='CelebA')

parser.add_argument('--last_back', action='store_true')
parser.add_argument('--rt_norm', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--split', type=bool, default=True)

opt = parser.parse_args()

save_dir = os.path.join(opt.output_dir, opt.curriculum, f"seed_{opt.seed}")
os.makedirs(save_dir, exist_ok=True)

curriculum = curriculums.get(opt.curriculum)
curriculum['num_steps'] = curriculum[0]['num_steps'] * opt.ray_step_multiplier
# curriculum['num_steps_surface'] = curriculum[0]['num_steps_surface'] * opt.ray_step_multiplier
# curriculum['num_steps_coarse'] = curriculum['num_steps_coarse'] * opt.ray_step_multiplier
# curriculum['num_steps_fine'] = curriculum['num_steps_fine'] * opt.ray_step_multiplier
curriculum['img_size'] = opt.image_size
curriculum['psi'] = 0.7
curriculum['v_stddev'] = 0
curriculum['h_stddev'] = 0
curriculum['lock_view_dependence'] = opt.lock_view_dependence
# curriculum['last_back'] = curriculum.get('eval_last_back', False)
curriculum['white_back'] = False
curriculum['last_back'] = opt.last_back
curriculum['num_frames'] = opt.num_frames
curriculum['nerf_noise'] = 0
# curriculum['sample_near_mesh'] = False
curriculum = {key: value for key, value in curriculum.items() if type(key) is str}
if 'interval_min' in curriculum:
    curriculum['interval'] = curriculum['interval_min']
rt_norm = opt.rt_norm

def img_normalize(img):
    img = torch.clamp(img, min=-1, max=1)
    img = img / 2 + 0.5

    return img

def ellipse(i, num, x_max, y_max):
    theta = i / (num-1) * math.pi * 2 
    yaw = math.cos(theta) * x_max + math.pi / 2
    pitch = math.sin(theta) * y_max * 0.8 + math.pi / 2
    return yaw, pitch

SIREN = getattr(siren, curriculum['model'])
generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim']).to(device)
ema_file = opt.path.split('generator')[0] + 'ema.pth'
ema = torch.load(ema_file)
ema.copy_to(generator.parameters())
generator.set_device(device)
generator.eval()

face_recon = init_face_recon(device)

torch.manual_seed(opt.seed)
z = torch.randn((1, 411), device=device)
ellipse_points = []
for i in range(opt.num_frames):
    yaw, pitch = ellipse(
        2*i,
        opt.num_frames,
        np.min((opt.pose_range*i*2/(opt.num_frames-1), opt.pose_range*2-opt.pose_range*i*2/(opt.num_frames-1))),
        np.min((opt.pose_range*i*2/(opt.num_frames-1), opt.pose_range*2-opt.pose_range*i*2/(opt.num_frames-1)))
    )
    ellipse_points.append([yaw, pitch])
even_ellipse_points = curve(np.array(ellipse_points), opt.num_frames)
for i in tqdm(range(opt.num_frames)):
    yaw, pitch = even_ellipse_points[i]
    curriculum['h_mean'] = yaw
    curriculum['v_mean'] = pitch
    with torch.no_grad():
        fetch_data = {}
        outputs = generator.staged_forward(
            z, face_recon=face_recon, rt_norm=rt_norm,
            zy_data=fetch_data, **curriculum)
    # tensor_img = outputs['imgs']
    tensor_img = outputs[0]
    
    save_image(img_normalize(tensor_img), os.path.join(save_dir, f"img_{i:03d}.png"))
    # save_image(single_img, os.path.join(save_dir, f"single_img_{i:03d}.png"))
    # if normals is not None:
    if rt_norm:
        # norm = generator.pred_normals
        
        n_root = os.path.join(save_dir, f'splits/normals/{opt.seed}')
        os.makedirs(n_root, exist_ok=True)
        n_path = os.path.join(n_root, f'normal_seed{opt.seed}_i{i:04d}.png')
        norm = generator.pred_normals.reshape(
                    opt.image_size, opt.image_size, 3).detach().cpu().numpy()
        # from IPython import embed; embed()
        norm = (norm + 1) / 2
        norm = cv2.cvtColor(norm, cv2.COLOR_RGB2BGR)*255
        cv2.imwrite(n_path, norm)
        
        
    # save_image(normals, os.path.join(save_dir, f"normal_{i:03d}.png"))

    # if opt.split:
    #     split_path = os.path.join(save_dir, f'splits/imgs/grid_seed{opt.seed}/row{i}/')
    #     os.makedirs(split_path, exist_ok=True)
    #     img_path = os.path.join(split_path, f'img_seed{opt.seed}_i{i}.png')
    #     img_vis = tensor_img[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
    #     img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())
    #     cv2.imwrite(img_path, img_vis*255)

    #     if opt.rt_face_recon:
    #         f_root = os.path.join(save_path, f'splits/face_recon/grid_seed{opt.seed}/row{i}/')
    #         os.makedirs(f_root, exist_ok=True)

    #         d_path = os.path.join(f_root, f'depth_seed{opt.seed}_i{i}.png')
    #         d = face_recon.d_pigan_input
    #         d[torch.logical_not(face_recon.m_pigan_input)] = face_recon.d_pigan_input[face_recon.m_pigan_input].mean()
    #         d = (d-d.min())/(d.max()-d.min())
    #         d = d.detach().cpu().numpy()[0, 0]
    #         d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    #         d = np.dstack([d,np.uint8(face_recon.m_pigan_input[0].permute(1,2,0).detach().cpu().numpy() * 1.0)])
    #         cv2.imwrite(d_path, d*255)

    #         f_path = os.path.join(f_root, f'face_seed{opt.seed}_i{i}.png')
    #         f = face_recon.f_pigan_input[0].permute(1,2,0)
    #         f[torch.logical_not(face_recon.m_pigan_input[0,0])] = face_recon.f_pigan_input[0].permute(1,2,0)[face_recon.m_pigan_input[0,0]].mean(0)
    #         f = (f-f.min())/(f.max()-f.min())
    #         f = f.detach().cpu().numpy()
    #         f = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    #         f = np.dstack([f,np.uint8(face_recon.m_pigan_input[0].permute(1,2,0).detach().cpu().numpy() * 1.0)])
    #         cv2.imwrite(f_path, f*255)

    #         m0 = np.uint8(face_recon.m_pigan_input[0].permute(1,2,0).detach().cpu().numpy() * 255.0)
    #         m0 = cv2.resize(m0, (256,256))
    #         lm = face_recon.pigan_landmark
    #         lm[..., 1] = 255-lm[...,1]
    #         mesh0 = face_recon.pigan_mesh[0].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
    #         mesh0 = np.dstack([mesh0,m0])
    #         cv2.imwrite(
    #             os.path.join(f_root,f'mesh_seed{opt.seed}_i{i}.png'), mesh0)
    #         save_img_with_landmarks(
    #             mesh0[...,[2,1,0,3]],
    #             os.path.join(f_root, f'mesh_lm_seed{opt.seed}_i{i}.png'), lm[0], radius=2, color=(0,255,0))
            


# generate frames
# python tools/eval/render/render_pose_video.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300_ckpt/generator.pth --lock_view_dependence --image_size 128 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_lm3d300 --num_frames 200 --pose_range 1.2
# from images to video
# ffmpeg -r 15 -f image2 -i xxx.png -c:v libx264 -crf 25 -pix_fmt yuv420p xxx.mp4
# ffmpeg -r 15 -f image2 -i vids/img_%03d.png -pix_fmt yuv420p 31.mp4