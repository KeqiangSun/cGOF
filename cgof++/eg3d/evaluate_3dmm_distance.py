# CUDA_VISIBLE_DEVICES=1 \
# python tools/eval/eval_pigan_.py \
# outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500/generator.pth \
# --range 0 2 1 --save_depth \
# --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500 \
# --output_dir evaluation_outputs \
# --using_cross_test

import argparse
from genericpath import exists
from re import A
import numpy as np
import sys
import os

import torch
from tqdm import tqdm

from easydict import EasyDict
from model.model_106.MobileNet_v29 import MobileNet
from model.model_106.dlib_sensetime_corres import dlib_corres as corres_106to68

from utils.utils import *

import cv2
from utils.mesh_io import save_obj_vertex, load_obj
from utils.utils import img2depth, z2depth, save_img_with_landmarks

# from deep3dfacerecon_opt import deep3dfacerecon_opt
from Deep3DFaceRecon_pytorch import init_face_recon

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
from utils.utils import ensure_dir

from training.volumetric_rendering.utils import get_depth_mask, close_mouth
import mcubes

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
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    # samples[:, 2] *= -1

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

class EvalPigan:
    # initialization
    def __init__(self, opt, ):
        self.debug = False
        self.opt = opt
        # self.opt.output_dir = os.path.join(
        #     self.opt.output_dir, self.opt.curriculum)
        print(f'self.opt.output_dir: {self.opt.output_dir}')
        os.makedirs(self.opt.output_dir, exist_ok=True)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.data = {}

        if self.opt.snm:
            self.opt.sigma_threshold = 0.1
        else:
            self.opt.sigma_threshold = 20
        print(f"sigma_threshold: {self.opt.sigma_threshold}")

        self._init_face_recon(torch.device(self.device))
        self._init_generator(network_pkl=opt.path, reload_modules=True)
        self._init_ldmk_detector()
        self._init_error_list()
        self._set_generating_param()

    def _init_face_recon(self, device=torch.device('cuda')):
        self.face_recon, self.visualizer = init_face_recon(device)
        self.recon_basic_size = [256, 256]
        self.recon_s = [0.9215189874]
        self.recon_t = [128, 109.5384615424]
        self.trans_params = self.recon_basic_size + self.recon_s + self.recon_t

    def _init_generator(self, network_pkl, reload_modules=False):
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device) # type: ignore
        if reload_modules:
            print("Reloading Modules!")
            G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(self.device)
            misc.copy_params_and_buffers(G, G_new, require_all=True)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            G = G_new
        self.G = G
        self.fov_deg = 18.837
        self.cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=self.device), radius=2.7, device=self.device)
        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=self.device)
        self.img_size = self.opt.image_size
        self.focal_length = (self.img_size/2) / np.tan(13.373/2*math.pi/180)

    def _init_ldmk_detector(self):
        self.config_ldmk = EasyDict(
            {
                "ckpt_path": "./model/model_106/ldmk_detector/exp_mobile_v17-ceph-long/ckpt_best_model.pth.tar",
                "model": {
                    "conv_channels": [1, 10, 16, 24, 32, 64, 64, 96, 128, 128],
                    "ip_channels": [128, 128],
                    "init_type": 'init'
                },
                "num_landmarks": 106,
                "crop_size": 112
            }
        )
        self.ldmk_detector = MobileNet(
            config=self.config_ldmk).to(self.device)
        checkpoint = torch.load(
            self.config_ldmk.ckpt_path, map_location=self.device)
        self.ldmk_detector.load_state_dict(
            checkpoint['state_dict'], strict=False)
        self.ldmk_detector.eval()

    def _set_generating_param(self):
        self.truncation_psi = 0.7
        self.truncation_cutoff = 14

        self.face_angles = [[0,0]]

        self.dim_id = 80
        self.dim_exp = 64
        self.dim_bg_geo = 80
        self.dim_tex = 80
        self.dim_gamma = 27
        self.dim_bg_tex = 80
        self.length = 80+64+80+80+27+80
        # self.z_manager = Z_Manager(self.length, self.device)
        
        if self.opt.range is not None:
            self.seeds = list(range(*self.opt.range))
        else:
            self.seeds = list(self.opt.seeds)

    def _init_error_list(self):
        # self.icp_errs = {}
        self.inp_errs = {}
        self.ldmk_errs = {}
        self.msra_lm_errs = {}
        self.correlations = {}
        self.msra_lm_correlations = {}

    def _init_seed(self):
        print(f"-----------\nseed: {self.seed}")
        # self.icp_errs[self.seed] = []
        self.inp_errs[self.seed] = []
        self.ldmk_errs[self.seed] = []
        self.msra_lm_errs[self.seed] = []
        self.correlations[self.seed] = []
        self.msra_lm_correlations[self.seed] = []

        torch.manual_seed(self.seed)
        self.img_path = os.path.join(
            self.opt.output_dir,
            f'imgs/seed{self.seed}',
            f'grid_seed{self.seed}.png')
        self.img_root = os.path.dirname(self.img_path)
        ensure_dir(self.img_path)

        self._init_list()

    def _init_list(self):
        self.img_vis_list = []
        self.depth_vis_list = []
        self.d_input_vis_list = []
        self.f_input_vis_list = []
        self.m_input_vis_list = []
        self.d_pigan_input_list = []
        self.m_pigan_input_list = []
        self.f_pigan_input_list = []
        self.l_pigan_input_list = []
        self.d_pigan_input_close_mouth_list = []
        self.m_pigan_input_close_mouth_list = []
        self.pigan_img_list = []
        self.pigan_depth_list = []

        self.img_yaws_list = []
        self.depth_yaws_list = []
        self.tensor_img_yaws_list = []
        self.norm_tensor_img_yaws_list = []

        self.input_depth_yaws_list = []
        self.input_face_yaws_list = []
        self.input_mask_yaws_list = []

        self.gen_ldmks_list = []

        self.d_recon_list = []
        self.m_recon_list = []
        self.d_input_list = []
        self.m_input_list = []
        self.pred_ldmk_list = []
        self.input_ldmk_list = []
        self.input_ldmk_pigan_list = []
        self.input_3d_ldmks_list = []
        
        self.pred_msra_lm_list = []
        self.input_msra_lm_list = []

        self.input_shape2pigan_list = []

        self.gen_shape_pigan_list = []
        # self.input_shape2pigan_icp_list = []

    # run data
    def generate_img(self, z, angle_y=0, angle_p=0):
        with torch.no_grad():
            cam_pivot = torch.tensor([0, 0, 0.2], device=self.device)
            cam_radius = self.G.rendering_kwargs.get('avg_camera_radius', 2.7)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=self.device)
            conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=self.device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)
            self.conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1)

            ws = self.G.mapping(z, self.conditioning_params, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff)
            outputs = self.G.synthesis(ws, camera_params, face_recon=self.face_recon)
            img, depth = outputs['image'], outputs['image_depth']

        return img, depth

    def sample_generator(self, z, max_batch=1000000, shape_res=256):

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=self.G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        samples = samples.to(z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        # save the sampled points.
        save_obj_vertex(
            os.path.join(
                self.opt.output_dir,
                f'sample_{self.seed:04d}.obj'),
                samples.reshape((shape_res, shape_res, shape_res, 3))[::32, ::32, ::32])

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, self.conditioning_params, truncation_psi=self.truncation_psi, truncation_cutoff=self.truncation_cutoff, noise_mode='const')['sigma']
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

        return sigmas, samples


    def get_mesh(self, max_batch=500000):
        if self.debug:
            print('generator sampling...')

        voxel_grid, samples = self.sample_generator(
            self.z, max_batch=max_batch, shape_res=self.opt.shape_res)

        self.m_pigan_input_close_mouth, self.d_pigan_input_close_mouth = \
            close_mouth(self.m_pigan_input, self.d_pigan_input)
        if self.opt.snm:
            d_mask, m_mask = get_depth_mask(
                samples, self.m_pigan_input_close_mouth,
                self.d_pigan_input_close_mouth,
                thick=0.005, fov=13.373, cam_distance=2.7,
                )
            v_mask = torch.logical_and(d_mask, m_mask)
            empty_mask = torch.logical_and(
                m_mask, torch.logical_not(v_mask))
            voxel_grid[empty_mask.cpu().numpy().reshape(voxel_grid.shape)] = \
                voxel_grid.min()

        voxel_grid = np.maximum(voxel_grid, 0)
        vertices, triangles = mcubes.marching_cubes(
            voxel_grid, self.opt.sigma_threshold)  # 0.59s

        save_obj_vertex(
            os.path.join(
                self.opt.output_dir, f'objs/seed{self.seed}',
                f'seed{self.seed}_volume_full.obj'),
            vertices/(self.opt.shape_res-1) - 0.5)  # 2.49s

        import kornia
        mask = kornia.morphology.erosion(
            self.m_pigan_input_close_mouth*1,
            torch.ones(9, 9).to(self.device)) > 0.5
        mask = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

        if mask is not None:
            def vert_erosion(vertices, mask):
                fov = 13.373
                cam_distance = 2.7
                mask_size = mask.shape[1]
                f = (mask_size/2)/np.tan((fov/2)/180*np.pi)
                ratio = (cam_distance - vertices[:, 2])/f
                mask_sampler = -vertices[:, :2] / np.expand_dims(ratio, 1)
                mask_sampler += mask_size/2
                mask_sampler = mask_sampler.round().astype(np.long)
                mask_sampler[np.logical_or(
                    mask_sampler < 0, mask_sampler >= mask_size
                )] = 0
                m = mask[mask_sampler[:, 1], mask_sampler[:, 0]]
                # mask_indices = np.where(np.logical_and(
                #     m > 0.5,
                #     np.logical_and(
                #         vertices[:, 2] > -0.03,
                #         vertices[:, 2] < 0.12,
                #     )
                # ))[0]
                mask_indices = m > 0.5
                return vertices[mask_indices]

            vertices = vertices/(self.opt.shape_res-1) - 0.5
            vertices = vert_erosion(vertices, mask)
            inp = self.input_shape2pigan.detach().cpu().numpy()
            inp = vert_erosion(inp, mask)

        save_obj_vertex(
            os.path.join(
                self.opt.output_dir, f'objs/seed{self.seed}',
                f'seed{self.seed}_volume.obj'),
            vertices)  # 2.49s
        self.gen_shape_pigan = vertices

        save_obj_vertex(
            os.path.join(
                self.opt.output_dir, f'objs/seed{self.seed}',
                f'seed{self.seed}_masked_input.obj'),
            inp)  # 2.49s
        self.masked_input_mesh = inp

    def get_multiview_imgs(self):
        img_yaws = []
        depth_yaws = []
        tensor_img_yaws = []
        norm_tensor_img_yaws = []

        input_depth_yaws = []
        input_face_yaws = []
        input_mask_yaws = []
        for angle_num, (yaw, pitch) in enumerate(self.face_angles):
        
            img, depth = self.generate_img(
                self.z, yaw, pitch)

            tensor_img_yaws.append(img.detach())
            norm_tensor_img_yaws.append(min_max_norm(img.detach()))

            img_vis = min_max_norm(img)*255
            img_vis = img_vis.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            img_yaws.append(img_vis)

            depth_vis = min_max_norm(depth, 2.25, 3.3)*255
            depth_vis = depth_vis.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            depth_yaws.append(depth_vis)

            d_input = self.face_recon.d_pigan_input[0]
            m_input = self.face_recon.m_pigan_input[0]
            f_input = self.face_recon.f_pigan_input[0]
            d_input[torch.logical_not(m_input)] = d_input[m_input].mean()
            d_input_vis = min_max_norm(d_input, 2.25, 3.3)*255
            d_input_vis = d_input_vis.permute(1, 2, 0).cpu().numpy()
            d_input_vis = cv2.cvtColor(d_input_vis, cv2.COLOR_GRAY2BGR)
            input_depth_yaws.append(d_input_vis)

            f_input[torch.logical_not(m_input).repeat(3, 1, 1)] = f_input[
                m_input.repeat(3, 1, 1)].mean()
            f_input_vis = min_max_norm(f_input)*255
            f_input_vis = f_input_vis.permute(1, 2, 0).cpu().numpy()
            f_input_vis = cv2.cvtColor(f_input_vis, cv2.COLOR_RGB2BGR)
            input_face_yaws.append(f_input_vis)

            m_input_vis = m_input.squeeze(0).cpu().numpy()*255.
            input_mask_yaws.append(m_input_vis)

            if angle_num == len(self.face_angles)//2:
                self.img_vis = img_vis
                self.depth_vis = depth_vis
                self.d_input_vis = d_input_vis
                self.f_input_vis = f_input_vis
                self.m_input_vis = m_input_vis
                self.d_pigan_input = self.face_recon.d_pigan_input
                self.m_pigan_input = self.face_recon.m_pigan_input
                self.f_pigan_input = self.face_recon.f_pigan_input
                self.l_pigan_input = self.face_recon.pigan_landmark.clone()
                self.pigan_depth = depth.detach()
                self.pigan_img = img.detach()

        self.img_yaws = np.concatenate(img_yaws, axis=-2)
        self.depth_yaws = np.concatenate(depth_yaws, axis=-2)
        self.tensor_img_yaws = torch.cat(tensor_img_yaws, dim=-1)
        self.norm_tensor_img_yaws = torch.cat(norm_tensor_img_yaws, dim=-1)

        self.input_depth_yaws = np.concatenate(input_depth_yaws, axis=-2)
        self.input_face_yaws = np.concatenate(input_face_yaws, axis=-2)
        self.input_mask_yaws = np.concatenate(input_mask_yaws, axis=-2)


        cv2.imwrite(os.path.join(self.img_root, f'img_z{self.z_num}.png'),
                    self.img_yaws)
        cv2.imwrite(os.path.join(self.img_root, f'depth_z{self.z_num}.png'),
                    self.depth_yaws)
        cv2.imwrite(os.path.join(self.img_root,
                                 f'input_depth_z{self.z_num}.png'),
                    self.input_depth_yaws)
        cv2.imwrite(os.path.join(self.img_root,
                                 f'input_face_z{self.z_num}.png'),
                    self.input_face_yaws)
        cv2.imwrite(os.path.join(self.img_root,
                                 f'input_mask_z{self.z_num}.png'),
                    self.input_mask_yaws)

    def get_input_shape(self):
        self.d_recon, self.m_recon = img2depth(
            self.face_recon, self.tensor_img_yaws, self.trans_params,
            self.recon_t, self.recon_s)
        self.pred_3d_ldmks = self.face_recon.get_3d_ldmks(
            self.face_recon.pred_coeffs_dict)
        self.d_input, self.m_input = z2depth(self.face_recon, self.z)
        self.input_3d_ldmks = self.face_recon.get_3d_ldmks(
            self.face_recon.input_coeffs_dict)

        self.pred_ldmk = self.face_recon.inv_affine_ldmks_torch(
            self.face_recon.pred_lm)/self.recon_basic_size[0]*self.img_size
        self.pred_ldmk[..., 1] = self.img_size - 1 - self.pred_ldmk[..., 1]
        self.pred_ldmk = torch.clip(
            self.pred_ldmk, 0, self.img_size - 1).squeeze(0).round().long()

        self.input_ldmk = self.face_recon.inv_affine_ldmks_torch(
            self.face_recon.input_lm)/self.recon_basic_size[0]*self.img_size
        self.input_ldmk[..., 1] = self.img_size - 1 - self.input_ldmk[..., 1]
        self.input_ldmk = torch.clip(
            self.input_ldmk, 0, self.img_size - 1).squeeze(0).round().long()

        self.input_ldmk_pigan = self.face_recon.pigan_landmark.clone()/self.recon_basic_size[0]*self.img_size
        self.input_ldmk_pigan[..., 1] = self.img_size - 1 - self.input_ldmk_pigan[..., 1]
        self.input_ldmk_pigan = torch.clip(
            self.input_ldmk_pigan, 0, self.img_size - 1).squeeze(0).round().long()

        self.vis_face_recon_results()

        save_img_with_landmarks(
            cv2.cvtColor(self.img_yaws, cv2.COLOR_BGR2RGB),
            os.path.join(self.img_root, f'pred_ldmks_z{self.z_num}.png'),
            self.pred_ldmk)
        save_img_with_landmarks(
            cv2.cvtColor(self.img_yaws, cv2.COLOR_BGR2RGB),
            os.path.join(self.img_root, f'input_ldmks_z{self.z_num}.png'),
            self.input_ldmk)
        save_img_with_landmarks(
            cv2.cvtColor(self.img_yaws, cv2.COLOR_BGR2RGB),
            os.path.join(self.img_root, f'input_ldmks_pigan_z{self.z_num}.png'),
            self.input_ldmk_pigan)
        # print('input_ldmks:', self.input_ldmk.flatten())
        # print('pred_ldmks:', self.pred_ldmk.flatten())
        # print('input_ldmks_pigan:', self.input_ldmk_pigan.flatten())

    def vis_face_recon_results(self):
        self.face_recon.compute_ori_visuals('input')
        self.face_recon.compute_ori_visuals('pred')

        visuals = self.face_recon.get_current_visuals()  # get image results
        save_folder = "eval_pigan"
        self.visualizer.display_current_results(
            visuals, 0, 20, dataset=save_folder,
            save_results=True, count=self.z_num, add_image=False)
        msra_obj_save_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'input_msra_seed{self.seed}_z{self.z_num}.obj')
        # os.makedirs(msra_obj_save_path, exist_ok=True)
        ensure_dir(msra_obj_save_path)
        # print(f"msra_obj_save_path: {msra_obj_save_path}")
        save_obj_vertex(msra_obj_save_path, self.face_recon.input_shape)

        pigan_obj_save_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'input_pigan_seed{self.seed}_z{self.z_num}.obj')
        input_shape2pigan = self.face_recon.msra2pigan(
            self.face_recon.input_shape)
        save_obj_vertex(pigan_obj_save_path, input_shape2pigan)
        self.input_shape2pigan = input_shape2pigan.squeeze(0)

    def get_gen_ldmks(self):
        ldmk_inputs = torch.nn.functional.interpolate(
            self.norm_tensor_img_yaws,
            [self.config_ldmk.crop_size, self.config_ldmk.crop_size],
            mode='bicubic', align_corners=False)
        ldmk_inputs = cvt_rgb_to_gray(ldmk_inputs)
        ldmk_inputs = normalize_img(ldmk_inputs)
        self.gen_ldmks = self.ldmk_detector(
            ldmk_inputs.to(self.device)
        )*self.G.img_resolution/self.config_ldmk.crop_size
        self.gen_ldmks = self.gen_ldmks.reshape(-1, 2)[corres_106to68]
        self.gen_ldmks = torch.clip(
            self.gen_ldmks.reshape(-1, 2).round().long(),
            0, self.G.img_resolution-1
        )
        save_img_with_landmarks(
            cv2.cvtColor(self.img_yaws, cv2.COLOR_BGR2RGB),
            os.path.join(self.img_root, f'gen_ldmks_z{self.z_num}.png'),
            self.gen_ldmks)

    def append_z_results(self):
        self.img_vis_list.append(copy.deepcopy(self.img_vis))
        self.depth_vis_list.append(copy.deepcopy(self.depth_vis))
        self.d_input_vis_list.append(copy.deepcopy(self.d_input_vis))
        self.f_input_vis_list.append(copy.deepcopy(self.f_input_vis))
        self.m_input_vis_list.append(copy.deepcopy(self.m_input_vis))
        self.d_pigan_input_list.append(copy.deepcopy(self.d_pigan_input))
        self.m_pigan_input_list.append(copy.deepcopy(self.m_pigan_input))
        self.f_pigan_input_list.append(copy.deepcopy(self.f_pigan_input))
        self.l_pigan_input_list.append(copy.deepcopy(self.l_pigan_input))
        if self.opt.get_shape_distance:
            self.d_pigan_input_close_mouth_list.append(
                copy.deepcopy(self.d_pigan_input_close_mouth))
            self.m_pigan_input_close_mouth_list.append(
                copy.deepcopy(self.m_pigan_input_close_mouth))
        self.pigan_img_list.append(copy.deepcopy(self.pigan_img))
        self.pigan_depth_list.append(copy.deepcopy(self.pigan_depth))

        self.img_yaws_list.append(copy.deepcopy(self.img_yaws))
        self.depth_yaws_list.append(copy.deepcopy(self.depth_yaws))
        self.tensor_img_yaws_list.append(copy.deepcopy(self.tensor_img_yaws))
        self.norm_tensor_img_yaws_list.append(
            copy.deepcopy(self.norm_tensor_img_yaws))

        self.input_depth_yaws_list.append(copy.deepcopy(self.input_depth_yaws))
        self.input_face_yaws_list.append(copy.deepcopy(self.input_face_yaws))
        self.input_mask_yaws_list.append(copy.deepcopy(self.input_mask_yaws))

        self.gen_ldmks_list.append(copy.deepcopy(self.gen_ldmks.detach()))

        self.d_recon_list.append(copy.deepcopy(self.d_recon.detach()))
        self.m_recon_list.append(copy.deepcopy(self.m_recon.detach()))
        self.d_input_list.append(copy.deepcopy(self.d_input.detach()))
        self.m_input_list.append(copy.deepcopy(self.m_input.detach()))
        self.pred_ldmk_list.append(copy.deepcopy(self.pred_ldmk.detach()))
        self.input_ldmk_list.append(copy.deepcopy(self.input_ldmk.detach()))
        self.input_ldmk_pigan_list.append(copy.deepcopy(self.input_ldmk_pigan.detach()))
        self.input_3d_ldmks_list.append(copy.deepcopy(self.input_3d_ldmks.detach()))

        self.input_msra_lm_list.append(copy.deepcopy(self.face_recon.input_lm.detach()))
        self.pred_msra_lm_list.append(copy.deepcopy(self.face_recon.pred_lm.detach()))
        
        self.input_shape2pigan_list.append(
            copy.deepcopy(self.input_shape2pigan))

        if self.opt.get_shape_distance:
            self.gen_shape_pigan_list.append(copy.deepcopy(self.gen_shape_pigan))
        # self.input_shape2pigan_icp_list.append(
        #     copy.deepcopy(self.input_shape2pigan_icp))

    def run_z(self):
        # generate images:
        # self.img_yaws, self.depth_yaws
        # self.tensor_img_yaws, self.norm_tensor_img_yaws
        # self.input_depth_yaws, self.input_face_yaws
        self.get_multiview_imgs()

        # get ldmks from imgs
        self.get_gen_ldmks()

        #  render input code z to images
        self.get_input_shape()

        # get mesh from implicit volume
        if self.opt.get_shape_distance:
            self.get_mesh(max_batch=200000)
        # self.icp()

        self.append_z_results()

    def run_seed(self, seed,
                 using_cross_test=False,
                 using_correlation_test=False):
        self.seed = seed
        self._init_seed()

        z_ori = torch.randn((1, self.length), device=self.device)
        self.zs = [z_ori]
        if using_cross_test or using_correlation_test:
            self.zs.append(torch.randn((1, self.length), device=self.device))

        for self.z_num, self.z in enumerate(self.zs):
            self.run_z()
            self.cal_error(name=f"z_num{self.z_num}")
            # self.icp_errs[seed].append(self.icp_mesh_error.item())
            if self.opt.get_shape_distance:
                self.inp_errs[seed].append(self.input_mesh_error.item())
            self.ldmk_errs[seed].append(self.ldmk_error.item())
            self.msra_lm_errs[seed].append(self.msra_lm_error.item())

        if using_cross_test:
            self.index_pairs = [[0, 1],
                                [1, 0]]
            for idx_p, idx_n in self.index_pairs:
                self.cal_cross_error(idx_p=idx_p, idx_n=idx_n)
                # self.icp_errs[seed].append(self.icp_mesh_error.item())
                self.inp_errs[seed].append(self.input_mesh_error.item())
                self.ldmk_errs[seed].append(self.ldmk_error.item())
                self.msra_lm_errs[seed].append(self.msra_lm_error.item())

        if using_correlation_test:
            self.cal_correlation_error()
            self.cal_msra_correlation_error()

        catimg_lines = np.concatenate(self.img_yaws_list, axis=0)
        input_depth_yaws_list = [
            cv2.resize(
                dep, [self.G.img_resolution, self.G.img_resolution])
            for dep in self.input_depth_yaws_list
        ]
        catinpdep_lines = np.concatenate(input_depth_yaws_list, axis=0)


        # catdepth_lines = np.concatenate(depth_lines, axis=0)
        output = np.concatenate([catinpdep_lines, catimg_lines], axis=1)
        cv2.imwrite(self.img_path, output)

    def loop_seeds(self):
        # for seed in tqdm(self.seeds):
        for seed in self.seeds:
            self.run_seed(
                seed,
                self.opt.using_cross_test,
                self.opt.using_correlation_test)

        self.print_errors()
        print("")

    # calculate errors
    def cal_error(self, name='error'):
        #   seed {self.seed} z_num {z_num}:
        if self.opt.get_shape_distance:
            gen_shape_pigan = torch.from_numpy(
                self.gen_shape_pigan).float().to(self.device).unsqueeze(0)
            # inp_icp = self.input_shape2pigan_icp.Xt
            # inp = self.input_shape2pigan.unsqueeze(0)
            inp = torch.from_numpy(np.expand_dims(self.masked_input_mesh, 0)).to(self.device)

        # self.icp_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp_icp)
        # self.icp_mesh_error_ig = self.cal_mesh_error(inp_icp, gen_shape_pigan)
        # self.icp_mesh_error = (self.icp_mesh_error_gi + self.icp_mesh_error_ig) / 2

        if self.opt.get_shape_distance:
            self.input_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp)
            self.input_mesh_error_ig = self.cal_mesh_error(inp, gen_shape_pigan)
            self.input_mesh_error = (self.input_mesh_error_gi + self.input_mesh_error_ig) / 2

        self.ldmk_error = self.cal_ldmk_error(
            self.pred_ldmk.squeeze(0), self.input_ldmk_pigan.squeeze(0))

        # from IPython import embed; embed()
        self.msra_lm_error = self.cal_ldmk_error(
            self.face_recon.input_lm.squeeze(0), self.face_recon.pred_lm.squeeze(0))
        print(
            f'exp - {name}:\n'
            f'\t -> input mesh error: \t{self.input_mesh_error_gi.item()} \t{self.input_mesh_error_ig.item()} \t{self.input_mesh_error.item()},\n' if self.opt.get_shape_distance else ""
            # f'\t -> icp mesh error: \t{self.icp_mesh_error_gi.item()} \t{self.icp_mesh_error_ig.item()} \t{self.icp_mesh_error.item()},'
            f'\t -> ldmk error: \t{self.ldmk_error.item()} .\n'
            f'\t -> msra ldmk error: \t{self.msra_lm_error.item()} .'
            )

    def cal_cross_error(self, name='cross error', idx_p=0, idx_n=0):
        #   seed {self.seed} z_num {z_num}:
        gen_shape_pigan = torch.from_numpy(
            self.gen_shape_pigan_list[idx_p]
        ).float().to(self.device).unsqueeze(0)
        # inp_icp = self.input_shape2pigan_icp_list[idx_n].Xt
        inp = self.input_shape2pigan_list[idx_n].unsqueeze(0)

        # self.icp_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp_icp)
        # self.icp_mesh_error_ig = self.cal_mesh_error(inp_icp, gen_shape_pigan)
        # self.icp_mesh_error = (self.icp_mesh_error_gi + self.icp_mesh_error_ig) / 2

        self.input_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp)
        self.input_mesh_error_ig = self.cal_mesh_error(inp, gen_shape_pigan)
        self.input_mesh_error = (self.input_mesh_error_gi + self.input_mesh_error_ig) / 2

        self.ldmk_error = self.cal_ldmk_error(
            self.pred_ldmk_list[idx_p].squeeze(0), self.input_ldmk_pigan_list[idx_n].squeeze(0))

        print(
            f'exp - {name}:'
            # f'\t -> icp mesh error: \t{self.icp_mesh_error.item()} ,'
            f'\t -> input mesh error: \t{self.input_mesh_error.item()} ,'
            f'\t -> ldmk error: \t{self.ldmk_error.item()} .'
            )

    def cal_mean_error(self, errs):
        return np.array([v for k, v in errs.items()]).mean(axis=0)

    def cal_mesh_error(self, X, Y):
        from pytorch3d.ops import knn_points
        from pytorch3d.ops import utils as oputil

        # make sure we convert input Pointclouds structures to
        # padded tensors of shape (N, P, 3)
        Xt, num_points_X = oputil.convert_pointclouds_to_tensor(X)
        Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

        b, size_X, dim = Xt.shape

        if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
            raise ValueError(
                "Point sets X and Y have to have the same "
                + "number of batches and data dimensions."
            )

        if ((num_points_Y < Yt.shape[1]).any() or (num_points_X < Xt.shape[1]).any()) and (
            num_points_Y != num_points_X
        ).any():
            # we have a heterogeneous input (e.g. because X/Y is
            # an instance of Pointclouds)
            mask_X = (
                torch.arange(size_X, dtype=torch.int64, device=Xt.device)[None]
                < num_points_X[:, None]
            ).type_as(Xt)
        else:
            mask_X = Xt.new_ones(b, size_X)

        # the main loop over ICP iterations
        Xt_nn_points = knn_points(
            Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True
        ).knn[:, :, 0, :]

        # compute the root mean squared error
        Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)
        rmse = oputil.wmean(Xt_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]
        return rmse

    def cal_ldmk_error(self, X, Y):
        diff = X - Y
        diff = diff.detach().cpu().numpy()
        norm = np.linalg.norm(diff, axis=1)
        error = np.mean(norm, axis=0)
        return error

    def depth_to_3d_ldmk(self, depth, ldmk2d):
        depth = depth.to(ldmk2d.device)
        ldmk_depth = depth[ldmk2d[:, 1], ldmk2d[:, 0]].unsqueeze(-1)
        ldmk2d_c = ldmk2d - self.img_size/2
        ldmk3d = ldmk2d_c / self.focal_length * ldmk_depth
        # ldmk3d = torch.stack([ldmk3d[..., 1], ldmk3d[..., 0]], dim=-1)
        ldmk3d[..., 1] = -ldmk3d[..., 1]
        ldmk3d = torch.cat([ldmk3d, 1-ldmk_depth], dim=-1)
        return ldmk3d

    def cal_correlation_error(self):
        # process pred ldmks
        # ldmks106_list = [torch.clip(l,0,255) for l in ldmks106_list]
        img = min_max_norm(self.pigan_depth_list[0])*255
        img = nn.functional.interpolate(
            img, size=(self.img_size, self.img_size))[0].squeeze()
        img[self.pred_ldmk_list[0][:, 1], self.pred_ldmk_list[0][:, 0]] = 255
        cv2.imwrite(os.path.join(self.img_root, 'gen_ldmk_z0.png'),
                    img.detach().cpu().numpy())

        img = min_max_norm(self.pigan_depth_list[1])*255
        img = nn.functional.interpolate(
            img, size=(self.img_size, self.img_size))[0].squeeze()
        img[self.pred_ldmk_list[1][:, 1], self.pred_ldmk_list[1][:, 0]] = 255
        cv2.imwrite(os.path.join(self.img_root, 'gen_ldmk_z1.png'),
                    img.detach().cpu().numpy())

        pigan_3d_ldmk_0 = self.depth_to_3d_ldmk(
            nn.functional.interpolate(
                self.pigan_depth_list[0],
                size=(self.img_size, self.img_size))[0].squeeze(),
            self.pred_ldmk_list[0])
        pigan_3d_ldmk_1 = self.depth_to_3d_ldmk(
            nn.functional.interpolate(
                self.pigan_depth_list[1],
                size=(self.img_size, self.img_size))[0].squeeze(),
            self.pred_ldmk_list[1])
        ldmk3d_save_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'ldmk3d_seed{self.seed}_z{self.z_num}.obj')
        save_obj_vertex(ldmk3d_save_path, pigan_3d_ldmk_0)

        pigan_diff = pigan_3d_ldmk_1 - pigan_3d_ldmk_0
        pigan_diff_numpy = pigan_diff[17:].detach().cpu().numpy().flatten()

        # process input ldmks
        # depth0, depth1 = get_intersection_depth(depth0, depth1)
        input_diff = self.input_3d_ldmks_list[1] - self.input_3d_ldmks_list[0]
        input_diff_numpy = input_diff[0][17:].detach().cpu().numpy().flatten()

        correlation = np.dot(
            pigan_diff_numpy, input_diff_numpy)/(
                np.linalg.norm(pigan_diff_numpy)
                * np.linalg.norm(input_diff_numpy))

        print(f'correlation: {correlation}')
        self.correlations[self.seed].append(correlation.item())
        print(f'mean correlation: {self.cal_mean_error(self.correlations)}')

    def cal_msra_correlation_error(self):

        
        pigan_diff = self.pred_msra_lm_list[1] - self.pred_msra_lm_list[0]
        pigan_diff_numpy = pigan_diff[0][17:].detach().cpu().numpy().flatten()

        # process input ldmks
        # depth0, depth1 = get_intersection_depth(depth0, depth1)
        input_diff = self.input_msra_lm_list[1] - self.input_msra_lm_list[0]
        input_diff_numpy = input_diff[0][17:].detach().cpu().numpy().flatten()

        # from IPython import embed; embed()
        correlation = np.dot(
            pigan_diff_numpy, input_diff_numpy)/(
                np.linalg.norm(pigan_diff_numpy)
                * np.linalg.norm(input_diff_numpy))

        print(f'msra lm correlation: {correlation}')
        self.msra_lm_correlations[self.seed].append(correlation.item())
        print(f'mean msra lm correlation: {self.cal_mean_error(self.msra_lm_correlations)}')

    def print_errors(self):

        # icp_errs = self.cal_mean_error(self.icp_errs)
        inp_errs = self.cal_mean_error(self.inp_errs)
        ldmk_errs = self.cal_mean_error(self.ldmk_errs)
        msra_lm_errs = self.cal_mean_error(self.msra_lm_errs)
        corr_errs = self.cal_mean_error(self.correlations)
        msra_corr_errs = self.cal_mean_error(self.msra_lm_correlations)
        # self_icp_errs = np.mean(icp_errs[:2])
        # cros_icp_errs = np.mean(icp_errs[2:])
        self_inp_errs = np.mean(inp_errs[:2])
        cros_inp_errs = np.mean(inp_errs[2:])
        self_ldmk_errs = np.mean(ldmk_errs[:2])
        cros_ldmk_errs = np.mean(ldmk_errs[2:])
        self_msra_lm_errs = np.mean(msra_lm_errs[:2])
        cros_msra_lm_errs = np.mean(msra_lm_errs[2:])

        print(
            '=============='
            'Final Results:\n'
            # f'icp errors: {self_icp_errs} {cros_icp_errs}\n'
            f'inp errors: {self_inp_errs} {cros_inp_errs}\n'
            f'ldmk errors: {self_ldmk_errs} {cros_ldmk_errs}\n'
            f'msra lm errors: {self_msra_lm_errs} {cros_msra_lm_errs}\n'
            f'corr errors: {corr_errs}\n'
            f'msra corr errors: {msra_corr_errs}\n'
        )


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--ray_step_multiplier', type=int, default=2)
    parser.add_argument('--save_depth', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--range', default=None, nargs=3, metavar=('start', 'end', 'step'),
                        type=int, help='specify a range')
    parser.add_argument('--split', type=bool, default=True)

    parser.add_argument('--shape_res', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='evaluation_outputs')
    parser.add_argument('--sigma_threshold',
                        type=float,
                        default=20, # snm:0.5, wo snm:20.0
                        help='threshold to consider a location is occupied')

    parser.add_argument('--using_cross_test', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--using_correlation_test', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--get_shape_distance', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--snm', type=str2bool, nargs='?', const=True, default=False)
    
    opt = parser.parse_args()

    return opt


def main():
    opt = parse()
    eval_pigan = EvalPigan(opt)
    eval_pigan.loop_seeds()


if __name__ == '__main__':
    main()
