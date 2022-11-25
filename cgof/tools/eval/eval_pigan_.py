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

import curriculums
from model.model_106.MobileNet_v29 import MobileNet
from model.model_106.dlib_sensetime_corres import dlib_corres as corres_106to68

from utils.utils import *
from losses.contrastive_id_loss import Z_Manager

import cv2
import mcubes
from utils.mesh_io import save_obj_vertex, load_obj
from utils.utils import img2depth, z2depth, save_img_with_landmarks

from pytorch3d.ops import points_alignment

from models import create_model
from util.visualizer import MyVisualizer
from deep3dfacerecon_opt import deep3dfacerecon_opt

from IPython import embed
from model.PerCostFormer.warpformer import WarpFormer
from easydict import EasyDict

from generators import generators
from siren import siren
from generators.volumetric_rendering import (
    get_depth_mask, close_mouth)


class EvalPigan:
    # initialization
    def __init__(self, opt, ):
        self.debug = False
        self.opt = opt
        self.opt.output_dir = os.path.join(
            self.opt.output_dir, self.opt.curriculum)
        print(f'self.opt.output_dir: {self.opt.output_dir}')
        os.makedirs(self.opt.output_dir, exist_ok=True)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.data = {}

        self._init_face_recon(torch.device(self.device))
        self._init_curriculum()
        self._init_generator()
        self._init_ldmk_detector()
        self._init_error_list()
        # self._init_warpformer()
        self._set_generating_param()

    def _init_face_recon(self, device=torch.device('cuda')):
        face_recon = create_model(deep3dfacerecon_opt)
        face_recon.setup(deep3dfacerecon_opt)
        face_recon.device = self.device
        face_recon.set_coeff_static()
        face_recon.parallelize()
        face_recon.eval()
        vis_opt = copy.deepcopy(deep3dfacerecon_opt)
        vis_opt.img_folder = self.opt.output_dir
        visualizer = MyVisualizer(vis_opt)

        self.recon_basic_size = [256, 256]
        self.recon_s = [0.8975788298782894]
        self.recon_t = [128.04371143331298, 89.95939537909564]
        trans_params = self.recon_basic_size + self.recon_s + self.recon_t

        self.face_recon = face_recon
        self.visualizer = visualizer
        self.trans_params = trans_params
        # return face_recon, visualizer, trans_params

    def _init_curriculum(self):
        self.curriculum = curriculums.get(self.opt.curriculum)
        self.curriculum['num_steps'] = (
            self.curriculum[0]['num_steps'] * self.opt.ray_step_multiplier)
        self.curriculum['img_size'] = self.opt.image_size
        self.curriculum['psi'] = 0.7
        self.curriculum['v_stddev'] = 0
        self.curriculum['h_stddev'] = 0
        self.curriculum['lock_view_dependence'] = self.opt.lock_view_dependence
        self.curriculum['last_back'] = True
        self.curriculum['nerf_noise'] = 0
        self.curriculum = {
            key: value
            for key, value in self.curriculum.items()
            if type(key) is str
        }

        self.fov = self.curriculum['fov']

    def _init_generator(self):
        self.metadata = curriculums.extract_metadata(self.curriculum, 0)
        SIREN = getattr(siren, self.metadata['model'])
        GENERATOR = getattr(generators, self.metadata['generator'])
        self.generator = GENERATOR(
            SIREN, self.metadata['latent_dim'], **self.metadata
        ).to(self.device)
        load_pretrained_model(
                self.generator,
                self.opt.path,
                device=self.device)

        self.generator.set_device(self.device)
        # self.generator_ddp = DDP(
        #     self.generator, device_ids=[self.rank],
        #     find_unused_parameters=True)
        # self.generator = self.generator_ddp.module

    def _init_ldmk_detector(self):
        self.config_ldmk = curriculums.config_ldmk
        self.ldmk_detector = MobileNet(
            config=self.config_ldmk).to(self.device)
        checkpoint = torch.load(
            self.config_ldmk.ckpt_path, map_location=self.device)
        self.ldmk_detector.load_state_dict(
            checkpoint['state_dict'], strict=False)
        self.ldmk_detector.eval()

    def _init_warpformer(self):
        vert_dim = 3

        encoder_depth = 4
        cost_latent_token_num = 64
        cost_latent_dim = 32
        cost_latent_input_dim = 32
        query_latent_dim = 32
        dropout = 0.1
        num_heads = 2

        cfg = {
            'encoder_depth': encoder_depth,
            'cost_latent_token_num': cost_latent_token_num,
            'cost_latent_dim': cost_latent_dim,
            'cost_latent_input_dim': cost_latent_input_dim,
            'query_latent_dim': query_latent_dim,
            'num_heads': num_heads,

            'vert_dim': vert_dim,
            'cost_latent_input_dim': cost_latent_input_dim,
            'cost_latent_dim': cost_latent_dim,

            'dropout': dropout
        }
        cfg = EasyDict(cfg)

        self.warpformer = WarpFormer(cfg)
        # self.warpformer.load_state_dict(torch.load("warpformer.pth"))
        self.warpformer = torch.load("warpformer.pth")
        self.warpformer.to(self.device)

    def _set_generating_param(self):
        self.face_angles = [0.]
        self.face_angles = [
            a + self.curriculum['h_mean'] for a in self.face_angles]

        self.dim_id = 80
        self.dim_exp = 64
        self.dim_bg_geo = 80
        self.dim_tex = 80
        self.dim_gamma = 27
        self.dim_bg_tex = 80
        self.length = 80+64+80+80+27+80
        self.z_manager = Z_Manager(self.length, self.device)
        self.img_size = self.opt.image_size
        self.focal_length = (self.img_size/2) / np.tan(self.fov/2*math.pi/180)

        if self.opt.range is not None:
            self.seeds = list(range(*self.opt.range))
        else:
            self.seeds = list(self.opt.seeds)

    def _init_error_list(self):
        self.icp_errs = {}
        self.inp_errs = {}
        self.ldmk_errs = {}
        self.correlations = {}

    def _init_seed(self):
        print(f"-----------\nseed: {self.seed}")
        self.icp_errs[self.seed] = []
        self.inp_errs[self.seed] = []
        self.ldmk_errs[self.seed] = []
        self.correlations[self.seed] = []

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

        self.input_shape2pigan_list = []

        self.gen_shape_pigan_list = []
        self.input_shape2pigan_icp_list = []

    # run data
    def generate_img(self, z, **kwargs):
        with torch.no_grad():
            img, depth = self.generator.staged_forward(
                z, face_recon=self.face_recon, zy_data=self.data, **kwargs)
        return img, depth

    def generate_warping_img(self, zs, **kwargs):
        with torch.no_grad():
            img1, depth_map = self.generator.staged_forward(
                zs[1].unsqueeze(0), zy_data=self.data, **kwargs)
            fine_points = self.data['fine_points']
            transformed_points = self.data['transformed_points']

            fine_points_msra = self.face_recon.pigan2msra(fine_points)
            transformed_points_msra = self.face_recon.pigan2msra(
                transformed_points)

            input_coeffs_dict = self.face_recon.get_input_coeff(zs)
            self.face_recon.facemodel.to(self.device)
            input_shapes = self.face_recon.facemodel.compute_shape(
                input_coeffs_dict['id'],
                input_coeffs_dict['exp']
            )
            fine_flow = self.warpformer(
                input_shapes[1].unsqueeze(0), input_shapes[0].unsqueeze(0), fine_points_msra)
            course_flow = self.warpformer(
                input_shapes[1].unsqueeze(0), input_shapes[0].unsqueeze(0), transformed_points_msra)

            fine_warp_msra = fine_points_msra + fine_flow
            coarse_warp_msra = transformed_points_msra + course_flow
            fine_warp = self.face_recon.msra2pigan(fine_warp_msra)
            coarse_warp = self.face_recon.msra2pigan(coarse_warp_msra)

            img1_warp, depth_map = self.generator.staged_forward(
                zs[1].unsqueeze(0), transformed_points_input=coarse_warp,
                fine_points_input=fine_points, **kwargs)
            img0, depth_map = self.generator.staged_forward(
                zs[0].unsqueeze(0), zy_data=self.data, **kwargs)
            
            img = torch.cat([img1, img1_warp, img0], dim=3)
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min)/(img_max-img_min)
            img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()

        return img1, img1_warp, img

    def create_samples(self, N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
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

    def sample_generator(self, z, max_batch=100000,
                         voxel_resolution=256, voxel_origin=[0, 0, 0],
                         cube_length=2.0, psi=0.5, mask=None,
                         fov=12, cam_distance=1):

        head = 0
        samples, voxel_origin, voxel_size = self.create_samples(voxel_resolution, voxel_origin, cube_length)
        samples = samples.to(z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)

        transformed_ray_directions_expanded = torch.zeros(
            (samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1
        if self.debug:
            print('getting truncated frequencies and phase shifts...')
        self.generator.generate_avg_frequencies()
        with torch.no_grad():
            raw_frequencies, raw_phase_shifts = self.generator.siren.mapping_network(z)
            truncated_frequencies = self.generator.avg_frequencies + psi * (raw_frequencies - self.generator.avg_frequencies)
            truncated_phase_shifts = self.generator.avg_phase_shifts + psi * (raw_phase_shifts - self.generator.avg_phase_shifts)
        if self.debug:
            print('getting coarse_output...')
        with torch.no_grad():
            while head < samples.shape[1]:
                if self.debug:
                    print(f'head/samples.shape[1]:{head}/{samples.shape[1]}')
                coarse_output = self.generator.siren.forward_with_frequencies_phase_shifts(samples[:, head:head+max_batch], truncated_frequencies, truncated_phase_shifts, ray_directions=transformed_ray_directions_expanded[:, :samples.shape[1]-head]).reshape(samples.shape[0], -1, 4)
                sigmas[:, head:head+max_batch] = coarse_output[:, :, -1:]
                head += max_batch

        # sigmas = sigmas[0] * m.unsqueeze(1)
        # sigmas = sigmas[0]
        sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
        # sigmas = m.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()*90

        # print("")
        return sigmas, samples

    def get_mesh(self, max_batch=500000):
        if self.debug:
            print('generator sampling...')

        voxel_grid, samples = self.sample_generator(
            self.z, max_batch=max_batch, cube_length=self.opt.cube_size,
            voxel_resolution=self.opt.voxel_resolution)
        self.m_pigan_input_close_mouth, self.d_pigan_input_close_mouth = \
            close_mouth(self.m_pigan_input, self.d_pigan_input)

        if self.opt.snm:
            d_mask, m_mask = get_depth_mask(
                samples, self.m_pigan_input_close_mouth,
                self.d_pigan_input_close_mouth)
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
            vertices/255*0.3 - 0.15)  # 2.49s

        import kornia
        mask = kornia.morphology.erosion(
            self.m_pigan_input_close_mouth*1,
            torch.ones(9, 9).to(self.device)) > 0.5
        mask = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

        if mask is not None:
            def vert_erosion(vertices, mask):
                fov = 12
                cam_distance = 1
                mask_size = mask.shape[1]
                f = (mask_size/2)/np.tan((fov/2)/180*np.pi)
                ratio = (cam_distance - vertices[:, 2])/f
                mask_sampler = -vertices[:, :2] / np.expand_dims(ratio, 1)
                mask_sampler += mask_size/2
                mask_sampler = mask_sampler.round().astype(np.long)
                # mask_sampler = np.clip(mask_sampler, 0, mask_size-1)
                mask_sampler[np.logical_or(
                    mask_sampler < 0, mask_sampler >= mask_size
                )] = 0
                m = mask[mask_sampler[:, 1], mask_sampler[:, 0]]
                mask_indices = np.where(np.logical_and(
                    m > 0.5,
                    np.logical_and(
                        vertices[:, 2] > -0.03,
                        vertices[:, 2] < 0.12,
                    )
                ))[0]
                return vertices[mask_indices]*2

            # if False:
            vertices = vertices/255*0.3 - 0.15
            vertices = vert_erosion(vertices, mask)
            # mark
            inp = self.input_shape2pigan.detach().cpu().numpy()/2
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

    def icp(self):
        inp = self.input_shape2pigan.unsqueeze(0)
        pigan = torch.from_numpy(
            self.gen_shape_pigan).float().to(self.device).unsqueeze(0)

        self.input_shape2pigan_icp = points_alignment.iterative_closest_point(
            inp, pigan, relative_rmse_thr=1e-3, verbose=self.debug)
        icp_result_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'icp_seed{self.seed}.obj')
        # os.makedirs(os.path.dirname(icp_result_path), exist_ok=True)
        ensure_dir(icp_result_path)
        save_obj_vertex(
            icp_result_path, self.input_shape2pigan_icp.Xt.cpu().numpy()[0])

    def get_multiview_imgs(self):
        img_yaws = []
        depth_yaws = []
        tensor_img_yaws = []
        norm_tensor_img_yaws = []

        input_depth_yaws = []
        input_face_yaws = []
        input_mask_yaws = []
        for yaw_num, yaw in enumerate(self.face_angles):
            self.curriculum['h_mean'] = yaw
            img, depth = self.generate_img(
                self.z, gt_depths=self.opt.save_depth,
                **self.curriculum)

            tensor_img_yaws.append(img.detach())
            norm_tensor_img_yaws.append(min_max_norm(img.detach()))


            img_vis = min_max_norm(img)*255
            img_vis = img_vis.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
            img_yaws.append(img_vis)

            depth_vis = min_max_norm(depth, 0.88, 1.12)*255
            depth_vis = depth_vis.squeeze(0).cpu().numpy()
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR)
            depth_yaws.append(depth_vis)

            d_input = self.face_recon.d_pigan_input[0]
            m_input = self.face_recon.m_pigan_input[0]
            f_input = self.face_recon.f_pigan_input[0]
            d_input[torch.logical_not(m_input)] = d_input[m_input].mean()
            d_input_vis = min_max_norm(d_input, 0.88, 1.12)*255
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

            if yaw_num == len(self.face_angles)//2:
                self.img_vis = img_vis
                self.depth_vis = depth_vis
                self.d_input_vis = d_input_vis
                self.f_input_vis = f_input_vis
                self.m_input_vis = m_input_vis
                self.d_pigan_input = self.face_recon.d_pigan_input
                self.m_pigan_input = self.face_recon.m_pigan_input
                self.f_pigan_input = self.face_recon.f_pigan_input
                self.l_pigan_input = self.face_recon.l_pigan_input
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
            self.face_recon.pred_lm)
        self.pred_ldmk[..., 1] = self.img_size - 1 - self.pred_ldmk[..., 1]
        self.pred_ldmk = torch.clip(
            self.pred_ldmk, 0, self.img_size - 1).squeeze(0).round().long()

        self.input_ldmk = self.face_recon.inv_affine_ldmks_torch(
            self.face_recon.input_lm)
        self.input_ldmk[..., 1] = self.img_size - 1 - self.input_ldmk[..., 1]
        self.input_ldmk = torch.clip(
            self.input_ldmk, 0, self.img_size - 1).squeeze(0).round().long()

        self.input_ldmk_pigan = self.face_recon.l_pigan_input
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
        )*self.curriculum['img_size']/self.config_ldmk.crop_size
        self.gen_ldmks = self.gen_ldmks.reshape(-1, 2)[corres_106to68]
        self.gen_ldmks = torch.clip(
            self.gen_ldmks.reshape(-1, 2).round().long(),
            0, self.curriculum['img_size']-1
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

        self.input_shape2pigan_list.append(
            copy.deepcopy(self.input_shape2pigan))

        self.gen_shape_pigan_list.append(copy.deepcopy(self.gen_shape_pigan))
        self.input_shape2pigan_icp_list.append(
            copy.deepcopy(self.input_shape2pigan_icp))

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
        self.get_mesh(max_batch=200000)
        self.icp()

        self.append_z_results()

    def run_seed(self, seed,
                 using_cross_test=False,
                 using_correlation_test=False,
                 using_warping=False):
        self.seed = seed
        self._init_seed()

        z_ori = torch.randn((1, self.length), device=self.device)
        self.zs = [z_ori]
        if using_cross_test or using_correlation_test:
            self.zs.append(torch.randn((1, self.length), device=self.device))

        for self.z_num, self.z in enumerate(self.zs):
            self.run_z()
            self.cal_error(name=f"z_num{self.z_num}")
            self.icp_errs[seed].append(self.icp_mesh_error.item())
            self.inp_errs[seed].append(self.input_mesh_error.item())
            self.ldmk_errs[seed].append(self.ldmk_error.item())

        if using_cross_test:
            self.index_pairs = [[0, 1],
                                [1, 0]]
            for idx_p, idx_n in self.index_pairs:
                self.cal_cross_error(idx_p=idx_p, idx_n=idx_n)
                self.icp_errs[seed].append(self.icp_mesh_error.item())
                self.inp_errs[seed].append(self.input_mesh_error.item())
                self.ldmk_errs[seed].append(self.ldmk_error.item())

        if using_correlation_test:
            self.cal_correlation_error()

        catimg_lines = np.concatenate(self.img_yaws_list, axis=0)
        catinpdep_lines = np.concatenate(self.input_depth_yaws_list, axis=0)

        # catdepth_lines = np.concatenate(depth_lines, axis=0)
        output = np.concatenate([catinpdep_lines, catimg_lines], axis=1)
        cv2.imwrite(self.img_path, output)

        # return input_vs, pigan_volume_vs, icp_results
        if using_warping:
            if len(zs) < 2:
                zs.append(torch.randn((1, self.length), device=self.device))
            z1, z2 = zs[0], zs[-1]

            z_pair = torch.cat([z1, z2], dim=0)
            img1, img1_warp, img = self.generate_warping_img(z_pair, **self.curriculum)

            warping_img_root = 'warping_imgs'
            img_path = os.path.join(warping_img_root, f'img_{seed}.png')
            ensure_dir(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img*255)
            print("")

    def loop_seeds(self):
        # for seed in tqdm(self.seeds):
        for seed in self.seeds:
            self.run_seed(
                seed,
                self.opt.using_cross_test,
                self.opt.using_correlation_test,
                self.opt.using_warping)

        self.print_errors()
        print("")

    # calculate errors
    def cal_error(self, name='error'):
        #   seed {self.seed} z_num {z_num}:
        gen_shape_pigan = torch.from_numpy(
            self.gen_shape_pigan).float().to(self.device).unsqueeze(0)
        inp_icp = self.input_shape2pigan_icp.Xt
        # inp = self.input_shape2pigan.unsqueeze(0)
        inp = torch.from_numpy(np.expand_dims(self.masked_input_mesh, 0)).to(self.device)

        self.icp_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp_icp)
        self.icp_mesh_error_ig = self.cal_mesh_error(inp_icp, gen_shape_pigan)
        self.icp_mesh_error = (self.icp_mesh_error_gi + self.icp_mesh_error_ig) / 2

        self.input_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp)
        self.input_mesh_error_ig = self.cal_mesh_error(inp, gen_shape_pigan)
        self.input_mesh_error = (self.input_mesh_error_gi + self.input_mesh_error_ig) / 2

        self.ldmk_error = self.cal_ldmk_error(
            self.pred_ldmk.squeeze(0), self.input_ldmk_pigan.squeeze(0))

        print(
            f'exp - {name}:'
            f'\t -> input mesh error: \t{self.input_mesh_error_gi.item()} \t{self.input_mesh_error_ig.item()} \t{self.input_mesh_error.item()},'
            f'\t -> icp mesh error: \t{self.icp_mesh_error_gi.item()} \t{self.icp_mesh_error_ig.item()} \t{self.icp_mesh_error.item()},'
            f'\t -> ldmk error: \t{self.ldmk_error.item()} .'
            )

    def cal_cross_error(self, name='cross error', idx_p=0, idx_n=0):
        #   seed {self.seed} z_num {z_num}:
        gen_shape_pigan = torch.from_numpy(
            self.gen_shape_pigan_list[idx_p]
        ).float().to(self.device).unsqueeze(0)
        inp_icp = self.input_shape2pigan_icp_list[idx_n].Xt
        inp = self.input_shape2pigan_list[idx_n].unsqueeze(0)

        self.icp_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp_icp)
        self.icp_mesh_error_ig = self.cal_mesh_error(inp_icp, gen_shape_pigan)
        self.icp_mesh_error = (self.icp_mesh_error_gi + self.icp_mesh_error_ig) / 2

        self.input_mesh_error_gi = self.cal_mesh_error(gen_shape_pigan, inp)
        self.input_mesh_error_ig = self.cal_mesh_error(inp, gen_shape_pigan)
        self.input_mesh_error = (self.input_mesh_error_gi + self.input_mesh_error_ig) / 2

        self.ldmk_error = self.cal_ldmk_error(
            self.pred_ldmk_list[idx_p].squeeze(0), self.input_ldmk_pigan_list[idx_n].squeeze(0))

        print(
            f'exp - {name}:'
            f'\t -> icp mesh error: \t{self.icp_mesh_error.item()} ,'
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

    def cal_correlation_error_(self):
        # ldmks106_list = [torch.clip(l,0,255) for l in ldmks106_list]
        img = min_max_norm(self.pigan_depth_list[0][0])*255
        img[self.gen_ldmks_list[0][:, 1], self.gen_ldmks_list[0][:, 0]] = 255
        cv2.imwrite(os.path.join(self.img_root, 'gen_ldmk_z0.png'),
                    img.cpu().numpy())

        pigan_depth_0 = self.pigan_depth_list[0][0].squeeze()[
            self.gen_ldmks_list[0][:, 1], self.gen_ldmks_list[0][:, 0]]
        pigan_depth_1 = self.pigan_depth_list[1][0].squeeze()[
            self.gen_ldmks_list[1][:, 1], self.gen_ldmks_list[1][:, 0]]
        pigan_depth_diff = pigan_depth_1 - pigan_depth_0
        pigan_depth_diff_numpy = pigan_depth_diff.detach().cpu().numpy()

        def cvt_depth_msra2pigan(d):
            depth = d.clone()
            mask = depth > 0
            # depth[mask] = depth[mask] - 1
            depth[torch.logical_not(mask)] = depth[mask].median()
            return depth

        def get_intersection_depth(depth0, depth1):
            mask = torch.logical_and(depth0 > 0, depth1 > 0)
            return depth0*mask, depth1*mask

        depth0 = self.d_pigan_input_close_mouth_list[0]
        depth1 = self.d_pigan_input_close_mouth_list[1]
        depth0 = nn.functional.interpolate(
            depth0, size=(256, 256), mode='nearest')[0][0]
        depth1 = nn.functional.interpolate(
            depth1, size=(256, 256), mode='nearest')[0][0]
        # depth0 = cvt_depth_msra2pigan(depth0)
        # depth1 = cvt_depth_msra2pigan(depth1)

        d_input = cvt_depth_msra2pigan(depth0)
        d_input = min_max_norm(d_input)*255
        d_input[self.input_ldmk_pigan_list[0][:, 1],
                self.input_ldmk_pigan_list[0][:, 0]] = 255
        cv2.imwrite(os.path.join(self.img_root, 'input_ldmk_pigan_z0.png'),
                    d_input.detach().cpu().numpy())

        # depth0, depth1 = get_intersection_depth(depth0, depth1)
        input_depth_0 = depth0.squeeze()[
            self.input_ldmk_pigan_list[0][:, 1], self.input_ldmk_pigan_list[0][:, 0]]
        input_depth_1 = depth1.squeeze()[
            self.input_ldmk_pigan_list[1][:, 1], self.input_ldmk_pigan_list[1][:, 0]]
        input_depth_0, input_depth_1 = get_intersection_depth(
            input_depth_0, input_depth_1)
        input_depth_diff = input_depth_1 - input_depth_0
        input_depth_diff_numpy = input_depth_diff.detach().cpu().numpy()

        idx = input_depth_diff_numpy != 0
        input_depth_diff_numpy = input_depth_diff_numpy[idx]
        pigan_depth_diff_numpy = pigan_depth_diff_numpy[idx]

        gen_ldmk_diff = self.gen_ldmks_list[1] - self.gen_ldmks_list[0]
        gen_ldmk_diff = gen_ldmk_diff.detach().cpu().numpy()[idx].flatten()
        input_ldmk_diff = self.input_ldmk_pigan_list[1] - self.input_ldmk_pigan_list[0]
        input_ldmk_diff = input_ldmk_diff.detach().cpu().numpy()[idx].flatten()

        # print(f'pigan_depth_diff_numpy: {pigan_depth_diff_numpy}')
        # print(f'input_depth_diff_numpy: {input_depth_diff_numpy}')
        # pigan_3dldmk_diff = np.concatenate([gen_ldmk_diff, pigan_depth_diff_numpy], axis=0)
        # input_3dldmk_diff = np.concatenate([input_ldmk_diff, input_depth_diff_numpy], axis=0)

        z_correlation = np.dot(
            pigan_depth_diff_numpy, input_depth_diff_numpy)/(
                np.linalg.norm(pigan_depth_diff_numpy)
                * np.linalg.norm(input_depth_diff_numpy))
        xy_correlation = np.dot(
            gen_ldmk_diff, input_ldmk_diff)/(
                np.linalg.norm(gen_ldmk_diff)
                * np.linalg.norm(input_ldmk_diff))
        correlation = z_correlation/3 + xy_correlation*2/3

        print(f'correlation: {correlation}')
        self.correlations[self.seed].append(correlation.item())
        print(f'mean correlation: {self.cal_mean_error(self.correlations)}')

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
        img = min_max_norm(self.pigan_depth_list[0][0])*255
        img[self.pred_ldmk_list[0][:, 1], self.pred_ldmk_list[0][:, 0]] = 255
        cv2.imwrite(os.path.join(self.img_root, 'gen_ldmk_z0.png'),
                    img.cpu().numpy())

        pigan_3d_ldmk_0 = self.depth_to_3d_ldmk(
            self.pigan_depth_list[0][0].squeeze(),
            self.pred_ldmk_list[0])
        pigan_3d_ldmk_1 = self.depth_to_3d_ldmk(
            self.pigan_depth_list[1][0].squeeze(),
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

    def print_errors(self):

        icp_errs = self.cal_mean_error(self.icp_errs)
        inp_errs = self.cal_mean_error(self.inp_errs)
        ldmk_errs = self.cal_mean_error(self.ldmk_errs)
        corr_errs = self.cal_mean_error(self.correlations)
        self_icp_errs = np.mean(icp_errs[:2])
        cros_icp_errs = np.mean(icp_errs[2:])
        self_inp_errs = np.mean(inp_errs[:2])
        cros_inp_errs = np.mean(inp_errs[2:])
        self_ldmk_errs = np.mean(ldmk_errs[:2])
        cros_ldmk_errs = np.mean(ldmk_errs[2:])

        print(
            '=============='
            'Final Results:\n'
            f'icp errors: {self_icp_errs} {cros_icp_errs}\n'
            f'inp errors: {self_inp_errs} {cros_inp_errs}\n'
            f'ldmk errors: {self_ldmk_errs} {cros_ldmk_errs}\n'
            f'corr errors: {corr_errs}\n'
        )


def parse():
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

    parser.add_argument('--using_cross_test', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--using_correlation_test', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--using_warping', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--snm', type=str2bool, nargs='?', const=True, default=False)
    opt = parser.parse_args()

    return opt


def main():
    opt = parse()
    eval_pigan = EvalPigan(opt)
    eval_pigan.loop_seeds()


if __name__ == '__main__':
    main()
