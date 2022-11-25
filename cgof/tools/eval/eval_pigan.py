# CUDA_VISIBLE_DEVICES=1 \
# python tools/eval/eval_pigan.py \
# outputs/pigan_recon4/generator.pth \
# --range 0 2 1 --curriculum pigan  --save_depth \
# --output_dir shapes \
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

from utils.utils import *
from losses.contrastive_id_loss import Z_Manager

import cv2
import mcubes
from utils.mesh_io import save_obj_vertex, load_obj
from utils.utils import img2depth, z2depth

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
    def __init__(self, opt, ):
        self.debug = False
        self.opt = opt
        os.makedirs(opt.output_dir, exist_ok=True)
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
        visualizer = MyVisualizer(deep3dfacerecon_opt)

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
        self.curriculum['last_back'] = self.curriculum.get(
            'eval_last_back', False)
        self.curriculum['nerf_noise'] = 0
        self.curriculum = {
            key: value
            for key, value in self.curriculum.items()
            if type(key) is str
        }

    def _init_generator(self):
        self.generator = torch.load(
            self.opt.path, map_location=torch.device(self.device))
        ema_file = self.opt.path.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file)
        ema.copy_to(
            [p for p in self.generator.parameters() if p.requires_grad])

        self.generator.set_device(torch.device(self.device))
        self.generator.eval()

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

        if self.opt.range is not None:
            self.seeds = list(range(*self.opt.range))
        else:
            self.seeds = list(self.opt.seeds)

    def _init_error_list(self):
        self.icp_errs = {}
        self.inp_errs = {}
        self.ldmk_errs = {}
        self.coorelations = {}

    def generate_img(self, gen, z, **kwargs):
        with torch.no_grad():
            img, depth_map, gt_depth, gt_ldmks, gt_wets = generator.staged_forward(z, zy_data=self.data, **kwargs)
            tensor_img = img.detach()

            img_min = img.min()
            img_max = img.max()
            img = (img - img_min)/(img_max-img_min)
            img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img, tensor_img, depth_map, gt_depth

    def generate_img_wo_gt_depth(self, z, **kwargs):
        with torch.no_grad():
            img, depth_map = self.generator.staged_forward(
                z, zy_data=self.data, **kwargs)
            tensor_img = img.detach()

            img_min = img.min()
            img_max = img.max()
            img = (img - img_min)/(img_max-img_min)
            img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img, tensor_img, depth_map

    def generate_img_snm(self, z, **kwargs):
        with torch.no_grad():
            img, depth_map = self.generator.staged_forward(
                z, face_recon=self.face_recon, zy_data=self.data, **kwargs)
            tensor_img = img.detach()

            img_min = img.min()
            img_max = img.max()
            img = (img - img_min)/(img_max-img_min)
            img = img.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return img, tensor_img, depth_map

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

    def get_mesh(self, z, max_batch=500000):
        if self.debug:
            print('generator sampling...')
        # mask = cv2.imread('face-parsing.PyTorch/res/test_res/img_0_mask.png', cv2.IMREAD_GRAYSCALE)
        # mask = mask/255
        mask = self.face_recon.mask
        mask = mask.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        mask = mask.mean(axis=-1)

        # embed()
        # mask_256 = cv2.resize(mask, (256, 256), cv2.INTER_CUBIC)
        # voxel_grid = voxel_grid.transpose(1, 2, 0)
        # voxel_grid *= mask_256
        # voxel_grid = voxel_grid.transpose(2, 0, 1)

        voxel_grid, samples = self.sample_generator(
            z, max_batch=max_batch, cube_length=self.opt.cube_size,
            voxel_resolution=self.opt.voxel_resolution)
        d_input, m_input, f_input, z_input_dict = self.face_recon.z_to_pigan_depth_image(
                z,
                torch.ones((z.shape[0], 1))*math.pi*0.5,
                torch.ones((z.shape[0], 1))*math.pi*0.5,
                224
                )
        m_input, self.d_input = close_mouth(m_input, d_input)

        if self.opt.snm:
            # from utils.mesh_io import save_obj_vertex_color
            # def vis_tensor_as_vert_color(color, position, path='z_diff.obj',):
            #     color = torch.abs(color)
            #     color = (color - color.min())/(color.max() - color.min())
            #     c = torch.cat(
            #         [position, color],
            #         dim=-1)
            #     save_obj_vertex_color(path, c)

            d_mask, m_mask = get_depth_mask(samples, m_input, d_input)
            v_mask = torch.logical_and(d_mask, m_mask)
            empty_mask = torch.logical_and(
                m_mask, torch.logical_not(v_mask))
            voxel_grid[empty_mask.cpu().numpy().reshape(voxel_grid.shape)] = voxel_grid.min()
            # vis_tensor_as_vert_color(
            #     v_mask[0][::32].repeat((1, 3))*1.0, samples[0][::32], "v_mask.obj")
            # vis_tensor_as_vert_color(
            #     empty_mask[0][::32].repeat((1, 3))*1.0, samples[0][::32], "empty_mask.obj")
            # vis_tensor_as_vert_color(
            #     m_mask[0][::32].repeat((1, 3))*1.0, samples[0][::32], "m_mask.obj")

        voxel_grid = np.maximum(voxel_grid, 0)

        vertices, triangles = mcubes.marching_cubes(
            voxel_grid, self.opt.sigma_threshold)  # 0.59s

        if mask is not None:
            vertices = vertices/255*0.3 - 0.15
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
            # mask = np.from_numpy(mask).to(device)
            m = mask[mask_sampler[:, 1], mask_sampler[:, 0]]
            # v = m.reshape((256,256,256)).cpu().numpy()
            # cv2.imwrite('vis_mask.png',v[...,0]*255)
            # print("")
            mask_indices = np.where(np.logical_and(
                m > 0.5,
                np.logical_and(
                    vertices[:, 2] > -0.03,
                    vertices[:, 2] < 0.12,
                )
            ))[0]
            # samples = samples[:, mask_indices]
            vertices = vertices[mask_indices]*2
            # vertices = (vertices+0.15)*255/0.3

        save_obj_vertex(
            os.path.join(
                self.opt.output_dir, f'objs/seed{self.seed}',
                f'seed{self.seed}_volume.obj'),
            vertices)  # 2.49s

        return vertices, triangles

    def get_imgs(self, z):
        images = []
        depths = []
        gt_depths = []
        imgs = []
        for yaw_num, yaw in enumerate(self.face_angles):
            self.curriculum['h_mean'] = yaw
            # img, tensor_img, depth_map = self.generate_img_wo_gt_depth(
            img, tensor_img, depth_map = self.generate_img_snm(
                z, gt_depths=self.opt.save_depth,
                **self.curriculum)
            images.append(tensor_img)
            depths.append(depth_map)
            imgs.append(img)

        # depths = torch.cat(depths, dim=-1)[0].numpy()
        # depths = (depths-depths.min())/(depths.max()-depths.min())
        # depths = cv2.cvtColor(depths, cv2.COLOR_GRAY2BGR)
        images = torch.cat(images, dim=-1)
        images_vis = images[0].permute(1, 2, 0)[..., [2, 1, 0]].numpy()
        images_vis = (images_vis-images_vis.min())/(images_vis.max()-images_vis.min())

        return images_vis, depths, tensor_img, imgs

    def get_input_shape(self, z, z_num, tensor_img):
        d_recon, m_recon = img2depth(
            self.face_recon, tensor_img, self.trans_params,
            self.recon_t, self.recon_s)
        d_input, m_input = z2depth(self.face_recon, z)
        self.pred_ldmk = self.face_recon.pred_lm
        self.input_ldmk = self.face_recon.input_lm

        input_shape2pigan = self.vis_face_recon_results(z_num).squeeze(0)
        # input_vertex_pigan = self.face_recon.pigan2cam(
        #     input_shape2pigan,
        #     yaw=np.pi/2*torch.ones((1, 1)).to(self.device),
        #     pitch=np.pi/2*torch.ones((1, 1)).to(self.device))
        # input_mask, input_depth, _ = self.face_recon.renderer_pigan(
        #     input_vertex_pigan, self.face_recon.facemodel.face_buf)
        return input_shape2pigan, d_input

    def vis_face_recon_results(self, z_num):
        self.face_recon.compute_ori_visuals('input')
        self.face_recon.compute_ori_visuals('pred')

        visuals = self.face_recon.get_current_visuals()  # get image results
        save_folder = "eval_pigan"
        self.visualizer.display_current_results(
            visuals, 0, 20, dataset=save_folder,
            save_results=True, count=z_num, add_image=False)
        msra_obj_save_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'input_msra_seed{self.seed}_z{z_num}.obj')
        # os.makedirs(msra_obj_save_path, exist_ok=True)
        ensure_dir(msra_obj_save_path)
        # print(f"msra_obj_save_path: {msra_obj_save_path}")
        save_obj_vertex(msra_obj_save_path, self.face_recon.input_shape)

        pigan_obj_save_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'input_pigan_seed{self.seed}_z{z_num}.obj')
        input_shape2pigan = self.face_recon.msra2pigan(
            self.face_recon.input_shape)
        save_obj_vertex(pigan_obj_save_path, input_shape2pigan)
        
        return input_shape2pigan

    def get_ldmks(self, imgs):

        ldmk_inputs = torch.from_numpy(np.array(imgs))
        ldmk_inputs = ldmk_inputs.permute(0, 3, 1, 2)
        ldmk_inputs = torch.nn.functional.interpolate(
            ldmk_inputs, [112, 112], mode='bicubic', align_corners=False)
        ldmk_inputs = cvt_rgb_to_gray(ldmk_inputs)
        ldmk_inputs = normalize_img(ldmk_inputs)
        ldmks = self.ldmk_detector(
            ldmk_inputs.to(self.device)
        )*self.curriculum['img_size']/self.config_ldmk.crop_size
        return ldmks

    def run_z(self, z, z_num):
        outputs = {}

        # generate images
        images_vis, depths, tensor_img, imgs = self.get_imgs(z)
        outputs['images_vis'] = images_vis
        outputs['depths'] = depths

        # get ldmks from imgs
        ldmks = self.get_ldmks(imgs)
        outputs['ldmks'] = ldmks

        #  render input code z to images
        input_vs, d_input = self.get_input_shape(z, z_num, tensor_img)
        outputs['input_vs'] = input_vs

        # get mesh from implicit volume
        vs, ts = self.get_mesh(z, max_batch=200000)
        outputs['vs'] = vs
        outputs['d_input'] = self.d_input

        return outputs

    def run_seed(self, seed,
                 using_cross_test=True,
                 using_correlation_test=True,
                 using_warping=False):
        self.seed = seed
        print(f"seed: {seed}")
        self.icp_errs[seed] = []
        self.inp_errs[seed] = []
        self.ldmk_errs[seed] = []
        self.coorelations[seed] = []

        torch.manual_seed(seed)
        # os.makedirs(self.opt.output_dir, exist_ok=True)
        img_path = os.path.join(
            self.opt.output_dir, f'imgs/seed{self.seed}', f'grid_seed{seed}.png')
        # os.makedirs(os.path.dirname(img_path), exist_ok=True)
        ensure_dir(img_path)

        z_ori = torch.randn((1, self.length), device=self.device)
        zs = [z_ori]

        if using_cross_test:
            zs.append(torch.randn((1, self.length), device=self.device))

        img_lines = []
        depth_lines = []

        input_vs_list = []
        pigan_volume_vs_list = []
        icp_results_list = []
        pred_ldmk_list = []
        input_ldmk_list = []
        ldmks106_list = []
        pigan_depth_list = []
        input_depth_list = []
        for z_num, z in enumerate(zs):
            # print(f'z: {z}')
            outputs = self.run_z(z, z_num)
            images_vis = outputs['images_vis']
            depths = outputs['depths']
            input_vs = outputs['input_vs']
            pigan_volume_vs = outputs['vs']
            ldmks106 = outputs['ldmks']
            d_input = outputs['d_input']
            # embed()
            icp_results = self.icp(
                input_vs.unsqueeze(0),
                torch.from_numpy(pigan_volume_vs).float().to(self.device).unsqueeze(0))

            img_lines.append(images_vis*255)
            depth_lines.append(depths*255)

            input_vs_list.append(input_vs)
            pigan_volume_vs_list.append(pigan_volume_vs)
            icp_results_list.append(icp_results)
            pred_ldmk_list.append(self.pred_ldmk)
            input_ldmk_list.append(self.input_ldmk)
            ldmks106_list.append(
                torch.clip(
                    ldmks106.reshape(-1, 2).round().long(),
                    0, self.curriculum['img_size']-1
                )
            )
            pigan_depth_list.append(depths)
            input_depth_list.append(d_input)

        for z_num in range(len(zs)):
            icp_err, inp_err, ldmk_err = self.cal_error(
                pigan_volume_vs_list[z_num],
                icp_results_list[z_num],
                input_vs_list[z_num],
                pred_ldmk_list[z_num],
                input_ldmk_list[z_num],
                name=f"z_num{z_num}"
            )
            self.icp_errs[seed].append(icp_err.item())
            self.inp_errs[seed].append(inp_err.item())
            self.ldmk_errs[seed].append(ldmk_err.item())

        if using_cross_test:
            self.index_pairs = [[0, 1],
                                [1, 0]]
            for idx_p, idx_n in self.index_pairs:
                icp_err, inp_err, ldmk_err = self.cal_error(
                    pigan_volume_vs_list[idx_p], icp_results_list[idx_n],
                    input_vs_list[idx_n], pred_ldmk_list[idx_p],
                    input_ldmk_list[idx_n], name=f"pigan{idx_p}_input{idx_n}")
                self.icp_errs[seed].append(icp_err.item())
                self.inp_errs[seed].append(inp_err.item())
                self.ldmk_errs[seed].append(ldmk_err.item())

        if using_correlation_test:

            # ldmks106_list = [torch.clip(l,0,255) for l in ldmks106_list]
            img = min_max_norm(pigan_depth_list[0][0][0])*255
            img[ldmks106_list[0][:, 1], ldmks106_list[0][:, 0]] = 255
            cv2.imwrite('d_pigan.png', img.cpu().numpy())

            pigan_depth_0 = pigan_depth_list[0][0].squeeze()[
                ldmks106_list[0][:, 1], ldmks106_list[0][:, 0]]
            pigan_depth_1 = pigan_depth_list[1][0].squeeze()[
                ldmks106_list[1][:, 1], ldmks106_list[1][:, 0]]
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

            # d_input = input_depth_list[0]
            # d_input = nn.functional.interpolate(
            #     d_input, size=(256, 256), mode='nearest')[0][0]
            # d_input = cvt_depth_msra2pigan(d_input)
            # d_input = min_max_norm(d_input)*255
            # d_input[ldmks106_list[0][:, 1], ldmks106_list[0][:, 0]] = 255
            # cv2.imwrite('d_input.png', d_input.detach().cpu().numpy())

            depth0 = input_depth_list[0]
            depth1 = input_depth_list[1]
            depth0 = nn.functional.interpolate(
                depth0, size=(256, 256), mode='nearest')[0][0]
            depth1 = nn.functional.interpolate(
                depth1, size=(256, 256), mode='nearest')[0][0]
            # depth0 = cvt_depth_msra2pigan(depth0)
            # depth1 = cvt_depth_msra2pigan(depth1)

            d_input = cvt_depth_msra2pigan(depth0)
            d_input = min_max_norm(d_input)*255
            d_input[ldmks106_list[0][:, 1], ldmks106_list[0][:, 0]] = 255
            cv2.imwrite('d_input.png', d_input.detach().cpu().numpy())

            # depth0, depth1 = get_intersection_depth(depth0, depth1)
            input_depth_0 = depth0.squeeze()[
                ldmks106_list[0][:, 1], ldmks106_list[0][:, 0]]
            input_depth_1 = depth1.squeeze()[
                ldmks106_list[1][:, 1], ldmks106_list[1][:, 0]]
            input_depth_0, input_depth_1 = get_intersection_depth(
                input_depth_0, input_depth_1)
            input_depth_diff = input_depth_1 - input_depth_0
            input_depth_diff_numpy = input_depth_diff.detach().cpu().numpy()

            idx = input_depth_diff_numpy != 0
            input_depth_diff_numpy = input_depth_diff_numpy[idx]
            pigan_depth_diff_numpy = pigan_depth_diff_numpy[idx]

            print(f'pigan_depth_diff_numpy: {pigan_depth_diff_numpy}')
            print(f'input_depth_diff_numpy: {input_depth_diff_numpy}')

            # coorelation = np.sum(np.diag(np.abs(np.corrcoef(
            #     np.stack([pigan_depth_diff_numpy, input_depth_diff_numpy], 0).T
            # ))))

            coorelation = np.dot(
                pigan_depth_diff_numpy, input_depth_diff_numpy)/(
                    np.linalg.norm(pigan_depth_diff_numpy)
                    * np.linalg.norm(input_depth_diff_numpy))

            # ldmk_diff = ldmks106_list[1] - ldmks106_list[0]

            print(f'coorelation: {coorelation}')  # -0.02433144 0.040261106184997106
            self.coorelations[seed].append(coorelation.item())
            print(f'mean coorelation: {self.cal_mean_error(self.coorelations)}')

        # from IPython import embed; embed()

        # catimg_lines = np.concatenate(img_lines, axis=0)
        # catdepth_lines = np.concatenate(depth_lines, axis=0)
        # output = np.concatenate([catdepth_lines, catimg_lines], axis=1)
        # cv2.imwrite(img_path, output)

        # if self.opt.split:
        #     split_path = os.path.join(
        #         self.opt.output_dir, f'imgs/seed{self.seed}', f'splits/grid_seed{seed}')
        #     # os.makedirs(split_path, exist_ok=True)
        #     for idx in range(len(img_lines)):
        #         img_path = os.path.join(split_path, f'img_line{idx}.png')
        #         d_path = os.path.join(split_path, f'depth_line{idx}.png')
        #         ensure_dir(img_path)
        #         cv2.imwrite(img_path, img_lines[idx])
        #         cv2.imwrite(d_path, depth_lines[idx])

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

    def cal_error(self, pigan_volume_vs, icp_results, input_vs,
                  pred_ldmk, input_ldmk, name=''):
        #   seed {self.seed} z_num {z_num}:
        icp_mesh_error = self.cal_mesh_error(
            torch.from_numpy(pigan_volume_vs).float().to(self.device).unsqueeze(0),
            icp_results.Xt)
        input_mesh_error = self.cal_mesh_error(
            torch.from_numpy(pigan_volume_vs).float().to(self.device).unsqueeze(0),
            input_vs.unsqueeze(0))
        ldmk_error = self.cal_ldmk_error(pred_ldmk.squeeze(0), input_ldmk.squeeze(0))

        print(
            f'exp - {name}:'
            f'\t -> icp mesh error: \t{icp_mesh_error.item()} ,'
            f'\t -> input mesh error: \t{input_mesh_error.item()} ,'
            f'\t -> ldmk error: \t{ldmk_error.item()} .'
            )

        return icp_mesh_error, input_mesh_error, ldmk_error

    def cal_mean_error(self, errs):
        return np.array([v for k, v in errs.items()]).mean(axis=0)

    def loop_seeds(self):
        # for seed in tqdm(self.seeds):
        for seed in self.seeds:
            self.run_seed(seed, self.opt.using_cross_test)
        print(
            f'icp errors: {self.cal_mean_error(self.icp_errs)}\n'
            f'inp errors: {self.cal_mean_error(self.inp_errs)}\n'
            f'ldmk errors: {self.cal_mean_error(self.ldmk_errs)}\n'
            f'coorelation errors: {self.cal_mean_error(self.coorelations)}\n'
        )
        
    def icp(self, input, pigan):
        # volume = load_obj('shapes/0.obj')[0][0]
        # pigan = load_obj('Deep3DFaceRecon_pytorch/checkpoints/face_recon_feat0.2_augment/results/eval_pigan/pigan_000.obj')[0][0]

        # volume = torch.from_numpy(volume).unsqueeze(0)
        # pigan = torch.from_numpy(pigan).unsqueeze(0)
        results = points_alignment.iterative_closest_point(
            input, pigan, relative_rmse_thr=1e-3, verbose=self.debug)
        icp_result_path = os.path.join(
            self.opt.output_dir, f'objs/seed{self.seed}',
            f'icp_seed{self.seed}.obj')
        # os.makedirs(os.path.dirname(icp_result_path), exist_ok=True)
        ensure_dir(icp_result_path)
        save_obj_vertex(icp_result_path, results.Xt.cpu().numpy()[0])
        return results

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
    parser.add_argument('--snm', type=str2bool, nargs='?', const=True, default=False)
    opt = parser.parse_args()

    return opt


def main():
    opt = parse()
    eval_pigan = EvalPigan(opt)
    eval_pigan.loop_seeds()


if __name__ == '__main__':
    main()
