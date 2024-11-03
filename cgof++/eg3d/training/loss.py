# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

#---------------------------------------------------------------
# Modified by Sun et al.
from utils.utils import (
    img2depth, resize, min_max_norm, draw_landmarks,
    optical_flow_warping, vis_landmarks, make_grid)
import kornia
import cv2
#---------------------------------------------------------------

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    #---------------------------------------------------------------
    # Modified by Sun et al.
    def __init__(self, device, G, D, face_recon, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased', ctrl_kwargs=None):
    #---------------------------------------------------------------

        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        #---------------------------------------------------------------
        # Modified by Sun et al.
        self.ctrl_kwargs        = ctrl_kwargs
        self.face_recon         = face_recon
        self.trans_params       = self.face_recon.eg3d_trans_params
        self.recon_basic_size   = self.trans_params[:2]
        self.recon_s            = self.trans_params[2:3]
        self.recon_t            = self.trans_params[3:5]
        self.dim_id             = 80
        self.dim_exp            = 64
        #---------------------------------------------------------------

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, face_recon=None):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, face_recon=face_recon, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    #---------------------------------------------------------------
    # Modified by Sun et al.
    def cal_recon_loss(self, gen_img):
        # # z -> norm coeff
        # input_coeffs_dict = self.face_recon.get_input_coeff(z) # z -> 3dmm coeff
        # self.face_recon.input_norm_dict = self.face_recon.norm_coeff(input_coeffs_dict) # 3dmm coeff -> norm
        # # run face reconstruction
        d_recon, m_recon = img2depth(
            self.face_recon, gen_img['image_raw'], self.trans_params,
            self.recon_t, self.recon_s) # gen_img['image'] ~ [-1, 1]
        self.d_recon, self.m_recon = self.min_max_normalization(d_recon, m_recon)
        self.pred_lm = self.face_recon.pred_lm
        # calculate recon_loss
        recon_loss_keys = ['id', 'exp', 'tex', 'gamma']
        self.recon_loss = self.face_recon.cal_recon_loss(
            recon_loss_keys, 'norm') * self.ctrl_kwargs.recon_lambda

    def cal_geo_reg(self, gen_img):
        gen_depth = gen_img['image_depth']
        gen_depth.requires_grad_()
        if len(gen_depth.shape) == 3:
            gen_depth = gen_depth.unsqueeze(1)
        v00 = gen_depth[..., :-1, :-1]
        v01 = gen_depth[..., :-1, 1:]
        v10 = gen_depth[..., 1:, :-1]
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
        # loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)
        # geo_reg = loss.mean()

        # geo_reg_mask = torch.logical_not(
        #     self.face_recon.m_pigan_input[..., :-1, :-1])
        geo_reg_mask = torch.nn.functional.interpolate(self.face_recon.m_pigan_input*1.0, loss.shape[-2:], mode='nearest')
        geo_reg_fg = loss[geo_reg_mask>0].sum() / (geo_reg_mask.count_nonzero(dim=(1,2,3))+1e-8)

        self.geo_reg = geo_reg_fg * self.ctrl_kwargs.geo_reg_lambda

    def cal_lm_loss(self, gen_img, visualize=True):
        # cal ldmk loss
        lm_input = self.G.lm_input.detach().clip(0, 255)
        lm_pred = self.face_recon.inv_affine_ldmks_torch_(
            self.face_recon.pred_lm).clip(0, 255)
        self.lm_loss = torch.nn.L1Loss()(lm_pred, lm_input) * self.ctrl_kwargs.lm3d_lambda

        if visualize:
            imgs_vis = []
            for save_index in range(gen_img['image_raw'].shape[0]):
                gen_imgs = (gen_img['image_raw'][save_index:save_index+1]+1)/2
                gen_imgs_256 = torch.nn.functional.interpolate(
                    gen_imgs, size=(256, 256), mode='bilinear')
                img_vis = gen_imgs_256[0].permute(
                    1,2,0).detach().cpu().numpy()*255

                lm_input_vis = lm_input.cpu().numpy()
                lm_pred_vis = lm_pred.detach().cpu().numpy()

                lm_input_vis[save_index][:,1] = 255 - lm_input_vis[save_index][:,1]
                img_vis = draw_landmarks(img_vis, lm_input_vis[save_index], (0, 255, 0))

                lm_pred_vis[save_index][:,1] = 255 - lm_pred_vis[save_index][:,1]
                img_vis = draw_landmarks(img_vis, lm_pred_vis[save_index], (255, 0, 0))
                imgs_vis.append(img_vis)
            imgs_vis = np.concatenate(imgs_vis, axis=1)

            cv2.imwrite('ldmks_gInput_rPred.png', imgs_vis[...,[2,1,0]])

    def min_max_normalization(self, d, m):
        """
            zy: min max norm to be fixed
        """
        d = resize(d, self.neural_rendering_resolution)
        d = min_max_norm(d)
        m = resize(m, self.neural_rendering_resolution)
        m = min_max_norm(m)
        return d, m

    def cal_dist_depr(self):
        self.dist_depr = self.G.renderer.dist_depr.mean() * self.ctrl_kwargs.dist_depr_lambda

    def generate_warping_img(self, bs=1):

        torch.manual_seed(183)
        zs = torch.randn(
            (1, self.metadata['latent_dim']), device=self.device).repeat(2, 1)
        torch.manual_seed(183)
        # z_exp = torch.randn((self.dim_exp))
        z_exp = torch.randn(
                (10, self.dim_exp), device=self.device)[6]
        zs[1, self.dim_id:self.dim_id+self.dim_exp] = z_exp
        z0, z1 = zs[::2], zs[1::2]

        cfg = self.metadata
        cfg['nerf_noise'] = 0
        # cfg['num_steps'] *= 2
        yaws, pitchs = sample_yaw_pitch(
            bs, self.h_mean, self.v_mean,
            self.h_stddev, self.v_stddev, self.device)
        self.data1 = {}

        with torch.no_grad():
            gen_outputs0 = self.generator_ddp(
                z0, return_depth=self.requiring_depth,
                yaw=yaws, pitch=pitchs, face_recon=self.face_recon,
                using_dist_depr=self.using_depr,
                using_norm_reg=self.using_norm_reg,
                narrow_ratio=self.narrow_ratio, **cfg)
            img0 = gen_outputs0[0]

            # get flow
            input_coeffs_dict = self.face_recon.get_input_coeff(zs)
            self.face_recon.facemodel.to(self.device)
            input_shapes = self.face_recon.facemodel.compute_shape(
                input_coeffs_dict['id'],
                input_coeffs_dict['exp']
            )
            input_shapes = self.face_recon.msra2pigan(input_shapes)*0.5

            pigan_vertex = self.face_recon.pigan2cam(
                input_shapes, self.yaw2theta(yaws), self.pitch2phi(pitchs))
            pos_diff = input_shapes[::2] - input_shapes[1::2]
            pos_diff = pos_diff.unsqueeze(1).repeat(1, 2, 1, 1).contiguous()
            B, C, N, D = pos_diff.shape
            pos_diff = pos_diff.reshape(B*C, N, D)
            mask, depth, flow = self.face_recon.renderer_pigan(
                pigan_vertex, self.face_recon.facemodel.face_buf,
                pos_diff)

            _, ms, fs = self.face_recon.render_pigan(
                zs, self.yaw2theta(yaws), self.pitch2phi(pitchs))

            f0 = fs[0].permute(1, 2, 0)[..., [2, 1, 0]].clip(0, 1).cpu().numpy() * 255.0
            m0 = np.uint8(ms[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0)
            f0 = np.dstack([f0, m0])
            cv2.imwrite('warping_imgs/f0.png', f0)
            lm = self.face_recon.pigan_landmark
            lm[..., 1] = 255-lm[..., 1]
            save_img_with_landmarks(f0[..., [2, 1, 0, 3]], 'warping_imgs/f0_lm.png', lm[0], radius=2, color=(0, 255,0))

            mesh0 = self.face_recon.pigan_mesh[0].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
            mesh0 = np.dstack([mesh0,m0])
            cv2.imwrite('warping_imgs/mesh0.png', mesh0)
            save_img_with_landmarks(mesh0[...,[2,1,0,3]], 'warping_imgs/mesh0_lm.png', lm[0], radius=2, color=(0,255,0))

            f1 = fs[1].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
            m1 = np.uint8(ms[1].permute(1,2,0).detach().cpu().numpy() * 255.0)
            f1 = np.dstack([f1,m1])
            cv2.imwrite('warping_imgs/f1.png', f1)
            save_img_with_landmarks(f1[...,[2,1,0,3]], 'warping_imgs/f1_lm.png', lm[1], radius=2, color=(0,255,0))

            mesh1 = self.face_recon.pigan_mesh[1].permute(1,2,0)[...,[2,1,0]].clip(0,1).cpu().numpy()*255.0
            mesh1 = np.dstack([mesh1, m1])
            cv2.imwrite('warping_imgs/mesh1.png', mesh1)
            save_img_with_landmarks(mesh1[...,[2,1,0,3]], 'warping_imgs/mesh1_lm.png', lm[1], radius=2, color=(0,255,0))

            flow_2to1 = flow[1::2]
            flow_2to1 = flow_2to1.to(self.device)
            flow_2to1 = torch.nn.functional.interpolate(
                flow_2to1, (self.img_size, self.img_size))
            flow_2to1 = flow_2to1.permute(0,2,3,1).contiguous().unsqueeze(-2).repeat(1,1,1,12,1).contiguous().reshape(1,64*64*cfg['num_steps'],3)

            loss_mask = ((mask[::2]-mask[1::2]) <= 0).float()
            loss_mask = loss_mask.to(self.device)
            loss_mask = torch.nn.functional.interpolate(
                loss_mask, (self.img_size, self.img_size)).repeat(1, 3, 1, 1)

            gen_outputs1 = self.generator_ddp(
                z1, zy_data=self.data1, return_depth=self.requiring_depth,
                yaw=yaws, pitch=pitchs, face_recon=self.face_recon,
                using_dist_depr=self.using_depr,
                using_norm_reg=self.using_norm_reg,
                narrow_ratio=self.narrow_ratio, **cfg)
            img1 = gen_outputs1[0]
            depth1 = gen_outputs1[2]


            points_cam = self.data1['points_cam']

            c_output_1 = self.data1['coarse_output']
            f_output_1 = self.data1['fine_output']
            f_output2_1 = self.data1['fine_output2']

            transformed_points1 = self.data1['transformed_points'].detach()
            fine_points1 = self.data1['fine_points'].detach()
            fine_points2_1 = self.data1['fine_points2'].detach()

            fine_warp = fine_points1 + flow_2to1
            coarse_warp = transformed_points1 + flow_2to1
            fine2_warp = fine_points2_1 + flow_2to1
            self.data1['transformed_points_input'] = coarse_warp
            self.data1['fine_points_input'] = fine_warp
            self.data1['fine_points2_input'] = fine2_warp

            img0_warp, _, depth0_warp = self.generator_ddp(
                z0, zy_data=self.data1, return_depth=self.requiring_depth,
                yaw=yaws, pitch=pitchs, face_recon=self.face_recon,
                using_dist_depr=self.using_depr,
                using_norm_reg=self.using_norm_reg,
                narrow_ratio=self.narrow_ratio, **cfg)

            c_output_0warp = self.data1['coarse_output']
            f_output_0warp = self.data1['fine_output']
            f_output2_0warp = self.data1['fine_output2']

        outputs = {
            "img0": img0,
            "img0_warp": img0_warp,
            "img1": img1,

            "c_output_0warp": c_output_0warp,
            "c_output_1": c_output_1,

            "f_output_0warp": f_output_0warp,
            "f_output_1": f_output_1,

            "f_output2_0warp": f_output2_0warp,
            "f_output2_1": f_output2_1,

            "depth0_warp": depth0_warp,
            "depth1": depth1,

            "loss_mask": loss_mask,
            "flow_2to1": flow_2to1,
            "points_cam": points_cam,
        }

        return outputs

    def cal_warping3d_loss(self, bs=1, vis_warping=False):
        # print("cal 3dwarping_loss...")
        outputs = self.generate_warping_img()
        imgs0, imgs0_warp, imgs1 = outputs["img0"], outputs["img0_warp"], outputs["img1"]

        f_output_0warp, f_output_1 = outputs["f_output_0warp"], outputs["f_output_1"]
        f_output2_0warp, f_output2_1 = outputs["f_output2_0warp"], outputs["f_output2_1"]
        loss_mask = outputs["loss_mask"]
        flow_2to1 = outputs["flow_2to1"]
        points_cam = outputs["points_cam"]
        volume_loss_mask = get_volume_mask(
                points_cam, loss_mask, self.ray_start, self.ray_end)

        imgs1_blur = kornia.filters.gaussian_blur2d(
            imgs1, (5, 5), (2.0, 2.0))
        imgs0_warp_blur = kornia.filters.gaussian_blur2d(
            imgs0_warp, (5, 5), (2.0, 2.0))

        pixel_warping_loss = torch.sum(
            torch.sqrt(((imgs1_blur - imgs0_warp_blur)**2) + 1e-8)
            * loss_mask)/loss_mask.sum()
        volume_fine_warping_loss = torch.sum(
            torch.sqrt(((f_output_1 - f_output_0warp)**2) + 1e-8)
            * volume_loss_mask)/volume_loss_mask.sum()
        volume_fine2_warping_loss = torch.sum(
            torch.sqrt(((f_output2_1 - f_output2_0warp)**2) + 1e-8)
            * volume_loss_mask)/volume_loss_mask.sum()
        self.warping3d_loss = (
            pixel_warping_loss + volume_fine_warping_loss
            + volume_fine2_warping_loss
        ) * self.warping3d_lambda

        if self.rank == 0 and self.discriminator.step % self.opt.sample_interval == 0:
            imgs0_grid = make_grid(
                imgs0, nrow=1, padding=0, normalize=True,
                value_range=None, scale_each=False, pad_value=1).to(self.device)
            imgs1_grid = make_grid(
                imgs1, nrow=1, padding=0, normalize=True,
                value_range=None, scale_each=False, pad_value=1).to(self.device)
            imgs0_warp_grid = make_grid(
                imgs0_warp, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            imgs1_blur_grid = make_grid(
                imgs1_blur, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            imgs0_warp_blur_grid = make_grid(
                imgs0_warp_blur, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            loss_mask_grid = make_grid(
                loss_mask, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            flow = flow_2to1.permute(0, 2, 1).contiguous().reshape(bs, 3, self.img_size, self.img_size, self.num_steps)[...,0]
            flow_2to1_grid_x = make_grid(
                flow[:,0:1,:,:].repeat(1,3,1,1),
                nrow=1, padding=0, normalize=True, value_range=(-0.02, 0.02),
                scale_each=False, pad_value=1).to(self.device)
            flow_2to1_grid_y = make_grid(
                flow[:,1:2,:,:].repeat(1,3,1,1),
                nrow=1, padding=0, normalize=True, value_range=(-0.02, 0.02),
                scale_each=False, pad_value=1).to(self.device)
            flow_2to1_grid_z = make_grid(
                flow[:,2:3,:,:].repeat(1,3,1,1),
                nrow=1, padding=0, normalize=True, value_range=(-0.02, 0.02),
                scale_each=False, pad_value=1).to(self.device)
            flow_2to1_grid = make_grid(
                flow[:,:3,:,:],
                nrow=1, padding=0, normalize=True, value_range=(-0.02, 0.02),
                scale_each=False, pad_value=1).to(self.device)
            m = flow[:,:3,:,:].abs().sum(1)>0
            flow_2to1_grid = flow_2to1_grid * m
            # flow_2to1_grid_alpha = torch.cat([flow_2to1_grid, m.unsqueeze(1)], dim=1)
            images = torch.cat(
                [
                    imgs0_grid, imgs1_grid, imgs0_warp_grid,
                    imgs1_blur_grid, imgs0_warp_blur_grid, loss_mask_grid,
                    flow_2to1_grid_x, flow_2to1_grid_y, flow_2to1_grid_z], dim=-1
            ).to(self.device)
            images = images.permute(1, 2, 0).detach().cpu().numpy()

            cv2.imwrite(
                'warping_imgs/imgs0_grid.png', cv2.cvtColor(imgs0_grid.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/imgs1_grid.png', cv2.cvtColor(imgs1_grid.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/imgs0_warp_grid.png', cv2.cvtColor((imgs0_warp_grid*loss_mask[0]).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/flow_2to1_grid_x.png', cv2.cvtColor(flow_2to1_grid_x.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/flow_2to1_grid_y.png', cv2.cvtColor(flow_2to1_grid_y.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/flow_2to1_grid_z.png', cv2.cvtColor(flow_2to1_grid_z.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/flow_2to1_grid.png', cv2.cvtColor(flow_2to1_grid.permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_RGB2BGR)*255)
            cv2.imwrite(
                'warping_imgs/3d_warping.png', cv2.cvtColor(images, cv2.COLOR_RGB2BGR)*255)

            _d_recon, _m_recon, _pred_lm = self.render_recon_depth(torch.cat([imgs0, imgs1, imgs0_warp], dim=0))
            _pred_lm = self.face_recon.inv_affine_ldmks_torch_(_pred_lm).clip(0, 255)
            _pred_lm[...,1] = 255 - _pred_lm[...,1]
            imgs0_grid = torch.nn.functional.interpolate(imgs0_grid.unsqueeze(0), (255,255))[0]
            save_img_with_landmarks(
                imgs0_grid.permute(1, 2, 0).detach().cpu().numpy()*255,
                './warping_imgs/img0_lm.png',
                _pred_lm.detach().cpu().numpy()[0],
                radius=2, color=(0,0,255))
            imgs1_grid = torch.nn.functional.interpolate(imgs1_grid.unsqueeze(0), (255,255))[0]
            save_img_with_landmarks(
                imgs1_grid.permute(1, 2, 0).detach().cpu().numpy()*255,
                './warping_imgs/img1_lm.png',
                _pred_lm.detach().cpu().numpy()[1],
                radius=2, color=(0,0,255))
            
            self.tensorboard_writer.add_image(
                '3d expression warp', images,
                self.discriminator.step, dataformats='HWC')

        return self.warping3d_loss

    def cal_exp_warping_loss(self, bs):
        # print("cal exp_warping_loss...")
        zs = torch.randn(
            (bs*2, self.metadata['latent_dim']), device=self.device)
        zs[1::2, :self.dim_id] = zs[::2, :self.dim_id]
        zs[1::2, (self.dim_id+self.dim_exp):] = zs[::2, (self.dim_id+self.dim_exp):]

        gen_img, _gen_ws = self.run_G(
            zs, gen_c, swapping_prob=swapping_prob,
            neural_rendering_resolution=neural_rendering_resolution,
            face_recon=self.face_recon)
        
        gen_imgs = gen_outputs[0]
        gen_imgs.requires_grad_()
        gt_images = self.face_recon.f_pigan_input.to(self.device)
        flow_2to1, loss_mask = self.face_recon.get_warping_flow_bs(
            zs, self.yaw2theta(yaws), self.pitch2phi(pitchs))
        flow_2to1 = flow_2to1.to(self.device)
        flow_2to1_ = torch.nn.functional.interpolate(
            flow_2to1, (self.img_size, self.img_size))
        loss_mask = loss_mask.to(self.device)
        loss_mask = torch.nn.functional.interpolate(
            loss_mask, (self.img_size, self.img_size)).repeat(1, 3, 1, 1)
        flow_2to1 = flow_2to1_[:, :2, :, :]
        fake_1to2 = optical_flow_warping(
            gen_imgs[::2], flow_2to1, pad_mode="border")  # IMPORTANT!
        # fake2 = gaussian_blur(fake2,size=5,sigma=2)
        # fake_1to2 = gaussian_blur(fake_1to2,size=5,sigma=2)
        fake2_blur = kornia.filters.gaussian_blur2d(
            gen_imgs[1::2], (5, 5), (2.0, 2.0))
        fake_1to2_blur = kornia.filters.gaussian_blur2d(
            fake_1to2, (5, 5), (2.0, 2.0))

        self.exp_warping_loss = torch.sum(
            torch.sqrt(((fake2_blur - fake_1to2_blur)**2) + 1e-8)
            * loss_mask)/loss_mask.sum() * self.exp_warping_lambda

        # visualization
        if self.rank == 0 and self.discriminator.step % self.opt.sample_interval == 0:
            gen_img_grid = make_grid(
                gen_imgs, nrow=2, padding=0, normalize=True,
                value_range=None, scale_each=False, pad_value=1).to(self.device)
            fake_1to2_grid = make_grid(
                fake_1to2, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            fake2_blur_grid = make_grid(
                fake2_blur, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            fake_1to2_blur_grid = make_grid(
                fake_1to2_blur, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            loss_mask_grid = make_grid(
                loss_mask, nrow=1, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            gt_images_grid = make_grid(
                gt_images, nrow=2, padding=0, normalize=True,
                scale_each=False, pad_value=1).to(self.device)
            flow_2to1_grid_x = make_grid(
                flow_2to1_[:,0:1,:,:].repeat(1,3,1,1), nrow=1, padding=0, normalize=True, value_range=(-32,32),
                scale_each=False, pad_value=1).to(self.device)
            flow_2to1_grid_y = make_grid(
                flow_2to1_[:,1:2,:,:].repeat(1,3,1,1), nrow=1, padding=0, normalize=True, value_range=(-32,32),
                scale_each=False, pad_value=1).to(self.device)
            images = torch.cat(
                [gen_img_grid, fake_1to2_grid, fake2_blur_grid,
                    fake_1to2_blur_grid, loss_mask_grid,
                    gt_images_grid, flow_2to1_grid_x, flow_2to1_grid_y], dim=-1
            ).to(self.device)
            images = images.permute(1, 2, 0).detach().cpu().numpy()
            cv2.imwrite(
                'exp_warping.png', cv2.cvtColor(images, cv2.COLOR_RGB2BGR)*255)
            self.tensorboard_writer.add_image(
                'expression warp', images,
                self.discriminator.step, dataformats='HWC')

        return self.exp_warping_loss

    #---------------------------------------------------------------

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, verbose=False):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(
                    gen_z, gen_c, swapping_prob=swapping_prob,
                    neural_rendering_resolution=neural_rendering_resolution,
                    face_recon=self.face_recon)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

            #---------------------------------------------------------------
            # Modified by Sun et al.

            # calculate recon loss
            if self.ctrl_kwargs.using_recon_loss:
                # print("using recon_loss...")
                self.neural_rendering_resolution = neural_rendering_resolution
                with torch.autograd.profiler.record_function('Recon_forward'):
                    self.cal_recon_loss(gen_img)
                    training_stats.report('Loss/G/recon_loss', self.recon_loss)
                    loss_Gmain = loss_Gmain + self.recon_loss
            else:
                # print("not using recon_loss...")
                pass

            # calculate geo reg
            if self.ctrl_kwargs.using_geo_reg:
                # print("using geo_reg...")
                with torch.autograd.profiler.record_function('geo_reg_forward'):
                    self.cal_geo_reg(gen_img)
                    training_stats.report('Loss/G/geo_reg', self.geo_reg)
                    # print(f"geo_reg:{self.geo_reg}")
                    loss_Gmain = loss_Gmain + self.geo_reg
            else:
                # print("not using geo_reg...")
                pass

            # calculate ldmk loss
            if self.ctrl_kwargs.using_ldmk_loss:
                # print("using ldmk loss...")
                self.neural_rendering_resolution = neural_rendering_resolution
                with torch.autograd.profiler.record_function('Ldmk_forward'):
                    self.cal_lm_loss(gen_img)
                    training_stats.report('Loss/G/ldmk_loss', self.lm_loss)
                    loss_Gmain = loss_Gmain + self.lm_loss
            else:
                # print("not using ldmk loss...")
                pass

            # calculate dist_depr
            if self.ctrl_kwargs.using_dist_depr:
                # print("using dist_depr...")
                with torch.autograd.profiler.record_function('Depr_forward'):
                    self.cal_dist_depr()
                    training_stats.report('Loss/G/dist_depr', self.dist_depr)
                    loss_Gmain = loss_Gmain + self.dist_depr
            else:
                # print("not using dist_depr...")
                pass

            # calculate warping loss
            if self.ctrl_kwargs.using_warping_loss:
                # print("using warping loss...")
                with torch.autograd.profiler.record_function('Lwarp_forward'):
                    gt_images1 = self.face_recon.f_pigan_input.to(self.device)
                    pos_2d_1 = self.G.verts_input.detach()
                    verts_cam_1 = self.G.verts_cam.detach()
                    image1 = gen_img['image_raw']

                    with torch.no_grad():
                        gen_z2 = torch.randn(gen_z.shape, device=self.device)
                        gen_z2[:, :self.dim_id] = gen_z[:, :self.dim_id]
                        gen_z2[:, (self.dim_id+self.dim_exp):] = gen_z[:, (self.dim_id+self.dim_exp):]
                        gen_img2, _gen_ws2 = self.run_G(
                            gen_z2, gen_c, swapping_prob=swapping_prob,
                            neural_rendering_resolution=neural_rendering_resolution,
                            face_recon=self.face_recon)
                        gt_images2 = self.face_recon.f_pigan_input.to(self.device)
                        pos_2d_2 = self.G.verts_input.detach()
                        verts_cam_2 = self.G.verts_cam.detach()
                        image2 = gen_img2['image_raw']

                        pos_diff = pos_2d_1 - pos_2d_2
                        pos_diff = torch.stack(
                            [pos_diff[..., 0], -pos_diff[..., 1]], axis=-1)
                        pos_diff = torch.cat(
                            [pos_diff, torch.zeros_like(pos_diff[..., 0:1])], axis=-1).contiguous()
                        pos_diff = pos_diff.unsqueeze(1).repeat(1,2,1,1).contiguous()
                        B, C, N, D = pos_diff.shape
                        pos_diff = pos_diff.reshape(B*C, N, D)

                        verts_cam = torch.cat([verts_cam_1, verts_cam_2], dim=0)
                        mask, depth, flow = self.face_recon.renderer_pigan(
                            verts_cam,
                            self.face_recon.facemodel.face_buf,
                            pos_diff)
                        loss_mask = ((mask[::2]-mask[1::2])<=0).float()
                        flow_2to1 = flow[1::2]
                        flow_2to1 = flow_2to1.to(self.device)
                        flow_2to1_ = torch.nn.functional.interpolate(
                            flow_2to1, (self.G.neural_rendering_resolution, self.G.neural_rendering_resolution))/self.face_recon.renderer_pigan.rasterize_size*self.G.neural_rendering_resolution
                        loss_mask = loss_mask.to(self.device)
                        loss_mask = torch.nn.functional.interpolate(
                            loss_mask, (self.G.neural_rendering_resolution, self.G.neural_rendering_resolution)).repeat(1, 3, 1, 1)
                        flow_2to1 = flow_2to1_[:, :2, :, :]
                    fake_1to2 = optical_flow_warping(
                        image1, flow_2to1, pad_mode="border")  # IMPORTANT!
                    fake2_blur = kornia.filters.gaussian_blur2d(
                        image2, (5, 5), (2.0, 2.0))
                    fake_1to2_blur = kornia.filters.gaussian_blur2d(
                        fake_1to2, (5, 5), (2.0, 2.0))

                    self.exp_warping_loss = torch.sum(
                        torch.sqrt(((fake2_blur - fake_1to2_blur)**2) + 1e-8)
                        * loss_mask)/loss_mask.sum() * self.ctrl_kwargs.exp_warping_lambda
                    # training_stats.report('Loss/G/exp_warping_loss', self.exp_warping_loss)
                    loss_Gmain = loss_Gmain + self.exp_warping_loss

                    if verbose:
                        gen_img_grid = make_grid(
                            torch.cat([image1, image2],dim=-1), nrow=1, padding=0, normalize=True,
                            value_range=None, scale_each=False, pad_value=1).to(self.device)
                        fake_1to2_grid = make_grid(
                            fake_1to2, nrow=1, padding=0, normalize=True,
                            scale_each=False, pad_value=1).to(self.device)
                        fake2_blur_grid = make_grid(
                            fake2_blur, nrow=1, padding=0, normalize=True,
                            scale_each=False, pad_value=1).to(self.device)
                        fake_1to2_blur_grid = make_grid(
                            fake_1to2_blur, nrow=1, padding=0, normalize=True,
                            scale_each=False, pad_value=1).to(self.device)
                        loss_mask_grid = make_grid(
                            loss_mask, nrow=1, padding=0, normalize=True,
                            scale_each=False, pad_value=1).to(self.device)
                        gt_images_grid = make_grid(
                            torch.cat(
                                [
                                    torch.nn.functional.interpolate(gt_images1, [self.G.neural_rendering_resolution, self.G.neural_rendering_resolution]),
                                    torch.nn.functional.interpolate(gt_images2, [self.G.neural_rendering_resolution, self.G.neural_rendering_resolution]),
                                ], dim=-1
                            ), nrow=1, padding=0, normalize=True,
                            scale_each=False, pad_value=1).to(self.device)
                        flow_2to1_grid_x = make_grid(
                            flow_2to1_[:,0:1,:,:].repeat(1,3,1,1), nrow=1, padding=0, normalize=True, value_range=(-32,32),
                            scale_each=False, pad_value=1).to(self.device)
                        flow_2to1_grid_y = make_grid(
                            flow_2to1_[:,1:2,:,:].repeat(1,3,1,1), nrow=1, padding=0, normalize=True, value_range=(-32,32),
                            scale_each=False, pad_value=1).to(self.device)
                        images = torch.cat(
                            [gen_img_grid, fake_1to2_grid, fake2_blur_grid,
                                fake_1to2_blur_grid, loss_mask_grid,
                                gt_images_grid, flow_2to1_grid_x, flow_2to1_grid_y], dim=-1
                        ).to(self.device)
                        images = images.permute(1, 2, 0).detach().cpu().numpy()
                        self.exp_warping_img = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)*255

                    training_stats.report('Loss/G/warping', self.exp_warping_loss)
                # with torch.autograd.profiler.record_function('Lwarp_backward'):
                #     self.exp_warping_loss.mean().mul(gain).backward()
            else:
                # print("not using warping loss...")
                pass
            #---------------------------------------------------------------

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

            #---------------------------------------------------------------
            # Modified by Sun et al.
            #---------------------------------------------------------------


        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, face_recon=self.face_recon)

                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
