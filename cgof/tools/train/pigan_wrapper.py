from matplotlib.pyplot import step
from sklearn import neighbors
from torch import logical_not
from packages import *
from initializer import Initializer, z_sampler


class PiganWrapper(Initializer):
    """run G or D and calculate losses.
    """
    def __init__(self, rank, world_size, opt):
        super(PiganWrapper, self).__init__(rank, world_size, opt)
        self.data = {}

    # train Discriminator
    def gen_img_for_D(self):
        with torch.no_grad():
            self.z = z_sampler(
                (self.real_imgs.shape[0], self.metadata['latent_dim']),
                device=self.device, dist=self.metadata['z_dist'])
            self.split_batch_size = (
                self.z.shape[0] // self.metadata['batch_split'])
            gen_imgs = []
            gen_positions = []
            gen_depths = []

            for split in range(self.metadata['batch_split']):
                self.logger(
                    f'------ split {split} ------',
                    info_level='debug')
                self.subset_z = self.z[
                    split * self.split_batch_size:(split + 1)
                    * self.split_batch_size]

                gen_outputs = self.generator_ddp(
                    self.subset_z, return_depth=self.requiring_depth,
                    face_recon=self.face_recon,
                    narrow_ratio=self.narrow_ratio,
                    **self.metadata)
                g_imgs, g_pos = gen_outputs[0], gen_outputs[1]
                g_imgs, g_pos = g_imgs.detach(), g_pos.detach()
                # if torch.any(torch.isnan(g_imgs)):
                #     self.save_models()
                #     torch.save(self.subset_z, "debug/subset_z.pt")
                #     torch.save(g_pos, "debug/g_pose.pt")
                #     raise ValueError
                g_imgs.requires_grad = True  # ?????
                g_pos.requires_grad = True
                gen_imgs.append(g_imgs)
                gen_positions.append(g_pos)
                if self.requiring_depth:
                    gen_depth = gen_outputs[2].detach()
                    gen_depth.requires_grad = True
                    if len(gen_depth.shape) == 3:
                        gen_depth = gen_depth.unsqueeze(1)
                    gen_depths.append(gen_depth)
            self.gen_imgs = torch.cat(gen_imgs, axis=0)
            self.gen_positions = torch.cat(gen_positions, axis=0)
            if self.requiring_depth:
                self.gen_depths = torch.cat(gen_depths, axis=0)
                self.gen_depths.requires_grad = True

        self.real_imgs.requires_grad = True

    def D_g_preds(self):
        d_outputs = self.discriminator_ddp(
            self.gen_imgs, self.alpha, **self.metadata)
        self.g_preds, self.g_pred_latent, self.g_pred_position = d_outputs

    def D_r_preds(self):
        self.r_preds, _, _ = self.discriminator_ddp(
            self.real_imgs, self.alpha, **self.metadata)

    def cal_grad_penalty_for_D(self):
        if self.metadata['r1_lambda'] > 0:
            grad_real = torch.autograd.grad(
                outputs=self.scaler.scale(self.r_preds.sum()),
                inputs=self.real_imgs,
                create_graph=True)
            inv_scale = 1. / self.scaler.get_scale()
            grad_real = [p * inv_scale for p in grad_real][0]
            grad_penalty = (
                grad_real.view(
                    grad_real.size(0), -1).norm(2, dim=1)**2).mean()
            self.grad_penalty = 0.5 * self.metadata['r1_lambda'] * grad_penalty
        else:
            self.grad_penalty = torch.cuda.FloatTensor(1).fill_(0)

    def cal_id_penalty(self, z):
        if self.metadata['z_lambda'] > 0 or self.metadata['pos_lambda'] > 0:
            self.latent_penalty = torch.nn.MSELoss()(
                self.g_pred_latent, z) * self.metadata['z_lambda']
            self.position_penalty = torch.nn.MSELoss()(
                self.g_pred_position,
                self.gen_positions) * self.metadata['pos_lambda']
            self.identity_penalty = self.latent_penalty + self.position_penalty
        else:
            self.identity_penalty = torch.cuda.FloatTensor(1).fill_(0)

    def cal_d_loss(self):
        # Generate images for discriminator
        self.gen_img_for_D()

        # concat images
        self.D_r_preds()
        self.D_g_preds()

        # gradient penalty
        self.cal_grad_penalty_for_D()

        # identity penalty
        self.cal_id_penalty(self.z)

        # optimize discriminator
        self.d_loss = (
            torch.nn.functional.softplus(self.g_preds).mean()
            + torch.nn.functional.softplus(-self.r_preds).mean()
            + self.grad_penalty + self.identity_penalty)

    def discriminator_train_step(self):
        self.cal_d_loss()
        self.optimizer_D.zero_grad()
        self.scaler.scale(self.d_loss).backward()
        self.scaler.unscale_(self.optimizer_D)
        torch.nn.utils.clip_grad_norm_(
            self.discriminator_ddp.parameters(),
            self.metadata['grad_clip'])
        self.scaler.step(self.optimizer_D)

    def spade_d_train_step(self):
        # calculate spade_d_loss
        if self.using_spade_d:
            D_losses = {}
            self.logger('using spade_discriminator')
            # In Batch Normalization, the fake and real images are
            # recommended to be in the same batch to avoid disparate
            # statistics in fake and real images.
            # So both fake and real images are fed to D all at once.
            fake_and_real = torch.cat(
                [self.gen_imgs, self.real_imgs], dim=0)  # 2, 3, 128, 128
            discriminator_out = self.spade_discriminator_ddp(fake_and_real)
            pred_fake, pred_real = divide_pred(discriminator_out)
            D_losses['D_Fake'] = self.criterionGAN(
                pred_fake, False, for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(
                pred_real, True, for_discriminator=True)
            spade_d_loss = sum(
                D_losses.values()).mean() * self.metadata['spade_d_lambda']

            # optimize spade discriminator
            self.optimizer_spade_D.zero_grad()
            self.scaler.scale(spade_d_loss).backward()
            self.scaler.unscale_(self.optimizer_spade_D)
            torch.nn.utils.clip_grad_norm_(
                self.spade_discriminator_ddp.parameters(),
                self.metadata['grad_clip'])
            self.scaler.step(self.optimizer_spade_D)

    def depth_d_train_step(self):
        # calculate depth discriminator loss
        if self.using_depth_d:
            # logger('using deep3dfacerecon!')
            D_losses = {}

            d_recon, m_recon = img2depth(
                self.face_recon, self.gen_imgs,
                self.trans_params, self.recon_t,
                self.recon_s)
            d_input, m_input = z2depth(self.face_recon, self.z)
            # visualize face recon
            """
                zy: min max norm to be fixed
            """
            d_input = resize(d_input, self.metadata['img_size'])
            d_input = min_max_norm(d_input)
            d_recon = resize(d_recon, self.metadata['img_size'])
            d_recon = min_max_norm(d_recon)

            m_input = resize(m_input, self.metadata['img_size'])
            m_input = min_max_norm(m_input)
            m_recon = resize(m_recon, self.metadata['img_size'])
            m_recon = min_max_norm(m_recon)

            # m_input = torch.from_numpy(
            #   m_input)[..., 0:1].permute(0, 3, 1, 2).to(self.device)
            fake_concat = torch.cat([d_input, self.gen_depths], dim=1)
            real_concat = torch.cat([d_recon, self.gen_depths], dim=1)
            fake_and_real = torch.cat(
                [fake_concat, real_concat], dim=0)  # 2, 3, 128, 128

            discriminator_out = self.depth_discriminator_ddp(fake_and_real)
            pred_fake, pred_real = divide_pred(discriminator_out)
            D_losses['D_Fake'] = self.criterionGAN(
                pred_fake, False, for_discriminator=True)
            D_losses['D_real'] = self.criterionGAN(
                pred_real, True, for_discriminator=True)
            depth_d_loss = sum(
                D_losses.values()).mean() * self.metadata['depth_d_lambda']

            # optimize depth discriminator
            self.optimizer_depth_D.zero_grad()
            self.scaler.scale(depth_d_loss).backward()
            # embed()
            self.scaler.unscale_(self.optimizer_depth_D)
            torch.nn.utils.clip_grad_norm_(
                self.depth_discriminator_ddp.parameters(),
                self.metadata['grad_clip'])
            self.scaler.step(self.optimizer_depth_D)

    def run_discriminator_one_step(self):
        self.discriminator_train_step()
        self.spade_d_train_step()
        self.depth_d_train_step()
        # save the inference results
        if self.rank == 0:
            if self.discriminator.step % self.opt.sample_interval == 0:
                f = min_max_norm(self.gen_imgs[0])
                r = min_max_norm(self.real_imgs[0])

                # d = d.expand([3,-1,-1])
                c = torch.cat([f, r], dim=2)
                save_image(
                    c,
                    os.path.join(
                        self.opt.output_dir,
                        f"{self.discriminator.step}_debug.png"),
                    nrow=1)

    # train Generator
    def sample_z_for_G(self):
        self.z = z_sampler(
            (self.imgs.shape[0], self.metadata['latent_dim']),
            device=self.device,
            dist=self.metadata['z_dist'])
        # def sample_yaw_pitch(
        #   n, horizontal_mean, vertical_mean,
        #   horizontal_stddev, vertical_stddev, device):
        self.split_batch_size = self.z.shape[0] // self.metadata['batch_split']

        self.extra_split_for_contrastive = 0
        extra_split_start_idx_in_batch = self.z.shape[0]
        self.extra_zs = [
            z_sampler(
                (1, self.metadata['latent_dim']),
                device=self.device,
                dist=self.metadata['z_dist'])
        ]
        contrastive_idx_count = 1
        contrastive_loss_index = {}

        if self.using_cid_loss:
            zs = self.z_manager.make_contrast_id(self.extra_zs[0])
            self.extra_zs = self.extra_zs + zs

            contrastive_loss_index['id'] = contrastive_idx_count
            contrastive_idx_count = contrastive_idx_count + 2
            self.extra_split_for_contrastive = 1

        if self.using_cgeo_loss:
            zs = self.z_manager.make_contrast_geo(self.extra_zs[0])
            self.extra_zs = self.extra_zs + zs

            contrastive_loss_index['geo'] = contrastive_idx_count
            contrastive_idx_count = contrastive_idx_count + 2
            self.extra_split_for_contrastive = 1

        if self.using_ctex_loss:
            zs = self.z_manager.make_contrast_tex(self.extra_zs[0])
            self.extra_zs = self.extra_zs + zs

            contrastive_loss_index['tex'] = contrastive_idx_count
            contrastive_idx_count = contrastive_idx_count + 2
            self.extra_split_for_contrastive = 1

        if self.using_ctex_gram_loss:
            zs = self.z_manager.make_contrast_tex(self.extra_zs[0])
            self.extra_zs = self.extra_zs + zs

            contrastive_loss_index['tex_gram'] = contrastive_idx_count
            contrastive_idx_count = contrastive_idx_count + 2
            self.extra_split_for_contrastive = 1

        if self.using_cface_loss:
            zs = self.z_manager.make_contrast_face(self.extra_zs[0])
            self.extra_zs = self.extra_zs + zs

            contrastive_loss_index['face'] = contrastive_idx_count
            contrastive_idx_count = contrastive_idx_count + 2
            self.extra_split_for_contrastive = 1

    def sample_yaw_pitch_for_G(self):
        # ensure to use extra split
        if self.extra_split_for_contrastive == 1:
            self.extra_zs.insert(0, self.z)
            z = torch.cat(self.extra_zs, dim=0)

            self.yaw, self.pitch = sample_yaw_pitch(
                z.shape[0],
                self.metadata['h_mean'],
                self.metadata['v_mean'],
                self.metadata['h_stddev'],
                self.metadata['v_stddev'],
                self.device)

            if self.using_cgeo_loss:  # fix yaw and pitch
                idx = self.contrastive_loss_index['geo']
                self.yaw[
                    self.extra_split_start_idx_in_batch + idx, :
                ] = self.yaw[self.extra_split_start_idx_in_batch]
                self.yaw[
                    self.extra_split_start_idx_in_batch + idx + 1, :
                ] = self.yaw[self.extra_split_start_idx_in_batch]
                self.pitch[
                    self.extra_split_start_idx_in_batch + idx, :
                ] = self.pitch[self.extra_split_start_idx_in_batch]
                self.pitch[
                    self.extra_split_start_idx_in_batch + idx + 1, :
                ] = self.pitch[self.extra_split_start_idx_in_batch]

            if self.using_cface_loss:  # fix yaw and pitch
                idx = self.contrastive_loss_index['face']
                self.yaw[
                    self.extra_split_start_idx_in_batch + idx, :
                ] = self.yaw[self.extra_split_start_idx_in_batch]
                self.yaw[
                    self.extra_split_start_idx_in_batch + idx + 1, :
                ] = self.yaw[self.extra_split_start_idx_in_batch]
                self.pitch[
                    self.extra_split_start_idx_in_batch + idx, :
                ] = self.pitch[self.extra_split_start_idx_in_batch]
                self.pitch[
                    self.extra_split_start_idx_in_batch + idx + 1, :
                ] = self.pitch[self.extra_split_start_idx_in_batch]

        else:
            self.yaw, self.pitch = sample_yaw_pitch(
                self.z.shape[0],
                self.metadata['h_mean'],
                self.metadata['v_mean'],
                self.metadata['h_stddev'],
                self.metadata['v_stddev'],
                self.device)

    def yaw2theta(self, yaw):
        return yaw * self.metadata['h_stddev'] + self.metadata['h_mean']

    def pitch2phi(self, pitch):
        return pitch * self.metadata['v_stddev'] + self.metadata['v_mean']

    def gen_img_for_G(self):
        # generate a subset of images
        self.subset_z = self.z[
            self.split * self.split_batch_size:(self.split + 1)
            * self.split_batch_size]
        subset_yaw = self.yaw[
            self.split * self.split_batch_size:(self.split + 1)
            * self.split_batch_size]
        subset_pitch = self.pitch[
            self.split * self.split_batch_size:(self.split + 1)
            * self.split_batch_size]
        subset_real_imgs = self.real_imgs[
            self.split * self.split_batch_size:(self.split + 1)
            * self.split_batch_size]
        zy_data = {}

        # if self.requiring_depth:
        #     self.d_pigan_input, self.m_pigan_input, self.z_input_dict = \
        #         self.render_input_depth(
        #             self.subset_z,
        #             self.yaw2theta(subset_yaw),
        #             self.pitch2phi(subset_pitch)
        #         )

        gen_outputs = self.generator_ddp(
            self.subset_z,
            zy_data=zy_data,
            return_depth=self.requiring_depth,
            yaw=subset_yaw,
            pitch=subset_pitch,
            face_recon=self.face_recon,
            using_dist_depr=self.using_depr,
            using_norm_reg=self.using_norm_reg,
            narrow_ratio=self.narrow_ratio,
            **self.metadata)
        # gen_imgs.requires_grad_()
        # gen_positions.requires_grad_()

        self.gen_imgs, self.gen_positions = gen_outputs[0], gen_outputs[1]
        self.gen_imgs.requires_grad_()
        self.gen_positions.requires_grad_()

        if self.requiring_depth:
            self.gen_depth_variance = zy_data['depth_variance']
            self.gen_depth_variance.requires_grad_()
            self.gen_depth = gen_outputs[2]
            self.gen_depth.requires_grad_()

            gdepth = min_max_norm(self.gen_depth.unsqueeze(1), 0.88, 1.12)
            fdepth = min_max_norm(
                self.face_recon.d_pigan_input
                * self.face_recon.m_pigan_input,
                # + torch.logical_not(self.face_recon.m_pigan_input)
                # * torch.median(
                #     self.face_recon.d_pigan_input[
                #         self.face_recon.m_pigan_input
                #     ]),
                0.88, 1.12)
            gf_diff = torch.abs(gdepth - fdepth)
            gf_diff_mask = gf_diff * self.face_recon.m_pigan_input
            dep_cat = torch.cat(
                [gdepth, fdepth, gf_diff, gf_diff_mask], dim=-1)

            if self.rank == 0:
                self.tensorboard_writer.add_image(
                    'dep_cat',
                    torch.cat(
                        [dep_cat[i] for i in range(dep_cat.shape[0])], dim=-2),
                    self.discriminator.step)
            if len(self.gen_depth.shape) == 3:
                self.gen_depth = self.gen_depth.unsqueeze(1)

    def run_D_for_G(self):
        # D inference
        self.D_g_preds()
        topk_percentage = max(
            0.99**(self.discriminator.step / self.metadata['topk_interval']),
            self.metadata['topk_v']
            ) if (
                'topk_interval' in self.metadata
                and 'topk_v' in self.metadata) else 1
        self.topk_num = math.ceil(topk_percentage * self.g_preds.shape[0])
        self.g_preds = torch.topk(self.g_preds, self.topk_num, dim=0).values

    def cal_tddfa_penalty(self):
        # calculate ldmk penalty
        if self.using_tddfa_loss:
            gen_imgs_255 = self.gen_imgs.mul(255).add(0.5).clamp(0, 255)
            gen_imgs_bgr = gen_imgs_255[:, [2, 1, 0], :, :]
            roi_box = [
                -3.57902386, 8.05658312, 67.7579577, 79.39356469
            ]
            cropped_imgs = crop_batch_tensor(gen_imgs_bgr, roi_box)
            gen_imgs_120 = nn.functional.interpolate(
                cropped_imgs, size=(120, 120), mode='bilinear')
            gen_imgs_norm = gen_imgs_120.sub(127.5).div(128)
            # gen_imgs_norm = torch.cat(gen_imgs_norm,dim=0)
            recon = self.tddfa_ddp(gen_imgs_norm)
            self.tddfa_loss = l1loss()(
                recon[:, -(self.dim_s + self.dim_e):],
                self.subset_z[:, -(self.dim_s + self.dim_e):].detach(
                )) * self.metadata['tddfa_lambda']
        else:
            self.tddfa_loss = torch.cuda.FloatTensor(1).fill_(0)

    def cal_spade_penalty(self):
        if self.using_spade_d:
            spade_g_loss = {}
            if self.metadata.get('spade_g_lambda', 0.0) > 0:
                self.logger('using spade g loss', info_level='debug')
                # real_concat = torch.cat(
                #   [gt_depths, subset_real_imgs], dim=1)
                # from IPython import embed; embed()
                fake_and_real = torch.cat(
                    [self.gen_imgs, self.subset_real_imgs], dim=0)
                discriminator_out = self.spade_discriminator_ddp(
                    fake_and_real)  #
                pred_fake, pred_real = divide_pred(
                    discriminator_out)

                spade_g_loss['GAN'] = self.criterionGAN(
                    pred_fake, True, for_discriminator=False
                ) * self.metadata['spade_g_lambda']

            if self.metadata.get('spade_gan_feature_lambda', 0.0) > 0:
                self.logger(
                    'using spade gan feature loss',
                    info_level='debug')
                num_D = len(pred_fake)
                GAN_Feat_loss = torch.cuda.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction,
                    # so we exclude it
                    num_intermediate_outputs = len(
                        pred_fake[i]) - 1
                    for j in range(
                            num_intermediate_outputs):  # for each layer output
                        unweighted_loss = torch.nn.L1Loss()(
                            pred_fake[i][j],
                            pred_real[i][j].detach())
                        GAN_Feat_loss += (
                            unweighted_loss
                            * self.metadata['spade_vgg_lambda']
                        ) / num_D
                spade_g_loss['GAN_Feat'] = GAN_Feat_loss

            if self.metadata.get('spade_vgg_lambda', 0.0) > 0:
                self.logger('using spade vgg loss', info_level='debug')
                spade_g_loss['VGG'] = self.criterionVGG(
                    self.gen_imgs, self.subset_real_imgs
                ) * self.metadata['spade_vgg_lambda']
            self.spade_g_loss = sum(spade_g_loss.values()).mean()
        else:
            self.spade_g_loss = 0

    def render_recon_depth(self, gen_imgs):
        d_recon, m_recon = self.img_to_depth(gen_imgs)
        pred_lm = self.face_recon.pred_lm
        return d_recon, m_recon, pred_lm

    def render_input_depth(self, subset_z, yaw, pitch):
        d_input, m_input = self.z_to_pigan_depth(subset_z, yaw, pitch)
        z_input_dict = self.face_recon.get_z_dict(subset_z)
        return d_input, m_input, z_input_dict

    def min_max_normalization(self, d, m):
        """
            zy: min max norm to be fixed
        """
        d = resize(d, self.metadata['img_size'])
        d = min_max_norm(d)
        m = resize(m, self.metadata['img_size'])
        m = min_max_norm(m)
        return d, m

    def img_to_depth(self, gen_imgs):
        d_recon, m_recon = img2depth(
            self.face_recon, gen_imgs, self.trans_params,
            self.recon_t, self.recon_s)
        # self.lm_pred = self.face_recon.pred_lm
        d_recon, m_recon = self.min_max_normalization(d_recon, m_recon)
        return d_recon, m_recon

    def z_to_depth(self, subset_z):
        d_input, m_input = z2depth(self.face_recon, subset_z)
        d_input, m_input = self.min_max_normalization(d_input, m_input)
        return d_input, m_input

    def z_to_pigan_depth(self, subset_z, yaw, pitch):
        d_input, m_input = z2pigandepth(self.face_recon, subset_z, yaw, pitch)
        # depth_numpy = d_input.detach().cpu().permute(0,2,3,1).expand(-1,-1,-1,3).numpy()
        # mask_numpy = m_input.detach().cpu().permute(0,2,3,1).expand(-1,-1,-1,3).numpy()
        # depth_numpy = mask_numpy*depth_numpy+(1-mask_numpy)*np.median(depth_numpy[mask_numpy.astype(np.bool8)])
        # depth_numpy = min_max_norm(depth_numpy) * 255.
        # cv2.imwrite('debug/pigan_depth_1.png', depth_numpy[0])
        # # d_input, m_input = self.min_max_normalization(d_input, m_input)
        d_input = resize(d_input, self.metadata['img_size'])
        # m_input = resize(m_input, self.metadata['img_size'])
        m_input = d_input > 0
        # print("")
        # depth_numpy = d_input.detach().cpu().permute(0,2,3,1).expand(-1,-1,-1,3).numpy()
        # mask_numpy = m_input.detach().cpu().permute(0,2,3,1).expand(-1,-1,-1,3).numpy()
        # depth_numpy = mask_numpy*depth_numpy+(1-mask_numpy)*np.median(depth_numpy[mask_numpy.astype(np.bool8)])
        # depth_numpy = min_max_norm(depth_numpy) * 255.
        # cv2.imwrite('debug/pigan_depth_2.png', depth_numpy[0])
        return d_input, m_input

    def _init_g_losses(self):
        self.rel_d_loss = 0
        self.depth_smooth_laplacian_loss = 0
        self.depth_smooth_edge_aware_loss = 0
        self.variance_loss = 0
        self.recon_loss = 0
        self.lm_loss = 0
        self.lm3d_loss = 0
        self.depth_g_loss = 0
        self.contrastive_id_loss = 0
        self.contrastive_geo_loss = 0
        self.contrastive_tex_loss = 0
        self.contrastive_tex_gram_loss = 0
        self.contrastive_face_loss = 0
        self.imitative_texture_loss = 0
        self.warping3d_loss = 0
        self.exp_warping_loss = 0
        self.dist_depr = 0
        self.bg_depr = 0
        self.norm_reg = 0
        self.geo_reg = 0
        self.depth_consistency_loss = 0

    def cal_g_loss(self):
        self._init_g_losses()
        if self.using_recon_loss:
            self.logger(
                'using msra19 reconstrunction loss!',
                info_level="debug")
            self.cal_recon_loss()

        if self.using_lm_loss:
            self.cal_lm_loss()

        if self.using_lm3d_loss:
            self.cal_lm3d_loss()

        if self.using_depth_d:
            self.cal_depth_d_loss()

        if self.using_rel_depth_consistency:
            self.cal_rel_depth_consistency()

        if self.using_depth_smooth_laplacian_loss:
            self.cal_depth_smooth_laplacian_loss()
            # depth_smooth_laplacian_loss = (
            #   torch.cuda.FloatTensor(1).fill_(0))

        if self.using_depth_smooth_edge_aware_loss:
            self.cal_depth_smooth_edge_aware_loss()

        if self.using_variance_loss:
            self.cal_variance_loss()

        if self.using_itex_loss:
            self.cal_itex_loss()

        if self.using_cid_loss and self.split == self.metadata['batch_split']:
            self.cal_cid_loss()

        if self.using_cgeo_loss and self.split == self.metadata['batch_split']:
            self.cal_cgeo_loss()

        if self.using_ctex_loss and self.split == self.metadata['batch_split']:
            self.cal_ctex_loss()

        if (self.using_ctex_gram_loss
                and self.split == self.metadata['batch_split']):
            self.cal_ctex_gram_loss()

        if (self.using_cface_loss
                and self.split == self.metadata['batch_split']):
            self.cal_cface_loss()

        if self.using_depth_consistency_loss:
            self.cal_depth_consistency_loss()

        if self.using_dist_depr or self.using_bg_depr:
            self.cal_depr()

        if self.using_norm_reg:
            self.cal_norm_reg()

        if self.using_geo_reg:
            self.cal_geo_reg()

        # integrate g_loss and bp gradient
        self.g_loss = (
            torch.nn.functional.softplus(-self.g_preds).mean()
            + self.identity_penalty
            + self.tddfa_loss + self.spade_g_loss + self.depth_g_loss
            + self.rel_d_loss + self.variance_loss
            + self.depth_smooth_laplacian_loss
            + self.depth_smooth_edge_aware_loss
            + self.recon_loss + self.lm_loss + self.lm3d_loss
            + self.contrastive_id_loss
            + self.contrastive_geo_loss
            + self.contrastive_tex_loss + self.contrastive_tex_gram_loss
            + self.contrastive_face_loss + self.imitative_texture_loss
            + self.dist_depr + self.bg_depr + self.norm_reg + self.geo_reg
            + self.depth_consistency_loss
            + self.exp_warping_loss
            + self.warping3d_loss
        )

    def cal_recon_loss(self):
        recon_loss_keys = self.metadata.get(
            'recon_loss_keys', ['id', 'exp', 'tex', 'gamma'])
        recon_norm_mode = self.metadata.get(
            'recon_norm_mode', 'denorm')
        self.recon_split = self.metadata.get('recon_split', False)

        if self.recon_split:
            # print('split recon loss!')
            recon_lambbdas = self.metadata.get(
                'recon_lambdas', {
                    'id': 0.25,
                    'exp': 0.25,
                    'tex': 0.25,
                    'gamma': 0.25
                })
            recon_losses = self.face_recon.split_recon_loss(
                recon_loss_keys, recon_norm_mode)
            recon_id_loss = recon_losses['id'] * recon_lambbdas['id']
            recon_exp_loss = recon_losses['exp'] * recon_lambbdas['exp']
            recon_tex_loss = recon_losses['tex'] * recon_lambbdas['tex']
            recon_gamma_loss = recon_losses['gamma'] * recon_lambbdas['gamma']
            # print("---")
            # print("recon_id_loss", recon_id_loss)
            # print("recon_exp_loss", recon_exp_loss)
            # print("recon_tex_loss", recon_tex_loss)
            # print("recon_gamma_loss", recon_gamma_loss)
            self.recon_loss = (
                recon_id_loss + recon_exp_loss +
                recon_tex_loss + recon_gamma_loss
            ) * self.metadata['recon_lambda']
        else:
            self.recon_loss = self.face_recon.cal_recon_loss(
                recon_loss_keys,
                recon_norm_mode) * self.metadata['recon_lambda']

    def cal_lm_loss(self):
        
        lm_input = self.generator_ddp.module.lm_input.detach().clip(0, 255)
        lm_pred = self.face_recon.inv_affine_ldmks_torch_(
            self.face_recon.pred_lm).clip(0, 255)
        self.lm_loss = torch.nn.L1Loss()(lm_pred, lm_input) * self.lm_lambda

        if self.rank == 0 and self.discriminator.step % self.opt.sample_interval == 0:
            imgs_vis = []
            for save_index in range(self.gen_imgs.shape[0]):
                gen_imgs = (self.gen_imgs+1)/2
                gen_imgs_256 = nn.functional.interpolate(
                    gen_imgs, size=(256, 256), mode='bilinear')
                img_vis = gen_imgs_256[save_index].permute(
                    1,2,0).detach().cpu().numpy()*255

                lm_input_vis = lm_input.cpu().numpy()
                lm_pred_vis = lm_pred.detach().cpu().numpy()

                lm_input_vis[save_index][:, 1] = 255 - lm_input_vis[save_index][:, 1]
                img_vis = draw_landmarks(img_vis, lm_input_vis[save_index], (0, 255, 0))

                lm_pred_vis[save_index][:, 1] = 255 - lm_pred_vis[save_index][:, 1]
                img_vis = draw_landmarks(img_vis, lm_pred_vis[save_index], (255, 0, 0))
                imgs_vis.append(img_vis)
            imgs_vis = np.concatenate(imgs_vis, axis=1)

            self.tensorboard_writer.add_image(
                'ldmk: green-input, red-pred',
                imgs_vis/255., self.discriminator.step, dataformats='HWC')

    def cal_lm3d_loss(self):
        lm_input = self.generator_ddp.module.lm_input.detach().clip(0, 255)
        lm_pred = self.face_recon.inv_affine_ldmks_torch_(
            self.face_recon.pred_lm).clip(0, 255)
        lm_loss_2d = torch.nn.L1Loss()(lm_pred, lm_input)

        lm3d_input = self.generator_ddp.module.lm3d_input.detach()
        ldmk2d = self.face_recon.inv_affine_ldmks_torch_(
            self.face_recon.pred_lm).clip(0, 255)
        ldmk2d = (ldmk2d/255*self.img_size).round().long()

        ldmk_depths = []
        for b in range(self.gen_depth.shape[0]):
            ldmk_depths.append(self.gen_depth[b][..., ldmk2d[b, :, 1], ldmk2d[b, :, 0]].unsqueeze(-1))
        ldmk_depths = torch.cat(ldmk_depths, dim=0)
        ldmk2d_c = ldmk2d - self.img_size/2
        ldmk3d = ldmk2d_c / self.focal_length * ldmk_depths

        ldmk3d = torch.cat([ldmk3d, 1-ldmk_depths], dim=-1)
        ldmk3d = self.face_recon.cam2pigan(ldmk3d)

        lm_loss_3d = torch.nn.L1Loss()(
            ldmk3d[:, 17:, :], lm3d_input[:, 17:, :])

        self.lm3d_loss = (lm_loss_2d + lm_loss_3d) * self.lm3d_lambda

    def cal_warping3d_loss(self, bs):
        # print("cal 3dwarping_loss...")
        self.data = {}
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
            os.makedirs('./warping_imgs', exist_ok=True)
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
        self.data = {}
        # torch.manual_seed(self.debug_counter+1)
        self.debug_counter += 1
        zs = torch.randn(
            (bs*2, self.metadata['latent_dim']), device=self.device)
        zs[1::2, :self.dim_id] = zs[::2, :self.dim_id]
        zs[1::2, (self.dim_id+self.dim_exp):] = zs[::2, (self.dim_id+self.dim_exp):]

        yaws, pitchs = sample_yaw_pitch(
            bs*2, self.h_mean, self.v_mean,
            self.h_stddev, self.v_stddev, self.device)
        yaws[1::2] = yaws[::2]
        pitchs[1::2] = pitchs[::2]

        gen_outputs = self.generator_ddp(
            zs, return_depth=self.requiring_depth,
            yaw=yaws, pitch=pitchs,
            face_recon=self.face_recon,
            using_dist_depr=self.using_depr,
            using_norm_reg=self.using_norm_reg,
            narrow_ratio=self.narrow_ratio,
            **self.metadata)
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

    def diff_Zs_Ze(self, z0, z1):
        z1[:, (self.dim_id+self.dim_exp):] = z0[:, (self.dim_id+self.dim_exp):]
        return z0, z1

    def diff_Ze(self, z0, z1):
        z1[..., :self.dim_id] = z0[..., :self.dim_id]
        z1[:, (self.dim_id+self.dim_exp):] = z0[:, (self.dim_id+self.dim_exp):]
        return z0, z1

    def generate_warping_img(self, bs=1):

        torch.manual_seed(self.debug_counter+1)
        self.debug_counter += 1
        zs = torch.randn(
            (bs*2, self.metadata['latent_dim']), device=self.device)
        zs[1::2, :self.dim_id] = zs[::2, :self.dim_id]
        zs[1::2, (self.dim_id+self.dim_exp):] = zs[::2, (self.dim_id+self.dim_exp):]
        z0, z1 = zs[::2], zs[1::2]

        cfg = self.metadata
        cfg['nerf_noise'] = 0
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

            flow_2to1 = flow[1::2]
            flow_2to1 = flow_2to1.to(self.device)
            flow_2to1 = torch.nn.functional.interpolate(
                flow_2to1, (self.img_size, self.img_size))
            flow_2to1 = flow_2to1.permute(0,2,3,1).contiguous().unsqueeze(-2).repeat(1,1,1,12,1).contiguous().reshape(1,64*64*12,3)

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

        img1.requires_grad_()
        depth1.requires_grad_()

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

    def cal_depth_d_loss(self):
        depth_g_loss = {}
        fake_concat = torch.cat([self.d_input, self.gen_depth], dim=1)
        real_concat = torch.cat([self.d_recon, self.gen_depth], dim=1)
        fake_and_real = torch.cat(
            [fake_concat, real_concat], dim=0)  # 2, 3, 128, 128

        discriminator_out = self.depth_discriminator_ddp(
            fake_and_real)
        pred_fake, pred_real = divide_pred(discriminator_out)
        depth_g_loss['GAN'] = self.criterionGAN(
            pred_fake, True, for_discriminator=False)
        self.depth_g_loss = sum(
            depth_g_loss.values()
        ).mean() * self.metadata['depth_g_lambda']

    def cal_rel_depth_consistency(self):
        input = {
            'depth_3dmm': self.d_input,
            'depth_pigan': self.gen_depth,
            'mask': self.m_input
        }
        self.rel_d_loss = self.rel_depth_lambda * self.rel_depth_loss(input)

    def cal_depth_smooth_laplacian_loss(self):
        self.depth_smooth_laplacian_loss = self.laplacianer(
            self.gen_depth) * self.depth_smooth_laplacian_lambda

    def cal_depth_smooth_edge_aware_loss(self):
        self.depth_smooth_edge_aware_loss = self.edge_aware_smoother(
                self.gen_imgs,
                self.gen_depth) * self.depth_smooth_edge_aware_lambda

    def cal_variance_loss(self):
        self.variance_loss = (
            self.variance_loss_lambda * self.gen_depth_variance.mean())

    def cal_depth_consistency_loss(self):
        d_con = torch.abs(self.gen_depth - self.face_recon.d_pigan_input)
        d_con = d_con[self.face_recon.m_pigan_input].mean()
        self.depth_consistency_loss = d_con * self.depth_consistency_lambda

    def cal_itex_loss(self):
        recon_tex = self.face_recon.input_face
        mask = self.face_recon.input_mask
        mask_padding = self.face_recon.inv_affine_tensors(mask) >= 1.0
        mask = mask_padding.int()
        recon_tex = self.face_recon.inv_affine_tensors(recon_tex)

        gen_imgs = min_max_norm(self.gen_imgs)
        gen_imgs = nn.functional.interpolate(
            gen_imgs, recon_tex.shape[-2:], mode='bicubic')

        if self.rank == 0:
            cat = torch.cat([gen_imgs, recon_tex, mask], dim=-1)
            img2tb(
                self.tensorboard_writer, 'gen_imgs - recon_tex - mask',
                cat, self.discriminator.step)

        self.imitative_texture_loss = self.itex_loss(
            gen_imgs, recon_tex, mask
        ) * self.itex_lambda

    def cal_cid_loss(self):
        idx = self.contrastive_loss_index['id']
        indices_tensor = torch.tensor(
            [0, idx, idx+1], device=self.device)
        imgs = self.gen_imgs[indices_tensor]
        trans_m = estimate_norm_torch(
            self.pred_lm[indices_tensor], self.gen_imgs.shape[-2])
        input = {
            'imgs': imgs,
            'trans_m': trans_m,
            "shape_input": self.z_input_dict['id'][indices_tensor],
            "tex_input": self.z_input_dict['tex'][indices_tensor],
            'shape':
                self.face_recon.pred_norm_dict[
                    'id'][indices_tensor],
            'tex':
                self.face_recon.pred_norm_dict[
                    'tex'][indices_tensor],
            'p_lambda': self.cid_p_lambda,
            'c_lambda': self.cid_c_lambda
        }
        self.contrastive_id_loss = self.cid_loss(input) * self.cid_lambda

    def cal_cgeo_loss(self):
        idx = self.contrastive_loss_index['geo']
        indices_tensor = torch.tensor(
            [0, idx, idx+1], device=self.device)
        depths = self.gen_depth[indices_tensor]
        self.contrastive_geo_loss = self.cgeo_loss(depths) * self.cgeo_lambda

    def cal_ctex_loss(self):
        idx = self.contrastive_loss_index['tex']
        indices_tensor = torch.tensor(
            [0, idx, idx+1], device=self.device)
        input = {
            "gamma_input":
                self.z_input_dict['gamma'][indices_tensor],
            "tex_input":
                self.z_input_dict['tex'][indices_tensor],
            'gamma':
                self.face_recon.pred_norm_dict['gamma'][indices_tensor],
            'tex':
                self.face_recon.pred_norm_dict['tex'][indices_tensor],
            'p_lambda': self.ctex_p_lambda
        }
        # contrastive_tex_loss = ctex_loss(
        #     tex, gm) * ctex_lambda
        self.contrastive_tex_loss = self.ctex_loss(input) * self.ctex_lambda

    def cal_ctex_gram_loss(self):
        idx = self.contrastive_loss_index['tex_gram']
        indices_tensor = torch.tensor(
            [0, idx, idx+1], device=self.device)
        vgg_feature = self.ctex_gram_loss.vgg(
            self.gen_imgs[indices_tensor])
        input = {
            # "gamma_input":
            #   z_input_dict['gamma'][indices_tensor],
            # "tex_input":
            #   z_input_dict['tex'][indices_tensor],
            # "bg_tex_input":
            #   z_input_dict['bg_tex'][indices_tensor],
            # 'gamma':
            #   face_recon.pred_norm_dict['gamma'][indices_tensor],
            # 'tex':
            #   face_recon.pred_norm_dict['tex'][indices_tensor],
            # "bg_tex":
            #   face_recon.pred_norm_dict['bg_tex'][indices_tensor]
            "vgg_feature": vgg_feature
        }
        # contrastive_tex_loss = (
        #   ctex_loss(tex, gm) * ctex_lambda)
        self.contrastive_tex_gram_loss = (
            self.ctex_gram_loss(input) * self.ctex_lambda)

    def cal_cface_loss(self):
        idx = self.contrastive_loss_index['face']
        indices_tensor = torch.tensor(
            [0, idx, idx+1], device=self.device)
        imgs_face = self.gen_imgs[indices_tensor]
        masks_face = self.m_recon[indices_tensor]
        depths_face = self.d_recon[indices_tensor]
        self.contrastive_face_loss = self.cface_loss(
            imgs_face, masks_face, depths_face) * self.cface_lambda

    def cal_depr(self):
        ray_start = self.metadata['ray_start']
        ray_end = self.metadata['ray_end']
        img_size = self.metadata['img_size']
        thick_ratio = self.metadata['thick_ratio']
        thick_ratio = 1.0 + (thick_ratio - 1.0) * self.w_narrow_ratio

        weights = self.generator_ddp.module.all_weights
        points_cam = self.generator_ddp.module.all_points_cam
        points = self.generator_ddp.module.all_points
        m_input = self.generator_ddp.module.depr_m_input
        d_input = self.generator_ddp.module.depr_d_input
        volume_mask = self.generator_ddp.module.volume_mask
        yaw = self.generator_ddp.module.depr_yaws
        pitch = self.generator_ddp.module.depr_pitch
        # bg_depths = self.generator_ddp.module.bg_depths
        # bg_depths = torch.nn.functional.interpolate(
        #     bg_depths, size=(img_size, img_size))
        d_input[torch.logical_not(m_input)] = torch.mean(d_input[m_input])

        batch_size = weights.shape[0]
        if volume_mask is None:
            volume_mask = get_volume_mask(
                points_cam, m_input, ray_start, ray_end)
        # depth_map = d_input * m_input + bg_depths * torch.logical_not(m_input)
        # d_input_flatten = depth_map.reshape(
        d_input_flatten = d_input.reshape(
            batch_size, img_size*img_size, 1, 1)
        z_diff = (-points_cam[..., -1:]) - d_input_flatten
        z_diff_fg = z_diff[volume_mask].detach()

        if self.using_dist_depr:
            weights_fg = weights[volume_mask]
            mesh_thick = (ray_end - ray_start) * thick_ratio
            dist_depr = self.dist_depr_fn(weights_fg, z_diff_fg, mesh_thick)
            self.dist_depr = dist_depr.mean() * self.dist_depr_lambda * self.narrow_ratio

            if (self.discriminator.step % self.opt.model_save_interval == 0
                    and self.rank == 0):
                z_diff_crop_0 = z_diff[0][volume_mask[0]].detach()
                color = z_diff_crop_0.unsqueeze(-1).repeat((1, 3))
                position = points_cam[0][volume_mask[0].squeeze(-1)]
                # vis_tensor_as_vert_color(color, position, 'z_diff.obj')
                weights_crop_0 = weights[0][volume_mask[0]].flatten()
                color = weights_crop_0.unsqueeze(-1).repeat((1, 3))
                # vis_tensor_as_vert_color(color, position, 'weights.obj')
                dist_depr_0 = self.dist_depr_fn(
                    weights_crop_0, z_diff_crop_0, mesh_thick)
                color = dist_depr_0.unsqueeze(-1).repeat((1, 3))
                # vis_tensor_as_vert_color(color, position, 'weights_depr.obj')

        if self.using_bg_depr:
            min_d_input = torch.min(d_input[m_input])
            d_bg = -points_cam[..., -1:][torch.logical_not(volume_mask)]
            weights_bg = weights[torch.logical_not(volume_mask)]
            d_bg_diff = (min_d_input-ray_start)*0.9 + ray_start - d_bg
            bg_depr_mask = d_bg_diff > 0
            if torch.any(bg_depr_mask):
                bg_depr = weights_bg[bg_depr_mask] * d_bg_diff[bg_depr_mask]
            else:
                bg_depr = torch.tensor(0.)

            max_face_z = torch.max(points[..., -1:][volume_mask])
            z_bg = points[..., -1:][torch.logical_not(volume_mask)]
            z_bg_diff = z_bg - max_face_z
            bg_z_depr_mask = z_bg_diff > 0
            if torch.any(bg_z_depr_mask):
                bg_z_depr = weights_bg[bg_z_depr_mask] * z_bg_diff[bg_z_depr_mask]
            else:
                bg_z_depr = torch.tensor(0.)

            self.bg_depr = (
                bg_depr.mean() + bg_z_depr.mean()) * self.bg_depr_lambda

    def dist_depr_fn(self, weights_crop, z_diff_crop, mesh_thick):
        w = torch.nn.functional.relu(weights_crop)
        z = torch.nn.functional.relu(
            torch.abs(z_diff_crop)-mesh_thick/2)

        dist_depr = (w * (torch.exp(20*z) - 1))
        # dist_depr = (w * z)
        return dist_depr

    def cal_norm_reg(self):
        normals = self.generator_ddp.module.pred_normals[:, :, 0, :]
        neighbor_normals = self.generator_ddp.module.pred_normals[:, :, 1, :]
        self.norm_reg = torch.nn.MSELoss()(normals, neighbor_normals) * self.norm_reg_lambda

    def cal_geo_reg(self):

        self.gen_depth.requires_grad_()
        if len(self.gen_depth.shape) == 3:
            self.gen_depth = self.gen_depth.unsqueeze(1)
        v00 = self.gen_depth[..., :-1, :-1]
        v01 = self.gen_depth[..., :-1, 1:]
        v10 = self.gen_depth[..., 1:, :-1]
        loss = ((v00 - v01) ** 2) + ((v00 - v10) ** 2)
        # loss = torch.abs(v00 - v01) + torch.abs(v00 - v10)

        geo_reg_mask = torch.logical_not(
            self.face_recon.m_pigan_input[..., :-1, :-1])
        geo_reg_bg = loss[geo_reg_mask].sum() / (geo_reg_mask.sum()+1e-8)
        geo_reg = loss.mean()

        self.geo_reg = (geo_reg + geo_reg_bg) * self.geo_reg_lambda

    def generator_train_one_split(self):

        self.gen_img_for_G()
        # fake_concat = torch.cat([gt_depths, gen_imgs], dim=1)
        self.run_D_for_G()

        # calculate identity penalty
        self.cal_id_penalty(self.subset_z)

        # calculate tddfa penalty
        self.cal_tddfa_penalty()

        # calculate spade g loss
        self.cal_spade_penalty()

        if self.requiring_depth:
            # self.render_depth()
            (
                self.d_recon, self.m_recon,
                self.pred_lm
            ) = self.render_recon_depth(self.gen_imgs)
            # self.face_recon.compute_ori_visuals('pred')
            # self.face_recon.compute_ori_visuals('input')

        self.cal_g_loss()

        self.scaler.scale(self.g_loss).backward()
        if self.using_cid_loss:
            self.cid_loss.net_recog_ddp.zero_grad()
        # print('g_loss.bp')
        # embed()

    def contrastive_loss_one_split(self):
        contrastive_z = self.subset_z.clone()
        contrastive_yaw = self.subset_yaw.clone()
        contrastive_pitch = self.subset_pitch.clone()
        if len(contrastive_z) != 3:
            warnings.warn(
                "The contrastive_z should have a batchsize of 3!")
        contrastive_z[1, self.dim_t:] = contrastive_z[0, self.dim_t:]
        contrastive_z[2, :self.dim_t] = contrastive_z[0, :self.dim_t]
        contrastive_yaw = contrastive_yaw[0:1].repeat([3, 1])
        contrastive_pitch = contrastive_pitch[0:1].repeat([3, 1])
        contrastive_imgs, contrastive_positions, depths = self.generator_ddp(
            contrastive_z,
            yaw=contrastive_yaw,
            pitch=contrastive_pitch,
            delt_v=self.subset_delt_v,
            return_depth=True,
            **self.metadata)
        depths.requires_grad_()
        contrastive_imgs.requires_grad_()
        depthloss_type = self.metadata.get(
            'depthloss_type', 'l1loss').lower()
        if depthloss_type in ['l1loss', 'l1_loss']:
            depth_loss = l1loss()(depths[0], depths[1])
        elif depthloss_type in ['wingloss', 'wing_loss']:
            depth_loss = WingLoss()(depths[0], depths[1])

        style_loss = self.criterionVGG.cal_style_loss(
            contrastive_imgs[0:1], contrastive_imgs[2:3])
        contrastive_loss = (
            depth_loss * self.depth_lambda
            + style_loss * self.style_lambda)
        self.scaler.scale(contrastive_loss).backward()

    def run_generator_one_step(self):
        self.sample_z_for_G()
        self.sample_yaw_pitch_for_G()

        for self.split in range(
                self.metadata['batch_split'] +
                self.extra_split_for_contrastive):
            self.logger(f'split: {self.split}', info_level='debug')

            self.generator_train_one_split()
            # Cal contrastive loss
            # To avoid OOM,
            # compute contrastive loss after g_loss.backward().
            if self.using_contrastive_loss:
                self.contrastive_loss_step()

            if self.using_warping3d_loss:
                self.cal_warping3d_loss(bs=1)
                self.scaler.scale(self.warping3d_loss).backward()

            if self.using_exp_warping_loss:
                self.cal_exp_warping_loss(bs=1)
                self.scaler.scale(self.exp_warping_loss).backward()

        # optimize G
        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(
            self.generator_ddp.parameters(),
            self.metadata.get('grad_clip', 0.3))
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        # self.scaler.unscale_(self.optimizer_W)
        # torch.nn.utils.clip_grad_norm_(
        #     self.warpformer_ddp.parameters(),
        #     self.metadata.get('grad_clip', 0.3))
        # self.scaler.step(self.optimizer_W)
        # self.scaler.update()

        self.optimizer_G.zero_grad()
        # self.optimizer_W.zero_grad()

        self.ema.update(
            [p for p in self.generator_ddp.parameters() if p.requires_grad])
        self.ema2.update(
            [p for p in self.generator_ddp.parameters() if p.requires_grad])

        # for v in self.generator_ddp.parameters():
        #     pass
