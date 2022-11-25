from packages import *
from tools.train.pigan_wrapper import PiganWrapper


def cleanup():
    dist.destroy_process_group()


class PiganTrainer(PiganWrapper):
    """main trianing pipeline
    """
    def __init__(self, rank, world_size, opt):
        super(PiganTrainer, self).__init__(rank, world_size, opt)

    def train(self):
        torch.manual_seed(self.rank)
        self.dataloader = None
        self.total_progress_bar = tqdm(total=self.opt.n_epochs,
                                       desc="Total progress",
                                       dynamic_ncols=True)
        self.total_progress_bar.update(self.discriminator.epoch)
        self.interior_step_bar = tqdm(dynamic_ncols=True)
        self.criterionGAN = GANLoss(
            gan_mode='hinge', device=self.device).to(self.device)
        self.criterionVGG = VGGLoss(self.rank).to(self.device)
        self.L1Criterion = torch.nn.L1Loss()
        self.MSECriterion = torch.nn.MSELoss()
        self.debug_counter = 0

        # begin training
        for n_epoch in range(self.opt.n_epochs):
            self.logger(
                f'----------------- epoch: {n_epoch} -----------------',
                info_level='debug')

            self.epoch_step()

            if self.interior_step_bar.n > self.interior_step_bar.total:
                break
        self._terminate_all()

    def epoch_step(self):

        self.total_progress_bar.update(1)

        # update configs
        metadata = curriculums.extract_metadata(
            self.curriculum, self.discriminator.step)

        # update learning rates
        update_param(self.optimizer_G, metadata)
        update_param(self.optimizer_D, metadata)
        if self.using_spade_d:
            update_param(self.optimizer_spade_D, metadata)
        if self.using_depth_d:
            update_param(self.optimizer_depth_D, metadata)

        # update dataloader or batchsize
        if (not self.dataloader
                or self.dataloader.batch_size != metadata['batch_size']):
            self.logger('update dataloader!')
            self.dataloader, CHANNELS = datasets.get_dataset_distributed(
                metadata['dataset'], self.world_size, self.rank, **metadata)

            self.step_next_upsample = curriculums.next_upsample_step(
                self.curriculum, self.discriminator.step)
            self.step_last_upsample = curriculums.last_upsample_step(
                self.curriculum, self.discriminator.step)

            self.interior_step_bar.reset(
                total=(self.step_next_upsample - self.step_last_upsample))
            self.interior_step_bar.set_description(f"Progress to next stage")
            self.interior_step_bar.update(
                (self.discriminator.step - self.step_last_upsample))

        # loop iterations.
        for self.i, (self.imgs, _) in enumerate(self.dataloader):
            # from IPython import embed; embed()
            self.logger(
                f'-------------------- iter: {self.i} --------------------',
                info_level='debug')
            if self.dataloader.batch_size != metadata['batch_size']:
                break

            # with torch.autograd.detect_anomaly():
            self.iter_step()

            self.interior_step_bar.update(1)
            if self.interior_step_bar.n > self.interior_step_bar.total:
                break

        self.discriminator.epoch += 1
        self.generator.epoch += 1

    def iter_step(self):
        # save models
        if (self.discriminator.step % self.opt.model_save_interval == 0
                and self.rank == 0):
            self.save_models()

        # update training setting
        self.update_training_setting()

        # move data to GPU
        self.real_imgs = self.imgs.to(self.device, non_blocking=True)

        # TRAIN DISCRIMINATOR
        self.run_discriminator_one_step()

        # TRAIN GENERATOR
        self.run_generator_one_step()

        # visualization
        if self.rank == 0:
            if self.i % 10 == 0:
                self.write_tb()
            if self.discriminator.step % self.opt.sample_interval == 0:
                self.visualize()
                self.save_models()

        # evaluate FID
        if (self.opt.eval_freq > 0 and self.discriminator.step > 0
                and self.discriminator.step % self.opt.eval_freq == 0
                or self.opt.fid):
            self.cal_fid()
            # self.eval_recon_error()
            torch.cuda.empty_cache()

        self.discriminator.step += 1
        self.generator.step += 1
        self.narrow_ratio = min(self.generator.step/(self.shrink_step_num+1e-9), 1)
        self.w_narrow_ratio = min(self.generator.step/(self.w_shrink_step_num+1e-9), 1)

    def eval_recon_error(self):

        (self.eval_d_recon, self.eval_m_recon,
         self.eval_d_input, self.eval_m_input,
         self.eval_pred_lm, self.eval_norm_input_z_dict
         ) = self.render_depth(self.eval_zs, self.eval_imgs)
        self.eval_input_coeff_dict = self.face_recon.get_input_coeff(
            self.eval_zs)

        self.face_recon.compute_ori_visuals('pred')
        self.face_recon.compute_ori_visuals('input')

        norm_z_error = cal_z_error(
            self.eval_norm_input_z_dict, self.face_recon.pred_norm_dict)
        denorm_z_error = cal_z_error(
            self.eval_input_coeff_dict, self.face_recon.pred_coeff_dict)
        shape_error = cal_mesh_error(
            self.face_recon.pred_shape,
            self.face_recon.input_shape
        )
        vertex_error = cal_mesh_error(
            self.face_recon.pred_vertex,
            self.face_recon.input_vertex
        )
        return norm_z_error, denorm_z_error, shape_error, vertex_error

    def save_models(self):
        now = datetime.now()
        now = now.strftime("%d--%H:%M--")
        torch.save(
            self.ema,
            os.path.join(self.opt.output_dir, now + 'ema.pth'))
        torch.save(
            self.ema2, os.path.join(self.opt.output_dir,
                                    now + 'ema2.pth'))
        torch.save(
            self.generator_ddp.module,
            os.path.join(self.opt.output_dir, now + 'generator.pth'))
        torch.save(
            self.discriminator_ddp.module,
            os.path.join(self.opt.output_dir, now + 'discriminator.pth'))
        torch.save(
            self.optimizer_G.state_dict(),
            os.path.join(self.opt.output_dir, now + 'optimizer_G.pth'))
        torch.save(
            self.optimizer_D.state_dict(),
            os.path.join(self.opt.output_dir, now + 'optimizer_D.pth'))
        torch.save(
            self.scaler.state_dict(),
            os.path.join(self.opt.output_dir, now + 'scaler.pth'))
        if self.using_spade_d:
            torch.save(
                self.spade_discriminator_ddp.module,
                os.path.join(self.opt.output_dir,
                             now + 'spade_discriminator.pth'))
        if self.using_depth_d:
            torch.save(
                self.depth_discriminator_ddp.module,
                os.path.join(self.opt.output_dir,
                             now + 'depth_discriminator.pth'))

    def write_tb(self):
        tqdm.write(
            f"[Experiment: {self.opt.output_dir}]"
            f"[GPU: {os.environ['CUDA_VISIBLE_DEVICES']}]"
            f"[Epoch: {self.discriminator.epoch}/{self.opt.n_epochs}]"
            f"[D loss: {self.d_loss.item()}]"
            f"[G loss: {self.g_loss.item()}]"
            f"[Step: {self.discriminator.step}] [Alpha: {self.alpha:.2f}]"
            f"[Img Size: {self.metadata['img_size']}]"
            f"[Batch Size: {self.metadata['batch_size']}]"
            f"[TopK: {self.topk_num}]"
            f"[Scale: {self.scaler.get_scale()}]"
        )
        # tensorboard_writer.add_scalar('depth_penalty',depth_penalty.item(),global_step=discriminator.step)
        self.tensorboard_writer.add_scalar(
            'latent_penalty',
            self.latent_penalty.item(),
            global_step=self.discriminator.step)
        self.tensorboard_writer.add_scalar(
            'position_penalty',
            self.position_penalty.item(),
            global_step=self.discriminator.step)
        self.tensorboard_writer.add_scalar(
            'g_loss',
            self.g_loss.item(),
            global_step=self.discriminator.step)
        self.tensorboard_writer.add_scalar(
            'd_loss',
            self.d_loss.item(),
            global_step=self.discriminator.step)

        if self.using_tddfa_loss and self.tddfa_loss != (
                torch.cuda.FloatTensor(1).fill_(0)):
            self.tensorboard_writer.add_scalar(
                'tddfa_loss',
                self.tddfa_loss.item(),
                global_step=self.discriminator.step)
        if self.using_recon_loss and self.recon_loss != 0:
            self.tensorboard_writer.add_scalar(
                'recon_loss',
                self.recon_loss.item(),
                global_step=self.discriminator.step)
            if self.recon_split:
                self.tensorboard_writer.add_scalar(
                    'recon_id_loss',
                    self.recon_id_loss.item(),
                    global_step=self.discriminator.step)
                self.tensorboard_writer.add_scalar(
                    'recon_exp_loss',
                    self.recon_exp_loss.item(),
                    global_step=self.discriminator.step)
                self.tensorboard_writer.add_scalar(
                    'recon_tex_loss',
                    self.recon_tex_loss.item(),
                    global_step=self.discriminator.step)
                self.tensorboard_writer.add_scalar(
                    'recon_gamma_loss',
                    self.recon_gamma_loss.item(),
                    global_step=self.discriminator.step)
        if self.using_lm_loss:
            self.tensorboard_writer.add_scalar(
                'lm_loss',
                self.lm_loss.item(),
                global_step=self.discriminator.step)
        if self.using_lm3d_loss:
            self.tensorboard_writer.add_scalar(
                'lm3d_loss',
                self.lm3d_loss.item(),
                global_step=self.discriminator.step)
        if self.depth_lambda > 0:
            self.tensorboard_writer.add_scalar(
                'depth_loss',
                self.depth_loss.item(),
                global_step=self.discriminator.step)
        if self.style_lambda > 0:
            self.tensorboard_writer.add_scalar(
                'style_loss',
                self.style_loss.item(),
                global_step=self.discriminator.step)
        if self.using_depth_d:
            self.tensorboard_writer.add_scalar(
                'depth_g_loss',
                self.depth_g_loss.item(),
                global_step=self.discriminator.step)
            self.tensorboard_writer.add_scalar(
                'depth_d_loss',
                self.depth_d_loss.item(),
                global_step=self.discriminator.step)
        if self.using_depth_smooth_laplacian_loss:
            self.tensorboard_writer.add_scalar(
                'depth_smooth_laplacian_loss',
                self.depth_smooth_laplacian_loss.item(),
                global_step=self.discriminator.step)
        if self.using_depth_smooth_edge_aware_loss:
            self.tensorboard_writer.add_scalar(
                'depth_smooth_edge_aware_loss',
                self.depth_smooth_edge_aware_loss.item(),
                global_step=self.discriminator.step)
        if self.using_rel_depth_consistency:
            self.tensorboard_writer.add_scalar(
                'rel_depth_consistency_loss',
                self.rel_d_loss.item(),
                global_step=self.discriminator.step)
        if self.using_variance_loss:
            self.tensorboard_writer.add_scalar(
                'variance_loss',
                self.variance_loss.item(),
                global_step=self.discriminator.step)
        if self.using_cid_loss:
            self.tensorboard_writer.add_scalar(
                'contrastive_id_loss',
                self.contrastive_id_loss.item(),
                global_step=self.discriminator.step)
        if self.using_cgeo_loss:
            self.tensorboard_writer.add_scalar(
                'contrastive_geo_loss',
                self.contrastive_geo_loss.item(),
                global_step=self.discriminator.step)
        if self.using_ctex_loss:
            self.tensorboard_writer.add_scalar(
                'contrastive_tex_loss',
                self.contrastive_tex_loss.item(),
                global_step=self.discriminator.step)
        if self.using_ctex_gram_loss:
            self.tensorboard_writer.add_scalar(
                'contrastive_tex_gram_loss',
                self.contrastive_tex_gram_loss.item(),
                global_step=self.discriminator.step)
        if self.using_cface_loss:
            self.tensorboard_writer.add_scalar(
                'contrastive_face_loss',
                self.contrastive_face_loss.item(),
                global_step=self.discriminator.step)
        if self.using_itex_loss:
            self.tensorboard_writer.add_scalar(
                'imitative_tex_loss',
                self.imitative_texture_loss.item(),
                global_step=self.discriminator.step)
        if self.using_warping3d_loss:
            self.tensorboard_writer.add_scalar(
                'warping3d_loss',
                self.warping3d_loss.item(),
                global_step=self.discriminator.step)
        if self.using_exp_warping_loss:
            self.tensorboard_writer.add_scalar(
                'exp_warping_loss',
                self.exp_warping_loss.item(),
                global_step=self.discriminator.step)
        if self.using_dist_depr:
            self.tensorboard_writer.add_scalar(
                'dist_depr',
                self.dist_depr.item(),
                global_step=self.discriminator.step)
        if self.using_bg_depr:
            self.tensorboard_writer.add_scalar(
                'bg_depr',
                self.bg_depr.item(),
                global_step=self.discriminator.step)
        if self.using_norm_reg:
            self.tensorboard_writer.add_scalar(
                'norm_reg',
                self.norm_reg.item(),
                global_step=self.discriminator.step)
        if self.using_geo_reg:
            self.tensorboard_writer.add_scalar(
                'geo_reg',
                self.geo_reg.item(),
                global_step=self.discriminator.step)
        if self.using_depth_consistency_loss:
            self.tensorboard_writer.add_scalar(
                'depth_consistency',
                self.depth_consistency_loss.item(),
                global_step=self.discriminator.step)

    def visualize(self):
        self.generator_ddp.eval()

        if self.requiring_depth:
            visual_face_recon = self.face_recon.visual_imgs
            self.visualizer.tb_current_results(
                self.tensorboard_writer,
                visual_face_recon,
                self.discriminator.step)

        # Evaluate generator and save images.
        eval_modes = ["fixed", "tilted"]
        if self.requiring_depth:
            eval_modes.append("ctrl")
        for mode in eval_modes:
            self.eval(
                z=self.fixed_z,
                config=self.metadata,
                opt=self.opt,
                mode=mode,
                global_step=self.discriminator.step,
                face_recon=self.face_recon,
                sample_near_mesh=self.sample_near_mesh,
                tensorboard_writer=self.tensorboard_writer)

        # Evaluate ema and save images.
        eval_modes = ["ema_fixed", "ema_tilted", "ema_random"]
        if self.requiring_depth:
            eval_modes.append("ema_ctrl")
        self.ema.store([
            p for p in self.generator_ddp.parameters()
            if p.requires_grad
        ])
        self.ema.copy_to([
            p for p in self.generator_ddp.parameters()
            if p.requires_grad
        ])
        self.generator_ddp.eval()
        for mode in eval_modes:
            self.eval(
                z=self.fixed_z,
                config=self.metadata,
                opt=self.opt,
                mode=mode,
                global_step=self.discriminator.step,
                face_recon=self.face_recon,
                tensorboard_writer=self.tensorboard_writer)
        self.ema.restore([
            p for p in self.generator_ddp.parameters()
            if p.requires_grad
        ])

    def eval(
            self, z, config, opt, mode, sr_g_ddp=None,
            global_step=0, face_recon=None, sample_near_mesh=False,
            tensorboard_writer=None):
        device = z.device
        copied_metadata = copy.deepcopy(config)
        copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
        copied_metadata['img_size'] = 128
        n_row = 5

        if mode in ['fixed', 'ema_fixed']:
            pass
        elif mode in ['tilted', 'ema_tilted']:
            copied_metadata['h_mean'] += 0.5
        elif mode in ['ctrl', 'ema_ctrl']:
            z = create_cmp_zs(z, 80, 64, 80, 80, 27, 80)
            n_row = 7
        elif mode in ['random', 'ema_random']:
            copied_metadata['psi'] = 0.7

        with torch.no_grad():
            zy_data = {}
            gen_imgs, gen_depths = self.generator_ddp.module.staged_forward(
                z.to(device), gt_depths=opt.save_depth, zy_data=zy_data,
                face_recon=face_recon,
                # sample_near_mesh=sample_near_mesh,
                using_dist_depr=self.using_depr,
                narrow_ratio=self.narrow_ratio,
                **copied_metadata)

        img_name = f"{global_step}_{mode}.png"
        dep_name = f"{global_step}_{mode}_gen_depth.png"
        save_image(gen_imgs[:len(z)], os.path.join(opt.output_dir, img_name), nrow=n_row, normalize=True)

        if tensorboard_writer is not None:
            img2tb(tensorboard_writer, mode, gen_imgs[:len(z)], global_step, nrow=n_row, normalize=True)

        if sr_g_ddp is not None:
            sr_imgs = sr_g_ddp(gen_imgs)
            sr_name = f"{global_step}_{mode}_sr.png"
            save_image(sr_imgs[:len(z)], os.path.join(opt.output_dir, sr_name), nrow=n_row, normalize=True)
            if tensorboard_writer is not None:
                img2tb(tensorboard_writer, mode+"_sr", sr_imgs[:len(z)], global_step, nrow=n_row, normalize=True)
        
        if opt.save_depth:
            gen_depths = (gen_depths-gen_depths.min())/(gen_depths.max()-gen_depths.min())
            gen_depths_var = zy_data['variance_map']
            gen_depths_var = (gen_depths_var-gen_depths_var.min())/(gen_depths_var.max()-gen_depths_var.min())
            if len(gen_depths.shape)==3:
                gen_depths = gen_depths.unsqueeze(1)
                gen_depths_var = gen_depths_var.unsqueeze(1)
            save_image(gen_depths[:len(z)], os.path.join(opt.output_dir, dep_name), nrow=n_row, normalize=True)
            save_image(gen_depths_var[:len(z)], os.path.join(opt.output_dir, "var"+dep_name), nrow=n_row, normalize=True)
            if tensorboard_writer is not None:
                img2tb(tensorboard_writer, mode+"_depth", gen_depths[:len(z)], global_step, nrow=n_row, normalize=True)

    def save_models(self):
        # Save checkpoints
        torch.save(
            self.ema, os.path.join(self.opt.output_dir, 'ema.pth'))
        torch.save(
            self.ema2, os.path.join(self.opt.output_dir, 'ema2.pth'))
        torch.save(
            self.generator_ddp.module,
            os.path.join(self.opt.output_dir, 'generator.pth'))
        torch.save(
            self.discriminator_ddp.module,
            os.path.join(self.opt.output_dir, 'discriminator.pth'))
        torch.save(
            self.optimizer_G.state_dict(),
            os.path.join(self.opt.output_dir, 'optimizer_G.pth'))
        torch.save(
            self.optimizer_D.state_dict(),
            os.path.join(self.opt.output_dir, 'optimizer_D.pth'))
        torch.save(
            self.scaler.state_dict(),
            os.path.join(self.opt.output_dir, 'scaler.pth'))
        if self.using_spade_d:
            torch.save(
                self.spade_discriminator_ddp.module,
                os.path.join(self.opt.output_dir,
                             'spade_discriminator.pth'))
            torch.save(
                self.optimizer_spade_D.state_dict(),
                os.path.join(self.opt.output_dir,
                             'optimizer_spade_D.pth'))
        if self.using_depth_d:
            torch.save(
                self.depth_discriminator_ddp.module,
                os.path.join(self.opt.output_dir,
                             'depth_discriminator.pth'))
            torch.save(
                self.optimizer_depth_D.state_dict(),
                os.path.join(self.opt.output_dir,
                             'optimizer_depth_D.pth'))

    def cal_fid(self):
        self.generated_dir = os.path.join(
            self.opt.output_dir, 'evaluation/generated')

        if self.rank == 0:
            fid_evaluation.setup_evaluation(
                self.metadata['dataset'],
                generated_dir=self.generated_dir,
                dataset_path=self.metadata['dataset_path'],
                target_size=128)
        dist.barrier()
        self.ema.store(
            [
                p for p in self.generator_ddp.parameters()
                if p.requires_grad
            ])
        self.ema.copy_to(
            [
                p for p in self.generator_ddp.parameters()
                if p.requires_grad
            ])
        self.generator_ddp.eval()
        self.eval_zs, self.eval_imgs = fid_evaluation.output_images(
            self.generator_ddp,
            self.metadata,
            self.rank,
            self.world_size,
            self.generated_dir,
            face_recon=self.face_recon,
            num_imgs=self.opt.fid_output_num)
        self.ema.restore(
            [
                p for p in self.generator_ddp.parameters()
                if p.requires_grad
            ])
        dist.barrier()
        if self.rank == 0:
            fid = fid_evaluation.calculate_fid(
                self.metadata['dataset'],
                self.generated_dir,
                target_size=128)
            with open(os.path.join(
                    self.opt.output_dir, f'fid.txt'), 'a') as f:
                f.write(f'\n{self.discriminator.step}:{fid}')

    def update_training_setting(self):
        self.metadata = curriculums.extract_metadata(
            self.curriculum, self.discriminator.step)

        if self.scaler.get_scale() < 1:
            self.scaler.update(1.)
        self.generator_ddp.train()
        self.discriminator_ddp.train()
        if self.using_spade_d:
            self.spade_discriminator_ddp.train()
        if self.using_depth_d:
            self.depth_discriminator_ddp.train()
        self.alpha = min(
            1,
            (self.discriminator.step - self.step_last_upsample)
            / (self.metadata['fade_steps'])
        )
        self.metadata['nerf_noise'] = max(
            0, 1. - self.discriminator.step / 5000.)
        self.using_tddfa_loss = (
            self.discriminator.step
            >= self.metadata.get('awake_tddfa_loss', 2e4)
            and self.metadata.get('tddfa_lambda', 0) > 0)
        self.using_recon_loss = (
            self.discriminator.step
            >= self.metadata.get('awake_recon_loss', 2e4)
            and self.metadata.get('recon_lambda', 0) > 0)

    def _terminate_all(self):
        if self.rank == 0:
            self.tensorboard_writer.close()
        cleanup()
