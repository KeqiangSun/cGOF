from packages import *

def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


class Initializer():
    """initialize basic settings, models, and loss criterions
    """
    def __init__(self, rank, world_size, opt):
        # super(PiganTrainer, self).__init__()
        self.rank = rank
        self.world_size = world_size
        self.opt = opt
        # set basic environment
        torch.manual_seed(0)

        torch.cuda.set_device(self.rank)
        self.device = torch.device(self.rank)

        self._init_setting()
        self._init_models()
        self._init_losses()

        with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
            f.write(str(opt))
            f.write('\n\n')
            f.write(str(self.generator))
            f.write('\n\n')
            f.write(str(self.discriminator))
            f.write('\n\n')
            f.write(str(self.curriculum))

    def _init_setting(self):
        self._init_logger()
        self._init_curriculum()
        self._init_dims()
        self._init_fixed_z()
        self._init_lambdas()

    def _init_models(self):
        self._init_G()
        self._init_D()
        self._init_emas()
        self._init_face_recon()
        # if self.using_warping_loss:
        #     self._init_warpformer()
        self._init_optimizer()
        self._init_reld()

    def _init_losses(self):
        self._init_depth_smooth_laplacian_loss()
        self._init_depth_smooth_edge_aware_loss()
        self._init_contrastive_loss()
        self._init_immitative_loss()

    def _init_logger(self):
        # initiate tensorboard
        if self.rank == 0:
            TIMESTAMP = "{0:%Y%m%d_%H%M%S}".format(datetime.now())
            tb_dir = os.path.join(
                self.opt.output_dir, 'tensorboard', TIMESTAMP)
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tb_dir)

        # init logger
        self.logger = LOGGER(self.rank, self.opt.print_level)

    def _init_curriculum(self):
        self.curriculum = curriculums.get(self.opt.curriculum)
        self.metadata = curriculums.extract_metadata(self.curriculum, 0)
        self.metadata['logger'] = self.logger
        self.enable_scaler = not self.metadata.get('disable_scaler', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_scaler)
        self.h_mean = self.metadata['h_mean']
        self.h_stddev = self.metadata['h_stddev']
        self.v_mean = self.metadata['v_mean']
        self.v_stddev = self.metadata['v_stddev']
        self.img_size = self.metadata['img_size']
        self.num_steps = self.metadata['num_steps']
        self.ray_start = self.metadata['ray_start']
        self.ray_end = self.metadata['ray_end']
        
        self.fov = self.metadata['fov']
        self.focal_length = (self.img_size/2) / np.tan(self.fov/2*math.pi/180)


    def _init_dims(self):
        self.dim_id = self.metadata.get('dim_id', 80)
        self.dim_exp = self.metadata.get('dim_exp', 64)
        self.dim_bg_geo = self.metadata.get('dim_bg_geo', 80)
        self.dim_tex = self.metadata.get('dim_tex', 80)
        self.dim_gamma = self.metadata.get('dim_gamma', 27)
        self.dim_bg_tex = self.metadata.get('dim_bg_tex', 80)

        if self.metadata['latent_dim'] != (
                self.dim_id + self.dim_exp + self.dim_bg_geo
                + self.dim_tex + self.dim_gamma + self.dim_bg_tex):
            warnings.warn(("latent_dim should be equal to dim_id + dim_exp",
                           "+ dim_bg_geo + dim_tex + dim_gamma + dim_bg_te !",
                           "Using metadata['latent_dim']."))

    def _init_fixed_z(self):
        fixed_bs = 25
        self.fixed_z = z_sampler(
            (fixed_bs, self.metadata['latent_dim']),
            device='cpu',
            dist=self.metadata['z_dist']).to(self.device)

    def _load_model(self, model, path):
        if self.opt.load_dir != '':
            load_pretrained_model(
                model,
                os.path.join(self.opt.load_dir, path),
                device=self.device)

    def _init_G(self):
        SIREN = getattr(siren, self.metadata['model'])
        GENERATOR = getattr(generators, self.metadata['generator'])
        self.generator = GENERATOR(
            SIREN, self.metadata['latent_dim'], **self.metadata
        ).to(self.device)
        self._load_model(self.generator, "generator.pth")

        self.generator_ddp = DDP(
            self.generator, device_ids=[self.rank],
            find_unused_parameters=True)
        self.generator = self.generator_ddp.module

        if self.opt.set_step is not None:
            self.generator.step = self.opt.set_step
        self.generator.set_device(self.device)

    def _init_D(self):
        DISCRIMINATOR = getattr(
            discriminators, self.metadata['discriminator'])
        self.discriminator = DISCRIMINATOR(
            latent_dim=self.metadata['latent_dim']
        ).to(self.device)
        self._load_model(self.discriminator, "discriminator.pth")

        self.discriminator_ddp = DDP(
            self.discriminator, device_ids=[self.rank],
            find_unused_parameters=True,
            broadcast_buffers=False)
        self.discriminator = self.discriminator_ddp.module
        if self.opt.set_step is not None:
            self.discriminator.step = self.opt.set_step

    def _init_emas(self):
        self.ema = ExponentialMovingAverage(
            [p for p in self.generator.parameters() if p.requires_grad],
            decay=0.999)
        self.ema2 = ExponentialMovingAverage(
            [p for p in self.generator.parameters() if p.requires_grad],
            decay=0.9999)

        self._load_model(self.ema, "ema.pth")
        self._load_model(self.ema2, "ema2.pth")

    def _init_face_recon(self):
        self.face_recon = create_model(deep3dfacerecon_opt)
        self.face_recon.setup(deep3dfacerecon_opt)
        self.face_recon.device = self.device
        self.face_recon.set_coeff_static()
        self.face_recon.parallelize()
        self.face_recon.eval()
        self.visualizer = MyVisualizer(deep3dfacerecon_opt)

        self.recon_basic_size = [256, 256]
        self.recon_s = [0.8975788298782894]
        self.recon_t = [128.04371143331298, 89.95939537909564]
        self.trans_params = self.recon_basic_size + self.recon_s + self.recon_t

        if self.using_recon_loss:
            recon_loss_criterion = self.metadata.get(
                'recon_loss_criterion', 'l1')
            self.face_recon.set_recon_criterion(recon_loss_criterion)

    def _init_optimizer(self):
        # exclude_names <= mapping_network params
        exclude_names = []
        if hasattr(self.generator_ddp.module.siren,
                   'mapping_network'):
            mapping_network_param_names = [
                name for name, _ in self.generator_ddp.module.siren
                .mapping_network.named_parameters()
            ]
        else:
            mapping_network_param_names = []
        exclude_names += mapping_network_param_names

        self.logger('mapping_network_param_names:'
                    f'{mapping_network_param_names}')

        # get mapping and generator params.
        mapping_network_parameters = [
            p for n, p in self.generator_ddp.named_parameters()
            if n in mapping_network_param_names
        ]
        generator_parameters = [
            p for n, p in self.generator_ddp.named_parameters()
            if n not in exclude_names
        ]
        params = [
            {
                'params': generator_parameters,
                'name': 'generator'
            },
            {
                'params': mapping_network_parameters,
                'name': 'mapping_network',
            },
        ]
        # get warping parameters
        # if self.using_warping_loss:
        #     params.append(
        #         {
        #             'params': self.warpformer_ddp.parameters(),
        #             'name': 'warpformer'
        #         }
        #     )
        # set learning rate
        if self.metadata.get('unique_lr', False):
            self.logger('Using unique lr!')

            # initiate optimizer_G with different learning rate
            gen_lr = self.metadata['gen_lr']
            mapping_network_lr = gen_lr * 5e-2
            for i_param in range(len(params)):
                if params[i_param]['name'] == 'mapping_network':
                    params[i_param]['lr'] = mapping_network_lr

            self.logger('Initialize mapping_network_lr with'
                        f'{self.mapping_network_lr}')
            self.optimizer_G = torch.optim.Adam(
                params,
                lr=gen_lr, betas=self.metadata['betas'],
                weight_decay=self.metadata['weight_decay']
            )
        else:
            self.optimizer_G = torch.optim.Adam(
                # self.generator_ddp.parameters(),
                params,
                lr=self.metadata['gen_lr'],
                betas=self.metadata['betas'],
                weight_decay=self.metadata['weight_decay'])
        self.optimizer_D = torch.optim.Adam(
            self.discriminator_ddp.parameters(),
            lr=self.metadata['disc_lr'],
            betas=self.metadata['betas'],
            weight_decay=self.metadata['weight_decay'])
        # if self.using_warping_loss:
        #     self.optimizer_W = torch.optim.Adam(
        #         self.warpformer_ddp.parameters(),
        #         lr=self.metadata['gen_lr'],
        #         betas=self.metadata['betas'],
        #         weight_decay=self.metadata['weight_decay'])

        # load optimizer weights
        if self.opt.load_dir != '':
            load_pretrained_optimizer(
                self.optimizer_G,
                os.path.join(self.opt.load_dir, 'optimizer_G.pth'))
            load_pretrained_optimizer(
                self.optimizer_D,
                os.path.join(self.opt.load_dir, 'optimizer_D.pth'))
            if self.enable_scaler:
                self.scaler.load_state_dict(
                    torch.load(os.path.join(self.opt.load_dir, 'scaler.pth')))

    def _init_lambdas(self):
        self.depth_lambda = self.metadata.get('depth_lambda', 0)
        self.style_lambda = self.metadata.get('style_lambda', 0)
        self.spade_d_lambda = self.metadata.get('spade_d_lambda', 0)
        self.depth_d_lambda = self.metadata.get('depth_d_lambda', 0)
        self.depth_g_lambda = self.metadata.get('depth_g_lambda', 0)
        self.rel_depth_lambda = self.metadata.get(
            'rel_depth_consistency_lambda', 0)
        self.variance_loss_lambda = self.metadata.get('variance_lambda', 0)
        self.depth_smooth_laplacian_lambda = self.metadata.get(
            'depth_smooth_laplacian_lambda', 0)
        self.depth_smooth_edge_aware_lambda = self.metadata.get(
            'depth_smooth_edge_aware_lambda', 0)
        self.cid_lambda = self.metadata.get('cid_lambda', 0)
        self.cid_p_lambda = self.metadata.get('cid_p_lambda', 0)
        self.cid_c_lambda = self.metadata.get('cid_c_lambda', 0)
        self.cgeo_lambda = self.metadata.get('cgeo_lambda', 0)
        self.ctex_lambda = self.metadata.get('ctex_lambda', 0)
        self.ctex_p_lambda = self.metadata.get('ctex_p_lambda', 0)
        self.ctex_gram_lambda = self.metadata.get('ctex_gram_lambda', 0)
        self.itex_lambda = self.metadata.get('itex_lambda', 0)
        self.cface_lambda = self.metadata.get('cface_lambda', 0)
        self.recon_lambda = self.metadata.get('recon_lambda', 0)
        self.lm_lambda = self.metadata.get('lm_lambda', 0)
        self.lm3d_lambda = self.metadata.get('lm3d_lambda', 0)
        self.warping3d_lambda = self.metadata.get('warping3d_lambda', 0)
        self.exp_warping_lambda = self.metadata.get('exp_warping_lambda', 0)

        self.sample_near_mesh = self.metadata.get('sample_near_mesh', False)
        self.thick_ratio = self.metadata.get('thick_ratio', 0.1)
        self.shrink_step_num = self.metadata.get('shrink_step_num', 5000.0)
        self.w_shrink_step_num = self.metadata.get(
            'w_shrink_step_num', self.shrink_step_num)
        self.narrow_ratio = 0.0
        self.w_narrow_ratio = 0.0
        self.dist_depr_lambda = self.metadata.get('dist_depr_lambda', 0)
        self.bg_depr_lambda = self.metadata.get('bg_depr_lambda', 0)
        self.norm_reg_lambda = self.metadata.get('norm_reg_lambda', 0)
        self.geo_reg_lambda = self.metadata.get('geo_reg_lambda', 0)

        self.depth_consistency_lambda = self.metadata.get('depth_consistency_lambda', 0)

        self.using_contrastive_loss = (
            self.depth_lambda > 0 or self.style_lambda > 0)
        self.using_spade_d = self.spade_d_lambda > 0
        self.using_depth_d = self.depth_d_lambda and self.depth_g_lambda
        self.using_rel_depth_consistency = self.rel_depth_lambda > 0
        self.using_proportional_reld = self.metadata.get(
            'using_proportional_reld', 0)
        self.using_variance_loss = self.variance_loss_lambda > 0
        self.using_depth_smooth_laplacian_loss = (
            self.depth_smooth_laplacian_lambda > 0)
        self.using_depth_smooth_edge_aware_loss = (
            self.depth_smooth_edge_aware_lambda > 0)
        self.using_cid_loss = self.cid_lambda > 0
        self.using_cgeo_loss = self.cgeo_lambda > 0
        self.using_ctex_loss = self.ctex_lambda > 0
        self.using_ctex_gram_loss = self.ctex_gram_lambda > 0
        self.using_itex_loss = self.itex_lambda > 0
        self.using_cface_loss = self.cface_lambda > 0
        self.using_recon_loss = self.recon_lambda > 0
        self.using_lm_loss = self.lm_lambda > 0
        self.using_lm3d_loss = self.lm3d_lambda > 0
        self.using_warping3d_loss = self.warping3d_lambda > 0
        self.using_exp_warping_loss = self.exp_warping_lambda > 0
        self.using_dist_depr = self.dist_depr_lambda > 0
        self.using_bg_depr = self.bg_depr_lambda > 0
        self.using_depr = self.using_dist_depr or self.using_bg_depr
        self.using_norm_reg = self.norm_reg_lambda > 0
        self.using_geo_reg = self.geo_reg_lambda > 0
        self.using_depth_consistency_loss = self.depth_consistency_lambda > 0

        self.requiring_depth = (
            self.using_depth_d
            or self.using_depth_smooth_laplacian_loss
            or self.using_depth_smooth_edge_aware_loss
            or self.using_rel_depth_consistency
            or self.using_recon_loss
            or self.using_cid_loss
            or self.using_cgeo_loss
            or self.using_ctex_loss
            or self.using_cface_loss
            or self.ctex_gram_lambda
            or self.using_itex_loss
            or self.using_depth_consistency_loss
            or self.using_geo_reg
        )

    def _init_spade_d(self):
        if self.using_spade_d:
            return
        spade_discriminator_opt = edict(
            self.metadata['spade_discriminator_opt'])
        self.spade_discriminator = MultiscaleDiscriminator(
            spade_discriminator_opt).to(self.device)
        load_pretrained_model(
            self.spade_discriminator,
            os.path.join(self.opt.load_dir,
                         'spade_discriminator.pth'),
            device=self.device)
        self.spade_discriminator_ddp = DDP(
            self.spade_discriminator,
            device_ids=[self.rank],
            find_unused_parameters=True,
            broadcast_buffers=False)
        self.spade_discriminator = self.spade_discriminator_ddp.module
        self.optimizer_spade_D = torch.optim.Adam(
            self.spade_discriminator_ddp.parameters(),
            lr=self.metadata['disc_lr'],
            betas=self.metadata['betas'],
            weight_decay=self.metadata['weight_decay'])
        load_pretrained_optimizer(
            self.optimizer_spade_D,
            os.path.join(self.opt.load_dir, 'optimizer_spade_D.pth'))

    def _init_depth_discriminator(self):
        if self.using_depth_d:
            self.logger('using depth discriminator!')
            depth_discriminator_opt = edict(
                self.metadata['depth_discriminator_opt'])
            self.depth_discriminator = MultiscaleDiscriminator(
                depth_discriminator_opt).to(self.device)
            load_pretrained_model(
                self.depth_discriminator,
                os.path.join(self.opt.load_dir,
                             'depth_discriminator.pth'),
                device=self.device)
            self.depth_discriminator_ddp = DDP(
                self.depth_discriminator,
                device_ids=[self.rank],
                find_unused_parameters=True,
                broadcast_buffers=False)
            depth_discriminator = self.depth_discriminator_ddp.module
            self.optimizer_depth_D = torch.optim.Adam(
                self.depth_discriminator_ddp.parameters(),
                lr=self.metadata['disc_lr'],
                betas=self.metadata['betas'],
                weight_decay=self.metadata['weight_decay'])
            load_pretrained_optimizer(
                self.optimizer_depth_D,
                os.path.join(self.opt.load_dir, 'optimizer_depth_D.pth'))

    def _init_reld(self):
        if self.using_rel_depth_consistency:
            self.logger('using relative depth consistency!')
            num_sample_pairs = self.metadata.get('num_sample_pairs', 5000)
            sample_epsilon = self.metadata.get('sample_epsilon', 5000)
            self.rel_depth_loss = RelDConsistencyLoss(
                num_sample_pairs,
                epsilon=sample_epsilon,
                using_proportional_reld=self.using_proportional_reld)

    def _init_depth_smooth_laplacian_loss(self):
        if self.using_depth_smooth_laplacian_loss:
            self.laplacianer = LaplacianLoss().to(self.device)

    def _init_depth_smooth_edge_aware_loss(self):
        if self.using_depth_smooth_edge_aware_loss:
            self.edge_aware_smoother = EdgeAwareDepthSmoothLoss()
            self.edge_aware_smoother = self.edge_aware_smoother.to(self.device)

    def _init_contrastive_loss(self):
        self.z_manager = Z_Manager(
            self.metadata['latent_dim'], device=self.device)

        if self.using_cid_loss:
            self.cid_loss = ContrastiveIDLoss(
                length=self.metadata['latent_dim'], device=self.device)
            self.cid_loss.ddp(self.rank)

        if self.using_cgeo_loss:
            cgeo_overall_contrast = self.metadata.get(
                'cgeo_overall_contrast', 0)
            self.cgeo_loss = ContrastiveGeoLoss(
                length=self.metadata['latent_dim'],
                device=self.device, margin=self.metadata['cgeo_margin'],
                overall_contrast=cgeo_overall_contrast)

        if self.using_ctex_loss:
            self.ctex_loss = ContrastiveTexLoss(
                length=self.metadata['latent_dim'], device=self.device,
                margin=self.metadata['ctex_margin'])

        if self.using_ctex_gram_loss:
            self.ctex_gram_loss = ContrastiveTexGramLoss(
                length=self.metadata['latent_dim'], device=self.device,
                margin=self.metadata['ctex_gram_margin'])

        if self.using_cface_loss:
            cface_img_lambda = self.metadata.get('cface_img_lambda', 0)
            cface_d_lambda = self.metadata.get('cface_depth_lambda', 0)
            cface_img_margin = self.metadata.get('cface_img_margin', 0)
            cface_d_margin = self.metadata.get('cface_depth_margin', 0)
            cface_overall_contrast = self.metadata.get(
                'cface_overall_contrast', 0)
            self.cface_loss = ContrastiveFaceLoss(
                length=self.metadata['latent_dim'],
                device=self.device,
                lambda_img=cface_img_lambda,
                lambda_d=cface_d_lambda,
                margin_m=0,
                margin_img=cface_img_margin,
                margin_d=cface_d_margin,
                overall_contrast=cface_overall_contrast)

    def _init_immitative_loss(self):
        if self.using_itex_loss:
            self.itex_loss = ImitativeTexLoss()

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
        self.warpformer.load_state_dict(
            torch.load(
                "warping_models/warpformer_190000.pth",
                map_location=self.device))
        # self.warpformer = torch.load("warpformer.pth")
        self.warpformer.to(self.device)

        self.warpformer_ddp = DDP(
            self.warpformer, device_ids=[self.rank],
            find_unused_parameters=True)
        self.warpformer_ddp.train()
        self.warpformer = self.warpformer_ddp.module
