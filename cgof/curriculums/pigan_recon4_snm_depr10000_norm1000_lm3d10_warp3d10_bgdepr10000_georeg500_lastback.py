import math
pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4,
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(8e4): {},

    'param_path': '/home/kqsun/Data/param_crop_celeba/*.npz',
    'dataset_path': '/home/kqsun/Data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 5000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),

    'weight_decay': 0,
    'r1_lambda': 0.2,

    # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'latent_dim': 80+64+80+80+27+80,

    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,  # 15,
    'pos_lambda': 15,
    'last_back': True,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0,  # 1.5
    'depth_g_lambda': 0,  # 1.5 # using depth d

    "rel_depth_consistency_lambda": 0,  # 1e3
    "num_sample_pairs": 5000,

    "variance_lambda": 0,  # 1e1
    "sample_epsilon": 1e-5,

    'depth_smooth_edge_aware_lambda': 0,  # 5

    "tddfa_lambda": 0,
    "sample_near_mesh": True,
    "dist_depr_lambda": 10000,
    "bg_depr_lambda": 10000,
    "norm_reg_lambda": 1000,
    "geo_reg_lambda": 500,
    "warping3d_lambda": 10,
    "exp_warping_lambda": 0,
    "thick_ratio": 0.005,
    "shrink_step_num": 200.0,
    "recon_lambda": 4,
    "lm_lambda": 0,
    "lm3d_lambda": 10,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id', 'exp', 'tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0,  # 1.0,
    'spade_g_lambda': 0,  # 1.0,
    'spade_gan_feature_lambda': 0,  # 10,
    'spade_vgg_lambda': 0,  # 10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': False,
    'transformer_lr': 0.0,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt': {
        'lambda_feat': 10.0,  # weight for feature matching loss
        'lambda_vgg':  10.0,  # weight for vgg loss
        # if specified, do *not* use discriminator feature matching loss
        'no_ganFeat_loss': True,
        # if specified, do *not* use VGG feature matching loss
        'no_vgg_loss': True,
        'gan_mode': 'hinge',  # (ls|original|hinge)
        'netD': 'multiscale',  # (n_layers|multiscale|image)
        'lambda_kld': 0.05,
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64,  # of discrim filters in first conv layer
        # instance normalization or batch normalization
        'norm_D': 'spectralinstance',
        # of input label classes without unknown class.
        # If you have unknown class as class label,
        # specify --contain_dopntcare_label.
        'label_nc': 0,
        # 'if the label map contains dontcare label (dontcare=255)')
        'contain_dontcare_label': False,
        'output_nc': 3,  # of output image channels
        'no_instance': True,  # instance normalization or batch normalization
    },
    'depth_discriminator_opt': {
        'lambda_feat': 10.0,  # weight for feature matching loss
        'lambda_vgg':  10.0,  # weight for vgg loss
        # if specified, do *not* use discriminator feature matching loss
        'no_ganFeat_loss': True,
        # if specified, do *not* use VGG feature matching loss
        'no_vgg_loss': True,
        'gan_mode': 'hinge',  # (ls|original|hinge)
        'netD': 'multiscale',  # (n_layers|multiscale|image)
        'lambda_kld': 0.05,
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64,  # of discrim filters in first conv layer
        # instance normalization or batch normalization
        'norm_D': 'spectralinstance',
        # of input label classes without unknown class.
        # If you have unknown class as class label,
        # specify --contain_dopntcare_label.
        'label_nc': 1,
        # 'if the label map contains dontcare label (dontcare=255)')
        'contain_dontcare_label': False,
        'output_nc': 1,  # of output image channels
        'no_instance': True,  # instance normalization or batch normalization
    },
    # activate alpha
    'activate_mode': None,
    # tddfa options
    'tddfa_opt': {
        "arch": "resnet22",
        "tddfa_ckpt_fp": "tddfa/weights/resnet22.pth",
        "face_detector_ckpt_fp": "tddfa/FaceBoxes/weights/FaceBoxesProd.pth",
        "bfm_fp": "configs/bfm_noneck_v3.pkl",
        "size": 120,
        "num_params": 62
    },
    'random_pose': True,
}
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train/train.py --output_dir outputs/pigan_recon4_snm_depr10000_norm10000_lm10_expwarp20 --load_dir outputs/pigan_recon4_snm_depr10000_norm1000_lm10/ --curriculum pigan_recon4_snm_depr10000_norm10000_lm10_expwarp20 --save_depth