"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
"""

import math
from re import A
from easydict import EasyDict
recon4_minv1e1_reld1e3 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon2_minv1e1_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 2,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon1_split1111 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 1,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon08_split1211 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.8,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 2,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon07_split1311 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.7,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 3,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon05_split1511 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 5,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon08_split1211_minv1e1_reld1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.8,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 2,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon07_split1311_minv1e1_reld1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.7,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 3,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon05_split1511_minv1e1_reld1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0.5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    "recon_split": True,
    "recon_lambdas": {
        'id': 1,
        'exp': 5,
        'tex': 1,
        'gamma': 1,
    },
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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



recon4_minv1e1_reld5e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 5e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2_newcface10d0over = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2,
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 10,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 0, #1e1
    'cface_overall_contrast': 1,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2_newctex5 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2,
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 5,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2_newcface5d0over = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2,
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 5,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 0, #1e1
    'cface_overall_contrast': 1,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2_cgeo3e2mg1e_2over = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2,
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 5,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 3e2,
    'cgeo_margin': 1e-2,
    'cgeo_overall_contrast': 1,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1_reld1e2_bk = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_reld1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e2, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_minv1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 
        'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


# debug fid
bare_z0_bs32 = {
    0: {'batch_size': 32, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
bare_z0_bs16 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_bs32_split4 = {
    0: {'batch_size': 32, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},


    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
}
CelebA_bs32 = {
    0: {'batch_size': 32, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},


    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
}
CelebA_bs16 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},


    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
}
bare = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
bare_z0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
ori = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
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
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA = {
    0: {'batch_size': 4 * 2, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
}



recon4_norm_glbmean_reld2e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 2e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_reld1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_reld1e2_largernet = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_minv5e1_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 5e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_minv1e1_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_glbmean_minv1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon4_norm_glbmean_minv1e1_reld1e2_newcid5 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 5,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon1_norm_newcid = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 1,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon4_norm_glbmean_minv1e1_reld1e2_newctex5 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 5,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon1_norm_newctex = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 1,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


recon4_norm_glbmean_minv1e1_reld1e2_newcface4d0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 4,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 0, #1e1
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_newcfaced0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 1,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 0, #1e1
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon4_norm_newcface = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 1,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 1e1,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_newcface = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 1,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5,
    'cface_depth_lambda': 1e1,
    'ctex_lambda': 0,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon1_norm_debug = {
    0: {'batch_size': 6, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_img_margin': 0.1,
    'cface_depth_margin': 0.02,
    'cface_img_lambda': 5e1,
    'cface_depth_lambda': 1e2,
    'ctex_lambda': 1,
    'ctex_p_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    'cid_p_lambda': 1,
    'cid_c_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


recon5_norm_glbmean = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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



recon3_norm_glbmean = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 3,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon4L2_norm_glbmean = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_loss_criterion": 'L2',
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5L2_norm_glbmean = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_loss_criterion": 'L2',
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_ctex2_cgeo2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 2,
    'ctex_margin': 0.5,
    'cgeo_lambda': 2,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_cgeo3l2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 3,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


recon4_norm_cgeo5e1mg1e_2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 5e1,
    'cgeo_margin': 1e-2,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 4,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
recon1_norm_cgeo5e1mg1e_2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 5e1,
    'cgeo_margin': 1e-2,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


recon1_norm_ctex4 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 4,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_ctex3 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 3,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_cface3 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 3,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_cface2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 2,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,

    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_cid = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 0,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm_minv1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon2_denorm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 2,
    "recon_norm_mode": "denorm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon2_norm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 2,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_denorm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "denorm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon1_norm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 1,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_cface = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'cface_lambda': 1,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_cgeo_ctex = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 1,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_dd = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_cgeo = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 1,
    'cgeo_margin': 0.5,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon2p5_norm_reld1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e1, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 2.5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon2p5_norm_reld1e2 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e2, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 2.5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_reld1e3 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm_minv1e1 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_denorm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "denorm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

recon5_norm = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

pretrained_wo_gamma = {
    0: {'batch_size': 10, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 1,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_loss_keys": ['id','exp','tex'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

pretrained_using_cid = {
    0: {'batch_size': 10, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 1,
    'cid_lambda': 1,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

pretrained_using_ctex = {
    0: {'batch_size': 10, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 1,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 1,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

pretrained_using_cgeo = {
    0: {'batch_size': 10, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 1,
    'cgeo_margin': 1,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa0_recon5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3_minv1e1_edgeaware5_zy = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "cid_lambda": 0, #contrastive id loss
    "cgeo_lambda": 1, #contrastive id loss
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 1,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

debug_lab = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    'cface_lambda': 1,
    'cface_margin': 1.0,
    'ctex_lambda': 0,
    'ctex_margin': 0.5,
    'cgeo_lambda': 0,
    'cgeo_margin': 1,
    'cid_lambda': 0,
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    "recon_norm_mode": "norm",
    "recon_loss_keys": ['id','exp','tex', 'gamma'],
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon5_depth0_style0_wingloss_depthd0_depthg0_reld0_minv0_edgeaware5 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 1,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon5_depth0_style0_wingloss_depthd0_depthg0_reld0_minv1e1_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon5_depth0_style0_wingloss_depthd0_depthg0_reld1e3_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

#recon5_pretrain
CelebA_pigan_recon5_depth0_style0_wingloss_depthd0_depthg0_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon10_depth0_style0_wingloss_depthd0_depthg0_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 10,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon3_depth0_style0_wingloss_depthd0_depthg0_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 3,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_recon0_depth0_style0_wingloss_depthd0_depthg0_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 16, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'disc_lr': 2e-4},
    int(4e4): {},

    'param_path': 'data/param_crop_celeba/*.npz',
    'dataset_path': 'data/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CropCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 0,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 0,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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


CelebA_pigan_spade0_tddfa0_recon5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3_minv1e1_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    # 'dataset': 'TddfaCelebA',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 1,
    'awake_tddfa_loss': 0,
    'awake_recon_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa0_recon10_depth0_style0_wingloss_depthd0_depthg0_reld1e0_minv0_edgeaware0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 80+64+80+80+27+80, # id80, exp64, bg_geo80, tex80, gamma27, gb_tex80
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 0,
    "recon_lambda": 10,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    'awake_recon_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

# ablation study
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3_minv1e1_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd0_depthg0_reld1e3_minv1e1_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld0_minv1e1_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3_minv0_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3_minv1e1_edgeaware0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd0_depthg0_reld1e0_minv0_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 5, #5
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd0_depthg0_reld1e0_minv1e1_edgeaware0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 1e1, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0,
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd0_depthg0_reld1e3_minv0_edgeaware0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 0, #1.5
    'depth_g_lambda': 0, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 1e3, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0,
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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
CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld0_minv0_edgeaware0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(2e4): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_d_lambda': 1.5, #1.5
    'depth_g_lambda': 1.5, #1.5 # using depth d
    
    "rel_depth_consistency_lambda": 0, #1e3
    "num_sample_pairs": 5000,
    
    "variance_lambda": 0, #1e1
    "sample_epsilon": 1e-5,
    
    'depth_smooth_edge_aware_lambda': 0, #2
    
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_smooth_laplacian_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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




CelebA_pigan_spade0_tddfa1p5_depth0_style0_wingloss_depthd0_depthg0_reld1e4 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "rel_depth_consistency_lambda": 10000,
    "num_sample_pairs": 5000,
    "sample_epsilon": 1e-5,
    "tddfa_lambda": 1.5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 0,
    'depth_g_lambda': 0, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa1p5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e3 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "rel_depth_consistency_lambda": 1000,
    "num_sample_pairs": 5000,
    "sample_epsilon": 1e-5,
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd0_depthg0_reld1e4 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "rel_depth_consistency_lambda": 10000,
    "num_sample_pairs": 5000,
    "sample_epsilon": 1e-5,
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 0,
    'depth_g_lambda': 0, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1_depthg2_woraydirction = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1,
    'depth_g_lambda': 2, # using depth d
    'awake_tddfa_loss': 1,
    'wo_ray_direction': True,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa10_depth0_style0_wingloss_depthd0_depthg0_woraydirction = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 0,
    'depth_g_lambda': 0, # using depth d
    'awake_tddfa_loss': 1,
    'wo_ray_direction': True,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa0_depth0_style0_wingloss_depthd0_depthg0_woraydirction = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 0,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 0,
    'depth_g_lambda': 0, # using depth d
    'awake_tddfa_loss': 1,
    'wo_ray_direction': True,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_lap1_edgeaware2_sr = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebAHD',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, # 1.0,
    'spade_g_lambda': 0, # 1.0,
    'spade_gan_feature_lambda': 0, # 10,
    'spade_vgg_lambda': 0, # 10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 1,
    'depth_smooth_edge_aware_lambda': 2,
    'awake_tddfa_loss': 1,
    # sr
    'hd_img_size': 256,
    'sr_arch': "NeuralRenderer",
    "sr_d_opt": {
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'spade_d_lambda': 1,
    'sr_g_lambda': 1,
    'sr_feature_lambda': 1,
    'sr_vgg_lambda': 1,
    
    
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_lap1_edgeaware2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, # 1.0,
    'spade_g_lambda': 0, # 1.0,
    'spade_gan_feature_lambda': 0, # 10,
    'spade_vgg_lambda': 0, # 10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 1,
    'depth_smooth_edge_aware_lambda': 2,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_edgeaware5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 0,
    'depth_smooth_edge_aware_lambda': 5,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_edgeaware2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, # 1.0,
    'spade_g_lambda': 0, # 1.0,
    'spade_gan_feature_lambda': 0, # 10,
    'spade_vgg_lambda': 0, # 10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 0,
    'depth_smooth_edge_aware_lambda': 2,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_edgeaware1 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 0,
    'depth_smooth_edge_aware_lambda': 1,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_lap2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 2,
    'depth_smooth_edge_aware_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_lap0p5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, # 1.0,
    'spade_g_lambda': 0, # 1.0,
    'spade_gan_feature_lambda': 0, # 10,
    'spade_vgg_lambda': 0, # 10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 0.5,
    'depth_smooth_edge_aware_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1p5_depthg1p5_lap1 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'depth_smooth_laplacian_lambda': 1,
    'depth_smooth_edge_aware_lambda': 0,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa2_depth1_style1_wingloss_depthd2_depthg2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 2,
    'depth_lambda': 1,
    "style_lambda": 1,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 2,
    'depth_g_lambda': 2, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd1_depthg2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1,
    'depth_g_lambda': 2, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa10_depth0_style0_wingloss_depthd0_depthg0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 0,
    'depth_g_lambda': 0, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa3_depth0_style0_wingloss_depthd3_depthg3 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 3,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 3,
    'depth_g_lambda': 3, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

# here
CelebA_pigan_spade0_tddfa1p5_depth0_style0_wingloss_depthd1p5_depthg1p5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa1p5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "rel_depth_consistency_lambda": 10000,
    "num_sample_pairs": 5000,
    "sample_epsilon": 1e-5,
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa1p5_depth0_style0_wingloss_depthd1p5_depthg1p5_reld1e2_minv1e1 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "rel_depth_consistency_lambda": 1e2,
    "num_sample_pairs": 5000,
    "variance_lambda": 1e1,
    "sample_epsilon": 1e-5,
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 1.5,
    'depth_g_lambda': 1.5, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa2_depth0_style0_wingloss_depthd2_depthg2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 2,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 2,
    'depth_g_lambda': 2, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa5_depth0_style0_wingloss_depthd5_depthg5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 5,
    'depth_g_lambda': 5, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa10_depth0_style0_wingloss_depthd10_depthg10 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 0,
    "style_lambda": 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'depth_d_lambda': 10,
    'depth_g_lambda': 10, # using depth d
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'spade_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
    },
    'depth_discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 1, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade0_tddfa10_depth10_style10_wingloss = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 10,
    "style_lambda": 10,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # depthloss type
    'depthloss_type': 'wingloss',
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa10_depth1_style1 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 1,
    "style_lambda": 1,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa10_depth3_style3 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 3,
    "style_lambda": 3,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa10_depth5_style5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 5,
    "style_lambda": 5,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa3_depth3_style3 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 5,
    "style_lambda": 5,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa10_depth10_style10 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 10,
    "style_lambda": 10,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa5_depth5_style5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 5,
    "style_lambda": 5,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa5 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 5,
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade0_tddfa10 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    "tddfa_lambda": 10,
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
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

CelebA_pigan_spade_tddfa10 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'tddfa_lambda': 10,
    'spade_d_lambda': 1.0, #1.0,
    'spade_g_lambda': 1.0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        # 'lambda_feat': 10.0, # weight for feature matching loss
        # 'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        # 'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade_tddfa_vgg = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'tddfa_lambda': 1,
    'spade_d_lambda': 1.0, #1.0,
    'spade_g_lambda': 1.0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 10, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        # 'lambda_feat': 10.0, # weight for feature matching loss
        # 'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        # 'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade_tddfa = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'tddfa_lambda': 1,
    'spade_d_lambda': 1.0, #1.0,
    'spade_g_lambda': 1.0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_tddfa_loss': 1,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        # 'lambda_feat': 10.0, # weight for feature matching loss
        # 'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        # 'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
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

CelebA_pigan_spade_bfm_exp2 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),

    'weight_decay': 0,
    'r1_lambda': 0.5,
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 15, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 1.0, #1.0,
    'spade_g_lambda': 1.0, #1.0,
    'spade_gan_feature_lambda': 5, #10,
    'spade_vgg_lambda': 5, #10,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        # 'lambda_feat': 10.0, # weight for feature matching loss
        # 'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        # 'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

CelebA_pigan_spade_bfm_exp1 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 5, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 1.0, #1.0,
    'spade_g_lambda': 1.0, #1.0,
    'spade_gan_feature_lambda': 10, #10,
    'spade_vgg_lambda': 10, #10,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        # 'lambda_feat': 10.0, # weight for feature matching loss
        # 'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        # 'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

CelebA_pigan_spade_bfm = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+40+10, # texture 256, shape 40, expression 10
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'PoseImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'TddfaCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0, #15,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 0, #1.0,
    'spade_g_lambda': 0, #1.0,
    'spade_gan_feature_lambda': 0, #10,
    'spade_vgg_lambda': 0, #10,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 0, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

CelebA_pigan_spade_activate = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'RGBDEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 100,
    'ldmk_lambda': 0,
    'spade_d_lambda': 1.0,
    'spade_g_lambda': 1.0,
    'spade_gan_feature_lambda': 5,
    'spade_vgg_lambda': 5,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': "tanh",
}

CelebA_spade = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),

    'weight_decay': 0,
    'r1_lambda': 0,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'RGBDEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'spade_d_lambda': 1.0,
    'spade_g_lambda': 1.0,
    'spade_gan_feature_lambda': 10,
    'spade_vgg_lambda': 10,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

CelebA_pigan_spade = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'RGBDEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 100,
    'ldmk_lambda': 0,
    'spade_d_lambda': 1.0,
    'spade_g_lambda': 1.0,
    'spade_gan_feature_lambda': 5,
    'spade_vgg_lambda': 5,
    'awake_ldmk_loss': 9999999,
    # transform mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

ControlableCelebAWoPointnet_controlableSpade_noactivate = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'RGBDEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

ControlableCelebAWoPointnet_Spade_noactivate = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': None,
}

ControlableCelebAWoPointnet_Spade = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': 'tanh',
}

ControlableCelebAWoPointnet_Spade_debug = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'param_path': '/media/SSD/kqsun/param_crop_celeba_sample/*.npz',
    'dataset_path': '/media/SSD/kqsun/img_crop_celeba_sample/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'ParamCelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr': 0.0,
        'gen_lr': 6e-5, # 2e-4,
    'transformer_lr_gamma': 0,
    # discriminator:
    'discriminator_opt':{
        'lambda_feat': 10.0, # weight for feature matching loss
        'lambda_vgg':  10.0, # weight for vgg loss
        'no_ganFeat_loss': True, # if specified, do *not* use discriminator feature matching loss
        'no_vgg_loss': True, # if specified, do *not* use VGG feature matching loss
        'gan_mode': 'hinge', # (ls|original|hinge)
        'netD': 'multiscale', # (n_layers|multiscale|image)
        'lambda_kld': 0.05, 
        # MultiscaleDiscriminator
        'netD_subarch': 'n_layer',
        'num_D': 1,
        # NLayerDiscriminator
        'n_layers_D': 4,
        'ndf': 64, # of discrim filters in first conv layer
        'norm_D': 'spectralinstance', # instance normalization or batch normalization
        'label_nc': 1, # of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.
        'contain_dontcare_label': False, # 'if the label map contains dontcare label (dontcare=255)')
        'output_nc': 3, # of output image channels
        'no_instance': True, # instance normalization or batch normalization
        
    },
    # activate alpha
    'activate_mode': 'tanh',
}

ControlableCelebAWoPointnet_depth0_ldmk0_gamma0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr':0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0
}

ControlableCelebAWoPointnet_depth300_ldmk0_gamma0 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 300,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr':0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0
}

ControlableCelebAWoPointnet_depth300_ldmk0_gamma0_activalpha = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr':0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # activate alpha
    'activate_mode': 'tanh'
}

ControlableCelebAWoPointnet_depth300_ldmk0_gamma0_activalpha_uniform = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),

    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'awake_ldmk_loss': 9999999,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr':0.0,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0,
    # activate alpha
    'activate_mode': 'tanh'
}

ControlableCelebAWoPointnet_depth300_ldmk10_finetune_gamma094 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 300,
    'ldmk_lambda': 10,
    'awake_ldmk_loss': 0,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0.94
}

ControlableCelebAWoPointnet_depth300_ldmk10_awake8e3_gamma0965 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 300,
    'ldmk_lambda': 10,
    'awake_ldmk_loss': 8000,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
    'transformer_lr_gamma': 0.965
}

ControlableCelebAWoPointnet_depth300_ldmk10_finetune_awake1000 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 300,
    'ldmk_lambda': 10,
    'awake_ldmk_loss': 1000,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
}

ControlableCelebAWoPointnet_depth300_ldmk10 = {
    0: {'batch_size': 9, 'num_steps': 12, 'img_size': 64, 'batch_split': 3, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 300,
    'ldmk_lambda': 10,
    'awake_ldmk_loss': 10000,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
}

CelebAwoAMP = {
    0: {'batch_size': 4 * 2, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 1,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA_depth10 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 10,
    'ldmk_lambda': 10,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA_depth8000 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 8000,
    'ldmk_lambda': 500,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlablePretrain = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA_depth8000_ldmk500 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 8000,
    'ldmk_lambda': 500,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA_depth8000_ldmk0 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 8000,
    'ldmk_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True
}

ControlableCelebA_depth550_ldmk20 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 550,
    'ldmk_lambda': 20,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,
    'rotation_mode': "Quaternion"
}

ControlableCelebA_depth550_ldmk20_trans6e6 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 550,
    'ldmk_lambda': 20,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,
    'rotation_mode': "Quaternion",

    'unique_lr': True,
    'transformer_lr':6e-6,
    'gen_lr': 6e-5,
}

ControlableCelebA_depth5500_ldmk0 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 5500,
    'ldmk_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,
    'rotation_mode': "Quaternion"
}

ControlableCelebA_depth0_ldmk0 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINE',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    'unique_lr': True,
        'transformer_lr':6e-6,
        'pointnet_lr':6e-5,
        'gen_lr': 6e-5,
    'awake_ldmk_loss': 10000
}

ControlableCelebAWoPointnet_depth0_ldmk0 = {
    0: {'batch_size': 8, 'num_steps': 12, 'img_size': 64, 'batch_split': 4, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'depth_lambda': 0,
    'ldmk_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    'unique_lr': True,
        'transformer_lr':6e-6,
        'gen_lr': 6e-5,
    'awake_ldmk_loss': 10000
}

ControlableCelebAFitDepth_depth500 = {
    0: {'batch_size': 1, 'num_steps': 12, 'img_size': 64, 'batch_split': 1, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 500,
    'ldmk_lambda': 20,
    'awake_ldmk_loss': 10000,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': False,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
}

ControlableCelebAFitDepths_depth500 = {
    0: {'batch_size': 1, 'num_steps': 12, 'img_size': 64, 'batch_split': 1, 'disc_lr': 2e-4},
    int(200e3): {},

    'dataset_path': '/media/SSD/kqsun/img_align_celeba/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
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
    'latent_dim': 256+199+50, # texture 256, shape 199, expression 50
    'grad_clip': 10,
    'model': 'CONTROLABLESIRENBASELINEWOPOINTNET',
    'generator': 'ControlableImplicitGenerator3d',
    'discriminator': 'CCSEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'disable_scaler': True,

    # loss lambda
    'depth_lambda': 500,
    'ldmk_lambda': 20,
    'awake_ldmk_loss': 10000,
    # transfrom mode
    'rotation_mode': "Quaternion",
    'rotation_trainable': True,
    # learning rate
    'unique_lr': True,
        'transformer_lr':6e-5,
        'gen_lr': 6e-5,
}



CARLA = {
    0: {'batch_size': 30, 'num_steps': 48, 'img_size': 32, 'batch_split': 1, 'gen_lr': 4e-5, 'disc_lr': 4e-4},
    int(10e3): {'batch_size': 14, 'num_steps': 48, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(55e3): {'batch_size': 10, 'num_steps': 48, 'img_size': 128, 'batch_split': 5, 'gen_lr': 10e-6, 'disc_lr': 10e-5},
    int(200e3): {},

    'dataset_path': '/home/marcorm/S-GAN/data/cats_bigger_than_128x128/*.jpg',
    'fov': 30,
    'ray_start': 0.7,
    'ray_end': 1.3,
    'fade_steps': 10000,
    'sample_dist': 'spherical_uniform',
    'h_stddev': math.pi,
    'v_stddev': math.pi/4 * 85/90,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi/4 * 85/90,
    'topk_interval': 1000,
    'topk_v': 1,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 10,
    'latent_dim': 256,
    'grad_clip': 1,
    'model': 'TALLSIREN',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Carla',
    'white_back': True,
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'learnable_dist': False
}

CATS = {
    0: {'batch_size': 28, 'num_steps': 24, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(200e3): {},


    'dataset_path': '/home/ericryanchan/graf-beta/data/carla/carla/*.png',
    'fov': 12,
    'ray_start': 0.8,
    'ray_end': 1.2,
    'fade_steps': 10000,
    'h_stddev': 0.5,
    'v_stddev': 0.4,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'StridedDiscriminator',
    'dataset': 'Cats',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
}

config_ldmk = EasyDict(
    {
        "ckpt_path": "./models/model_106/ldmk_detector/exp_mobile_v17-ceph-long/ckpt_best_model.pth.tar",
        "model": {
            "conv_channels": [1, 10, 16, 24, 32, 64, 64, 96, 128, 128],
            "ip_channels": [128, 128],
            "init_type": 'init'
        },
        "num_landmarks": 106,
        "crop_size": 112
    }
)

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage,
    # i.e. the epoch it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict

def check_diff(a, b):
    for key in a.keys():
        val = a[key]
        if key in b.keys():
            if a[key] != b[key]:
                if isinstance(a[key], dict) and isinstance(b[key], dict):
                    check_diff(a[key], b[key])
                else:
                    print(f"{key} changed from {a[key]} to {b[key]}.")
        else:
            print(f"{key} not found.")

if __name__ == "__main__":
    check_diff(recon4_minv1e1_reld1e1, recon4_minv1e1_reld1e2)
    check_diff(recon4_minv1e1_reld1e2, recon4_minv1e1_reld1e1)
    # check_diff(recon4_norm_glbmean_minv1e1_reld1e2_newctex5, recon4_minv1e1_reld1e2_cgeo3e2mg1e_2)
     
    # check_diff(CelebA_pigan_spade_tddfa_gan, CelebA_pigan_spade_tddfa10)
