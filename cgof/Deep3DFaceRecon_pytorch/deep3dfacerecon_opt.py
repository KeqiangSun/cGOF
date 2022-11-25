import numpy as np
from easydict import EasyDict as edict

deep3dfacerecon_dict = {
    "add_image": True,
    "bfm_folder": 'Deep3DFaceRecon_pytorch/BFM',
    "bfm_model": 'BFM_model_front.mat',
    "camera_d": 10.0,
    "center": 112.0,
    "checkpoints_dir": 'Deep3DFaceRecon_pytorch/checkpoints',
    "dataset_mode": None,
    "ddp_port": '12355',
    "display_per_batch": True,
    "epoch": '20',
    "eval_batch_nums": np.inf,
    "focal": 1015.0,
    "gpu_ids": '0',
    # "img_folder": 'Deep3DFaceRecon_pytorch/datasets/pigan/imgs',
    "img_folder": './shapes/pigan/splits/grid_seed0/',
    "init_path": 'Deep3DFaceRecon_pytorch/checkpoints/init_model/resnet50-0676ba61.pth',
    "isTrain": False,
    "model": 'facereconfitting',
    "name": 'face_recon_feat0.2_augment',
    "net_recon": 'resnet50',
    "phase": 'test',
    "suffix": '',
    "use_ddp": False,
    "use_last_fc": False,
    "verbose": False,
    "vis_batch_nums": 1,
    "world_size": 1,
    "z_far": 15.0,
    "z_near": 5.0,
    "coeff_static_path": "Deep3DFaceRecon_pytorch/coeff_distribution_L.npz",

    "camera_distance_pigan": 0.0,
    "focal_pigan": 1217.838650140491,
    "center_pigan": 128.0,
    "fov_pigan": 12.0,
    # "z_near_pigan": -0.12,
    # "z_far_pigan": 0.12,
    "z_near_pigan": 0.88,
    "z_far_pigan": 1.12,
}
deep3dfacerecon_opt = edict(deep3dfacerecon_dict)
