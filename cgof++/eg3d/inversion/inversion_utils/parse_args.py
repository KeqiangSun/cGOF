import os
from configs import global_config, paths_config, hyperparameters
import configargparse


def parse_args():
    p = configargparse.ArgumentParser()

    # config file, output directories
    p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
    p.add_argument('--experiment_name', type=str, default='train_img', required=True,
                   help='path to directory where checkpoints & tensorboard events will be saved.')
    p.add_argument('--input_pose_path', type=str, default='./cameras.json',
                   help='path to pose json file')
    p.add_argument('--input_data_path', type=str, default='./results_small_vanilla/000',
                   help='path to folder with images')
    p.add_argument('--gpu', type=int, default=0,
                   help='gpu to use')

    # general training options
    p.add_argument('--latent_ball_num_of_samples', type=int, default=1)
    p.add_argument('--locality_regularization_interval', type=int, default=1)
    p.add_argument('--use_locality_regularization', type=bool, default=False)
    p.add_argument('--use_noise_regularization', type=bool, default=False)
    p.add_argument('--use_mouth_inpainting', dest='mouth_inpainting', action='store_true')
    p.add_argument('--no_mouth_inpainting', dest='mouth_inpainting', action='store_false')
    p.set_defaults(mouth_inpainting=True)
    p.add_argument('--temporal_consistency_loss', type=bool, default=False)

    p.add_argument('--use_stylegan2d', dest='use_stylegan2d', action='store_true')
    p.add_argument('--use_stylegan3d', dest='use_stylegan2d', action='store_false')
    p.set_defaults(use_stylegan2d=False)

    # regularizer strengths
    p.add_argument('--regularizer_l2_lambda', type=float, default=0.1)
    p.add_argument('--regularizer_lpips_lambda', type=float, default=0.1)
    p.add_argument('--regularizer_alpha', type=float, default=30)
    p.add_argument('--pt_l2_lambda', type=float, default=1)
    p.add_argument('--pt_lpips_lambda', type=float, default=1)
    p.add_argument('--pt_temporal_photo_lambda', type=float, default=0)
    p.add_argument('--pt_temporal_depth_lambda', type=float, default=0)

    # optimization
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--first_inv_steps', type=int, default=2000)
    p.add_argument('--first_inv_lr', type=float, default=5e-4)

    p.add_argument('--max_pti_steps', type=int, default=2000)
    p.add_argument('--pti_learning_rate', type=float, default=1e-3)

    opt = p.parse_args()

    paths_config.logdir = os.path.join(opt.logging_root, opt.experiment_name)
    paths_config.input_pose_path = opt.input_pose_path
    paths_config.input_data_path = opt.input_data_path

    global_config.cuda_visible_devices = str(opt.gpu)
    global_config.run_stylegan2d = opt.use_stylegan2d

    hyperparameters.latent_ball_num_of_samples = opt.latent_ball_num_of_samples
    hyperparameters.locality_regularization_interval = opt.locality_regularization_interval
    hyperparameters.use_locality_regularization = opt.use_locality_regularization
    hyperparameters.use_noise_regularization = opt.use_noise_regularization
    hyperparameters.use_mouth_inpainting = opt.mouth_inpainting
    hyperparameters.temporal_consistency_loss = opt.temporal_consistency_loss

    hyperparameters.regularizer_l2_lambda = opt.regularizer_l2_lambda
    hyperparameters.regularizer_lpips_lambda = opt.regularizer_lpips_lambda
    hyperparameters.regularizer_alpha = opt.regularizer_alpha
    hyperparameters.pt_l2_lambda = opt.pt_l2_lambda
    hyperparameters.pt_lpips_lambda = opt.pt_lpips_lambda
    hyperparameters.pt_temporal_photo_lambda = opt.pt_temporal_photo_lambda
    hyperparameters.pt_temporal_depth_lambda = opt.pt_temporal_depth_lambda

    hyperparameters.batch_size = opt.batch_size
    hyperparameters.first_inv_steps = opt.first_inv_steps
    hyperparameters.first_inv_lr = opt.first_inv_lr
    hyperparameters.max_pti_steps = opt.max_pti_steps
    hyperparameters.pti_learning_rate = opt.pti_learning_rate

    # back up configs
    os.makedirs(paths_config.logdir, exist_ok=True)
    p.write_config_file(opt, [os.path.join(paths_config.logdir, 'config.ini')])

    with open(os.path.join(paths_config.logdir, 'params.txt'), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
