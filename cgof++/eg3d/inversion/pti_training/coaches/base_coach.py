import abc
import os
import pickle
from argparse import Namespace
import wandb
import os.path
from criteria.localitly_regulizer import Space_Regulizer
import torch
from torchvision import transforms
from lpips import LPIPS
from pti_training.projectors import w_projector, w_projector_grayscale
from configs import global_config, paths_config, hyperparameters
from criteria import l2_loss
from inversion_utils.log_utils import log_image_from_w
from inversion_utils.models_utils import toogle_grad, load_stylegan2d, load_3dgan
import numpy as np
import json

def dfs_freeze(model): 
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)

class BaseCoach:
    def __init__(self, data_loader, use_wandb, reload_modules=False, use_face_recon=False):

        self.use_wandb = use_wandb
        self.data_loader = data_loader
        self.w_pivots = {}
        self.image_counter = 0
        self.reload_modules = reload_modules
        self.use_face_recon = use_face_recon

        # Initialize loss
        self.lpips_loss = LPIPS(net=hyperparameters.lpips_type).to(global_config.device).eval()

        self.restart_training()

        # Initialize checkpoint dir
        self.checkpoint_dir = paths_config.checkpoints_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def restart_training(self):

        # Initialize networks
        self.G = load_3dgan(self.reload_modules)
        toogle_grad(self.G, True)
        self.original_G = load_3dgan(self.reload_modules)

        self.space_regulizer = Space_Regulizer(self.original_G, self.lpips_loss)
        self.optimizer = self.configure_optimizers()

        if self.use_face_recon:
            from Deep3DFaceRecon_pytorch import init_face_recon
            self.face_recon, _ = init_face_recon(global_config.device)
        else:
            self.face_recon = None

    def get_inversion(self, w_path_dir, image_name, image):
        embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
        os.makedirs(embedding_dir, exist_ok=True)

        w_pivot = None

        if hyperparameters.use_last_w_pivots:
            w_pivot = self.load_inversions(w_path_dir, image_name)

        if not hyperparameters.use_last_w_pivots or w_pivot is None:
            w_pivot = self.calc_inversions(image, image_name)
            torch.save(w_pivot, f'{embedding_dir}/0.pt')

        w_pivot = w_pivot.to(global_config.device)
        return w_pivot

    def load_inversions(self, w_path_dir, image_name):
        if image_name in self.w_pivots:
            return self.w_pivots[image_name]

        if hyperparameters.first_inv_type == 'w+':
            w_potential_path = f'{w_path_dir}/{paths_config.e4e_results_keyword}/{image_name}/model_{image_name}.pt'
        else:
            w_potential_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}/model_{image_name}.pt'

        if not os.path.isfile(w_potential_path):
            return None
        w = torch.load(w_potential_path).to(global_config.device)
        self.w_pivots[image_name] = w
        return w

    def calc_inversions(self, image, image_name, embedding_dir, grayscale=False, mask=None,
                        initial_w=None, writer=None, num_steps=hyperparameters.first_inv_steps,
                        write_video=False):

        if hyperparameters.first_inv_type == 'w+':
            w = self.get_e4e_inversion(image)

        else:
            id_image = torch.squeeze((image.to(global_config.device) + 1) / 2) * 255
            if grayscale:
                id_image = id_image.unsqueeze(0)
                w = w_projector_grayscale.project(self.G, id_image, embedding_dir, device=torch.device(global_config.device), w_avg_samples=600,
                        num_steps=hyperparameters.first_inv_steps, w_name=image_name,
                        use_wandb=self.use_wandb)
            else:
                w = w_projector.project(self.G, self.face_recon, id_image, embedding_dir,
                                        device=torch.device(global_config.device), w_avg_samples=600,
                                        num_steps=num_steps, w_name=image_name,
                                        use_wandb=self.use_wandb, mask=mask, initial_w=initial_w,
                                        writer=writer, write_video=write_video)

        return w

    @abc.abstractmethod
    def train(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.G.parameters(), lr=hyperparameters.pti_learning_rate)
        return optimizer

    def calc_loss(self, generated_images, real_images, log_name, new_G, use_ball_holder, w_batch,
                  depth=None, temporal_mask=None):
        loss = 0.0

        if hyperparameters.pt_l2_lambda > 0:
            l2_loss_val = l2_loss.l2_loss(generated_images, real_images)
            if self.use_wandb:
                wandb.log({f'MSE_loss_val_{log_name}': l2_loss_val.detach().cpu()}, step=global_config.training_step)
            loss += l2_loss_val * hyperparameters.pt_l2_lambda
        if hyperparameters.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.mean(loss_lpips)
            if self.use_wandb:
                wandb.log({f'LPIPS_loss_val_{log_name}': loss_lpips.detach().cpu()}, step=global_config.training_step)
            loss += loss_lpips * hyperparameters.pt_lpips_lambda

        if use_ball_holder and hyperparameters.use_locality_regularization:
            ball_holder_loss_val = self.space_regulizer.space_regulizer_loss(new_G, w_batch, use_wandb=self.use_wandb)
            loss += ball_holder_loss_val

        if hyperparameters.pt_temporal_photo_lambda > 0:
            loss_tc = l2_loss.l2_loss(temporal_mask[::2] * generated_images[::2],
                                      temporal_mask[1::2] * generated_images[1::2])

            loss += loss_tc * hyperparameters.pt_temporal_photo_lambda

        if hyperparameters.pt_temporal_depth_lambda > 0:
            loss_depth_tc = l2_loss.l2_loss(temporal_mask[::2] * depth[::2],
                                            temporal_mask[1::2] * depth[1::2])

            loss += loss_depth_tc * hyperparameters.pt_temporal_depth_lambda

        return loss, l2_loss_val, loss_lpips

    def forward(self, w):
        if os.path.basename(paths_config.input_pose_path).split(".")[1] == "json":
            f = open(paths_config.input_pose_path)
            target_pose = np.asarray(json.load(f)[paths_config.input_id]['pose']).astype(np.float32)
            f.close()
            o = target_pose[0:3, 3]
            o = 2.7 * o / np.linalg.norm(o)
            target_pose[0:3, 3] = o
            target_pose = np.reshape(target_pose, -1)    
        else:
            target_pose = np.load(paths_config.input_pose_path).astype(np.float32)
            target_pose = np.reshape(target_pose, -1)

        intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
        target_pose = np.concatenate([target_pose, intrinsics])
        target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)
        generated_images = self.G.synthesis(w, target_pose, noise_mode='const', force_fp32=True)
        return generated_images

