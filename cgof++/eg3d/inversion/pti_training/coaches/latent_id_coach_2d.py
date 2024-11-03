import os
import sys
sys.path.append("../../../")

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler
from models.deca import DECA
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from pti_training.coaches.base_coach import BaseCoach
from inversion_utils.log_utils import log_images_from_w
from inversion_utils.ImagesDataset import RandomPairSampler
from PIL import Image
import imageio
import numpy as np
import pickle
from camera_utils import LookAtPoseSampler
from criteria import l2_loss
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib
import cv2
from inversion_utils import log_utils
from kornia import morphology as morph
from kornia.filters import gaussian_blur2d
from torchvision.utils import save_image

#matplotlib.use('TkAgg')


class LatentIDCoach2D(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

        self.logdir = paths_config.logdir
        self.deca = DECA(device=global_config.device)
        self.dataloader = data_loader
        self.frame_idx = 0

    def generate_image(self, w):
        # broadcast pose to batch dimension
        output = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        output = F.interpolate(output, size=(512, 512), mode='area')
        return output

    def tune_generator(self):

        # load optimized 'w' values
        self.dataloader.dataset.load_w(self.logdir)

        # create dataset with shuffled dataloader
        if hyperparameters.temporal_consistency_loss:
            self.dataloader = DataLoader(self.dataloader.dataset,
                                         batch_sampler=BatchSampler(RandomPairSampler(self.dataloader.dataset),
                                                              drop_last=True,
                                                              batch_size=hyperparameters.batch_size))
        else:
            self.dataloader = DataLoader(self.dataloader.dataset,
                                         batch_size=self.dataloader.batch_size,
                                         shuffle=True)

        sample_generator = iter(self.dataloader)

        for i in tqdm(range(hyperparameters.max_pti_steps)):

            # set learning rate
            lr_rampdown_length = 0.25
            lr_rampup_length = 0.05
            t = i / hyperparameters.max_pti_steps
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = hyperparameters.pti_learning_rate * lr_ramp

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # get batch
            try:
                sample = next(sample_generator)
            except StopIteration:
                sample_generator = iter(self.dataloader)
                sample = next(sample_generator)

            face_mask = sample['face_mask'].to(global_config.device)
            face_bg_mask = sample['face_bg_mask'].to(global_config.device)
            face_bg_img = sample['face_bg_img'].to(global_config.device)
            w = sample['w'].to(global_config.device)
            w_img = sample['w_img'].to(global_config.device)

            # get image
            generated_images = self.generate_image(w)

            if hyperparameters.temporal_consistency_loss:
                temporal_mask = 1-face_mask - (1 - face_bg_mask)
            else:
                temporal_mask = None

            if hyperparameters.use_mouth_inpainting:
                face_bg_mask = morph.erosion(face_bg_mask, torch.ones(32, 32, device=global_config.device))
                face_bg_mask = gaussian_blur2d(face_bg_mask, (9, 9), (3, 3))
                face_bg_img = face_bg_mask * face_bg_img + (1 - face_bg_mask) * w_img
                face_bg_mask = torch.ones_like(face_bg_mask)

            loss, _, _ = self.calc_loss(face_bg_mask * generated_images, face_bg_img,
                                        None, self.G, False, w, temporal_mask=temporal_mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            global_config.training_step += 1

            if ((i+1) % 2000) == 0 or (i+1) == hyperparameters.max_pti_steps:
                self.render_frames(i+1)
                torch.save(self.G, f'{self.logdir}/output_{i+1}/model.pt')

        #self.render_frames(i+1)
        #torch.save(self.G, f'{self.logdir}/output_{i+1}/model.pt')


    def render_frames(self, it):

        # make output dir
        outdir = os.path.join(self.logdir, f'output_{it}')
        os.makedirs(outdir, exist_ok=True)

        # render each frame and save
        for idx, sample in enumerate(tqdm(self.dataloader.dataset)):
            if idx >= len(self.dataloader.dataset):
                break
            w = sample['w'].unsqueeze(0).to(global_config.device)
            generated_images = self.generate_image(w)

            # write out image
            generated_images = torch.clamp((generated_images + 1) / 2, 0, 1).squeeze().detach().cpu().numpy()
            generated_images = (generated_images.transpose(1, 2, 0) * 255).astype(np.uint8)
            skimage.io.imsave(os.path.join(outdir, f'{idx:04d}.png'), generated_images)

    def train(self):
        np.random.seed(1989)
        torch.manual_seed(1989)

        # step 1: fit 'w' for each frame of the target image
        if not os.path.exists(os.path.join(self.logdir, 'w.npy')):
            output_path = os.path.join(self.logdir, "w_output")
            os.makedirs(output_path, exist_ok=True)

            w_opt = []
            w_next = None
            for idx, sample in enumerate(tqdm(self.dataloader)):

                face_mask = sample['face_mask'].to(global_config.device)
                face_img = sample['face_img'].to(global_config.device)

                #face_bg_mask = sample['face_bg_mask'].to(global_config.device)
                #face_bg_img = sample['face_bg_img'].to(global_config.device)
                # optimize for w
                self.restart_training()

                if idx == 0:
                    w_pivot, noise_bufs, all_w_opt = self.calc_inversions(face_img, None, self.logdir,
                                                                          mask=face_mask, initial_w=None,
                                                                          writer=None,
                                                                          num_steps=hyperparameters.first_inv_steps)
                else:
                    w_pivot, noise_bufs, all_w_opt = self.calc_inversions(face_img, None, self.logdir,
                                                                          mask=face_mask, initial_w=w_next,
                                                                          writer=None,
                                                                          num_steps=hyperparameters.first_inv_steps//100)

                w_next = w_pivot[-1, 0, :][None, None, :].detach().cpu().numpy()
                w = w_pivot.to(global_config.device).detach()
                w.requires_grad = False
                w_opt.append(w)

                with torch.no_grad():
                    generated_images = self.generate_image(w)

                # save image
                for i, img in enumerate(generated_images):
                    index = sample['idx'][i]
                    img = ((img.detach().cpu().permute(1, 2, 0).numpy()+1)*255/2)
                    img = np.clip(img, 0, 255).astype(np.uint8)
                    skimage.io.imsave(os.path.join(self.logdir, 'w_output', f'{index:04d}.png'), img)

            w_opt = torch.cat(w_opt, dim=0)
            np.save(os.path.join(self.logdir, 'w.npy'), w_opt.detach().cpu().numpy())

        # step 2: finetune the generator on all the w_opt 
        # load the optimized 'w' and generated 'w' images to the dataset
        self.tune_generator()

