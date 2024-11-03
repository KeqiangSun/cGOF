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


class LatentIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

        self.logdir = paths_config.logdir
        # self.writer = SummaryWriter(os.path.join(self.logdir, 'summaries'))
        self.writer = None
        self.deca = DECA(device=global_config.device)
        self.dataloader = data_loader
        self.set_pose()
        self.frame_idx = 0
        self.image_name = self.dataloader.dataset.image_name

        # self.G.rendering_kwargs['depth_resolution'] *= 4
        # self.G.rendering_kwargs['depth_resolution_importance'] *= 4

    def set_pose(self):
        if self.dataloader.dataset.pose is not None:
            target_pose = self.dataloader.dataset.pose
            o = target_pose[0:3, 3]
            o = 2.7 * o / np.linalg.norm(o)
            target_pose[0:3, 3] = o
            target_pose = np.reshape(target_pose, -1)
        else:
            # get pose default
            target_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=global_config.device), radius=2.7, device=global_config.device).reshape(16)
            target_pose = target_pose.detach().cpu().numpy()

        intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
        target_pose = np.concatenate([target_pose, intrinsics])
        self.target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)

    def sample_w(self):
        with torch.no_grad():
            # sample an image
            z_sample = np.random.RandomState(123).randn(1, self.G.z_dim)
            z_sample = torch.from_numpy(z_sample).to(global_config.device)
            w = self.G.mapping(z_sample, self.target_pose, truncation_psi=0.7)  # [N, L, C]

        # w = torch.from_numpy(np.load('seed0000.npy')).to(global_config.device)
        w = torch.nn.Parameter(w, requires_grad=True)

        return w
    
    def get_attributes(self, image):
        image = (image + 1) * 0.5

        # need to resize image to 224 x 224
        image = F.interpolate(image, size=(224, 224), mode='bilinear')
        dat = self.deca.encode(image)

        return dat['exp'], dat['shape'], dat['light'], dat['pose']

    def generate_image(self, w, return_depth=False):
        # broadcast pose to batch dimension
        target_pose = self.target_pose.repeat(w.shape[0], 1)
        output = self.G.synthesis(w, target_pose, noise_mode='const', force_fp32=True)

        if return_depth:
            depth = F.interpolate(output['image_depth'], size=(512, 512),
                                  mode='bilinear', align_corners=False)

            return output['image'], depth
        else:
            return output['image']

    def load_attributes(self, fname, fname_png):
        dat = np.load(fname, allow_pickle=True)[()]
        exp = dat['exp'].to(global_config.device)
        pose = dat['pose'].to(global_config.device)
        img = skimage.io.imread(fname_png)
        img = (skimage.transform.resize(img, (512, 512)) * 255).astype(np.uint8)

        return exp, img, pose

    def expr_loss(self, expr, target_expr, ident, orig_ident, light, orig_light, pose, target_pose):
        loss = l2_loss.l2_loss(expr, target_expr)
        loss += 5 * l2_loss.l2_loss(light, orig_light)
        loss += 0.5 * l2_loss.l2_loss(ident, orig_ident)
        loss += 10 * l2_loss.l2_loss(pose[:, 3:], target_pose[:, 3:])

        return loss

    def photo_loss(self, output, target, target_mask):
        return l2_loss.l2_loss(output*target_mask, target)

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
            generated_images, generated_depths = self.generate_image(w, return_depth=True)

            if hyperparameters.temporal_consistency_loss:
                temporal_mask = 1-face_mask - (1 - face_bg_mask)
            else:
                temporal_mask = None

            if hyperparameters.use_mouth_inpainting:
                face_bg_mask = morph.erosion(face_bg_mask, torch.ones(32, 32, device=global_config.device))
                face_bg_mask = gaussian_blur2d(face_bg_mask, (9, 9), (3, 3))
                #save_image(face_bg_mask, "/orion/u/connorzl/projects/S-GAN/stylegan3/pti_inversion/logs/puppet/face_bg_mask2.png")
                face_bg_img = face_bg_mask * face_bg_img + (1 - face_bg_mask) * w_img
                #save_image((face_bg_img+1)/2, "/orion/u/connorzl/projects/S-GAN/stylegan3/pti_inversion/logs/puppet/face_bg_img2.png")
                face_bg_mask = torch.ones_like(face_bg_mask)

            # debug
            # import matplotlib.pyplot as plt
            # plt.subplot(131)
            # # plt.imshow(temporal_mask[0].detach().cpu().permute(1, 2, 0).numpy())
            # plt.imshow(face_bg_mask[0].detach().cpu().permute(1, 2, 0).numpy())
            # plt.subplot(132)
            # plt.imshow((face_bg_img[0].detach().cpu().permute(1, 2, 0).numpy()+1)/2)
            # plt.subplot(133)
            # plt.imshow((generated_images[0].detach().cpu().permute(1, 2, 0).numpy()+1)/2)
            # plt.show()

            loss, _, _ = self.calc_loss(face_bg_mask * generated_images, face_bg_img,
                                        None, self.G, False, w, temporal_mask=temporal_mask,
                                        depth=generated_depths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            global_config.training_step += 1

            #log_utils.tb_log_images(torch.cat(((face_bg_img+1)/2, (generated_images+1)/2), dim=-1),
            #                        self.writer,  i, label='gan_finetune')

            if i % 2000 == 0 and i > 0:
                self.render_frames(i)
                torch.save(self.G, f'{self.logdir}/output_{i}/model.pt')

        self.render_frames(i+1)
        torch.save(self.G, f'{self.logdir}/output_{i+1}/model.pt')


    def render_frames(self, it):

        # make output dir
        outdir = os.path.join(self.logdir, f'output_{it}')
        os.makedirs(outdir, exist_ok=True)

        # render each frame and save
        for idx, sample in enumerate(tqdm(self.dataloader.dataset)):
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
            os.makedirs(os.path.join(self.logdir, 'w_output'), exist_ok=True)

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
                    w_pivot, noise_bufs, all_w_opt = self.calc_inversions(face_img, self.image_name, self.logdir,
                                                                          mask=face_mask, initial_w=None,
                                                                          writer=None, write_video=True,
                                                                          num_steps=hyperparameters.first_inv_steps)
                else:
                    w_pivot, noise_bufs, all_w_opt = self.calc_inversions(face_img, self.image_name, self.logdir,
                                                                          mask=face_mask, initial_w=w_next,
                                                                          writer=None,
                                                                          write_video=False,
                                                                          num_steps=hyperparameters.first_inv_steps//20)

                w_next = w_pivot[-1, 0, :][None, None, :].detach().cpu().numpy()
                w = w_pivot.to(global_config.device).detach()
                w.requires_grad = False
                w_opt.append(w)

                with torch.no_grad():
                    generated_images = self.generate_image(w)

                # log image
                #log_utils.tb_log_images(torch.cat(((face_img+1)/2, (generated_images+1)/2), dim=-1),
                #                                   self.writer, idx, label='w_proj')

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


    # deprecated
    def train_expression(self):
        np.random.seed(1989)
        torch.manual_seed(1989)

        # sample some w to manipulate
        w = self.sample_w()
        w_orig = w.clone().detach()

        with torch.no_grad():
            orig_image = (self.generate_image(w) + 1) * 255/2
            orig_image = orig_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

        
        # hardcode target expression for now
        for frame in range(1, 300, 10):
            print(f'FRAME: {frame}')
            target_npy = f'/home/lindell/workspace/DECA/biden/{frame:03d}/{frame:03d}_attr.npy'
            target_png = f'/home/lindell/workspace/DECA/biden/{frame:03d}/{frame:03d}_inputs.jpg'
            target_exp, target_img = self.load_attributes(target_npy, target_png)

            # set up output directories
            embedding_dir = f'./embeddings/latent'
            os.makedirs(embedding_dir, exist_ok=True)
            vid_path = f'{embedding_dir}' + '/' + 'rgb_proj.mp4'
            rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

            # initialize generator
            self.restart_training()

            w = w_orig.clone().detach()
            w.requires_grad = True
            self.optimizer = torch.optim.Adam([w], lr=hyperparameters.first_inv_lr)
            # self.optimizer = torch.optim.Adam(list(self.G.parameters()) + [w], lr=hyperparameters.pti_learning_rate)



            for i in tqdm(range(hyperparameters.first_inv_steps)):

                # get image
                generated_images = self.generate_image(w)

                # import matplotlib
                # import matplotlib.pyplot as plt
                # matplotlib.use('TkAgg')
                # plt.imshow(generated_images.cpu().detach().squeeze().permute(1, 2, 0))
                # plt.show()
                # break

                # get expression
                exp, ident, light = self.get_attributes(generated_images)

                if i == 0:
                    orig_ident = ident.clone().detach()
                    orig_light = light.clone().detach()

                # TODO: need identity preserving expression loss
                loss = self.expr_loss(exp, target_exp, ident, orig_ident, light, orig_light)

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                tqdm.write(f'loss: {loss:.02f}')

                if i % 5 == 0:
                    synth_image = generated_images
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    rgb_video.append_data(np.concatenate([orig_image, synth_image, target_img], axis=1))

                if i == hyperparameters.max_pti_steps - 1:
                    synth_image = generated_images
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    skimage.io.imsave(f'{embedding_dir}' + '/' + f'w_{frame:03d}.png', np.concatenate([orig_image, synth_image, target_img], axis=1))

                global_config.training_step += 1

            ### PTI
            self.optimizer = torch.optim.Adam(list(self.G.parameters()), lr=hyperparameters.pti_learning_rate)
            for i in tqdm(range(hyperparameters.max_pti_steps)):

                # get image
                generated_images = self.generate_image(w)

                # import matplotlib
                # import matplotlib.pyplot as plt
                # matplotlib.use('TkAgg')
                # plt.imshow(generated_images.cpu().detach().squeeze().permute(1, 2, 0))
                # plt.show()
                # break

                # get expression
                exp, ident, light = self.get_attributes(generated_images)

                if i == 0:
                    orig_ident = ident.clone().detach()
                    orig_light = light.clone().detach()

                # TODO: need identity preserving expression loss
                loss = self.expr_loss(exp, target_exp, ident, orig_ident, light, orig_light)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tqdm.write(f'loss: {loss:.02f}')

                if i == hyperparameters.max_pti_steps - 1:
                    synth_image = generated_images
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    skimage.io.imsave(f'{embedding_dir}' + '/' + f'pti_{frame:03d}.png', np.concatenate([orig_image, synth_image, target_img], axis=1))

                global_config.training_step += 1


            rgb_video.close()

            torch.save(self.G, f'{embedding_dir}/model.pt')
