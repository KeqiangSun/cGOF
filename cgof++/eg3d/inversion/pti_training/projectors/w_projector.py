# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import os
import copy
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from configs import global_config, hyperparameters, paths_config
from inversion_utils import log_utils
import dnnlib
import imageio
import json
from camera_utils import LookAtPoseSampler
from PIL import Image

def project_plus(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        embedding_dir,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.025,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,  # DL: originally 1e5 from stylegan2 paper
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        mask=None,
        writer=None
):
    if mask is None:
        mask = torch.ones_like(target)

    np.random.seed(1989)
    torch.manual_seed(1989)
    
    if w_name is None:
        target_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=global_config.device), radius=2.7, device=global_config.device).reshape(4, 4)
        target_pose = target_pose.cpu().numpy()
    else:
        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), \
            'only batch size==1  supported'
        f = open(paths_config.input_pose_path)
        target_pose = np.asarray(json.load(f)[paths_config.input_id]['pose']).astype(np.float32)
        f.close()

    o = target_pose[0:3, 3]
    print("norm of origin before normalization:", np.linalg.norm(o))
    o = 2.7 * o / np.linalg.norm(o)
    target_pose[0:3, 3] = o
    target_pose = np.reshape(target_pose, -1)    

    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.tensor(target_pose, device=device).unsqueeze(0)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    if target.ndim == 4:
        batchsize = target.shape[0]
        target_images = target.to(device).float()
        bs = target.shape[0]
    else:
        batchsize = 1
        target_images = target.unsqueeze(0).to(device).to(torch.float32)

    real_image = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    target_images_orig = target_images.clone()
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        mask = F.interpolate(mask, size=(256, 256), mode='area')

    target_features = vgg16(mask*target_images, resize_images=False, return_lpips=True)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), target_pose.repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    start_w = start_w.repeat(batchsize, 0)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # broadcast target pose to batch size
    target_pose = target_pose.repeat(batchsize, 1)

    start_w = np.repeat(start_w, G.backbone.mapping.num_ws, axis=1)
    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    all_w_opt = []
    num_ws = G.backbone.mapping.num_ws
    vid_path = f'{embedding_dir}' + '/' + 'w_rgb_proj.mp4'
    rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = w_opt + w_noise
        all_w_opt.append(w_opt.detach())

        synth_images_orig = G.synthesis(ws, target_pose, noise_mode='const', force_fp32=True)['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images_orig + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(mask*synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        if hyperparameters.use_noise_regularization:
            reg_loss = 1.0
        else:
            reg_loss = 0.0

        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % 5 == 0:
            synth_image = (synth_images_orig + 1) * (255/2)

            if writer is not None:
                log_utils.tb_log_images(torch.cat((target_images_orig / 255., (synth_images_orig+1)/2), dim=-1),
                                         writer,  step, label='w')

            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            rgb_video.append_data(np.concatenate([real_image, synth_image], axis=1))
            
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    rgb_video.close()
    all_w_opt = torch.cat(all_w_opt, 0)
    del G
    return w_opt, noise_bufs, all_w_opt

def project(
        G,
        face_recon,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        embedding_dir,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.025,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,  # DL: originally 1e5 from stylegan2 paper
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        w_name: str,
        mask=None,
        writer=None,
        write_video=False
):
    if mask is None:
        mask = torch.ones_like(target)

    np.random.seed(1989)
    torch.manual_seed(1989)
    
    if w_name is None:
        target_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=global_config.device), radius=2.7, device=global_config.device).reshape(4, 4)
        target_pose = target_pose.cpu().numpy()
    else:
        assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution), \
            'only batch size==1  supported'
        if os.path.basename(paths_config.input_pose_path).split(".")[1] == "json":
            f = open(paths_config.input_pose_path)
            target_pose = np.asarray(json.load(f)[paths_config.input_id]['pose']).astype(np.float32)
            f.close()
            o = target_pose[0:3, 3]
            print("norm of origin before normalization:", np.linalg.norm(o))
            o = 2.7 * o / np.linalg.norm(o)
            target_pose[0:3, 3] = o
            target_pose = np.reshape(target_pose, -1)  
        else:
            target_pose = np.load(paths_config.input_pose_path).astype(np.float32)
            target_pose = np.reshape(target_pose, -1)  

    intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
    target_pose = np.concatenate([target_pose, intrinsics])
    target_pose = torch.tensor(target_pose, device=device).unsqueeze(0)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    if target.ndim == 4:
        batchsize = target.shape[0]
        target_images = target.to(device).float()
        bs = target.shape[0]
    else:
        batchsize = 1
        target_images = target.unsqueeze(0).to(device).to(torch.float32)

    real_image = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    target_images_orig = target_images.clone()
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        mask = F.interpolate(mask.unsqueeze(0), size=(256, 256), mode='area')

    target_features = vgg16(mask*target_images, resize_images=False, return_lpips=True)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), target_pose.repeat(w_avg_samples, 1))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    start_w = start_w.repeat(batchsize, 0)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    # broadcast target pose to batch size
    target_pose = target_pose.repeat(batchsize, 1)

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    all_w_opt = []
    num_ws = G.backbone.mapping.num_ws

    if write_video:
        vid_path = f'{embedding_dir}' + '/' + 'w_rgb_proj.mp4'
        rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

    # load z
    if face_recon is not None:
        
        import scipy.io
        mat = scipy.io.loadmat(paths_config.z_path)
        mat = np.concatenate(
            [
                mat['id'], mat['exp'], np.zeros_like(mat['id']),
                mat['tex'], mat['gamma'], np.zeros_like(mat['id'])
            ], axis=1)
        mat = torch.from_numpy(mat).to(device)
        z = face_recon.split_z(mat)
        z = face_recon.norm_coeff(z)
        z['bg_geo'] = torch.zeros_like(z['id'])
        z['bg_tex'] = torch.zeros_like(z['id'])
        z = torch.cat([z['id'], z['exp'], z['bg_geo'], z['tex'], z['gamma'], z['bg_tex']], dim=1)
        G.z = z

    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        # don't add noise if we're optimizing from an initial 'w'
        # we want to stay somewhat close to that
        if initial_w is not None:
            w_noise_scale = 0.

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.backbone.mapping.num_ws, 1])
        all_w_opt.append(w_opt.detach().repeat([1, num_ws, 1]))

        synth_images_orig = G.synthesis(ws, target_pose, face_recon=face_recon, noise_mode='const', force_fp32=True)['image']

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images_orig + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(mask*synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        if hyperparameters.use_noise_regularization:
            reg_loss = 1.0
        else:
            reg_loss = 0.0

        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % 5 == 0:
            synth_image = (synth_images_orig + 1) * (255/2)

            if writer is not None:
                log_utils.tb_log_images(torch.cat((target_images_orig / 255., (synth_images_orig+1)/2), dim=-1),
                                         writer,  step, label='w')

            if write_video:
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                rgb_video.append_data(np.concatenate([real_image, synth_image], axis=1))
            
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    if write_video:
        rgb_video.close()
    all_w_opt = torch.cat(all_w_opt, 0)
    del G
    return w_opt.repeat([1, num_ws, 1]), noise_bufs, all_w_opt


def project2d(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        embedding_dir,
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.025,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,  # DL: originally 1e5 from stylegan2 paper
        verbose=False,
        device: torch.device,
        use_wandb=False,
        initial_w=None,
        image_log_step=global_config.image_rec_result_log_snapshot,
        mask=None,
        writer=None,
        write_video=False
):
    if mask is None:
        mask = torch.ones_like(target)

    np.random.seed(1989)
    torch.manual_seed(1989)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    if target.ndim == 4:
        batchsize = target.shape[0]
        target_images = target.to(device).float()
        bs = target.shape[0]
    else:
        batchsize = 1
        target_images = target.unsqueeze(0).to(device).to(torch.float32)

    real_image = target_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

    target_images_orig = target_images.clone()
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        mask = F.interpolate(mask, size=(256, 256), mode='area')

    target_features = vgg16(mask*target_images, resize_images=False, return_lpips=True)

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    start_w = start_w.repeat(batchsize, 0)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device,
                         requires_grad=True)  # pylint: disable=not-callable

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999),
                                 lr=hyperparameters.first_inv_lr)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    all_w_opt = []
    num_ws = G.mapping.num_ws
    if write_video:
        vid_path = f'{embedding_dir}' + '/' + 'w_rgb_proj.mp4'
        rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

    for step in tqdm(range(num_steps)):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp

        # don't add noise if we're optimizing from an initial 'w'
        # we want to stay somewhat close to that
        if initial_w is not None:
            w_noise_scale = 0.

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        all_w_opt.append(w_opt.detach().repeat([1, num_ws, 1]))

        synth_images_orig = G.synthesis(ws, noise_mode='const', force_fp32=True)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images_orig + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(mask*synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        if hyperparameters.use_noise_regularization:
            reg_loss = 1.0
        else:
            reg_loss = 0.0

        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % 5 == 0:
            if writer is not None:
                log_utils.tb_log_images(torch.cat((target_images_orig / 255., (synth_images_orig+1)/2), dim=-1),
                                         writer,  step, label='w')

            if write_video:
                synth_image = (synth_images_orig + 1) * (255/2)
                synth_image = F.interpolate(synth_image, size=(512, 512), mode='area')
                synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                rgb_video.append_data(np.concatenate([real_image, synth_image], axis=1))
            
        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
    if write_video:
        rgb_video.close()
    all_w_opt = torch.cat(all_w_opt, 0)
    del G
    return w_opt.repeat([1, num_ws, 1]), noise_bufs, all_w_opt
