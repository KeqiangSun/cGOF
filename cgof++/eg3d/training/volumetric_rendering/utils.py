"""
Differentiable volumetric implementation used by pi-GAN generator.
"""

from contextlib import closing
from itertools import count
import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random

from .math_utils import *
# from math_utils_torch import *

import kornia
import cv2


def fancy_integration(
        rgb_sigma, z_vals, device, noise_std=0.5, last_back=False,
        white_back=False, clamp_mode=None, fill_mode=None,
        activate_mode=None, zy_data={}, **kwargs):
    """Performs NeRF volumetric rendering."""

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    if activate_mode == "tanh":
        alphas = (torch.tanh(2 * (2 * alphas - 1)) + 1) / 2
    else:
        alphas = alphas

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    depth_variance = torch.sum(weights * z_vals ** 2, -2) - depth_final ** 2
    if isinstance(zy_data, dict):
        zy_data['depth_variance'] = depth_variance

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor(
            [1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    if torch.any(torch.isinf(weights)) or torch.any(torch.isnan(weights)):
        torch.save(
            {
                "rgb_sigma": rgb_sigma,
                "z_vals": z_vals,
                "device": device,
                "noise_std": noise_std,
                "last_back": last_back,
                "white_back": white_back,
                "clamp_mode": clamp_mode,
                "fill_mode": fill_mode,
                "activate_mode": activate_mode,
                "noise": noise,
            },
            "debug/fancy_integration_input.pt"
        )
        # raise ValueError

    return rgb_final, depth_final, weights


def check_tensor(tensor):
    return f"max: {tensor.max()}  " + f"min: {tensor.min()}  " + f"shape: {tensor.shape}"


def sort_by_z_vals(tensor, z_vals):
    sorted_z_vals, indices = sort_z_vals(z_vals)
    sorted_tensor = torch.gather(
        tensor, -2, indices.expand(-1, -1, -1, 4))
    return sorted_tensor, sorted_z_vals


def sort_z_vals(z_vals):
    _, indices = torch.sort(z_vals, dim=-2)
    sorted_z_vals = torch.gather(z_vals, -2, indices)
    return sorted_z_vals, indices


def fancy_integration_debug(
        rgb_sigma, z_vals, device, noise_std=0.5, last_back=False,
        white_back=False, clamp_mode=None, fill_mode=None,
        activate_mode=None, zy_data={}, **kwargs):
    """Performs NeRF volumetric rendering."""
    device = rgb_sigma.device

    rgbs = rgb_sigma[..., :3]
    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    if activate_mode == "tanh":
        alphas = (torch.tanh(2 * (2 * alphas - 1)) + 1) / 2
    else:
        alphas = alphas

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    rgb_final = torch.sum(weights * rgbs, -2)
    depth_final = torch.sum(weights * z_vals, -2)
    depth_variance = torch.sum(weights * z_vals ** 2, -2) - depth_final ** 2
    # zy_data['depth_variance'] = depth_variance

    if white_back:
        rgb_final = rgb_final + 1-weights_sum

    if fill_mode == 'debug':
        rgb_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor(
            [1., 0, 0], device=rgb_final.device)
    elif fill_mode == 'weight':
        rgb_final = weights_sum.expand_as(rgb_final)

    if torch.any(torch.isinf(weights)) or torch.any(torch.isnan(weights)):
        torch.save(
            {
                "rgb_sigma": rgb_sigma,
                "z_vals": z_vals,
                "device": device,
                "noise_std": noise_std,
                "last_back": last_back,
                "white_back": white_back,
                "clamp_mode": clamp_mode,
                "fill_mode": fill_mode,
                "activate_mode": activate_mode,
                "noise": noise,
            },
            "debug/fancy_integration_input.pt"
        )
        # raise ValueError

    return rgb_final, depth_final, weights


def cal_weights(
    rgb_sigma, z_vals, device, noise_std=0.5, last_back=False,
    clamp_mode=None, activate_mode=None, **kwargs):
    """Performs NeRF volumetric rendering."""

    sigmas = rgb_sigma[..., 3:]

    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise "Need to choose clamp mode"

    if activate_mode == "tanh":
        alphas = (torch.tanh(2 * (2 * alphas - 1)) + 1) / 2
    else:
        alphas = alphas

    alphas_shifted = torch.cat(
        [torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back:
        weights[:, :, -1] += (1 - weights_sum)

    return alphas, weights


def get_initial_rays_trig(
        n, num_steps, device, fov, resolution, ray_start, ray_end, norm_mode='vec_len'):
    """Returns sample points, z_vals, and ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan(
        (2 * math.pi * fov / 360) / 2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1), norm_mode=norm_mode)

    z_vals = torch.linspace(
        ray_start, ray_end, num_steps, device=device
    ).reshape(1, num_steps, 1).repeat(W*H, 1, 1)  # W*H, num_steps, 1
    points = rays_d_cam.unsqueeze(1).repeat(
        1, num_steps, 1) * z_vals  # W*H, num_steps, 3

    points = torch.stack(
        n * [points])  # batchsize:n, sample num_steps steps from ray_start to ray_end.
    z_vals = torch.stack(n * [z_vals])  # linspace from ray_start to ray_end.
    rays_d_cam = torch.stack(n * [rays_d_cam]).to(device)  # unit vector emitted from the camera
    # (set as the origin of the coordinate system.)

    return points, z_vals, rays_d_cam


def vis_tensor_as_vert(path, points):
    from utils import mesh_io
    o = points.reshape(-1, 3)
    mesh_io.save_obj_vertex(path, o)


def vis_tensor_as_img(path, tensor):
    import cv2
    tensor = tensor.detach().cpu().numpy()
    b, c, H, W = tensor.shape
    for i in range(b):
        img = tensor[i].transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path+f'{i}.png', img*255)


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def transform_sampled_points(
        points, z_vals, ray_directions, device, yaw=None, pitch=None,
        h_stddev=1, v_stddev=1, h_mean=math.pi*0.5, v_mean=math.pi*0.5,
        mode='normal', return_cam2word=False):
    """Samples a camera position and maps points in camera space to world space."""

    n, num_rays, num_steps, channels = points.shape

    if z_vals is not None:
        # print('perturbing z_vals...')
        points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    # torch.manual_seed(np.random.randint(999))
    # phi: pitch
    # theta: yaw
    camera_origin, pitch, yaw = sample_camera_positions(
        n=points.shape[0], r=1, horizontal_stddev=h_stddev,
        vertical_stddev=v_stddev, horizontal_mean=h_mean,
        vertical_mean=v_mean, yaw=yaw, pitch=pitch, device=device, mode=mode)

    forward_vector = normalize_vecs(-camera_origin)

    camera_origin[..., 1] -= 0.012
    # camera_origin[..., 2] -= 0.12 * camera_origin[..., 0].abs()

    cam2world_matrix = create_cam2world_matrix(
        forward_vector, camera_origin, device=device)

    points_homogeneous = torch.ones(
        (points.shape[0], points.shape[1],
         points.shape[2], points.shape[3] + 1), device=device)
    points_homogeneous[:, :, :, :3] = points

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    transformed_points = torch.bmm(
        cam2world_matrix,
        points_homogeneous.reshape(n, -1, 4).permute(0, 2, 1)
    ).permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)

    transformed_ray_directions = torch.bmm(
        cam2world_matrix[..., :3, :3],
        ray_directions.reshape(n, -1, 3).permute(0, 2, 1)
    ).permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(
        cam2world_matrix, homogeneous_origins
    ).permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    if return_cam2word:
        return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, cam2world_matrix, pitch, yaw
    else:
        return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal', yaw=None, pitch=None):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    # if yaw is not None and pitch is not None:
    #     theta = math.pi / 2 + math.pi / 180.0 * yaw
    #     phi = math.pi / 2 - math.pi / 180.0 * pitch
    if yaw is not None and pitch is not None:
        theta = yaw * horizontal_stddev + horizontal_mean
        phi = pitch * vertical_stddev + vertical_mean

    else:
        if mode == 'uniform':
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

        elif mode == 'normal' or mode == 'gaussian': # CelebA
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

        elif mode == 'hybrid':
            if random.random() < 0.5:
                theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
                phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
            else:
                theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
                phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

        elif mode == 'truncated_gaussian':
            theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
            phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

        elif mode == 'spherical_uniform':
            theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
            v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
            v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
            v = torch.clamp(v, 1e-5, 1 - 1e-5)
            phi = torch.arccos(1 - 2 * v)

        else:
            # Just use the mean.
            theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
            phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device) # given phi=pi/2 and theta=pi/2, (x,y,z) = (0,0,1)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    # phi -> up
    # theta -> left
    return output_points, phi, theta


def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world


def create_world2cam_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    # weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / (torch.sum(weights, -1, keepdim=True) + eps) # (N_rays, N_samples_)
    if torch.any(torch.isnan(pdf)):
        pdf = weights / (torch.sum(weights, -1, keepdim=True) + 2*eps) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    if torch.any(torch.isnan(samples)):
        torch.save(
            {
                "bins": bins,
                "weights": weights,
                "samples": samples,
                "N_importance": N_importance,
                "det": det,
                "u": u,
            },
            "debug/sample_pdf_input.pt"
        )
        raise ValueError
    return samples


def sample_pdf_debug(bins, weights, N_importance, u, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    pdf = weights / (torch.sum(weights, -1, keepdim=True) + eps) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])

    return samples


def close_mouth(m_input, d_input):

    n, _, H, W = m_input.shape
    # get volume_mask
    m_f = 1 - m_input*1.0
    m_f = m_f.squeeze(1).detach().cpu().numpy()
    # vis_tensor_as_img('depth_before', d_input)

    for b in range(n):
        m = m_f[b]
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(m.astype(np.uint8), 4)
        labels = torch.from_numpy(labels).to(m_input.device).unsqueeze(0)
        # close mouth
        if np.any(stats[:, 4] < H*W/20):
            small_area_labels = np.where(stats[:, 4] < H*W/20)[0]
            mouth = torch.zeros_like(m_input[b])
            for label in small_area_labels.flatten():
                mouth = torch.logical_or(mouth, labels == label)
            m_input[b][mouth] = True
            mouth = mouth.unsqueeze(0)
            mouth_dilation = kornia.morphology.dilation(
                mouth, torch.ones(3, 3).to(mouth.device),
                border_type='constant')
            mouth_edge = torch.logical_xor(
                mouth, mouth_dilation).squeeze(0)
            mouth = mouth.squeeze(0)
            d_input[b][mouth] = d_input[b][mouth_edge].mean()
    return m_input, d_input


def close_mouth_kornia(m_input, d_input):
    n, _, H, W = m_input.shape
    # get volume_mask
    m_f = 1 - m_input*1.0
    # vis_tensor_as_img('depth_before', d_input)
    labels_out = kornia.contrib.connected_components(
        m_f, num_iterations=150)

    for b in range(n):
        # close mouth
        labels = torch.unique(labels_out[b])
        count_label_pixel = torch.tensor(
            [torch.count_nonzero(labels_out[b] == label)
                for label in labels])
        if torch.any(count_label_pixel < H*W/20):
            labels_remove = labels[count_label_pixel < H*W/20]
            mouth = torch.zeros_like(m_input[b])
            for label in labels_remove:
                mouth = torch.logical_or(mouth, labels_out[b] == label)
            m_input[b][mouth] = True
            mouth = mouth.unsqueeze(0)
            mouth_dilation = kornia.morphology.dilation(
                mouth, torch.ones(3, 3).to(mouth.device),
                border_type='constant')
            mouth_edge = torch.logical_xor(
                mouth, mouth_dilation).squeeze(0)
            mouth = mouth.squeeze(0)
            d_input[b][mouth] = d_input[b][mouth_edge].mean()
    return m_input, d_input


def get_volume_mask(points, m_input, ray_start=0, ray_end=0,
                    fov=12, cam_distance=1):
    n, _, H, W = m_input.shape
    points_ = points.clone()

    # set points
    points_[..., 2] = points_[..., 2] + (ray_end + ray_start)/2
    mask_size = m_input.shape[-2]
    f = (mask_size/2)/np.tan((fov/2)/180*np.pi)
    ratio = (cam_distance - points_[..., 2:])/f
    mask_sampler = points_[..., :2] / ratio
    mask_sampler[..., 1] = - mask_sampler[..., 1]
    mask_sampler += mask_size/2
    # mask_sampler = mask_sampler.round()
    # mask_sampler[torch.logical_or(
    #     mask_sampler < 0, mask_sampler >= mask_size
    # )] = 0
    mask_sampler = mask_sampler.floor().long()
    mask_sampler = torch.clip(mask_sampler, 0, mask_size-1)
    volume_mask = []
    for b in range(n):
        volume_mask.append(
            m_input[b, 0][mask_sampler[b][..., 1], mask_sampler[b][..., 0]]
        )
    volume_mask = torch.stack(volume_mask).unsqueeze(-1)
    return volume_mask


def get_depth_mask(points, m_input, d_input, ray_start=0, ray_end=0,
                   thick=0.0012, fov=12, cam_distance=1):
    n, _, H, W = m_input.shape
    points_ = points.clone()

    # set points
    points_[..., 2] = points_[..., 2] + (ray_end + ray_start)/2
    mask_size = m_input.shape[-2]
    f = (mask_size/2)/np.tan((fov/2)/180*np.pi)
    points_depth = cam_distance - points_[..., 2:]
    ratio = points_depth/f
    mask_sampler = points_[..., :2] / ratio
    mask_sampler[..., 1] = - mask_sampler[..., 1]
    mask_sampler += mask_size/2
    # mask_sampler = mask_sampler.round()
    # mask_sampler[torch.logical_or(
    #     mask_sampler < 0, mask_sampler >= mask_size
    # )] = 0
    mask_sampler = mask_sampler.floor().long()
    mask_sampler = torch.clip(mask_sampler, 0, mask_size-1)
    m_masks = []
    d_masks = []
    for b in range(n):
        d_mask = torch.abs(
            d_input[b, 0][mask_sampler[b][..., 1], mask_sampler[b][..., 0]]
            - points_depth[b].squeeze(-1)) <= thick
        m_mask = m_input[b, 0][
            mask_sampler[b][..., 1], mask_sampler[b][..., 0]]
        d_masks.append(d_mask)
        m_masks.append(m_mask)
    d_masks = torch.stack(d_masks).unsqueeze(-1)
    m_masks = torch.stack(m_masks).unsqueeze(-1)
    return d_masks, m_masks


def get_initial_rays_trig_from_depth(
        n, num_steps, device, fov, resolution, ray_start, ray_end,
        d_input=None, m_input=None, narrow_ratio=1.0, norm_mode='vec_len'):

    """Returns sample points, z_vals, and ray directions in camera space."""
    start_time = time.time()

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan(
        (2 * math.pi * fov / 360) / 2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1), norm_mode=norm_mode)

    z_vals = torch.linspace(
        ray_start, ray_end, num_steps, device=device
    ).reshape(1, num_steps, 1).repeat(W*H, 1, 1)  # W*H, num_steps, 1
    points = rays_d_cam.unsqueeze(1).repeat(
        1, num_steps, 1) * z_vals  # W*H, num_steps, 3

    points = torch.stack(
        n * [points])  # batchsize:n, sample num_steps steps from ray_start to ray_end.
    rays_d_cam = torch.stack(n * [rays_d_cam]).to(device)  # unit vector emitted from the camera

    pigan_points_time = time.time()
    if d_input is None or m_input is None:
        z_vals = torch.stack(n * [z_vals])  # linspace from ray_start to ray_end.
        # (set as the origin of the coordinate system.)
    else:
        m_input, d_input = close_mouth(m_input, d_input)
        volume_mask = get_volume_mask(
            points, m_input, ray_start, ray_end)
        # get depth_z_vals
        thick_ratio = 0.005
        thickness = (ray_end - ray_start) * thick_ratio
        unit_vals = torch.linspace(
            0, 1, num_steps, device=device
        ).reshape(1, 1, num_steps, 1).repeat(n, W*H, 1, 1)
        unit_vals -= 0.5
        depth_z_vals = unit_vals * thickness  # (W*H, num_steps, 1)
        depth_z_vals = depth_z_vals + d_input.reshape(n, W*H, 1, 1)

        depth_points = rays_d_cam.unsqueeze(2).repeat(
            1, 1, num_steps, 1) * depth_z_vals

        fg = depth_points * volume_mask
        # bg = points * (torch.logical_not(volume_mask))
        # bg[..., 2] = bg[..., 2] - (ray_end + ray_start)/2
        # points = bg
        points[volume_mask.squeeze(-1) > 0] = (
            fg[volume_mask.squeeze(-1) > 0] * narrow_ratio
            + points[volume_mask.squeeze(-1) > 0] * (1 - narrow_ratio))

        # for i in range(n):
        #     vis_tensor_as_vert(f'fg_{i}.obj', fg[i])
        #     vis_tensor_as_vert(f'bg_{i}.obj', bg[i])
        #     vis_tensor_as_vert(f'vlm_{i}.obj', points[i])

        # vis_tensor_as_img('depth_after', d_input)

        z_vals = torch.norm(points, dim=-1).unsqueeze(-1)

        my_time = time.time()

    # print(
    #     f"get init pigan points: {pigan_points_time - start_time},",
    #     f"connected components: {connected_components_time - pigan_points_time},",
    #     f"generate sampling from depth: {my_time - connected_components_time}.",
    # )
    return points, z_vals, rays_d_cam


def get_initial_rays_trig_from_depth2(
        n, num_steps, device, fov, resolution, ray_start, ray_end,
        d_input, m_input, thick_ratio, cam_distance=2.7, norm_mode='vec_len'):

    """Returns sample points, z_vals, and ray directions in camera space."""
    start_time = time.time()

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device),
                          torch.linspace(1, -1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()
    z = -torch.ones_like(x, device=device) / np.tan(
        (2 * math.pi * fov / 360) / 2)

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1), norm_mode=norm_mode)

    z_vals = torch.linspace(
        ray_start, ray_end, num_steps, device=device
    ).reshape(1, num_steps, 1).repeat(W*H, 1, 1)  # W*H, num_steps, 1
    points = rays_d_cam.unsqueeze(1).repeat(
        1, num_steps, 1) * z_vals  # W*H, num_steps, 3

    points = torch.stack(
        n * [points])  # batchsize:n, sample num_steps steps from ray_start to ray_end.
    rays_d_cam = torch.stack(n * [rays_d_cam]).to(device)  # unit vector emitted from the camera

    m_input, d_input = close_mouth(m_input, d_input)
    volume_mask = get_volume_mask(
        points, m_input, ray_start, ray_end, fov, cam_distance)
    # get depth_z_vals
    # thick_ratio = 0.01
    thickness = (ray_end - ray_start) * thick_ratio
    unit_vals = torch.linspace(
        0, 1, num_steps, device=device
    ).reshape(1, 1, num_steps, 1).repeat(n, W*H, 1, 1)
    unit_vals -= 0.5
    depth_z_vals = unit_vals * thickness  # (W*H, num_steps, 1)
    depth_z_vals = depth_z_vals + d_input.reshape(n, W*H, 1, 1)

    depth_points = rays_d_cam.unsqueeze(2).repeat(
        1, 1, num_steps, 1) * depth_z_vals

    return depth_points, depth_z_vals, volume_mask


def get_fine_points(
        coarse_points, coarse_output, transformed_ray_origins,
        transformed_ray_directions, z_vals, batch_size,
        img_size, num_steps, clamp_mode='relu', noise_std=0):
    coarse_points = coarse_points.reshape(
        batch_size, img_size * img_size, num_steps, 3)
    _, _, weights = fancy_integration(
        coarse_output, z_vals, device=coarse_output.device,
        clamp_mode=clamp_mode, noise_std=noise_std)

    weights = weights.reshape(
        batch_size * img_size * img_size, num_steps) + 1e-5

    #### Start new importance sampling
    z_vals = z_vals.reshape(
        batch_size * img_size * img_size, num_steps)
    z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
    z_vals = z_vals.reshape(
        batch_size, img_size * img_size, num_steps, 1)
    fine_z_vals = sample_pdf(
        z_vals_mid, weights[:, 1:-1],
        num_steps, det=False).detach()
    fine_z_vals = fine_z_vals.reshape(
        batch_size, img_size * img_size, num_steps, 1)
    fine_z_vals, _ = sort_z_vals(fine_z_vals)
    # print(f"fine_z_vals.min(): {fine_z_vals.min()}")
    # print(f"fine_z_vals.max(): {fine_z_vals.max()}")
    # fine_z_vals = torch.clip(fine_z_vals, 0.88, 1.12)

    fine_points = (
        transformed_ray_origins.unsqueeze(2).contiguous()
        + transformed_ray_directions.unsqueeze(2).contiguous()
        * fine_z_vals.expand(-1, -1, -1, 3).contiguous())
    fine_points = fine_points.reshape(
        batch_size, img_size*img_size*num_steps, 3)
    return fine_points, fine_z_vals


if __name__ == "__main__":
    # get_initial_rays_trig(
    #     n=1, num_steps=2, device=torch.device('cpu'),
    #     fov=12, resolution=[3,3], ray_start=0, ray_end=1
    # )

    # data = torch.load("debug/sample_pdf_input.pt", map_location=torch.device(0))
    # bins = data["bins"]
    # weights = data["weights"]
    # samples = data["samples"]
    # N_importance = data["N_importance"]
    # det = data["det"]
    # u = data["u"]
    # sample_pdf_debug(bins, weights, N_importance, u, det)

    data = torch.load("debug/fancy_integration_input.pt", map_location=torch.device(0))
    rgb_sigma = data["rgb_sigma"]
    z_vals = data["z_vals"]
    device = data["device"]
    noise_std = data["noise_std"]
    last_back = data["last_back"]
    white_back = data["white_back"]
    clamp_mode = data["clamp_mode"]
    fill_mode = data["fill_mode"]
    activate_mode = data["activate_mode"]
    noise = data["noise"]
    fancy_integration_debug(
        rgb_sigma, z_vals, device, noise_std, last_back, white_back,
        clamp_mode, fill_mode, activate_mode, noise)
