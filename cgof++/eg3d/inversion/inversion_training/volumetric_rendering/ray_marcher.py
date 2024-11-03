import torch
import torch.nn as nn
import torch.nn.functional as F

class RayMarcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_std = 0.5

    def forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        delta_inf = 1e10 * torch.ones_like(deltas[:, :, :1])

        deltas = torch.cat([deltas, delta_inf], -2)

#         noise = torch.randn(densities.shape, device=colors.device) * self.noise_std if self.noise_std > 0 else 0

        noise = torch.randn(densities.shape, device=densities.device) * rendering_options.get('nerf_noise', 0) if rendering_options.get('nerf_noise', 0) > 0 else 0

        if rendering_options['clamp_mode'] == 'softplus':
            alphas = 1 - torch.exp(-deltas * (F.softplus(densities + rendering_options.get('volume_init', -1) + noise)))
        elif rendering_options['clamp_mode'] == 'relu':
            alphas = 1 - torch.exp(-deltas * (F.relu(densities + noise)))
        elif rendering_options['clamp_mode'] == 'x^2':
            alphas = 1 - torch.exp(-deltas * (densities**2))

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-10], -2)
        weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]

        rgb_final = torch.sum(weights * colors, -2)
        depth_final = torch.sum(weights * depths, -2)/weights.sum(2)
        depth_final = torch.nan_to_num(depth_final, float('inf'))
        depth_final = torch.clamp(depth_final, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            weights_sum = weights.sum(2)
            rgb_final = rgb_final + 1-weights_sum

        self.noise_std -= 0.5/5000 # reduce noise over 5000 steps

        rgb_final = rgb_final * 2 - 1 # Scale to (-1, 1)
        depth_final = depth_final * 2 - 1

        return rgb_final, depth_final, weights

class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas

        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights