"""Implicit generator for 3D volumes"""

import random
import torch.nn as nn
import torch
import time
from torch.cuda.amp import autocast

from .volumetric_rendering import *

class PoseImplicitGenerator3d(nn.Module):
    '''
    Based on ImplicitGenerator3d but supporting yaw and pitch control.
    '''
    def __init__(self, siren, z_dim, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.siren = siren(
            output_dim=4, z_dim=self.z_dim, input_dim=3, device=None)
        self.epoch = 0
        self.step = 0

    def set_device(self, device):
        self.device = device
        self.siren.device = device

        self.generate_avg_frequencies()

    def forward(
            self, z, img_size, fov, ray_start, ray_end, num_steps, h_stddev,
            v_stddev, h_mean, v_mean, hierarchical_sample, zy_data=None,
            yaw=None, pitch=None, sample_dist=None, lock_view_dependence=False,
            face_recon=None, sample_near_mesh=False, using_dist_depr=False,
            narrow_ratio=1.0, thick_ratio=0.01, using_norm_reg=False, 
            camera_shift=(0, 0, 0), **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        """

        batch_size = z.shape[0]
        using_input_points = isinstance(zy_data, dict) \
            and 'transformed_points_input' in zy_data \
            and 'fine_points_input' in zy_data \
            and 'fine_points2_input' in zy_data

        # z = torch.load("debug/subset_z.pt", map_location=z.device)
        # g_pose = torch.load("debug/g_pose.pt", map_location=z.device)
        # phi = g_pose[:, 0:1]
        # theta = g_pose[:, 1:2]
        # pitch = (phi - v_mean) / v_stddev
        # yaw = (theta - h_mean) / h_stddev

        # Generate initial camera rays and sample points.
        with torch.no_grad():
            if yaw is None or pitch is None:
                _, phi, theta = sample_camera_positions(
                    n=batch_size, r=1, horizontal_stddev=h_stddev,
                    vertical_stddev=v_stddev, horizontal_mean=h_mean,
                    vertical_mean=v_mean, device=self.device, mode=sample_dist)
                pitch = (phi - v_mean) / v_stddev
                yaw = (theta - h_mean) / h_stddev

            if face_recon is not None:
                d_input, m_input, f_input, z_input_dict = \
                    face_recon.z_to_pigan_depth_image(
                        z,
                        face_recon.yaw2theta(yaw, h_mean, h_stddev),
                        face_recon.pitch2phi(pitch, v_mean, v_stddev),
                        img_size
                    )
                face_recon.d_pigan_input = d_input
                face_recon.m_pigan_input = m_input
                face_recon.f_pigan_input = f_input
                self.verts_input = face_recon.pigan_face_proj.detach()
                self.lm_input = face_recon.pigan_landmark.detach()
                self.lm3d_input = face_recon.pigan_lm3d.detach()

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                norm_mode='z_val')  # batch_size, pixels, num_steps, 1

            (
                transformed_points, z_vals,
                transformed_ray_directions,
                transformed_ray_origins, pitch, yaw
            ) = transform_sampled_points(
                points_cam,
                z_vals,
                rays_d_cam,
                yaw=yaw,
                pitch=pitch,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                device=self.device,
                mode=sample_dist,
                camera_shift=camera_shift)

            transformed_ray_directions_expanded = torch.unsqueeze(
                transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(
                -1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(
                batch_size, img_size * img_size * num_steps, 3)
            transformed_points = transformed_points.reshape(
                batch_size, img_size * img_size * num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # Model prediction on course points
            if using_input_points:
                transformed_points = zy_data['transformed_points_input']
                z_vals = torch.norm(
                    transformed_points.reshape(
                        batch_size, img_size * img_size, num_steps, 3)
                    - transformed_ray_origins.unsqueeze(2).contiguous(),
                    dim=-1
                ).unsqueeze(-1)
            coarse_output = self.siren(transformed_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)
            if isinstance(zy_data, dict):
                zy_data['points_cam'] = points_cam
                zy_data['transformed_points'] = transformed_points
                zy_data['coarse_output'] = coarse_output

        # Re-sample fine points alont camera rays, as described in NeRF
        if hierarchical_sample:
            volume_mask = None
            with torch.no_grad():
                fine_points, fine_z_vals = get_fine_points(
                    transformed_points, coarse_output, transformed_ray_origins,
                    transformed_ray_directions,  z_vals, batch_size,
                    img_size, num_steps, clamp_mode=kwargs['clamp_mode'],
                    noise_std=kwargs['nerf_noise'])

                if using_input_points:
                    fine_points = zy_data['fine_points_input']
                    fine_z_vals = torch.norm(
                        fine_points.reshape(
                            batch_size, img_size * img_size, num_steps, 3)
                        - transformed_ray_origins.unsqueeze(2).contiguous(),
                        dim=-1
                    ).unsqueeze(-1)

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

                #### end new importance sampling

            # Model prediction on re-sampled find points
            fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
            if isinstance(zy_data, dict):
                zy_data['fine_points'] = fine_points
                zy_data['fine_output'] = fine_output

            if sample_near_mesh:
                with torch.no_grad():
                    fine_points2, fine_z_vals2 = get_fine_points(
                        fine_points, fine_output, transformed_ray_origins,
                        transformed_ray_directions, fine_z_vals, batch_size,
                        img_size, num_steps, kwargs['clamp_mode'],
                        noise_std=kwargs['nerf_noise'])

                    # if torch.distributed.get_rank() == 0 and self.step % 1000 == 0:
                    #     print(
                    #         f"thick_ratio: {thick_ratio},",
                    #         f"narrow_ratio: {narrow_ratio}")
                    surf_points_cam, surf_z_vals, volume_mask = \
                        get_initial_rays_trig_from_depth2(
                            batch_size,
                            num_steps,
                            resolution=(img_size, img_size),
                            device=self.device,
                            fov=fov,
                            ray_start=ray_start,
                            ray_end=ray_end,
                            d_input=d_input,
                            m_input=m_input,
                            thick_ratio=thick_ratio,
                            norm_mode='z_val'
                            )  # batch_size, pixels, num_steps, 1
                    # surf_z_vals = surf_z_vals.reshape(
                    #     batch_size, img_size * img_size, num_steps, 1)
                    surf_points = (
                        transformed_ray_origins.unsqueeze(2).contiguous()
                        + transformed_ray_directions.unsqueeze(2).contiguous()
                        * surf_z_vals.expand(-1, -1, -1, 3).contiguous())

                    # if (torch.distributed.get_rank() == 0
                    #         and self.step % 1000 == 0):
                        
                    #     from utils.mesh_io import save_obj_vertex
                    #     save_obj_vertex(
                    #         f'coarse_{self.step}.obj', transformed_points[0])
                    #     save_obj_vertex(
                    #         f'fine_{self.step}.obj', fine_points[0])
                    #     save_obj_vertex(
                    #         f'fine2_{self.step}.obj', fine_points2[0])
                    #     save_obj_vertex(
                    #         f'surf_{self.step}.obj', surf_points[0])
                    #     fp2 = fine_points2.reshape(
                    #         batch_size, img_size * img_size, num_steps, 3)
                    #     fp2[volume_mask.squeeze(-1) > 0] = \
                    #         surf_points[volume_mask.squeeze(-1) > 0]
                    #     save_obj_vertex(f'fine2_surf_{self.step}.obj', fp2[0])

                    fine_points2 = fine_points2.reshape(
                        batch_size, img_size * img_size, num_steps, 3)
                    fine_points2[volume_mask.squeeze(-1) > 0] = (
                        surf_points[volume_mask.squeeze(-1) > 0] * narrow_ratio
                        + fine_points2[volume_mask.squeeze(-1) > 0]
                        * (1 - narrow_ratio))
                    fine_points2 = fine_points2.reshape(
                        batch_size, img_size * img_size * num_steps, 3)

                    if using_input_points:
                        fine_points2 = zy_data['fine_points2_input']
                        fine_z_vals2 = torch.norm(
                            (fine_points2.reshape(
                                batch_size, img_size * img_size, num_steps, 3)
                            - transformed_ray_origins.unsqueeze(2).contiguous()),
                            dim=-1
                        ).unsqueeze(-1)

                fine_output2 = self.siren(
                    fine_points2, z,
                    ray_directions=transformed_ray_directions_expanded
                ).reshape(batch_size, img_size * img_size, -1, 4)
                if isinstance(zy_data, dict):
                    zy_data['fine_points2'] = fine_points2
                    zy_data['fine_output2'] = fine_output2

                all_outputs = torch.cat([fine_output, fine_output2], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, fine_z_vals2], dim=-2)
            else:
                # Combine course and fine points
                all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)

            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(
                all_outputs, -2, indices.expand(-1, -1, -1, 4))

        else:
            all_outputs = coarse_output
            all_z_vals = z_vals

        # Create images with NeRF
        pixels, depth, weights = fancy_integration(
            all_outputs, all_z_vals, zy_data=zy_data,
            device=self.device, white_back=kwargs.get('white_back', False),
            last_back=kwargs.get('last_back', False),
            clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise']
        )

        depth = depth.reshape(batch_size, img_size, img_size).contiguous()
        if isinstance(zy_data, dict):
            zy_data['depth_variance'] = zy_data['depth_variance'].reshape(batch_size, img_size, img_size).contiguous()

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        if using_dist_depr:
            all_points_cam = rays_d_cam.unsqueeze(2).repeat(
                1, 1, num_steps*2, 1) * all_z_vals  # W*H, num_steps, 3
            all_points = (
                transformed_ray_origins.unsqueeze(2).repeat(
                    1, 1, num_steps*2, 1).contiguous()
                + transformed_ray_directions.unsqueeze(2).repeat(
                    1, 1, num_steps*2, 1).contiguous()
                * all_z_vals.expand(-1, -1, -1, 3).contiguous())

            self.all_weights = weights
            self.all_points_cam = all_points_cam
            self.all_points = all_points
            self.depr_m_input = m_input
            self.depr_d_input = d_input
            self.volume_mask = None
            self.depr_yaws = yaw
            self.depr_pitch = pitch
            # self.bg_depths = face_recon.bg_depths

        if using_norm_reg:
            # print("")
            on_surface_points = (
                transformed_ray_origins.contiguous()
                + transformed_ray_directions.contiguous()
                * depth.reshape([-1, img_size*img_size]).unsqueeze(2).expand(-1, -1, 3).contiguous())
            random_dirs = torch.rand_like(on_surface_points) - 0.5
            on_surface_points_neighbor = (
                on_surface_points
                + random_dirs
                / random_dirs.norm(dim=-1, keepdim=True)
                * kwargs.get('h_sample', 1e-3))
            surface_points = torch.stack(
                [on_surface_points, on_surface_points_neighbor], dim=2
            ).reshape(batch_size, -1, 3)
            self.pred_normals = self.siren(
                surface_points, z, ray_directions=None, get_normal=True
            ).reshape(batch_size, img_size * img_size, 2, 3)

        if kwargs.get('return_depth', False) is False:
            return pixels, torch.cat([pitch, yaw], -1)
        else:
            return pixels, torch.cat([pitch, yaw], -1), depth

    def generate_avg_frequencies(self):
        """Calculates average frequencies and phase shifts"""

        z = torch.randn((10000, self.z_dim), device=self.siren.device)
        with torch.no_grad():
            frequencies, phase_shifts = self.siren.mapping_network(z)
        self.avg_frequencies = frequencies.mean(0, keepdim=True)
        self.avg_phase_shifts = phase_shifts.mean(0, keepdim=True)
        return self.avg_frequencies, self.avg_phase_shifts

    def staged_forward(
            self, z, img_size, fov, ray_start, ray_end, num_steps,
            h_stddev, v_stddev, h_mean, v_mean,
            hierarchical_sample=False, zy_data={},
            yaw=None, pitch=None, sample_dist=None, lock_view_dependence=False,
            transformed_points_input=None, fine_points_input=None, psi=1,
            max_batch_size=50000, depth_map=False, near_clip=0, far_clip=2,
            face_recon=None, sample_near_mesh=False, rt_norm=False,
            narrow_ratio=1.0, thick_ratio=0.01, camera_shift=(0, -0.012, 0), **kwargs):
        """
        Similar to forward but used for inference.
        Calls the model sequencially using max_batch_size to limit memory usage.
        """

        batch_size = z.shape[0]

        self.generate_avg_frequencies()
        # print('last_back:', kwargs.get('last_back', False))
        # print('white_back:', kwargs.get('white_back', False))

        with torch.no_grad():

            raw_frequencies, raw_phase_shifts = self.siren.mapping_network(z)

            truncated_frequencies = self.avg_frequencies + psi * (
                raw_frequencies - self.avg_frequencies)
            truncated_phase_shifts = self.avg_phase_shifts + psi * (
                raw_phase_shifts - self.avg_phase_shifts)

            if yaw is None or pitch is None:
                _, phi, theta = sample_camera_positions(
                    n=batch_size, r=1, horizontal_stddev=h_stddev,
                    vertical_stddev=v_stddev, horizontal_mean=h_mean,
                    vertical_mean=v_mean, device=self.device, mode=sample_dist)
                pitch = (phi - v_mean) / (v_stddev+1e-9)
                yaw = (theta - h_mean) / (h_stddev+1e-9)
                # note that h_stddev = v_stddev = 0 here.
                # pitch = (phi - v_mean) / v_stddev
                # yaw = (theta - h_mean) / h_stddev

            if face_recon is not None:
                d_input, m_input, f_input, z_input_dict = \
                    face_recon.z_to_pigan_depth_image(
                        z,
                        face_recon.yaw2theta(yaw, h_mean, h_stddev),
                        face_recon.pitch2phi(pitch, v_mean, v_stddev),
                        img_size
                    )
                face_recon.d_pigan_input = d_input
                face_recon.m_pigan_input = m_input
                face_recon.f_pigan_input = f_input
                face_recon.l_pigan_input = face_recon.pigan_landmark.clone()

            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                norm_mode='z_val')  # batch_size, pixels, num_steps, 1

            (
                transformed_points, z_vals,
                transformed_ray_directions,
                transformed_ray_origins, pitch, yaw
            ) = transform_sampled_points(
                points_cam,
                z_vals,
                rays_d_cam,
                yaw=yaw,
                pitch=pitch,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                device=self.device,
                mode=sample_dist,
                camera_shift=camera_shift,
                )

            transformed_ray_directions_expanded = torch.unsqueeze(
                transformed_ray_directions, -2)
            transformed_ray_directions_expanded = \
                transformed_ray_directions_expanded.expand(
                    -1, -1, num_steps, -1)
            transformed_ray_directions_expanded = \
                transformed_ray_directions_expanded.reshape(
                    batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(
                batch_size, img_size*img_size*num_steps, 3)

            if transformed_points_input is not None:
                transformed_points = transformed_points_input

            if lock_view_dependence:

                transformed_ray_directions_expanded = torch.zeros_like(
                    transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1
                # print('lock_view_dependence')

                # _, _, transformed_ray_directions_, _, _, _ = \
                #     transform_sampled_points(
                #         points_cam,
                #         z_vals,
                #         rays_d_cam,
                #         yaw=torch.zeros((batch_size,1), device=self.device),
                #         pitch=torch.zeros((batch_size,1), device=self.device),
                #         h_stddev=h_stddev,
                #         v_stddev=v_stddev,
                #         h_mean=h_mean,
                #         v_mean=v_mean,
                #         device=self.device,
                #         mode=sample_dist,
                #         camera_shift=camera_shift)

                # transformed_ray_directions_expanded = torch.unsqueeze(
                #     transformed_ray_directions_, -2)
                # transformed_ray_directions_expanded = \
                #     transformed_ray_directions_expanded.expand(
                #         -1, -1, num_steps, -1)
                # transformed_ray_directions_expanded = \
                #     transformed_ray_directions_expanded.reshape(
                #         batch_size, img_size*img_size*num_steps, 3)


            if kwargs.get('wo_ray_direction', False):
                # print('Fix ray direction to [0, 0, -1]!')
                transformed_ray_directions_expanded[..., 0].fill_(0)
                transformed_ray_directions_expanded[..., 1].fill_(0)
                transformed_ray_directions_expanded[..., 2].fill_(-1)
            # Sequentially evaluate siren with max_batch_size to avoid OOM
            coarse_output = torch.zeros(
                (batch_size, transformed_points.shape[1], 4),
                device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = \
                        self.siren.forward_with_frequencies_phase_shifts(
                            transformed_points[b:b+1, head:tail],
                            truncated_frequencies[b:b+1],
                            truncated_phase_shifts[b:b+1],
                            ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(
                batch_size, img_size * img_size, num_steps, 4)
            zy_data['transformed_points'] = transformed_points

            if hierarchical_sample:
                with torch.no_grad():
                    fine_points, fine_z_vals = get_fine_points(
                        transformed_points, coarse_output, transformed_ray_origins,
                        transformed_ray_directions,  z_vals, batch_size,
                        img_size, num_steps, clamp_mode=kwargs['clamp_mode'],
                        noise_std=kwargs['nerf_noise'])

                    if fine_points_input is not None:
                        fine_points = fine_points_input
                    #### end new importance sampling

                if lock_view_dependence:
                    # print('lock_view_dependence')
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                    # transformed_ray_directions_expanded = torch.unsqueeze(
                    #     transformed_ray_directions_, -2)
                    # transformed_ray_directions_expanded = \
                    #     transformed_ray_directions_expanded.expand(
                    #         -1, -1, num_steps, -1)
                    # transformed_ray_directions_expanded = \
                    #     transformed_ray_directions_expanded.reshape(
                    #         batch_size, img_size*img_size*num_steps, 3)

                # Sequentially evaluate siren with max_batch_size to avoid OOM

                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                zy_data['fine_points'] = fine_points
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size
                fine_output = fine_output.reshape(
                    batch_size, img_size * img_size, num_steps, 4)

                # fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                if sample_near_mesh:
                    fine_points2, fine_z_vals2 = get_fine_points(
                        fine_points, fine_output, transformed_ray_origins,
                        transformed_ray_directions, fine_z_vals, batch_size,
                        img_size, num_steps, kwargs['clamp_mode'],
                        noise_std=kwargs['nerf_noise'])
                    # print(
                    #     f"thick_ratio: {thick_ratio},",
                    #     f"narrow_ratio: {narrow_ratio}")
                    surf_points_cam, surf_z_vals, volume_mask = \
                        get_initial_rays_trig_from_depth2(
                            batch_size,
                            num_steps,
                            resolution=(img_size, img_size),
                            device=self.device,
                            fov=fov,
                            ray_start=ray_start,
                            ray_end=ray_end,
                            d_input=d_input,
                            m_input=m_input,
                            thick_ratio=thick_ratio,
                            norm_mode='z_val'
                            )  # batch_size, pixels, num_steps, 1
                    # surf_z_vals = surf_z_vals.reshape(
                    #     batch_size, img_size * img_size, num_steps, 1)
                    surf_points = (
                        transformed_ray_origins.unsqueeze(2).contiguous()
                        + transformed_ray_directions.unsqueeze(2).contiguous()
                        * surf_z_vals.expand(-1, -1, -1, 3).contiguous())
                    # surf_points = surf_points.reshape(
                    #     batch_size, img_size*img_size*num_steps, 3)

                    # fg = surf_points * volume_mask
                    fine_points2 = fine_points2.reshape(
                        batch_size, img_size * img_size, num_steps, 3)
                    fine_points2[volume_mask.squeeze(-1) > 0] = (
                        surf_points[volume_mask.squeeze(-1) > 0]
                        * narrow_ratio
                        + fine_points2[volume_mask.squeeze(-1) > 0]
                        * (1 - narrow_ratio))
                    fine_z_vals2 = torch.norm(
                        (fine_points2
                         - transformed_ray_origins.unsqueeze(2).contiguous()),
                        dim=-1
                    ).unsqueeze(-1)
                    fine_points2 = fine_points2.reshape(
                        batch_size, img_size * img_size * num_steps, 3)
                    zy_data['fine_points2'] = fine_points2

                    with torch.no_grad():
                        fine_output2 = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                        for b in range(batch_size):
                            head = 0
                            while head < fine_points.shape[1]:
                                tail = head + max_batch_size
                                fine_output2[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points2[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                                head += max_batch_size
                        fine_output2 = fine_output2.reshape(
                            batch_size, img_size * img_size, num_steps, 4)

                    all_outputs = torch.cat([fine_output, fine_output2], dim=-2)
                    all_z_vals = torch.cat([fine_z_vals, fine_z_vals2], dim=-2)
                else:
                    # Combine course and fine points
                    all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
                    all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(
                    all_outputs, -2, indices.expand(-1, -1, -1, 4))

            else:
                all_outputs = coarse_output
                all_z_vals = z_vals

            pixels, depth, weights = fancy_integration(
                all_outputs, all_z_vals, zy_data=zy_data, device=self.device,
                white_back=kwargs.get('white_back', False),
                clamp_mode=kwargs['clamp_mode'],
                last_back=kwargs.get('last_back', False),
                fill_mode=kwargs.get('fill_mode', None),
                noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(
                batch_size, img_size, img_size).contiguous().cpu()
            zy_data['variance_map'] = zy_data['depth_variance'].reshape(
                batch_size, img_size, img_size).contiguous().cpu()

            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

            if rt_norm:
                # print("return norm")
                surface_points = (
                    transformed_ray_origins.contiguous()
                    + transformed_ray_directions.contiguous()
                    * depth.reshape([-1, img_size*img_size]).unsqueeze(2).expand(-1, -1, 3).contiguous())
                self.pred_normals = self.siren(
                    surface_points, z, ray_directions=None, get_normal=True
                ).reshape(batch_size, img_size * img_size, 3)

        return pixels, depth_map

    # Used for rendering interpolations
    def staged_forward_with_frequencies(
        self, truncated_frequencies, truncated_phase_shifts, img_size,
        fov, ray_start, ray_end, num_steps, h_stddev, v_stddev, h_mean, v_mean,
        psi=0.7, lock_view_dependence=False, max_batch_size=50000,
        depth_map=False, near_clip=0, far_clip=2, sample_dist=None,
        hierarchical_sample=False,
        d_input=None, m_input=None,
        yaw=None, pitch=None,
        camera_shift=(0, 0, 0),
            **kwargs):
        batch_size = truncated_frequencies.shape[0]

        with torch.no_grad():
            if d_input is None or m_input is None:
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                    batch_size,
                    num_steps,
                    resolution=(img_size, img_size),
                    device=self.device,
                    fov=fov,
                    ray_start=ray_start,
                    ray_end=ray_end,
                    norm_mode='z_val')  # batch_size, pixels, num_steps, 1
            else:
                points_cam, z_vals, rays_d_cam = get_initial_rays_trig_from_depth(
                    batch_size,
                    num_steps,
                    resolution=(img_size, img_size),
                    device=self.device,
                    fov=fov,
                    ray_start=ray_start,
                    ray_end=ray_end,
                    d_input=d_input,
                    m_input=m_input,
                    norm_mode='z_val')  # batch_size, pixels, num_steps, 1

            (
                transformed_points, z_vals,
                transformed_ray_directions,
                transformed_ray_origins, pitch, yaw
            ) = transform_sampled_points(
                points_cam,
                z_vals,
                rays_d_cam,
                yaw=yaw,
                pitch=pitch,
                h_stddev=h_stddev,
                v_stddev=v_stddev,
                h_mean=h_mean,
                v_mean=v_mean,
                device=self.device,
                mode=sample_dist,
                camera_shift=camera_shift)

            transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
            transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
            transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

            if lock_view_dependence:
                transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                transformed_ray_directions_expanded[..., -1] = -1

            # BATCHED SAMPLE
            coarse_output = torch.zeros((batch_size, transformed_points.shape[1], 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_batch_size
                    coarse_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(transformed_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                    head += max_batch_size

            coarse_output = coarse_output.reshape(batch_size, img_size * img_size, num_steps, 4)
            # END BATCHED SAMPLE


            if hierarchical_sample:
                with torch.no_grad():
                    transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                    _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                    weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                    z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                    z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                    z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                        num_steps, det=False).detach().to(self.device) # batch_size, num_pixels**2, num_steps
                    fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                    fine_z_vals, _ = sort_z_vals(fine_z_vals)

                    fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                    fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                    #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1
                # fine_output = self.siren(fine_points, z, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)
                # BATCHED SAMPLE
                fine_output = torch.zeros((batch_size, fine_points.shape[1], 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_batch_size
                        fine_output[b:b+1, head:tail] = self.siren.forward_with_frequencies_phase_shifts(fine_points[b:b+1, head:tail], truncated_frequencies[b:b+1], truncated_phase_shifts[b:b+1], ray_directions=transformed_ray_directions_expanded[b:b+1, head:tail])
                        head += max_batch_size

                fine_output = fine_output.reshape(batch_size, img_size * img_size, num_steps, 4)
                # END BATCHED SAMPLE

                all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
                all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
                _, indices = torch.sort(all_z_vals, dim=-2)
                all_z_vals = torch.gather(all_z_vals, -2, indices)
                all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
            else:
                all_outputs = coarse_output
                all_z_vals = z_vals


            pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), clamp_mode = kwargs['clamp_mode'], last_back=kwargs.get('last_back', False), fill_mode=kwargs.get('fill_mode', None), noise_std=kwargs['nerf_noise'])
            depth_map = depth.reshape(batch_size, img_size, img_size).contiguous().cpu()


            pixels = pixels.reshape((batch_size, img_size, img_size, 3))
            pixels = pixels.permute(0, 3, 1, 2).contiguous().cpu() * 2 - 1

        return pixels, depth_map


    def forward_with_frequencies(
            self, frequencies, phase_shifts, img_size, fov, ray_start, ray_end,
            num_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample,
            sample_dist=None, lock_view_dependence=False,
            d_input=None, m_input=None, yaw=None, pitch=None,
            camera_shift=(0, 0, 0), **kwargs):

        batch_size = frequencies.shape[0]

        if d_input is None or m_input is None:
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                norm_mode='z_val')  # batch_size, pixels, num_steps, 1
        else:
            points_cam, z_vals, rays_d_cam = get_initial_rays_trig_from_depth(
                batch_size,
                num_steps,
                resolution=(img_size, img_size),
                device=self.device,
                fov=fov,
                ray_start=ray_start,
                ray_end=ray_end,
                d_input=d_input,
                m_input=m_input,
                norm_mode='z_val')  # batch_size, pixels, num_steps, 1

        (
            transformed_points, z_vals,
            transformed_ray_directions,
            transformed_ray_origins, pitch, yaw
        ) = transform_sampled_points(
            points_cam,
            z_vals,
            rays_d_cam,
            yaw=yaw,
            pitch=pitch,
            h_stddev=h_stddev,
            v_stddev=v_stddev,
            h_mean=h_mean,
            v_mean=v_mean,
            device=self.device,
            mode=sample_dist,
            camera_shift=camera_shift)

        transformed_ray_directions_expanded = torch.unsqueeze(transformed_ray_directions, -2)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.expand(-1, -1, num_steps, -1)
        transformed_ray_directions_expanded = transformed_ray_directions_expanded.reshape(batch_size, img_size*img_size*num_steps, 3)
        transformed_points = transformed_points.reshape(batch_size, img_size*img_size*num_steps, 3)

        if lock_view_dependence:
            transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
            transformed_ray_directions_expanded[..., -1] = -1

        coarse_output = self.siren.forward_with_frequencies_phase_shifts(transformed_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, num_steps, 4)

        if hierarchical_sample:
            with torch.no_grad():
                transformed_points = transformed_points.reshape(batch_size, img_size * img_size, num_steps, 3)
                _, _, weights = fancy_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * img_size * img_size, num_steps) + 1e-5
                #### Start new importance sampling
                # RuntimeError: Sizes of tensors must match except in dimension 1. Got 3072 and 6144 (The offending index is 0)
                z_vals = z_vals.reshape(batch_size * img_size * img_size, num_steps) # We squash the dimensions here. This means we importance sample for every batch for every ray
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
                z_vals = z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                 num_steps, det=False).detach() # batch_size, num_pixels**2, num_steps
                fine_z_vals = fine_z_vals.reshape(batch_size, img_size * img_size, num_steps, 1)
                fine_z_vals, _ = sort_z_vals(fine_z_vals)


                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(2).contiguous() * fine_z_vals.expand(-1,-1,-1,3).contiguous() # dimensions here not matching
                fine_points = fine_points.reshape(batch_size, img_size*img_size*num_steps, 3)
                #### end new importance sampling

                if lock_view_dependence:
                    transformed_ray_directions_expanded = torch.zeros_like(transformed_ray_directions_expanded)
                    transformed_ray_directions_expanded[..., -1] = -1

            fine_output = self.siren.forward_with_frequencies_phase_shifts(fine_points, frequencies, phase_shifts, ray_directions=transformed_ray_directions_expanded).reshape(batch_size, img_size * img_size, -1, 4)

            all_outputs = torch.cat([fine_output, coarse_output], dim = -2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim = -2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            # Target sizes: [-1, -1, -1, 4].  Tensor sizes: [240, 512, 12]
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        pixels, depth, weights = fancy_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

        pixels = pixels.reshape((batch_size, img_size, img_size, 3))
        pixels = pixels.permute(0, 3, 1, 2).contiguous() * 2 - 1

        return pixels, torch.cat([pitch, yaw], -1)
