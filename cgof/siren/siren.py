import numpy as np
# from pytorch3d.renderer.mesh import rasterizer
import torch.nn as nn
import torch
import math
from pprint import pprint
import torch.nn.functional as F

# from .bfm import BFM, Transform, DepthRendder, LandmarkTable
# from .pointnet_utils import SimplePointNetEncoder, PointNetEncoder

from utils.utils import batched_index_select

# from model.PerCostFormer.encoder import WarpingEncoder
# from model.PerCostFormer.decoder import MemoryDecoderLayer
from easydict import EasyDict


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)


def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


def norm_init(m):
    classname = m.__class__.__name__
    torch.nn.init.normal_(m.weight, mean=0, std=1)


class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def set_device(self, device=None):
        self.device = device

    def forward(self, z, **kwargs):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class ControlableMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, rotation_mode='Euler', rotation_trainable=True, **kwargs):
        super().__init__()
        self.logger = kwargs.get('logger',lambda x: None)

        self.logger(f"SIREN using rotation mode: {rotation_mode}")
        self.logger(f"Training parameters of Transformer: {rotation_trainable}")

        self.rotation_mode = rotation_mode
        self.rotation_trainable = rotation_trainable
        
        self.z_dim = z_dim
        self.dim_z_t = 256
        self.dim_z_s = 199
        self.dim_z_e = 50
        # print('length of z:',self.dim_z_t, self.dim_z_s, self.dim_z_e)
        assert self.z_dim == self.dim_z_t + self.dim_z_s + self.dim_z_e, \
            "The equation self.z_dim = self.dim_z_t + self.dim_z_s + self.dim_z_e should always holds. \
             Please check z_dim."

        self.bfm = BFM(chosen_file='./models/bfm_models/index_2106pts_bs.txt')
        self.depth_renderer = DepthRendder(self.bfm.faces)
        self.transformer = Transform(
                                     rotation_mode=self.rotation_mode,
                                     rotation_trainable=self.rotation_trainable)
        # self.pointnet = SimplePointNetEncoder(global_feat=True, output_dim=self.dim_z_t)
        self.pointnet = PointNetEncoder(global_feat=True, feature_transform=False, output_dim=self.dim_z_t)
        
        self.ldmk_table = LandmarkTable()

        self.network = nn.Sequential(nn.Linear(self.dim_z_t+self.pointnet.output_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        # self.transformer.apply(norm_init)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def set_device(self, device=None):
        self.device = device

    def forward(self, z, pitch=None, yaw=None, img_size=128):
        # split latent code z to z_t and z_e_s

        z_t = z[:,:self.dim_z_t]
        z_e_s = z[:,self.dim_z_t:self.dim_z_t+self.dim_z_e+self.dim_z_s]
        
        # generate meshes and transform to the pigan space
        meshes = self.bfm(z_e_s)
        # print('meshes.type:',type(meshes))
        self.meshes_trans = self.transformer(meshes)
        
        # print('meshes.requires_grad:',meshes.requires_grad)
        # tmp = self.transformer(self.transformer.debug_mesh)
        # print('tmp.requires_grad:',tmp.requires_grad)
        
        # self.meshes_trans.retain_grad()
        # apply pointnet to get mesh feature
        self.mesh_feature,_,_ = self.pointnet(self.meshes_trans) # Bx1024
        # print("meshes.shape",meshes.shape)
        # print("mesh_feature.shape",mesh_feature.shape)
        
        # map the latent code to frequence and phase shifts
        self.z_ = torch.cat([z_t, self.mesh_feature], 1)
        
        frequencies_offsets = self.network(self.z_)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]
        
        # self.meshes_trans.retain_grad()
        # self.mesh_feature.retain_grad()
        # self.z_.retain_grad()
        
        # generate depths
        if pitch is not None and yaw is not None:
            depths, masks, verts_ndc = self.depth_renderer(self.meshes_trans, pitch, yaw, img_size=img_size)
            vids, wets, idx = self.ldmk_table(yaw=yaw, pitch=pitch)
            # ldmks = batched_index_select(verts_ndc, 1, vids)
            self.logger("vids:",vids,info_level='debug')
            self.logger("wets:",wets,info_level='debug')
            
            ldmks = []
            # print('vids.shape:',vids.shape)
            # print('verts_ndc.shape:',verts_ndc.shape)
            
            for b, vid in enumerate(vids):
                ldmks.append(verts_ndc[b][vid])
            ldmks = torch.stack(ldmks, 0)
            
            ldmks = (ldmks[...,:2] + 1) / 2 * (img_size - 1)
            ldmks = (img_size - 1) - ldmks
            # ldmks[...,0] = (img_size-1)-ldmks[...,0]
            
            '''
            from utils.utils import save_img_with_landmarks
            img_ = depths[0]
            print('img_.shape:',img_.shape)
            img_ = (img_-img_.min())/(img_.max()-img_.min())
            torch.save(ldmks, './ldmks.pth')
            img_ = img_.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            import cv2
            img_ = cv2.resize(img_,(512,512))
            ldmks = (ldmks-0.5)*(512-1)/(img_size-1)+0.5
            save_img_with_landmarks(img_,f'./check_ldmk_{idx[0]}.png',ldmks[0].detach().to('cpu').numpy())
            '''
            
            return frequencies, phase_shifts, depths, masks, ldmks, wets
        else:
            return frequencies, phase_shifts


class ControlableMappingNetworkWoPointnet(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim, rotation_mode='Euler', rotation_trainable=True, **kwargs):
        super().__init__()
        self.logger = kwargs.get('logger',lambda x: None)

        self.logger(f"SIREN using rotation mode: {rotation_mode}")
        self.logger(f"Training parameters of Transformer: {rotation_trainable}")

        self.rotation_mode = rotation_mode
        self.rotation_trainable = rotation_trainable
        
        self.z_dim = z_dim
        self.dim_z_t = 256
        self.dim_z_s = 199
        self.dim_z_e = 50
        # print('length of z:',self.dim_z_t, self.dim_z_s, self.dim_z_e)
        assert self.z_dim == self.dim_z_t + self.dim_z_s + self.dim_z_e, \
            "The equation self.z_dim = self.dim_z_t + self.dim_z_s + self.dim_z_e should always holds. \
             Please check z_dim."

        self.bfm = BFM(chosen_file='./models/bfm_models/index_2106pts_bs.txt')
        self.depth_renderer = DepthRendder(self.bfm.faces)
        self.transformer = Transform(
                                     rotation_mode=self.rotation_mode,
                                     rotation_trainable=self.rotation_trainable)
        
        self.ldmk_table = LandmarkTable()

        self.network = nn.Sequential(nn.Linear(self.z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_hidden_dim),
                                    nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        # self.transformer.apply(norm_init)
        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def set_device(self, device=None):
        self.device = device
        
    def forward(self, z, pitch=None, yaw=None, img_size=128, denorm_3dmm=True, delt_v=None, **kwargs):
        # map the latent code to frequence and phase shifts
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]
        
        # generate depths
        if pitch is not None and yaw is not None:
            self.logger('yaw and pitch are not None.',info_level='debug')
            
            # generate meshes and transform to the pigan space
            z_e_s = z[:, self.dim_z_t:self.dim_z_t+self.dim_z_e+self.dim_z_s]
            self.logger(f'denorm_3dmm in siren: {denorm_3dmm}', info_level='debug')
            meshes = self.bfm(z_e_s, denorm_3dmm=denorm_3dmm)
            meshes_trans = self.transformer(meshes)
            
            if delt_v is not None:
                # from IPython import embed
                # embed()
                self.logger('delt_v is not None.',info_level='debug')
                meshes_trans += delt_v.unsqueeze(1)
                
            depths, masks, verts_ndc = self.depth_renderer(meshes_trans, pitch, yaw, img_size=img_size)
            vids, wets, idx = self.ldmk_table(yaw=yaw, pitch=pitch)
            # self.logger("vids:",vids,info_level='debug')
            # self.logger("wets:",wets,info_level='debug')
            
            ldmks = []
            for b, vid in enumerate(vids):
                ldmks.append(verts_ndc[b][vid])
            ldmks = torch.stack(ldmks, 0)
            
            ldmks = (ldmks[...,:2] + 1) / 2 * (img_size - 1)
            ldmks = (img_size - 1) - ldmks
            
            return frequencies, phase_shifts, depths, masks, ldmks, wets
        else:
            return frequencies, phase_shifts


class PerceiverIOMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_output_dim, vert_dim=3,
                 encoder_depth=4, cost_latent_token_num=64,
                 cost_latent_input_dim=32, cost_latent_dim=32,
                 dropout=0.1, num_heads=2, query_latent_dim=32,):
        super(PerceiverIOMappingNetwork, self).__init__()

        self.vert_mapper = nn.Linear(
            vert_dim, cost_latent_input_dim, bias=True)
        self.z_mapper = nn.Linear(
            z_dim, cost_latent_input_dim, bias=True)
        self.query_mapper = nn.Linear(
            vert_dim, cost_latent_input_dim, bias=True)
        self.out_mapper = nn.Linear(
            cost_latent_dim, map_output_dim, bias=True)

        cfg = EasyDict({
            "encoder_depth": encoder_depth,
            "cost_latent_token_num": cost_latent_token_num,
            "cost_latent_dim": cost_latent_dim,
            "cost_latent_input_dim": cost_latent_input_dim,
            "dropout": dropout,
            "num_heads": num_heads,
            "query_latent_dim": query_latent_dim,
        })
        self.encoder = WarpingEncoder(cfg)
        self.decoder = MemoryDecoderLayer(cfg)

    def set_device(self, device=None):
        self.device = device

    def forward(self, z, query, face_recon):
        split_bs = 1
        bs = z.shape[0]
        freqs = []
        shifts = []
        for b in range(math.ceil(bs/split_bs)):
            z_subset = z[b*split_bs:(b+1)*split_bs]
            q_subset = query[b*split_bs:(b+1)*split_bs]
            vert = face_recon.z_to_pigan_vert(z_subset)[:, ::15, :]
            vert = vert.detach()
            vert_map = self.vert_mapper(vert)  # dropout

            if z_subset.dim() == 2:
                z_subset = z_subset.unsqueeze(-2)
            z_map = self.z_mapper(z_subset)  # dropout
            inp_map = torch.cat([vert_map, z_map], dim=-2)

            qry_map = self.query_mapper(q_subset)
            latent = self.encoder(inp_map)
            output = self.decoder(qry_map, None, None, latent)[0]
            frequencies_offsets = self.out_mapper(output)
            frequencies = frequencies_offsets[
                ..., :frequencies_offsets.shape[-1]//2]
            phase_shifts = frequencies_offsets[
                ..., frequencies_offsets.shape[-1]//2:]
            freqs.append(frequencies)
            shifts.append(phase_shifts)

        freqs = torch.cat(freqs, dim=0)
        shifts = torch.cat(shifts, dim=0)

        return freqs, shifts


def frequency_init(freq):
    def init(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        if freq.dim() != x.dim():
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)
        # print('freq.shape:',freq.shape)
        # print('phase_shift.shape:',phase_shift.shape)
        # print('x.shape:',x.shape)
        
        return torch.sin(freq * x + phase_shift)


class TALLSIREN(nn.Module):
    """Primary SIREN  architecture used in pi-GAN generators."""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(input_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        x = input

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = self.color_layer_linear(rbg)

        return torch.cat([rbg, sigma], dim=-1)


class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength
        
    def forward(self, coordinates):
        return coordinates * self.scale_factor


class SPATIALSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)
    
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):

        with_grad = torch.is_grad_enabled()
        get_normal = kwargs.get('get_normal', False)

        with torch.set_grad_enabled(with_grad or get_normal):
            input.requires_grad_()
            frequencies = frequencies*15 + 30

            input = self.gridwarper(input)
            x = input

            for index, layer in enumerate(self.network):
                start = index * self.hidden_dim
                end = (index+1) * self.hidden_dim
                x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

            sigma = self.final_layer(x)

            if get_normal:
                input_grad = torch.autograd.grad(torch.sum(sigma), input, create_graph=with_grad)[0]
                normal = - input_grad  / (torch.norm(input_grad, dim=-1, keepdim=True) + 1e-7)
                normal = normal.clamp(min=-1, max=1)
                normal[torch.isnan(normal)] = 0.

                return normal
            else:
                rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
                rbg = torch.sigmoid(self.color_layer_linear(rbg))

                return torch.cat([rbg, sigma], dim=-1)


class CONTROLABLESIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, **kwargs):
        super().__init__()
        self.logger = kwargs.get('logger',lambda x: None)
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = ControlableMappingNetwork(self.z_dim,
                                                    256, #256
                                                    (len(self.network) + 1)*hidden_dim*2,
                                                    **kwargs)
        
        '''
        mesh = torch.from_numpy(np.array(
                    [[[0,0,0],[1,0,0],[0,1,0]]]
                )).float().to(self.device)
        mesh_t = self.mapping_network.transformer(mesh)
        print(mesh_t.requires_grad)
        print('-----------')
        '''
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        # Don't worry about this,
        # it was added to ensure compatibility with another model.
        # Shouldn't affect performance.
        self.gridwarper = UniformBoxWarp(0.24) 
        
    def set_device(self, device):
        self.device = device
        
    def forward(self, input, z, ray_directions, pitch=None, yaw=None, img_size=128, **kwargs):
        if pitch is not None and yaw is not None:
            self.frequencies, self.phase_shifts, gt_depths, masks, gt_ldmks, wets  = self.mapping_network(z, pitch, yaw, img_size)
            # self.frequencies.retain_grad()
            # self.phase_shifts.retain_grad()
            return  self.forward_with_frequencies_phase_shifts(
                        input, self.frequencies, self.phase_shifts, ray_directions, **kwargs), \
                    gt_depths, masks, gt_ldmks, wets
        else:
            frequencies, phase_shifts = self.mapping_network(z)
            return  self.forward_with_frequencies_phase_shifts(
                        input, frequencies, phase_shifts, ray_directions, **kwargs)
        
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        # x.retain_grad()

        for index, layer in enumerate(self.network):
            # print('layer:',index)
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        # sigma.retain_grad()
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        # rbg.retain_grad()
        return torch.cat([rbg, sigma], dim=-1)
    
    
class CONTROLABLESIRENBASELINEWOPOINTNET(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, **kwargs):
        super().__init__()
        self.logger = kwargs.get('logger',lambda x: None)
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = ControlableMappingNetworkWoPointnet(self.z_dim,
                                                    256, #256
                                                    (len(self.network) + 1)*hidden_dim*2,
                                                    **kwargs)
        
        
        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        # Don't worry about this,
        # it was added to ensure compatibility with another model.
        # Shouldn't affect performance.
        self.gridwarper = UniformBoxWarp(0.24) 
        
    def set_device(self, device):
        self.device = device
        self.mapping_network.set_device(device)
        
    def forward(self, input, z, ray_directions, pitch=None, yaw=None, img_size=128, **kwargs):
        if pitch is not None and yaw is not None:
            self.frequencies, self.phase_shifts, gt_depths, masks, gt_ldmks, wets  = self.mapping_network(z, pitch, yaw, img_size, **kwargs)
            # self.frequencies.retain_grad()
            # self.phase_shifts.retain_grad()
            return  self.forward_with_frequencies_phase_shifts(
                        input, self.frequencies, self.phase_shifts, ray_directions, **kwargs), \
                    gt_depths, masks, gt_ldmks, wets
        else:
            frequencies, phase_shifts = self.mapping_network(z)
            return  self.forward_with_frequencies_phase_shifts(
                        input, frequencies, phase_shifts, ray_directions, **kwargs)
        
    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
        # x.retain_grad()

        for index, layer in enumerate(self.network):
            # print('layer:',index)
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        # sigma.retain_grad()
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        # rbg.retain_grad()
        return torch.cat([rbg, sigma], dim=-1)


class PERCEIVERIOSIRENBASELINE(nn.Module):
    """Same architecture as TALLSIREN but adds a UniformBoxWarp to map input points to -1, 1"""

    def __init__(self, input_dim=2, z_dim=100, hidden_dim=256, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.network = nn.ModuleList([
            FiLMLayer(3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))
        
        self.mapping_network = PerceiverIOMappingNetwork(
            z_dim, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(first_layer_film_sine_init)
        
        self.gridwarper = UniformBoxWarp(0.24) # Don't worry about this, it was added to ensure compatibility with another model. Shouldn't affect performance.

    def forward(self, input, z, ray_directions, face_recon, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z, input, face_recon)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30
        
        input = self.gridwarper(input)
        x = input
            
        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])
        
        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))
        
        return torch.cat([rbg, sigma], dim=-1)

    
class UniformBoxWarp(nn.Module):
    def __init__(self, sidelength):
        super().__init__()
        self.scale_factor = 2/sidelength

    def forward(self, coordinates):
        return coordinates * self.scale_factor


def sample_from_3dgrid(coordinates, grid):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    coordinates = coordinates.float()
    grid = grid.float()
    
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=True)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features


def modified_first_sine_init(m):
    with torch.no_grad():
        # if hasattr(m, 'weight'):
        if isinstance(m, nn.Linear):
            num_input = 3
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class EmbeddingPiGAN128(nn.Module):
    """Smaller architecture that has an additional cube of embeddings. Often gives better fine details."""
    
    def __init__(self, input_dim=2, z_dim=100, hidden_dim=128, output_dim=1, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.network = nn.ModuleList([
            FiLMLayer(32 + 3, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
            FiLMLayer(hidden_dim, hidden_dim),
        ])
        pprint(self.network)

        self.final_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3))

        self.mapping_network = CustomMappingNetwork(z_dim, 256, (len(self.network) + 1)*hidden_dim*2)

        self.network.apply(frequency_init(25))
        self.final_layer.apply(frequency_init(25))
        self.color_layer_sine.apply(frequency_init(25))
        self.color_layer_linear.apply(frequency_init(25))
        self.network[0].apply(modified_first_sine_init)

        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 96, 96, 96)*0.01)
        
        # !! Important !! Set this value to the expected side-length of your scene. e.g. for for faces, heads usually fit in
        # a box of side-length 0.24, since the camera has such a narrow FOV. For other scenes, with higher FOV, probably needs to be bigger.
        self.gridwarper = UniformBoxWarp(0.24)

    def forward(self, input, z, ray_directions, **kwargs):
        frequencies, phase_shifts = self.mapping_network(z)
        return self.forward_with_frequencies_phase_shifts(input, frequencies, phase_shifts, ray_directions, **kwargs)

    def forward_with_frequencies_phase_shifts(self, input, frequencies, phase_shifts, ray_directions, **kwargs):
        frequencies = frequencies*15 + 30

        input = self.gridwarper(input)
        shared_features = sample_from_3dgrid(input, self.spatial_embeddings)
        x = torch.cat([shared_features, input], -1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer(x)
        rbg = self.color_layer_sine(torch.cat([ray_directions, x], dim=-1), frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
        rbg = torch.sigmoid(self.color_layer_linear(rbg))

        return torch.cat([rbg, sigma], dim=-1)

    
class EmbeddingPiGAN256(EmbeddingPiGAN128):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, hidden_dim=256)
        self.spatial_embeddings = nn.Parameter(torch.randn(1, 32, 64, 64, 64)*0.1)
        

if __name__ == "__main__":
    # z_dim = 256+199+50
    # mapping_network = ControlableMappingNetwork(z_dim,256,100)
    # z = torch.rand(8, z_dim)
    # mapped_latent_code = mapping_network(z)
    # print("shape of mapped_latent_code:",
    #       mapped_latent_code[0].shape,
    #       mapped_latent_code[1].shape)
    
    # model = CONTROLABLESIRENBASELINE(z_dim=z_dim)
    # inputs = torch.randn(8,100,3)
    # ray_directions = torch.rand(8,100,3)
    # output = model(inputs,z,ray_directions)
    # print("output shape when global_feat = False:", output.shape)

    # from siren.siren import PerceiverIOMappingNetwork
    mapping_net = PerceiverIOMappingNetwork(80+64, 17)
    # import torch
    verts = torch.rand(5, 100, 3)
    zs = torch.rand(5, 80+64)
    qry = torch.rand(5, 30, 3)
    mod = mapping_net(zs, verts, qry)