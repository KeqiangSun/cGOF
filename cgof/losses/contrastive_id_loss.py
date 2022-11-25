import sys
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torch
import math
import numpy as np

def sample_yaw_pitch(n, horizontal_mean, vertical_mean, horizontal_stddev, vertical_stddev, device):
    
    # theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
    # phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean
    # yaw = (theta - math.pi / 2) * 180.0 / math.pi
    # pitch = (math.pi / 2 - phi) * 180.0 / math.pi
    yaw = torch.randn((n, 1), device=device)
    pitch = torch.randn((n, 1), device=device)

    return yaw, pitch

class ZSampler:
    def __init__(self, length, dist="gaussian"):
        self.length = length
        self.dist = dist

    def produce(self, B, device):
        shape = (B, self.length)
        if self.dist == 'gaussian':
            z = torch.randn(shape, device=device)
        elif self.dist == 'uniform':
            z = torch.rand(shape, device=device) * 2 - 1
        return z

class Z_Manager:
    def __init__(self, length, device):
        self.length = length
        self.device = device
        self.sampler = ZSampler(length)
        self.dim_id = 0
        self.dim_exp = 80
        self.dim_bg_geo = 144
        self.dim_tex = 224
        self.dim_gamma = 304
        self.dim_bg_tex = 331

        self.attribute_lens = {
            'id' : 80,
            'exp' : 64,
            'bg_geo' : 80,
            'tex' : 80,
            'gamma' : 27,
            'bg_tex' : 80
        }

        self.id_mask = torch.zeros((1, length), device=device).bool()
        self.id_mask[:, self.dim_id:self.dim_exp] = True
        self.id_mask[:, self.dim_tex:self.dim_gamma] = True

        self.geo_mask = torch.zeros((1, length), device=device).bool()
        self.geo_mask[:, self.dim_id:self.dim_tex] = True

        self.tex_mask = torch.zeros((1, length), device=device).bool()
        self.tex_mask[:, self.dim_tex:self.dim_bg_tex] = True

        self.face_mask = torch.zeros((1, length), device=device).bool()
        self.face_mask[:, self.dim_id:self.dim_bg_geo] = True
        self.face_mask[:, self.dim_tex:self.dim_bg_tex] = True

        self.faceimg_mask = torch.zeros((1, length), device=device).bool()
        self.faceimg_mask[:, self.dim_id:self.dim_bg_geo] = True
        self.faceimg_mask[:, self.dim_tex:self.dim_bg_tex] = True

        self.faceforegeo_mask = torch.zeros((1, length), device=device).bool()
        self.faceforegeo_mask[:, self.dim_id:self.dim_bg_geo] = True

        self.facebackgeo_mask = torch.zeros((1, length), device=device).bool()
        self.facebackgeo_mask[:, self.dim_bg_geo:self.dim_tex] = True

    def split_z(self, z):
        return {
            "id": z[:, self.dim_id: self.dim_exp],
            "exp": z[:, self.dim_exp: self.dim_bg_geo],
            "bg_geo": z[:, self.dim_bg_geo: self.dim_tex],
            "tex": z[:, self.dim_tex: self.dim_gamma],
            "gamma": z[:, self.dim_gamma: self.dim_bg_tex],
            "bg_tex": z[:, self.dim_bg_tex:]
        }

    def resample_attribute(self, z, name):
        resampled_z = torch.randn((z.shape[0], self.attribute_lens[name]), device=z.device)
        if name == 'id':
            z[:, self.dim_id: self.dim_exp] = resampled_z
        elif name == 'exp':
            z[:, self.dim_exp: self.dim_bg_geo] = resampled_z
        elif name == 'bg_geo':
            z[:, self.dim_bg_geo: self.dim_tex] = resampled_z
        elif name == 'tex':
            z[:, self.dim_tex: self.dim_gamma] = resampled_z
        elif name == 'gamma':
            z[:, self.dim_gamma: self.dim_bg_tex] = resampled_z
        elif name == 'bg_tex':
            z[:, self.dim_bg_tex:] = resampled_z
        else:
            raise ValueError()
        return z

    def set_attribute(self, z, name, attr):
        if name == 'identity':
            z[:, self.dim_id: self.dim_exp] = attr[0]
            z[:, self.dim_tex: self.dim_gamma] = attr[1]
        elif name == 'exp':
            z[:, self.dim_exp: self.dim_bg_geo] = attr
        elif name == 'bg':
            z[:, self.dim_bg_geo: self.dim_tex] = attr[0]
            z[:, self.dim_bg_tex:] = attr[1]
        elif name == 'gamma':
            z[:, self.dim_gamma: self.dim_bg_tex] = attr
        else:
            raise ValueError()

    def make_contrast(self, z, mask):
        z_pos = self.sampler.produce(1, self.device)
        z_neg = self.sampler.produce(1, self.device)
        z_pos[mask] = z[mask]       # keep id coeffs in positive z
        z_neg[~mask] = z[~mask]     # keep other coeffs in negative z

        return [z_pos, z_neg]

    def make_contrast_id(self, z):
        return self.make_contrast(z, self.id_mask)

    def make_contrast_geo(self, z):
        return self.make_contrast(z, self.geo_mask)

    def make_contrast_tex(self, z):
        return self.make_contrast(z, self.tex_mask)
    
    def make_contrast_face(self, z):
        return self.make_contrast(z, self.face_mask)

    def make_contrast_faceimg(self, z):
        return self.make_contrast(z, self.face_mask)

    def make_contrast_facegeo(self, z):
        z_pos = self.sampler.produce(1, self.device)
        z_neg = self.sampler.produce(1, self.device)
        z_pos[self.faceforegeo_mask] = z[self.faceforegeo_mask]       # keep id coeffs in positive z
        z_neg[self.facebackgeo_mask] = z[self.facebackgeo_mask]     # keep other coeffs in negative z
        return [z_pos, z_neg]

class ContrastiveLoss(nn.Module):
    def __init__(self, length, device, tao = 0.07, margin = 1, overall_contrast=False):
        super(ContrastiveLoss, self).__init__()
        self.sampler = ZSampler(length)
        self.tao = tao
        self.margin = margin
        self.overall_contrast = overall_contrast
        self.pair_loss_type = 'l2'

    def contrastive_sim(self, sim_pos, sim_neg):
        # contrastive_loss = -torch.log(torch.exp(sim_pos / self.tao) / (torch.exp(sim_pos / self.tao) + torch.exp(sim_neg / self.tao)))
        contrastive_loss = torch.mean(-sim_pos + torch.relu(sim_neg - self.margin))
        return contrastive_loss

    def contrastive_dis(self, dis_pos, dis_neg, margin=None):
        if margin is None:
            margin = self.margin
        if self.overall_contrast:
            contrastive_loss = torch.mean(dis_pos) + torch.relu(margin - torch.mean(dis_neg))
        else:
            contrastive_loss = torch.mean(dis_pos + torch.relu(margin - dis_neg))
        return contrastive_loss

    def pair_consistency(self, recon_z, recon_z_pos, recon_z_neg, input_z_neg):
        if self.pair_loss_type == 'l1':
            loss_consistency = torch.mean(torch.abs(recon_z - recon_z_pos))
            loss_recon = torch.mean(torch.abs(recon_z_neg - input_z_neg))
        elif self.pair_loss_type == 'l2':
            loss_consistency = torch.nn.MSELoss()(recon_z, recon_z_pos)
            loss_recon = torch.nn.MSELoss()(recon_z_neg, input_z_neg)

        return (loss_consistency + loss_recon) / 2

class ContrastiveIDLoss(ContrastiveLoss):
    def __init__(self, length, device, margin=0.9, tao = 0.07):
        super(ContrastiveIDLoss, self).__init__(length, device, tao, margin)
        import os
        import models.networks as networks
        pretrained_path = 'Deep3DFaceRecon_pytorch/checkpoints/recog_model/ms1mv3_arcface_arcface_r50_fp16/backbone.pth'
        self.net_recog = networks.define_net_recog(
            net_recog='r50', pretrained_path=pretrained_path
            )
        self.net_recog.to(device).eval()
    
    def ddp(self, rank):
        self.net_recog_ddp = DDP(self.net_recog, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
        self.net_recog = self.net_recog_ddp.module

    def forward(self, input):
        """
        1 - cosine distance
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1) , RGB order
            imageB       --same as imageA
        """

        recon_attr = torch.cat([input['shape'], input['tex']], dim=1)
        input_attr = torch.cat([input['shape_input'], input['tex_input']], dim=1)
        recon_z, recon_z_pos, recon_z_neg, input_z_neg = recon_attr[0], recon_attr[1], recon_attr[2], input_attr[2]
        p_loss = self.pair_consistency(recon_z, recon_z_pos, recon_z_neg, input_z_neg) * input['p_lambda']

        imgs, trans_m = input['imgs'], input['trans_m']
        feats = self.net_recog_ddp(imgs, trans_m)

        id = feats[0]
        id_pos = feats[1]
        id_neg = feats[2]

        sim_pos = torch.sum(id * id_pos, dim=-1)
        sim_neg = torch.sum(id * id_neg, dim=-1)

        c_loss = self.contrastive_sim(sim_pos, sim_neg) * input['c_lambda']
        print(f"p loss = {p_loss}")
        print(f"c loss = {c_loss}")

        return p_loss + c_loss

class ContrastiveGeoLoss(ContrastiveLoss):
    def __init__(self, length, device, loss_type = 'l2', margin = 1, overall_contrast=False):
        super(ContrastiveGeoLoss, self).__init__(length, device, margin=margin, overall_contrast=overall_contrast)
        self.loss_type = loss_type

    def forward(self, depths):
        d = depths[0]
        d_pos = depths[1]
        d_neg = depths[2]
        if self.loss_type == 'l1':
            dis_pos = torch.abs(d - d_pos)
            dis_neg = torch.abs(d - d_neg)
        elif self.loss_type == 'l2':
            dis_pos = torch.nn.MSELoss()(d, d_pos)
            dis_neg = torch.nn.MSELoss()(d, d_neg)

        c_loss = self.contrastive_dis(dis_pos, dis_neg)

        return c_loss

class ContrastiveTexLoss(ContrastiveLoss):
    def __init__(self, length, device, loss_type ='l1', margin = 1):
        super(ContrastiveTexLoss, self).__init__(length, device, margin=margin)
        self.loss_type = loss_type

    def forward(self, input):
        recon_attr = torch.cat([input['tex'], input['gamma']], dim=1)
        input_attr = torch.cat([input['tex_input'], input['gamma_input']], dim=1)
        recon_z, recon_z_pos, recon_z_neg, input_z_neg = recon_attr[0], recon_attr[1], recon_attr[2], input_attr[2]
        p_loss = self.pair_consistency(recon_z, recon_z_pos, recon_z_neg, input_z_neg) * input['p_lambda']
        # tex, tex_pos, tex_neg = tex[0], tex[1], tex[2]
        # gm, gm_pos, gm_neg = gm[0], gm[1], gm[2]
        # dis_pos = torch.cat(
        #     [
        #         torch.abs(tex - tex_pos),
        #         torch.abs(gm - gm_pos)
        #     ], dim=0
        # )
        # dis_neg = torch.cat(
        #     [
        #         torch.abs(tex - tex_neg),
        #         torch.abs(gm - gm_neg)
        #     ], dim=0
        # )

        # print(f"tex pos dis = {dis_pos.mean()}")
        # print(f"tex neg dis = {dis_neg.mean()}")

        # return self.contrastive_dis(dis_pos, dis_neg)
        return p_loss

class ContrastiveTexGramLoss(ContrastiveLoss):
    def __init__(self, length, device, loss_type ='l1', margin = 1):
        super(ContrastiveTexLoss, self).__init__(length, device, margin=margin)
        self.loss_type = loss_type
        self.vgg = None

    def forward(self, input):
        recon_attr = torch.cat([input['tex'], input['gamma']], dim=1)
        input_attr = torch.cat([input['tex_input'], input['gamma_input']], dim=1)
        recon_z, recon_z_pos, recon_z_neg, input_z_neg = recon_attr[0], recon_attr[1], recon_attr[2], input_attr[2]
        p_loss = self.pair_consistency(recon_z, recon_z_pos, recon_z_neg, input_z_neg) * input['p_lambda']

        vgg_feature = input['vgg_feature']
        vgg_feature_ori = vgg_feature[0]
        vgg_feature_pos = vgg_feature[1]
        vgg_feature_neg = vgg_feature[2]

        gram_pos = self.gram(vgg_feature_ori, vgg_feature_pos)
        gram_neg = self.gram(vgg_feature_ori, vgg_feature_neg)

        
        # tex, tex_pos, tex_neg = tex[0], tex[1], tex[2]
        # gm, gm_pos, gm_neg = gm[0], gm[1], gm[2]
        # dis_pos = torch.cat(
        #     [
        #         torch.abs(tex - tex_pos),
        #         torch.abs(gm - gm_pos)
        #     ], dim=0
        # )
        # dis_neg = torch.cat(
        #     [
        #         torch.abs(tex - tex_neg),
        #         torch.abs(gm - gm_neg)
        #     ], dim=0
        # )

        # print(f"tex pos dis = {dis_pos.mean()}")
        # print(f"tex neg dis = {dis_neg.mean()}")

        # return self.contrastive_dis(dis_pos, dis_neg)
        return p_loss


class ContrastiveFaceLoss(ContrastiveLoss):
    def __init__(self, length, device, lambda_img, lambda_d, margin_m, margin_img, margin_d, overall_contrast):
        super(ContrastiveFaceLoss, self).__init__(length, device, overall_contrast=overall_contrast)
        self.margin_m = margin_m
        self.margin_img = margin_img
        self.margin_d = margin_d
        self.lambda_img = lambda_img
        self.lambda_d = lambda_d
        self.bce = nn.BCELoss()
        
    def forward(self, imgs, masks, depths):
        img, img_pos, img_neg = imgs[0], imgs[1], imgs[2]
        # masks_c3 = masks.expand([-1, imgs.shape[1], -1, -1])

        mask, mask_pos, mask_neg = masks[0], masks[1], masks[2]

        depth, depth_pos, depth_neg = depths[0], depths[1], depths[2]

        foreground_mask = mask * mask_pos * mask_neg
        background_mask = (1-mask) * (1-mask_pos) * (1-mask_neg)

        dis_pos = torch.abs((img - img_pos) * foreground_mask).mean()
        dis_neg = torch.abs((img - img_neg) * foreground_mask).mean()
        c_img_loss1 = self.contrastive_dis(dis_pos, dis_neg, self.margin_img) * self.lambda_img

        dis_pos = torch.abs((img - img_neg) * background_mask).mean()
        dis_neg = torch.abs((img - img_pos) * background_mask).mean()
        c_img_loss2 = self.contrastive_dis(dis_pos, dis_neg, self.margin_img) * self.lambda_img
        c_img_loss = (c_img_loss1 + c_img_loss2) / 2
        print(f"c img loss = {c_img_loss}")
        # print(f"dis pos img loss = {dis_pos}")
        # print(f"dis neg img loss = {dis_neg}")

        dis_pos = torch.abs((depth - depth_pos) * foreground_mask).mean()
        dis_neg = torch.abs((depth - depth_neg) * foreground_mask).mean()
        c_depth_loss1 = self.contrastive_dis(dis_pos, dis_neg, self.margin_d) * self.lambda_d

        dis_pos = torch.abs((depth - depth_neg) * background_mask).mean()
        dis_neg = torch.abs((depth - depth_pos) * background_mask).mean()
        c_depth_loss2 = self.contrastive_dis(dis_pos, dis_neg, self.margin_d) * self.lambda_d
        c_depth_loss = (c_depth_loss1 + c_depth_loss2) / 2
        print(f"c depth loss = {c_depth_loss}")
        # print(f"dis pos d loss = {dis_pos}")
        # print(f"dis neg d loss = {dis_neg}")

        return c_img_loss * self.lambda_img + c_depth_loss * self.lambda_d

class ContrastiveFaceImgLoss(ContrastiveLoss):
    def __init__(self, length, device, lambda_img, margin_m, margin_img):
        super(ContrastiveFaceImgLoss, self).__init__(length, device)
        self.margin_m = margin_m
        self.margin_img = margin_img
        self.lambda_img = lambda_img
        self.bce = nn.BCELoss()
        
    def forward(self, imgs, masks):
        img, img_pos, img_neg = imgs[0], imgs[1], imgs[2]
        # masks_c3 = masks.expand([-1, imgs.shape[1], -1, -1])

        mask, mask_pos, mask_neg = masks[0], masks[1], masks[2]

        depth, depth_pos, depth_neg = depths[0], depths[1], depths[2]

        foreground_mask = mask * mask_pos * mask_neg
        background_mask = (1-mask) * (1-mask_pos) * (1-mask_neg)

        dis_pos = torch.abs((img - img_pos) * foreground_mask).mean()
        dis_neg = torch.abs((img - img_neg) * foreground_mask).mean()
        c_img_loss1 = self.contrastive_dis(dis_pos, dis_neg, self.margin_img) * self.lambda_img

        dis_pos = torch.abs((img - img_neg) * background_mask).mean()
        dis_neg = torch.abs((img - img_pos) * background_mask).mean()
        c_img_loss2 = self.contrastive_dis(dis_pos, dis_neg, self.margin_img) * self.lambda_img
        c_img_loss = c_img_loss1 + c_img_loss2

        return c_img_loss * self.lambda_img

class ContrastiveFaceDepthLoss(ContrastiveLoss):
    def __init__(self, length, device, lambda_d, margin_m, margin_d):
        super(ContrastiveFaceDepthLoss, self).__init__(length, device)
        self.margin_m = margin_m
        self.margin_d = margin_d
        self.lambda_d = lambda_d
        
    def forward(self, imgs, masks, depths):
        mask, mask_pos, mask_neg = masks[0], masks[1], masks[2]

        depth, depth_pos, depth_neg = depths[0], depths[1], depths[2]

        foreground_mask = mask * mask_pos * mask_neg
        background_mask = (1-mask) * (1-mask_pos) * (1-mask_neg)

        dis_pos = torch.abs((depth - depth_pos) * foreground_mask).mean()
        dis_neg = torch.abs((depth - depth_neg) * foreground_mask).mean()
        c_depth_loss1 = self.contrastive_dis(dis_pos, dis_neg, self.margin_d) * self.lambda_d

        dis_pos = torch.abs((depth - depth_neg) * background_mask).mean()
        dis_neg = torch.abs((depth - depth_pos) * background_mask).mean()
        c_depth_loss2 = self.contrastive_dis(dis_pos, dis_neg, self.margin_d) * self.lambda_d
        c_depth_loss = c_depth_loss1 + c_depth_loss2

        return c_depth_loss * self.lambda_d


class ImitativeTexLoss(nn.Module):
    def __init__(self):
        super(ImitativeTexLoss, self).__init__()

    def forward(self, gen_tex, recon_tex, tex_mask):
        """
        1 - cosine distance
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1) , RGB order
            imageB       --same as imageA
        """

        return torch.mean(torch.abs(gen_tex*tex_mask - recon_tex*tex_mask))
