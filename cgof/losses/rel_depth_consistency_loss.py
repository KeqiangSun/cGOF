import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss

from losses import functional as F
import torch.nn.functional as nnF
# import functional as F
import numpy as np
import scipy.sparse as sp

from IPython import embed

class RelDConsistencyLoss(nn.Module):
    def __init__(self, num_sample_pairs=1000, epsilon=1e-5, using_proportional_reld=False):
        super(RelDConsistencyLoss, self).__init__()
        self.num_pairs = num_sample_pairs
        self.epsilon = epsilon
        self.using_proportional_reld = using_proportional_reld
        self.proportion_band = 10

    def pair_ranking_loss(self, input):
        depth_3dmm = input['depth_3dmm'].squeeze()      # B, 1, H, W
        depth_pigan = input['depth_pigan'].squeeze()    # B, 1, H, W
        
        # mask = torch.from_numpy(input['mask'][:, :, :, 0]).to(depth_3dmm.device)                # B, H, W, 3
        mask = input['mask'][:, 0, :, :].bool().to(depth_3dmm.device)    # B, H, W, 3

        B, H, W = mask.shape
        depth_3dmm = torch.masked_select(depth_3dmm, mask)
        depth_pigan = torch.masked_select(depth_pigan, mask)
        valid_num = torch.sum(mask.view(B, H*W), dim=1)
        batch_base = torch.cumsum(valid_num, dim=0) - valid_num

        # get a pair of points ranging from 0~valid_num
        max_num = torch.max(valid_num)
        sample1 = torch.randint(max_num, (B, self.num_pairs), device=depth_3dmm.device) % valid_num[:, None]
        sample2 = torch.randint(max_num, (B, self.num_pairs), device=depth_3dmm.device) % valid_num[:, None]

        # put the individual sampled index to the index in the batch
        sample1 = (sample1 + batch_base[:, None]).view(-1)
        sample2 = (sample2 + batch_base[:, None]).view(-1)

        # 3dmm[s1] > 3dmm[s2] while pigan[s1] < pigan[s2]
        data1 = torch.masked_select(
                depth_pigan[sample1] - depth_pigan[sample2],
                depth_3dmm[sample1] > (depth_3dmm[sample2] + self.epsilon)
            )
        loss1 = nnF.relu(-data1)

        # 3dmm[s1] < 3dmm[s2] while pigan[s1] > pigan[s2]
        data2 = torch.masked_select(
                depth_pigan[sample1] - depth_pigan[sample2],
                depth_3dmm[sample1] < (depth_3dmm[sample2] - self.epsilon)
            )
        loss2 = nnF.relu(data2)

        return (loss1.mean() + loss2.mean()) / 2

    # def triplet_proportion_loss(self, input):
    #     depth_3dmm = input['depth_3dmm'].squeeze()      # B, 1, H, W
    #     depth_pigan = input['depth_pigan'].squeeze()    # B, 1, H, W
        
    #     # mask = torch.from_numpy(input['mask'][:, :, :, 0]).to(depth_3dmm.device)                # B, H, W, 3
    #     mask = input['mask'][:, 0, :, :].bool().to(depth_3dmm.device)    # B, H, W, 3

    #     B, H, W = mask.shape
    #     depth_3dmm = torch.masked_select(depth_3dmm, mask)
    #     depth_pigan = torch.masked_select(depth_pigan, mask)
    #     valid_num = torch.sum(mask.view(B, H*W), dim=1)
    #     batch_base = torch.cumsum(valid_num, dim=0) - valid_num

    #     # get a pair of points ranging from 0~valid_num
    #     max_num = torch.max(valid_num)
    #     sample1 = torch.randint(max_num, (B, self.num_pairs), device=depth_3dmm.device) % valid_num[:, None]
    #     sample2 = torch.randint(max_num, (B, self.num_pairs), device=depth_3dmm.device) % valid_num[:, None]
    #     sample3 = torch.randint(max_num, (B, self.num_pairs), device=depth_3dmm.device) % valid_num[:, None]

    #     # put the individual sampled index to the index in the batch
    #     sample1 = (sample1 + batch_base[:, None]).view(-1)
    #     sample2 = (sample2 + batch_base[:, None]).view(-1)
    #     sample3 = (sample3 + batch_base[:, None]).view(-1)

    #     pigan_reld12 = depth_pigan[sample2] - depth_pigan[sample1]
    #     pigan_reld13 = depth_pigan[sample3] - depth_pigan[sample1]
    #     proportional_reld = pigan_reld13 / pigan_reld12

    #     mm_reld12 = depth_pigan[sample2] - depth_pigan[sample1]
    #     mm_reld13 = depth_pigan[sample3] - depth_pigan[sample1]
    #     proportional_mm = mm_reld13 / mm_reld12

    #     data1 = torch.masked_select(
    #             depth_pigan[sample1] - depth_pigan[sample2],
    #             depth_3dmm[sample1] > (depth_3dmm[sample2] + self.epsilon)
    #         )
    #     loss1 = nnF.relu(-data1)

    #     # 3dmm[s1] < 3dmm[s2] while pigan[s1] > pigan[s2]
    #     data2 = torch.masked_select(
    #             depth_pigan[sample1] - depth_pigan[sample2],
    #             depth_3dmm[sample1] < (depth_3dmm[sample2] - self.epsilon)
    #         )
    #     loss2 = nnF.relu(data2)

    #     return (loss1.mean() + loss2.mean()) / 2


    def forward(self, input):
        # if self.using_proportional_reld:
        # else:
        return self.pair_ranking_loss(input)