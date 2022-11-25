import math
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['nme_loss', 'l2_loss', 'euclidean_loss', 'wing_loss', 'smooth_wing_loss', 'wider_wing_loss', 'normalized_wider_wing_loss']


def l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    reduction="mean"
):
    """Calculate the average l2 loss for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        prediction (Tensor): Predictions (B x 2N)
        target (Tensor): Ground truth target (B x 2N)
    """
    assert prediction.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = prediction - target
    dist_sq = diff.pow(2)
    dist_sq = dist_sq.mean(-1, keepdim=False)
    dist = dist_sq.sqrt() # shape (B x L)

    if reduction == "sum":
        dist = dist.sum()

    if reduction == "mean":
        dist = dist.mean()

    return dist 


def euclidean_loss(
    prediction: torch.Tensor, 
    target: torch.Tensor,
    reduction="mean"
):
    """Calculate the average Euclidean loss for multi-point samples.
    Each sample must contain `n` points, each with `d` dimensions. For example,
    in the MPII human pose estimation task n=16 (16 joint locations) and
    d=2 (locations are 2D).
    Args:
        prediction (Tensor): Predictions (B x L x D)
        target (Tensor): Ground truth target (B x L x D)
    """

    assert prediction.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = prediction - target
    diff = diff.reshape(diff.size(0), -1, 2)
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt() # shape (B x L)

    if reduction == "sum":
        dist = dist.sum()

    if reduction == "mean":
        dist = dist.mean()

    return dist 


def nme_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_points,
    image_size=256,
    reduction="mean"
):
    """
    normalized_wider_wing = 
        1. C/d * |x|                           , if |x| < thres1
        2. C/d * (thres2*ln(1+|x|/curvature) + B), if |x| >= thres2
    """
    assert prediction.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = prediction - target
    diff = diff.reshape(diff.size(0), -1, 2)
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    loss = dist_sq.sqrt() # shape (B x N)

    if num_points == 29:
        le_idx = [8]
        re_idx = [9]

    if num_points == 68:
        le_idx = [36] # le = left eye
        re_idx = [45] # re = right eye

    if num_points == 98:
        le_idx = [60]
        re_idx = [72]

    target = target.reshape(-1, num_points, 2)
    le_loc = torch.mean(target[:,le_idx,:], dim=1) # batchsize x 2
    re_loc = torch.mean(target[:,re_idx,:], dim=1)  

    norm_dist = torch.sqrt(torch.sum((le_loc - re_loc)**2, dim=1))  # batchsize
    factor = image_size/norm_dist

    if reduction == "sum":
        loss = loss.sum(dim=1)
        loss *= factor
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean(dim=1)
        loss *= factor
        loss = loss.mean()

    return loss


def wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    width=5,
    curvature=0.5,
    reduction="mean",
):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param prediction:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - prediction).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss

def smooth_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thres1=2,
    thres2=20,
    curvature=2,
    reduction="mean"
):
    """
    smooth_wing = 
        1. |x|                           , if |x| < thres1
        2. thres2*ln(1+|x|/curvature) + B, if A <= |x| < thres2
        3. |x| - C                       , if |x| >= thres2
    """
    loss = (target - prediction).abs()

    idx_small = loss < thres1
    idx_normal = (loss >= thres1) * (loss < thres2)
    idx_big = loss >= thres2

    B = thres1 - thres2 * math.log(1 + thres1 / curvature)
    loss[idx_normal] = thres2 * torch.log(1 + loss[idx_normal] / curvature) + B

    C = thres2 - thres2 * math.log(1 + thres2 / curvature) - B
    loss[idx_big] = loss[idx_big] - C

    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss


def wider_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thres1=2,
    thres2=20,
    curvature=2,
    reduction="mean"
):
    """
    wider_wing = 
        1. |x|                           , if |x| < thres1
        2. thres2*ln(1+|x|/curvature) + B, if |x| >= thres2
    """
    loss = (target - prediction).abs()

    idx_small = loss < thres1
    idx_big = loss >= thres1

    B = thres1 - thres2 * math.log(1 + thres1 / curvature)
    loss[idx_big] = thres2 * torch.log(1 + loss[idx_big] / curvature) + B


    if reduction == "sum":
        loss = loss.sum()
    if reduction == "mean":
        loss = loss.mean()

    return loss


def euclidean_wider_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thres1=2,
    thres2=20,
    curvature=2,
    reduction="mean"
):
    assert prediction.size() == target.size(), 'input tensors must have the same size'

    # Calculate Euclidean distances between actual and target locations
    diff = prediction - target
    diff = diff.reshape(diff.size(0), -1, 2)
    dist_sq = diff.pow(2).sum(-1, keepdim=False)
    dist = dist_sq.sqrt() # shape (B x L)

    pass

    
def normalized_wider_wing_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_points,
    image_size=256,
    thres1=2,
    thres2=20,
    curvature=2,
    reduction="mean"
):
    """
    normalized_wider_wing = 
        1. C/d * |x|                           , if |x| < thres1
        2. C/d * (thres2*ln(1+|x|/curvature) + B), if |x| >= thres2
    """
    loss = (target - prediction).abs()

    idx_small = loss < thres1
    idx_big = loss >= thres1

    B = thres1 - thres2 * math.log(1 + thres1 / curvature)
    loss[idx_big] = thres2 * torch.log(1 + loss[idx_big] / curvature) + B

    if num_points == 29:
        le_idx = [8]
        re_idx = [9]

    if num_points == 68:
        le_idx = [36] # le = left eye
        re_idx = [45] # re = right eye

    if num_points == 98:
        le_idx = [60]
        re_idx = [72]

    target = target.reshape(-1, num_points, 2)
    le_loc = torch.mean(target[:,le_idx,:], dim=1) # batchsize x 2
    re_loc = torch.mean(target[:,re_idx,:], dim=1)  

    norm_dist = torch.sqrt(torch.sum((le_loc - re_loc)**2, dim=1))  # batchsize
    factor = image_size/norm_dist

    if reduction == "sum":
        loss = loss.sum(dim=1)
        loss *= factor
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean(dim=1)
        loss *= factor
        loss = loss.mean()


    return loss