import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from losses import functional as F
import numpy as np
import scipy.sparse as sp


def gen_laplacian_matrix(num_points, top_k):
    """
    Return 106 x 106 matrix
    """
    if num_points == 98:
        connection = np.loadtxt(f'utils/face_graph/{num_points}pts_top{top_k}.txt')
    else:
        connection = np.loadtxt(f'utils/face_graph/{num_points}pts.txt')

    return adj_mx_from_list(connection, num_points)


def adj_mx_from_list(connection, num_points):
    edges = connection[:, :2]
    return adj_mx_from_weighted_edges(num_points, edges)


def adj_mx_from_weighted_edges(num_pts, edges):
    edges = np.array(edges, dtype=np.int32)
    weights = np.ones(edges.shape[0])
    data, i, j = weights, edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32).todense()

    np.fill_diagonal(adj_mx, 0)
    row_sum = adj_mx.sum(axis=1)
    row_sum[row_sum==0] = 1
    adj_mx /= row_sum
    np.fill_diagonal(adj_mx, -1)

    adj_mx = torch.tensor(adj_mx, dtype=torch.float)
    return adj_mx.unsqueeze(0)

        
class LaplacianLoss(_Loss):
    def __init__(self, loss, num_points, reduction="mean"):
        super(LaplacianLoss, self).__init__(reduction=reduction)
        self.loss = loss
        self.num_points = num_points
        self.laplacian_matrix = gen_laplacian_matrix(num_points, 3).cuda() # 106x106

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.num_points, 2)
        target = target.reshape(-1, self.num_points, 2)
        prediction = torch.matmul(self.laplacian_matrix, prediction)
        target = torch.matmul(self.laplacian_matrix, target)
        prediction = prediction.reshape(-1, self.num_points*2)
        target = target.reshape(-1, self.num_points*2)

        return self.loss(prediction, target)


class NMELoss(_Loss):
    """docstring for NMELoss"""
    def __init__(self, num_points, image_size, reduction="mean"):
        super(NMELoss, self).__init__(reduction=reduction)
        self.num_points = num_points
        self.image_size = image_size

    def forward(self, prediction, target):
        return F.nme_loss(prediction, target, 
                          num_points=self.num_points, 
                          image_size=self.image_size,
                          reduction=self.reduction)
  
        
class L2Loss(_Loss):
    """docstring for L2Loss"""
    def __init__(self, reduction="mean"):
        super(L2Loss, self).__init__(reduction=reduction)
    
    def forward(self, prediction, target):
        return F.l2_loss(prediction, target, self.reduction)


class EuclideanLoss(_Loss):
    """docstring for L2Loss"""
    def __init__(self, reduction="mean"):
        super(EuclideanLoss, self).__init__(reduction=reduction)
    
    def forward(self, prediction, target):
        return F.euclidean_loss(prediction, target, self.reduction)

        
class WingLoss(_Loss):
    def __init__(self, width=5, curvature=0.5, reduction="mean"):
        super(WingLoss, self).__init__(reduction=reduction)
        self.width = width
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wing_loss(
            prediction, target, self.width, self.curvature, self.reduction
        )

class SmoothWingLoss(_Loss):
    """docstring for SmoothWingLoss"""
    def __init__(self, thres1=2, thres2=20, curvature=2, reduction="mean"):
        super(SmoothWingLoss, self).__init__(reduction=reduction)
        self.thres1 = thres1
        self.thres2 = thres2
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.smooth_wing_loss(
            prediction, target, self.thres1, self.thres2, self.curvature, self.reduction
        )
    
        
class WiderWingLoss(_Loss):
    """docstring for WiderWingLoss"""
    def __init__(self, thres1=2, thres2=20, curvature=2, reduction="mean"):
        super(WiderWingLoss, self).__init__(reduction=reduction)
        self.thres1 = thres1
        self.thres2 = thres2
        self.curvature = curvature

    def forward(self, prediction, target):
        return F.wider_wing_loss(
            prediction, target, self.thres1, self.thres2, self.curvature, self.reduction
        )

class NormalizedWiderWingLoss(_Loss):
    """docstring for NormalizedWiderWingLoss"""
    def __init__(self, thres1=2, thres2=20, curvature=2, num_points=98, image_size=256, reduction="mean"):
        super(NormalizedWiderWingLoss, self).__init__(reduction=reduction)
        self.thres1 = thres1
        self.thres2 = thres2
        self.curvature = curvature

        self.num_points = num_points
        self.image_size = image_size

    def forward(self, prediction, target):
        return F.normalized_wider_wing_loss(
            prediction, target, self.num_points, self.image_size, self.thres1, self.thres2, self.curvature, self.reduction
        )