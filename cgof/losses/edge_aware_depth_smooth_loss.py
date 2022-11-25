import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss

from losses import functional as F
# import functional as F
import numpy as np
import scipy.sparse as sp

class EdgeAwareDepthSmoothLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(EdgeAwareDepthSmoothLoss, self).__init__(reduction=reduction)
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        weight = torch.from_numpy(np.array([[1,0,-1],[2,0,-2],[1,0,-1]])).float()
        weight.requires_grad_(False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        
    def laplacian_filter(self, input):
        return self.conv(input)
    
    def forward(self, input, target=None):
        if target is None:
            target = self.target
            target = target.expand(input.shape)
            # print(target)
        target = target.to(input.device)
        laplacian_output = self.laplacian_filter(input)
        # print(laplacian_output)
        return self.loss(laplacian_output, target)

def main():
    import cv2
    dep = torch.from_numpy(cv2.imread('../debug_dep.png')[...,0]).unsqueeze(0).unsqueeze(0).float()
    # from IPython import embed; embed()
    lpls = LaplacianLoss()
    # dep = torch.ones(3,1,5,5)
    loss = lpls(dep)
    print(loss)
    
    

if __name__ == "__main__":
    main()
    