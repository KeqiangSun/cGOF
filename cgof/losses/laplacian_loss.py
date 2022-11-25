import torch
import torch.nn as nn
from torch.nn import L1Loss
from torch.nn.modules.loss import _Loss

from losses import functional as F
# import functional as F
import numpy as np
import scipy.sparse as sp

class ConvOp(nn.Conv2d):
    def __init__(self, mode, in_channels=1, padding=1, padding_mode='zeros', bias=True):
        
        kernel_size = 3
        out_channels = 1
        super(ConvOp, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, padding=padding,
                                padding_mode=padding_mode, bias=bias)
        
        if mode == 'laplacian_4':
            weight = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        elif mode == 'laplacian_8':
            weight = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        elif mode == 'sobel_x':
            weight = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        elif mode == 'sobel_y':
            weight = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        else:
            raise ValueError(f'Unknown Operation mode. mode should be among: \
                                laplacian_4, laplacian_8, sobel_x, sobel_y')
        
        weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0). \
                    expand(-1, in_channels, -1, -1).float().to(self.weight.device)
        # from IPython import embed; embed()
        with torch.no_grad():
            self.weight.copy_(weight)
        self.weight.requires_grad_(False)
        
class LaplacianLoss(_Loss):
    def __init__(self, mode='laplacian_4', loss=L1Loss, reduction="mean"):
        super(LaplacianLoss, self).__init__(reduction=reduction)
        self.loss = loss()
        self.target = torch.FloatTensor(1).fill_(0)
        self.filter = ConvOp(mode=mode, padding_mode='replicate', bias=False)
        
    def forward(self, input, target=None):
        if target is None:
            target = self.target
            target = target.expand(input.shape)
            # print(target)
        target = target.to(input.device)
        lap = self.filter(input)
        # print(laplacian_output)
        return self.loss(lap, target)

class EdgeAwareDepthSmoothLoss(_Loss):
    def __init__(self, reduction="mean"):
        super(EdgeAwareDepthSmoothLoss, self).__init__(reduction=reduction)
        self.sobel_img_x = ConvOp(mode='sobel_x', in_channels=3,
                                  padding_mode='replicate', bias=False)
        self.sobel_img_y = ConvOp(mode='sobel_y', in_channels=3,
                                  padding_mode='replicate', bias=False)
        self.sobel_dep_x = ConvOp(mode='sobel_x', in_channels=1,
                                  padding_mode='replicate', bias=False)
        self.sobel_dep_y = ConvOp(mode='sobel_y', in_channels=1,
                                  padding_mode='replicate', bias=False)
    
    def l1_norm(self, x):
        return torch.abs(x).mean()
        
    def forward(self, img, depth):
        dimg_dx = self.l1_norm(self.sobel_img_x(img))
        dimg_dy = self.l1_norm(self.sobel_img_y(img))
        ddep_dx = self.l1_norm(self.sobel_dep_x(depth))
        ddep_dy = self.l1_norm(self.sobel_dep_y(depth))
        
        return ddep_dx * torch.exp(-dimg_dx) + ddep_dy * torch.exp(-dimg_dy)
    

def main():
    bs = 3
    img_channels = 3
    dep_channels = 1
    size = 5
    img = torch.rand(bs, img_channels, size, size)
    dep = torch.ones(bs, dep_channels, size, size)
    
    # test ConvOp
    conv = ConvOp('laplacian_8', in_channels=img_channels,
                  padding_mode='replicate', bias=False)
    output = conv(img)
    print(f"img conv output: {output}")
    
    conv = ConvOp('laplacian_8', in_channels=dep_channels,
                  padding_mode='replicate', bias=False)
    output = conv(dep)
    print(f"dep conv output: {output}")
    
    # test LaplacianLoss
    lap = LaplacianLoss()
    lap_loss = lap(dep)
    print(f"lap_loss: {lap_loss}")
    
    # test EdgeAwareDepthSmoothLoss
    edsmooth = EdgeAwareDepthSmoothLoss()
    edsmooth_loss = edsmooth(img, dep)
    print(f"edsmooth_loss: {edsmooth_loss}")
    

if __name__ == "__main__":
    main()
    