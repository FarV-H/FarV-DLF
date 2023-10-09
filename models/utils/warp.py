import torch
from torch import nn

class backWarp3D(nn.Module):
    def __init__(self, nx, ny, nz):
        super(backWarp3D, self).__init__()
        self.nx         = nx
        self.ny         = ny
        self.nz         = nz

    def forward(self, image, flow):
        xx, yy, zz      = torch.meshgrid(torch.arange(self.nx), torch.arange(self.ny), torch.arange(self.nz))
        xx              = xx.view(flow.size(0), 1, flow.size(2), flow.size(3), flow.size(4)).cuda() + flow[:, 0, :, :, :]
        yy              = yy.view(flow.size(0), 1, flow.size(2), flow.size(3), flow.size(4)).cuda() + flow[:, 1, :, :, :]
        zz              = zz.view(flow.size(0), 1, flow.size(2), flow.size(3), flow.size(4)).cuda() + flow[:, 2, :, :, :]
        xx              = 2 * (xx /  self.nx - 0.5)
        yy              = 2 * (yy /  self.ny - 0.5)
        zz              = 2 * (zz /  self.nz - 0.5)
        tp              = torch.cat([zz, yy, xx], dim=1)
        grid            = tp.permute(0, 2, 3, 4, 1).contiguous()
        return torch.nn.functional.grid_sample(image, grid, align_corners=True, mode='bilinear')