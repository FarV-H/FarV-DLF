import torch
import torch.nn as nn
from torchvision.models import vgg19

from . import contextual_loss as cl

import pdb

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.loss_weight    = loss_weight
        self.L1             = nn.L1Loss()

    def forward(self, pred, target):
        return self.loss_weight * self.L1(pred, target)

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight    = loss_weight
        self.MSE            = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss_weight * self.MSE(pred, target)

class VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

        #print( self.feature_extractor )
    def forward(self, x):                   # torch.Size([4, 3, 64, 64])
        out = self.feature_extractor(x)     # torch.Size([4, 512, 4, 4])
        
        return out

class ContextualLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(ContextualLoss, self).__init__()
        self.loss_weight            = loss_weight
        self.feature_extractor      = VGG_FeatureExtractor()
        self.criterion              = cl.ContextualLoss()
        
    def forward(self, pred, target):
        # contextual loss
        # input features
        num_z                       = 1
        if pred.ndim == 5:
            pred                    = pred.permute([0, 1, 4, 2, 3]).reshape([pred.shape[0], pred.shape[4], pred.shape[2], pred.shape[3]])
            target                  = target.permute([0, 1, 4, 2, 3]).reshape([target.shape[0], target.shape[4], target.shape[2], target.shape[3]])
            num_z                   = pred.shape[1]

        loss                        = 0
        for i in range(num_z):
            pred_T                  = pred[:, i, :, :].repeat(1, 3, 1, 1)
            target_T                = target[:, i, :, :].repeat(1, 3, 1, 1)
            pred_feature            = self.feature_extractor(pred_T)
            target_feature          = self.feature_extractor(target_T)
            loss                    += self.criterion(pred_feature, target_feature) / num_z

        return self.loss_weight * loss

#  loss_cfg = [MSE, MAE, VGG, SSIM, contextual_loss]
class LossFunc(nn.Module):
    def __init__(self):#, cfg=None):
        super(LossFunc, self).__init__()
        self.L1Loss         = L1Loss()
        self.ContextLoss    = ContextualLoss(loss_weight=0.01)

    def forward(self, pred, target):
        l_L1                = self.L1Loss(pred, target)
        l_Context           = self.ContextLoss(pred, target)
        loss                = l_L1 + l_Context
        return loss