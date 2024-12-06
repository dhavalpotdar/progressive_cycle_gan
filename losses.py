import torch
import torch.nn as nn


class CycleLoss(nn.Module):
    def __init__(self):
        super(CycleLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, recon, real):
        return self.l1(recon, real)


class LSGANLoss(nn.Module):
    """Least Squares GAN Loss"""

    def __init__(self):
        super(LSGANLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, predictions, target_is_real):
        target = (
            torch.ones_like(predictions)
            if target_is_real
            else torch.zeros_like(predictions)
        )
        return self.mse(predictions, target)
