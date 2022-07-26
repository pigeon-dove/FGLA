import torch
import torch.nn as nn
from torchvision import models


def Resnet50(pool: bool):
    model = models.resnet50(pretrained=True)
    if not pool:
        model.avgpool = nn.Sequential()
        model.fc = nn.Linear(2048 * 7 * 7, 1000)
    return model


class TVLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return self.loss_weight * (dx + dy)


class L2Loss(nn.Module):
    def __init__(self, loss_weight=1):
        super(L2Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        return self.loss_weight * torch.norm(x, p=2)
