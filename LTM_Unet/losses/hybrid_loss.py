# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:56:33 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        denom = pred.sum() + target.sum()

        dice = (2. * intersection + 1e-6) / (denom + 1e-6)

        return 1 - dice


class StableBoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

        # Predefine Sobel kernels (no recreation every forward)
        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],
                                    dtype=torch.float32).view(1,1,3,3)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],
                                    dtype=torch.float32).view(1,1,3,3)

    def forward(self, pred, target):
        device = pred.device

        sobel_x = self.sobel_x.to(device)
        sobel_y = self.sobel_y.to(device)

      
        pred_x = F.conv2d(pred, sobel_x, padding=1)
        pred_y = F.conv2d(pred, sobel_y, padding=1)

        tgt_x = F.conv2d(target, sobel_x, padding=1)
        tgt_y = F.conv2d(target, sobel_y, padding=1)

       
        pred_edge = torch.abs(pred_x) + torch.abs(pred_y)
        tgt_edge = torch.abs(tgt_x) + torch.abs(tgt_y)

        # Normalize (IMPORTANT)
        pred_edge = pred_edge / (pred_edge.max() + 1e-6)
        tgt_edge = tgt_edge / (tgt_edge.max() + 1e-6)

        return F.l1_loss(pred_edge, tgt_edge)


class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = StableBoundaryLoss()

        # REMOVE pos_weight here
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, epoch=None):

        device = pred.device

      
        pos_weight = torch.tensor([5.0], device=device)

        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, 1e-4, 1-1e-4)

        if epoch is not None:
            boundary_weight = min(0.2, epoch / 20 * 0.2)
        else:
            boundary_weight = 0.1

        loss = (
            0.5 * self.dice(pred_sigmoid, target) +
            0.4 * bce_loss(pred, target) +
            boundary_weight * self.boundary(pred_sigmoid, target)
        )

        return loss