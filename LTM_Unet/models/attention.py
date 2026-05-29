# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:22:59 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()

        hidden_channels = max(channels // reduction_ratio, 1)

        # Improved channel attention: shared MLP on both avg and max pooled features
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        )

        self.channel_gate = nn.Sigmoid()

        # Improved spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        # Lightweight refinement block (keeps same shape)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        identity = x

        # -------------------------
        # Channel Attention
        # -------------------------
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)

        ca_avg = self.mlp(avg_pool)
        ca_max = self.mlp(max_pool)

        ca = self.channel_gate(ca_avg + ca_max)
        x = x * ca

        # -------------------------
        # Spatial Attention
        # -------------------------
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)

        sa = torch.cat([avg_map, max_map], dim=1)
        sa = self.spatial_conv(sa)

        x = x * sa

        # -------------------------
        # Residual refinement
        # -------------------------
        x = self.refine(x) + identity

        return x


class SkipFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = SpatialChannelAttention(channels)

    def forward(self, enc, dec):
        enc = self.attn(enc)
        return enc + dec