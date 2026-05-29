# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:23:41 2026

@author: Santosh Prakash
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =========================================================
# Utilities
# =========================================================
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        # x: [B, C, L]
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


# =========================================================
# Selective SSM
# =========================================================
class SelectiveSSM(nn.Module):
    """
    Sequence module:
        input  : [B, L, D]
        output : [B, L, D]

    This is a self-contained selective state-space module inspired by Mamba-style parameterization,
    but implemented directly in PyTorch.
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
        dt_min=1e-4,
        dt_max=1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max

        # input projection -> x and gate z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # depthwise causal conv on sequence
        self.conv1d = CausalConv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )

        self.act = nn.SiLU()

        # continuous-time parameters
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # selective parameters
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.xavier_uniform_(self.B_proj.weight)
        nn.init.xavier_uniform_(self.C_proj.weight)
        nn.init.ones_(self.D)

    def selective_scan(self, u, delta, A, B, C, D):
        """
        u     : [B, L, D_in]
        delta : [B, L, D_in]
        A     : [D_in, N]
        B     : [B, L, N]
        C     : [B, L, N]
        D     : [D_in]

        returns:
            y: [B, L, D_in]
        """
        Bsz, L, D_in = u.shape
        N = A.shape[-1]

        # shapes for broadcasting
        delta = delta.unsqueeze(-1)                  # [B, L, D, 1]
        A = A.view(1, 1, D_in, N)                    # [1,1,D,N]
        B = B.unsqueeze(2)                           # [B,L,1,N]
        C = C.unsqueeze(2)                           # [B,L,1,N]

        # stabilize exponent
        deltaA = torch.exp(torch.clamp(delta * A, min=-8.0, max=8.0))   # [B,L,D,N]
        deltaB_u = delta * B * u.unsqueeze(-1)                           # [B,L,D,N]

        state = torch.zeros(Bsz, D_in, N, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(L):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = torch.einsum("bdn,bdn->bd", state, C[:, t])
            y_t = y_t + D * u[:, t]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)   # [B,L,D]
        return y

    def forward_once(self, x):
        """
        x: [B, L, D]
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)   # [B,L,d_inner], [B,L,d_inner]

        # depthwise causal conv over sequence
        x_part = rearrange(x_part, "b l d -> b d l")
        x_part = self.conv1d(x_part)
        x_part = rearrange(x_part, "b d l -> b l d")
        x_part = self.act(x_part)

        # discretization / selective params
        dt = self.dt_proj(x_part)
        dt = torch.sigmoid(dt)
        dt = self.dt_min + (self.dt_max - self.dt_min) * dt
        dt = torch.clamp(dt, self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log.float()).to(x_part.dtype)   # [d_inner, d_state]
        Bp = self.B_proj(x_part)                              # [B,L,d_state]
        Cp = self.C_proj(x_part)                              # [B,L,d_state]

        y = self.selective_scan(x_part, dt, A, Bp, Cp, self.D.to(x_part.dtype))

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y

    def forward(self, x):
        return self.forward_once(x)


# =========================================================
# Directional 2D scanning
# =========================================================
class DirectionalScan(nn.Module):
    """
    Applies the same SelectiveSSM over:
    - horizontal forward
    - horizontal backward
    - vertical forward
    - vertical backward

    Input:
        x: [B, L, D]
        h, w where L = h*w
    Output:
        [B, L, D]
    """
    def __init__(self, ssm_layer: SelectiveSSM, d_model: int, dropout=0.0):
        super().__init__()
        self.ssm_layer = ssm_layer
        self.fuse = nn.Linear(d_model * 4, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, h, w):
        b, l, d = x.shape
        assert l == h * w, f"Sequence length {l} does not match h*w = {h*w}"

        # horizontal forward
        x_hf = rearrange(x, "b (h w) d -> (b w) h d", h=h, w=w)
        y_hf = self.ssm_layer(x_hf)
        y_hf = rearrange(y_hf, "(b w) h d -> b (h w) d", b=b, h=h, w=w)

        # horizontal backward
        x_hb = rearrange(x, "b (h w) d -> (b w) h d", h=h, w=w)
        x_hb = torch.flip(x_hb, dims=[1])
        y_hb = self.ssm_layer(x_hb)
        y_hb = torch.flip(y_hb, dims=[1])
        y_hb = rearrange(y_hb, "(b w) h d -> b (h w) d", b=b, h=h, w=w)

        # vertical forward
        x_vf = rearrange(x, "b (h w) d -> (b h) w d", h=h, w=w)
        y_vf = self.ssm_layer(x_vf)
        y_vf = rearrange(y_vf, "(b h) w d -> b (h w) d", b=b, h=h, w=w)

        # vertical backward
        x_vb = rearrange(x, "b (h w) d -> (b h) w d", h=h, w=w)
        x_vb = torch.flip(x_vb, dims=[1])
        y_vb = self.ssm_layer(x_vb)
        y_vb = torch.flip(y_vb, dims=[1])
        y_vb = rearrange(y_vb, "(b h) w d -> b (h w) d", b=b, h=h, w=w)

        y = torch.cat([y_hf, y_hb, y_vf, y_vb], dim=-1)
        y = self.fuse(y)
        y = self.dropout(y)
        return y


# =========================================================
# Vision Mamba block
# =========================================================
class VisionMambaBlock(nn.Module):
    """
    Full vision block with:
    - pre-norm
    - directional SSM scan
    - residual connection
    - MLP branch
    - droppath

    Input:
        x: [B, L, D]
        h, w: spatial size
    Output:
        [B, L, D]
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        self.scan = DirectionalScan(self.ssm, d_model=d_model, dropout=dropout)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, h, w):
        """
        x: [B, L, D]
        """
        x = x + self.drop_path1(self.scan(self.norm1(x), h, w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


# =========================================================
# Optional 2D wrapper
# =========================================================
class VisionMambaBlock2D(nn.Module):
    """
    Convenience wrapper for feature maps.

    Input : [B, C, H, W]
    Output: [B, C, H, W]
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.block = VisionMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_seq = rearrange(x, "b c h w -> b (h w) c")
        x_seq = self.block(x_seq, h, w)
        x_out = rearrange(x_seq, "b (h w) c -> b c h w", h=h, w=w)
        return x_out


# =========================================================
# Sanity test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sequence version
    blk = VisionMambaBlock(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path=0.1,
    ).to(device)

    x = torch.randn(2, 16 * 16, 128).to(device)
    y = blk(x, 16, 16)
    print("Sequence input :", x.shape)
    print("Sequence output:", y.shape)

    # 2D wrapper version
    blk2d = VisionMambaBlock2D(
        d_model=128,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        dropout=0.1,
        drop_path=0.1,
    ).to(device)

    x2 = torch.randn(2, 128, 16, 16).to(device)
    y2 = blk2d(x2)
    print("2D input :", x2.shape)
    print("2D output:", y2.shape)