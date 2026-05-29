
import math
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None

class SS2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if selective_scan_fn is None:
            raise ImportError(
                "mamba_ssm is not installed or selective_scan_fn could not be imported. "
                "Install mamba_ssm first."
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        # input projection -> x and z branches
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # depthwise spatial conv
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 4 directional projections
        self.x_proj = (
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([p.weight for p in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([p.weight for p in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([p.bias for p in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = (dt_rank ** -0.5) * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n -> r n", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        returns 4 directional outputs
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4

        # horizontal and vertical flattening
        x_hwwh = torch.stack(
            [
                x.view(B, -1, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
            ],
            dim=1,
        ).view(B, 2, -1, L)

        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, 4, D, L]

        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(
            out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(
            inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3
        ).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        """
        x: [B, H, W, C]
        """
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()   # [B, C, H, W]
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        if self.dropout is not None:
            out = self.dropout(out)

        return out

class VisionMambaBlock(nn.Module):
    """
    A full residual Vision Mamba block using SS2D.
    Input/Output format: [B, C, H, W]
    """
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        mlp_ratio: float = 0.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.ssm = SS2D(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.use_mlp = mlp_ratio > 0
        if self.use_mlp:
            hidden_dim = int(dim * mlp_ratio)
            self.norm2 = norm_layer(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(drop),
            )

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        x_perm = x.permute(0, 2, 3, 1).contiguous()          # [B, H, W, C]
        x_perm = x_perm + self.drop_path(self.ssm(self.norm1(x_perm)))

        if self.use_mlp:
            x_perm = x_perm + self.drop_path(self.mlp(self.norm2(x_perm)))

        out = x_perm.permute(0, 3, 1, 2).contiguous()        # [B, C, H, W]
        return out

class MambaDecoderBlock(nn.Module):
    """
    Segmentation-friendly decoder block:
    - upsamples decoder feature if needed
    - concatenates with skip
    - fuses using conv
    - refines with Vision Mamba block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_ratio: float = 0.0,
    ):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.refine = VisionMambaBlock(
            dim=out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        """
        x   : decoder feature [B, C1, H1, W1]
        skip: skip feature    [B, C2, H2, W2]
        """
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.refine(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    block = VisionMambaBlock(
        dim=128,
        d_state=16,
        d_conv=3,
        expand=2,
        mlp_ratio=0.0,
        drop=0.0,
        drop_path=0.0,
    ).to(device)

    x = torch.randn(2, 128, 32, 32).to(device)
    y = block(x)
    print("VisionMambaBlock:", x.shape, "->", y.shape)

    dec = MambaDecoderBlock(
        in_channels=128 + 64,
        out_channels=128,
        d_state=16,
        d_conv=3,
        expand=2,
    ).to(device)

    x_dec = torch.randn(2, 128, 16, 16).to(device)
    x_skip = torch.randn(2, 64, 32, 32).to(device)
    y_dec = dec(x_dec, x_skip)
    print("MambaDecoderBlock:", y_dec.shape)
