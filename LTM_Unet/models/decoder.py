
import torch
import torch.nn as nn
import torch.nn.functional as F
from vssm import VisionMambaBlock
from einops import rearrange
# -*- coding: utf-8 -*-


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
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# =========================================================
# Selective SSM
# =========================================================
class SelectiveSSM(nn.Module):
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

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = CausalConv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            bias=True,
        )
        self.act = nn.SiLU()

        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(self.d_inner))

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
        bsz, seq_len, d_inner = u.shape
        d_state = A.shape[-1]

        delta = delta.unsqueeze(-1)                      # [B, L, D, 1]
        A = A.view(1, 1, d_inner, d_state)              # [1, 1, D, N]
        B = B.unsqueeze(2)                              # [B, L, 1, N]
        C = C.unsqueeze(2)                              # [B, L, 1, N]

        deltaA = torch.exp(torch.clamp(delta * A, min=-8.0, max=8.0))
        deltaB_u = delta * B * u.unsqueeze(-1)

        state = torch.zeros(bsz, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(seq_len):
            state = deltaA[:, t] * state + deltaB_u[:, t]
            y_t = torch.einsum("bdn,bdn->bd", state, C[:, t])
            y_t = y_t + D * u[:, t]
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)

    def forward(self, x):
        # x: [B, L, D]
        xz = self.in_proj(x)
        x_part, z = xz.chunk(2, dim=-1)

        x_part = rearrange(x_part, "b l d -> b d l")
        x_part = self.conv1d(x_part)
        x_part = rearrange(x_part, "b d l -> b l d")
        x_part = self.act(x_part)

        dt = self.dt_proj(x_part)
        dt = torch.sigmoid(dt)
        dt = self.dt_min + (self.dt_max - self.dt_min) * dt
        dt = torch.clamp(dt, self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log.float()).to(x_part.dtype)
        B = self.B_proj(x_part)
        C = self.C_proj(x_part)

        y = self.selective_scan(x_part, dt, A, B, C, self.D.to(x_part.dtype))
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


# =========================================================
# Directional Scan
# =========================================================
class DirectionalScan(nn.Module):
    def __init__(self, ssm_layer: SelectiveSSM, d_model: int, dropout=0.0):
        super().__init__()
        self.ssm_layer = ssm_layer
        self.fuse = nn.Linear(d_model * 4, d_model, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, h, w):
        b, l, d = x.shape
        assert l == h * w, f"Sequence length {l} does not match h*w={h*w}"

        x_hf = rearrange(x, "b (h w) d -> (b w) h d", h=h, w=w)
        y_hf = self.ssm_layer(x_hf)
        y_hf = rearrange(y_hf, "(b w) h d -> b (h w) d", b=b, h=h, w=w)

        x_hb = rearrange(x, "b (h w) d -> (b w) h d", h=h, w=w)
        x_hb = torch.flip(x_hb, dims=[1])
        y_hb = self.ssm_layer(x_hb)
        y_hb = torch.flip(y_hb, dims=[1])
        y_hb = rearrange(y_hb, "(b w) h d -> b (h w) d", b=b, h=h, w=w)

        x_vf = rearrange(x, "b (h w) d -> (b h) w d", h=h, w=w)
        y_vf = self.ssm_layer(x_vf)
        y_vf = rearrange(y_vf, "(b h) w d -> b (h w) d", b=b, h=h, w=w)

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
# Vision Mamba
# =========================================================
class VisionMambaBlock(nn.Module):
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
        x = x + self.drop_path1(self.scan(self.norm1(x), h, w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class VisionMambaBlock2D(nn.Module):
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
        return rearrange(x_seq, "b (h w) c -> b c h w", h=h, w=w)


# =========================================================
# Skip Fusion
# =========================================================
class SkipFusion(nn.Module):
    """
    Lightweight gated skip fusion.
    """
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, skip, dec):
        if dec.shape[-2:] != skip.shape[-2:]:
            dec = F.interpolate(dec, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        g = self.gate(torch.cat([skip, dec], dim=1))
        x = g * skip + (1.0 - g) * dec
        return self.out(x)


# =========================================================
# Decoder-ready block
# =========================================================
class MambaDecoderBlock(nn.Module):
    """
    Decoder-ready block for HybridSegNet.

    Inputs:
        x    : decoder feature [B, Cx, Hx, Wx]
        skip : encoder skip    [B, Cs, Hs, Ws]

    Steps:
        1. Upsample x to skip size
        2. Concatenate x and skip
        3. 1x1 fusion to out_channels
        4. Local conv refinement
        5. Vision Mamba refinement
        6. Gated skip fusion
        7. Final local refinement
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=2.0,
        dropout=0.1,
        drop_path=0.0,
        use_skip_fusion=True,
    ):
        super().__init__()

        self.use_skip_fusion = use_skip_fusion

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.local_refine1 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
        )

        self.mamba_refine = VisionMambaBlock2D(
            d_model=out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

        if self.use_skip_fusion:
            self.skip_fusion = SkipFusion(out_channels, dropout=dropout)

        self.local_refine2 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
        )

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.local_refine1(x)
        x = self.mamba_refine(x)

        if self.use_skip_fusion:
            x = self.skip_fusion(skip=x, dec=x)  # placeholder-safe path if no external skip wanted

        x = self.local_refine2(x)
        return x


# =========================================================
# Better decoder block with explicit projected skip fusion
# =========================================================
class MambaDecoderBlockV2(nn.Module):
    """
    Recommended version.

    This version explicitly projects skip to out_channels and fuses it again after Mamba refinement.
    """
    def __init__(
        self,
        dec_channels,
        skip_channels,
        out_channels,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=2.0,
        dropout=0.1,
        drop_path=0.0,
    ):
        super().__init__()

        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.input_fuse = nn.Sequential(
            nn.Conv2d(dec_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.local_refine1 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
        )

        self.mamba_refine = VisionMambaBlock2D(
            d_model=out_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.skip_fusion = SkipFusion(out_channels, dropout=dropout)

        self.local_refine2 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
            ConvBNAct(out_channels, out_channels, k=3, p=1, dropout=dropout),
        )

    def forward(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip_proj = self.skip_proj(skip)

        x = torch.cat([x, skip], dim=1)
        x = self.input_fuse(x)
        x = self.local_refine1(x)
        x = self.mamba_refine(x)
        x = self.skip_fusion(skip_proj, x)
        x = self.local_refine2(x)

        return x


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    block = MambaDecoderBlockV2(
        dec_channels=256,
        skip_channels=160,
        out_channels=160,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=2.0,
        dropout=0.1,
        drop_path=0.1,
    ).to(device)

    x = torch.randn(2, 256, 16, 16).to(device)
    skip = torch.randn(2, 160, 32, 32).to(device)

    y = block(x, skip)
    print("Decoder input :", x.shape)
    print("Skip input    :", skip.shape)
    print("Output        :", y.shape)
