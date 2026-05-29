
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand_tensor.floor_()
        return x.div(keep_prob) * rand_tensor


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=320):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Novelty: local positional enhancement with depthwise conv
        self.local_pos = nn.Conv2d(
            embed_dim, embed_dim,
            kernel_size=3, stride=1, padding=1,
            groups=embed_dim, bias=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                         # [B, C, H/P, W/P]
        x = x + self.local_pos(x)               # local positional enhancement
        x = x.flatten(2).transpose(1, 2)        # [B, N, C]
        x = self.norm(x)
        return x

class DepthwiseTokenMixing(nn.Module):
    """
    Converts token sequence [B, N, C] -> spatial -> DWConv -> tokens
    Keeps output shape unchanged.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1,
            groups=dim, bias=True
        )
        self.pwconv = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"N={N} is not a square number."

        feat = x.transpose(1, 2).reshape(B, C, H, W)
        feat = self.dwconv(feat)
        feat = self.pwconv(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return feat


class GEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x_proj, gate = self.fc1(x).chunk(2, dim=-1)
        x = x_proj * F.gelu(gate)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ExponentialLinearAttention(nn.Module):

    def __init__(self, dim, num_heads=8, eps=1e-6, use_token_mixing=True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.g_proj = nn.Linear(dim, dim)  # novelty: gating branch

        self.out_proj = nn.Linear(dim, dim)

        # learnable temperature per head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.use_token_mixing = use_token_mixing
        if use_token_mixing:
            self.token_mixer = DepthwiseTokenMixing(dim)

    def _reshape_heads(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [B, H, N, D]
        return x

    def _phi(self, x):
        x = x - x.amax(dim=-1, keepdim=True)
        return torch.exp(x)

    def forward(self, x):
        B, N, C = x.shape

        residual_tokens = x
        if self.use_token_mixing:
            x = x + self.token_mixer(x)

        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        g = self._reshape_heads(self.g_proj(x))

        q = self._phi(q * self.temperature)
        k = self._phi(k)

        # gated value
        v = v * torch.sigmoid(g)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)         # [B, H, D, D]
        k_sum = k.sum(dim=2)                              # [B, H, D]
        numerator = torch.einsum('bhnd,bhde->bhne', q, kv)
        denominator = torch.einsum('bhnd,bhd->bhn', q, k_sum).unsqueeze(-1) + self.eps

        out = numerator / denominator
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        out = self.out_proj(out)

        return out + 0.0 * residual_tokens  # keeps graph compatible, no shape change

class StageRefinement(nn.Module):
    """
    Lightweight refinement before stage output.
    Keeps feature shape [B, N, C].
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.norm(x)
        gated = self.proj(y) * self.gate(y)
        return x + gated

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
        layer_scale_init=1e-4
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ExponentialLinearAttention(dim, num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = GEGLU(dim, hidden_dim, dropout=dropout)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

        self.ls1 = LayerScale(dim, init_value=layer_scale_init)
        self.ls2 = LayerScale(dim, init_value=layer_scale_init)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=320,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path_rate=0.1
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.n_patches, embed_dim)
        )

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.depth = depth

        # Same 5-stage extraction logic
        self.stage_indices = [
            max(0, depth // 5 - 1),
            max(1, 2 * depth // 5 - 1),
            max(2, 3 * depth // 5 - 1),
            max(3, 4 * depth // 5 - 1),
            depth - 1
        ]

        self.stage_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(5)
        ])

        self.stage_refiners = nn.ModuleList([
            StageRefinement(embed_dim) for _ in range(5)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)         # [B, N, C]
        x = x + self.pos_embed          # [B, N, C]

        features = []
        stage_id = 0

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if i in self.stage_indices:
                feat = self.stage_norms[stage_id](x)
                feat = self.stage_refiners[stage_id](feat)
                features.append(feat)
                stage_id += 1

        if len(features) != 5:
            raise ValueError(f"Expected 5 features, got {len(features)}")

        return features[0], features[1], features[2], features[3], features[4]
