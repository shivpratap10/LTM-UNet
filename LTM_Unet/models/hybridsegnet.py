import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from vit_encoder import ViTEncoder
from attention import SkipFusion
from decoder import MambaDecoderBlock
from decoder import MambaDecoderBlockV2


# =========================================================
# Helper blocks
# =========================================================
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


class TokenToMap(nn.Module):
    def __init__(self, img_size=256, patch_size=16):
        super().__init__()
        self.H = img_size // patch_size
        self.W = img_size // patch_size

    def forward(self, x):
        """
        x: [B, N, D]
        returns: [B, D, H, W]
        """
        B, N, D = x.shape
        expected_tokens = self.H * self.W

        if N != expected_tokens:
            raise ValueError(
                f"Token count mismatch: got N={N}, expected {expected_tokens} "
                f"for feature map size ({self.H}, {self.W})"
            )

        return x.transpose(1, 2).reshape(B, D, self.H, self.W)


# =========================================================
# HybridSegNetStable
# =========================================================
class HybridSegNetStable(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        embed_dim=320,
        out_channels=1,
        logit_scale=8.0,
        dropout=0.1,
        drop_path=0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.logit_scale = logit_scale

        # -----------------------------------------------------
        # Encoder
        # -----------------------------------------------------
        self.encoder = ViTEncoder()
        self.token2map = TokenToMap(img_size=img_size, patch_size=patch_size)

        # -----------------------------------------------------
        # Channel reduction
        # Encoder feature maps: [B, embed_dim, Ht, Wt]
        # Reduced pyramid channels
        # -----------------------------------------------------
        self.reduce1 = nn.Sequential(
            nn.Conv2d(embed_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.reduce2 = nn.Sequential(
            nn.Conv2d(embed_dim, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.reduce3 = nn.Sequential(
            nn.Conv2d(embed_dim, 160, kernel_size=1, bias=False),
            nn.BatchNorm2d(160),
            nn.GELU()
        )
        self.reduce4 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.reduce5 = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # -----------------------------------------------------
        # Decoder: MambaDecoderBlockV2 end-to-end
        # -----------------------------------------------------
        self.dec4 = MambaDecoderBlockV2(
            dec_channels=256,
            skip_channels=256,
            out_channels=256,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=2.0,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.dec3 = MambaDecoderBlockV2(
            dec_channels=256,
            skip_channels=160,
            out_channels=160,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=2.0,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.dec2 = MambaDecoderBlockV2(
            dec_channels=160,
            skip_channels=96,
            out_channels=96,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=2.0,
            dropout=dropout,
            drop_path=drop_path,
        )

        self.dec1 = MambaDecoderBlockV2(
            dec_channels=96,
            skip_channels=64,
            out_channels=64,
            d_state=16,
            d_conv=4,
            expand=2,
            mlp_ratio=2.0,
            dropout=dropout,
            drop_path=drop_path,
        )

        # -----------------------------------------------------
        # Final refinement head
        # -----------------------------------------------------
        self.refine = nn.Sequential(
            ConvBNAct(64, 64, k=3, p=1, dropout=dropout),
            ConvBNAct(64, 32, k=3, p=1, dropout=dropout),
        )

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

        # Stabilized init
        nn.init.zeros_(self.final.weight)
        if self.final.bias is not None:
            nn.init.constant_(self.final.bias, -1.0)

    def forward(self, x):
        H, W = x.shape[2:]

        # -----------------------------------------------------
        # Encoder tokens
        # -----------------------------------------------------
        f1, f2, f3, f4, f5 = self.encoder(x)

        # -----------------------------------------------------
        # Token -> map
        # -----------------------------------------------------
        f1 = self.token2map(f1)
        f2 = self.token2map(f2)
        f3 = self.token2map(f3)
        f4 = self.token2map(f4)
        f5 = self.token2map(f5)

        # -----------------------------------------------------
        # Channel reduction
        # -----------------------------------------------------
        f1 = self.reduce1(f1)   # [B,  64, Ht, Wt]
        f2 = self.reduce2(f2)   # [B,  96, Ht, Wt]
        f3 = self.reduce3(f3)   # [B, 160, Ht, Wt]
        f4 = self.reduce4(f4)   # [B, 256, Ht, Wt]
        f5 = self.reduce5(f5)   # [B, 256, Ht, Wt]

        # -----------------------------------------------------
        # Decoder
        # -----------------------------------------------------
        d4 = self.dec4(f5, f4)   # -> 256
        d3 = self.dec3(d4, f3)   # -> 160
        d2 = self.dec2(d3, f2)   # -> 96
        d1 = self.dec1(d2, f1)   # -> 64

        # -----------------------------------------------------
        # Refinement + output
        # -----------------------------------------------------
        out = self.refine(d1)
        out = self.final(out)

        # bound logits to prevent explosion
        out = self.logit_scale * torch.tanh(out)

        # restore full image resolution
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out


# =========================================================
# Test
# =========================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HybridSegNetStable(
        img_size=224,
        patch_size=16,
        embed_dim=320,   # set to your actual ViTEncoder output dim
        out_channels=1,
        logit_scale=8.0,
        dropout=0.1,
        drop_path=0.1
    ).to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Pred range  :", y.min().item(), y.max().item())
