"""Vim-Tiny 冻结骨干：直接按 checkpoint 键名构建网络并加载 ImageNet-1K 预训练权重。

checkpoint 结构（hustvl/Vim-tiny-midclstok）：
    patch_embed.proj.{weight,bias}   # 16×16 conv, stride 16 → 14×14 token, dim 192
    cls_token (1,1,192), pos_embed (1,730,192)   # 729=27²+1，运行时插值到实际 token 数
    layers.{0..23}.norm.weight        # RMSNorm（无 bias）
    layers.{i}.mixer.{in_proj,conv1d,x_proj,dt_proj,A_log,D,
                      x_proj_b,dt_proj_b,A_b_log,D_b}
    norm_f.weight, head.{weight,bias}

mixer 的双向参数（*_b）与 mamba_ssm.mamba_simple.Mamba(bimamba_type="v") 的键名一一对应，
故直接复用该实现，无需 vendor Vim 源码。

多尺度特征（论文 2.x：28×28 / 14×14 / 7×7 三个网格，通道逐级增大）：
    F2 = 末层归一化 patch token 重排为 14×14×192（天然尺度）
    F1 = F2 经 PixelUnshuffle(2)     → 28×28×48
    F3 = F2 经 空间到深度(2)          → 7×7×768
两者为确定性重排，无可学参数，与冻结骨干策略一致。
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba

IMAGE_SIZE = 224
PATCH = 16
GRID = IMAGE_SIZE // PATCH  # 14
TOKENS = GRID * GRID  # 196
DIM = 192
DEPTH = 24


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype)).to(dtype)


def _interpolate_pos_embed(pos: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """把 pos_embed[:, 1:]（729 token）双线性插值到实际 token 数（Vim 官方做法）。"""
    if pos.shape[1] == n_tokens:
        return pos
    grid = int(pos.shape[1] ** 0.5)
    m = pos.reshape(1, grid, grid, -1).permute(0, 3, 1, 2)
    m = F.interpolate(m, size=GRID, mode="bicubic", align_corners=False)
    return m.flatten(2).permute(0, 2, 1)


class VimTiny(nn.Module):
    """双向 Mamba 视觉骨干。forward 返回归一化 patch token (B, 196, 192)。"""

    def __init__(self, drop_path_rate: float = 0.0):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, DIM, kernel_size=PATCH, stride=PATCH)
        self.pos_embed = nn.Parameter(torch.zeros(1, 27 * 27 + 1, DIM))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(DEPTH):
            self.norms.append(RMSNorm(DIM))
            self.blocks.append(Mamba(d_model=DIM, d_state=16, d_conv=4, expand=2,
                                     bimamba_type="v"))
        self.norm_f = RMSNorm(DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B,196,192)
        tokens = tokens + _interpolate_pos_embed(self.pos_embed[:, 1:], TOKENS)
        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1]
        mid = TOKENS // 2
        x = torch.cat([tokens[:, :mid], cls, tokens[:, mid:]], dim=1)  # midclstok
        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))
        x = self.norm_f(x)
        out = torch.cat([x[:, :mid], x[:, mid + 1:]], dim=1)  # 去掉 cls，仅保留 patch token
        return out

    @torch.no_grad()
    def load_pretrained(self, ckpt_path: str) -> None:
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = raw.get("model", raw) if isinstance(raw, dict) else raw
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        allowed_missing = {"head.weight", "head.bias"}
        bad = [k for k in missing if k not in allowed_missing]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint 加载异常 missing={bad} unexpected={unexpected}")


class MultiScaleFeatures(NamedTuple):
    f1: torch.Tensor  # (B, 28*28, 48)   细
    f2: torch.Tensor  # (B, 14*14, 192)  中
    f3: torch.Tensor  # (B, 7*7, 768)    粗


def to_multiscale(tokens: torch.Tensor) -> MultiScaleFeatures:
    """(B,196,192) → 三尺度序列特征。

    F1: pixel_shuffle(2)      (192,14,14) → (48,28,28)   上采样空间、减通道，保细节
    F3: pixel_unshuffle(2)    (192,14,14) → (768,7,7)    降采样空间、增通道，聚语义
    """
    b = tokens.shape[0]
    m = tokens.reshape(b, GRID, GRID, DIM).permute(0, 3, 1, 2)  # B,192,14,14
    f1 = F.pixel_shuffle(m, 2)  # B,48,28,28
    f3 = F.pixel_unshuffle(m, 2)  # B,768,7,7
    return MultiScaleFeatures(
        f1=f1.flatten(2).transpose(1, 2).contiguous(),
        f2=tokens.contiguous(),
        f3=f3.flatten(2).transpose(1, 2).contiguous(),
    )


class FrozenVimBackbone(nn.Module):
    """冻结 Vim-Tiny + 多尺度特征抽取。"""

    def __init__(self, ckpt_path: str):
        super().__init__()
        self.vim = VimTiny()
        self.vim.load_pretrained(ckpt_path)
        self.vim.eval()
        for p in self.vim.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        # 骨干永远处于 eval（冻结 BN/Dropout 语义），训练切换只影响其它模块
        super().train(mode)
        self.vim.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> MultiScaleFeatures:
        return to_multiscale(self.vim(x))


def forward_images(backbone, x: torch.Tensor) -> dict:
    """(…,3,H,W) 任意批量前导维 → 冻结骨干 → {尺度名: (…,L,C)}。

    nn.Conv2d 只接受 4D 输入，故先展平前导维再前向、后还原（修复训练/评测
    直接传 (b,n,K,3,H,W) 6D 张量的崩溃）。"""
    lead = tuple(x.shape[:-3])
    feats = backbone(x.reshape(-1, *x.shape[-3:]))._asdict()
    return {k: v.reshape(*lead, v.shape[-2], v.shape[-1]) for k, v in feats.items()}
