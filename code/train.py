"""
Main components
---------------
1. Generator:
   - 1D UNet-like convolutional encoder-decoder
   - Transformer bottleneck
   - skip connections
   - Tanh output

2. Discriminator / Critic:
   - 3 Conv1D layers with kernel sizes (65, 33, 17)
   - Transformer encoder with CLS token
   - scalar critic score output
   - optional self-supervised transformation classification head

3. Training:
   - WGAN-GP loss
   - gradient penalty
   - self-supervised pretraining support
   - GAN fine-tuning support

Expected tensor shapes
----------------------
PPG: (B, 1, T)
ECG: (B, 1, T)

"""

from __future__ import annotations

import os
import math
import copy
import random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================
# Configuration
# ============================================================

@dataclass
class ModelConfig:
    # signal
    sampling_rate: int = 130
    in_channels: int = 1
    out_channels: int = 1

    # UNet encoder/decoder depth
    num_unet_layers: int = 4
    base_channels: int = 8

    # transformer bottleneck in generator
    gen_transformer_dim: int = 16
    gen_transformer_layers: int = 2
    gen_transformer_heads: int = 2
    gen_transformer_mlp_ratio: int = 4
    gen_transformer_dropout: float = 0.1
    max_tokens: int = 4096

    # discriminator CNN encoder
    disc_conv_channels: Tuple[int, int, int] = (64, 128, 256)
    disc_conv_kernels: Tuple[int, int, int] = (65, 33, 17)
    disc_conv_stride: int = 1

    # discriminator transformer
    disc_model_dim: int = 256
    disc_transformer_layers: int = 2
    disc_transformer_heads: int = 2
    disc_ffn_dim: int = 512
    disc_dropout: float = 0.1
    disc_max_tokens: int = 8192

    # self-supervised pretraining head
    ssl_hidden_dim1: int = 128
    ssl_hidden_dim2: int = 128
    ssl_num_classes: int = 6  # six transformations

    # fine-tuning discriminator head
    finetune_hidden_dim1: int = 512
    finetune_hidden_dim2: int = 256

    # training
    batch_size: int = 128
    epochs_pretrain: int = 90
    epochs_gan: int = 90
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    betas: Tuple[float, float] = (0.5, 0.9)
    lambda_gp: float = 10.0
    n_critic: int = 5
    early_stopping_patience: int = 12

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = ModelConfig()


# ============================================================
# Preprocessing helpers
# ============================================================

def resample_signal_cubic(x: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if original_fs == target_fs:
        return x.copy()
    t_old = np.arange(len(x)) / original_fs
    duration = (len(x) - 1) / original_fs
    n_new = int(round(duration * target_fs)) + 1
    t_new = np.arange(n_new) / target_fs
    cs = CubicSpline(t_old, x)
    return cs(t_new)


def fir_bandpass_ecg(x: np.ndarray, fs: int, low: float = 0.5, high: float = 45.0, numtaps: int = 401):
    nyq = fs / 2.0
    taps = signal.firwin(numtaps, [low / nyq, high / nyq], pass_zero=False)
    return signal.filtfilt(taps, [1.0], x)


def butter_bandpass_ppg(x: np.ndarray, fs: int, low: float = 0.5, high: float = 8.0, order: int = 4):
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, x)


def median_denoise(x: np.ndarray, kernel_size: int = 5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return signal.medfilt(x, kernel_size=kernel_size)


def minmax_normalize_minus1_1(x: np.ndarray, eps: float = 1e-8):
    xmin = np.min(x)
    xmax = np.max(x)
    return 2.0 * (x - xmin) / (xmax - xmin + eps) - 1.0, xmin, xmax


def invert_minmax_minus1_1(x_norm: np.ndarray, xmin: float, xmax: float):
    return ((x_norm + 1.0) / 2.0) * (xmax - xmin) + xmin


def segment_with_overlap(x: np.ndarray, window_size: int, overlap_ratio: float = 0.2):
    step = max(1, int(window_size * (1.0 - overlap_ratio)))
    segments = []
    for start in range(0, len(x) - window_size + 1, step):
        segments.append(x[start:start + window_size])
    return np.stack(segments, axis=0) if segments else np.empty((0, window_size))


# ============================================================
# Positional embedding
# ============================================================

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, dim) * 0.02)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pos_embed[:, :T, :]


# ============================================================
# Transformer blocks
# ============================================================

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal: bool = False):
        # x: (B, T, D)
        residual = x
        x_norm = self.norm1(x)

        attn_mask = None
        if causal:
            T = x.shape[1]
            attn_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )

        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = residual + attn_out

        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, causal: bool = False):
        for blk in self.blocks:
            x = blk(x, causal=causal)
        return self.norm(x)


# ============================================================
# Generator modules
# ============================================================

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 1, padding: Optional[int] = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock1D(in_ch, out_ch)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down


class UpsampleBlock1D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock1D(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # match lengths if needed
        if x.size(-1) != skip.size(-1):
            min_len = min(x.size(-1), skip.size(-1))
            x = x[..., :min_len]
            skip = skip[..., :min_len]

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class GeneratorTransformerBottleneck(nn.Module):
    """
    CNN feature map -> channel expansion to transformer dim -> tokens -> transformer -> back
    """

    def __init__(self, in_channels: int, transformer_dim: int, depth: int, heads: int, mlp_ratio: int, dropout: float, max_tokens: int):
        super().__init__()
        self.proj_in = nn.Conv1d(in_channels, transformer_dim, kernel_size=1)
        self.pos_embed = LearnedPositionalEmbedding(max_tokens, transformer_dim)
        self.transformer = TransformerEncoder(
            dim=transformer_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.proj_out = nn.Conv1d(transformer_dim, in_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        x = self.proj_in(x)          # (B, D, T)
        x = x.transpose(1, 2)        # (B, T, D)
        x = self.pos_embed(x)
        x = self.transformer(x, causal=True)
        x = x.transpose(1, 2)        # (B, D, T)
        x = self.proj_out(x)         # (B, C, T)
        return x


class PPGtoECGGenerator(nn.Module):
    """
    UNet-inspired 1D CNN encoder-decoder with Transformer.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        base = cfg.base_channels

        # Encoder: 4 layers
        self.enc1 = DownsampleBlock1D(cfg.in_channels, base)          # 1 -> 8
        self.enc2 = DownsampleBlock1D(base, base * 2)                 # 8 -> 16
        self.enc3 = DownsampleBlock1D(base * 2, base * 4)             # 16 -> 32
        self.enc4 = DownsampleBlock1D(base * 4, base * 8)             # 32 -> 64

        self.bottleneck_conv = ConvBlock1D(base * 8, base * 8)

        # Transformer branch after CNN encoder
        # channels expanded from 8 to 16 after encoder feature map;
        # here we apply transformer with projection to transformer_dim.
        self.transformer_bottleneck = GeneratorTransformerBottleneck(
            in_channels=base * 8,
            transformer_dim=cfg.gen_transformer_dim,
            depth=cfg.gen_transformer_layers,
            heads=cfg.gen_transformer_heads,
            mlp_ratio=cfg.gen_transformer_mlp_ratio,
            dropout=cfg.gen_transformer_dropout,
            max_tokens=cfg.max_tokens,
        )

        # Decoder: 4 layers with skip connections
        self.dec4 = UpsampleBlock1D(base * 8, base * 8, base * 4)
        self.dec3 = UpsampleBlock1D(base * 4, base * 4, base * 2)
        self.dec2 = UpsampleBlock1D(base * 2, base * 2, base)
        self.dec1 = UpsampleBlock1D(base, base, base)

        self.final = nn.Sequential(
            nn.Conv1d(base, cfg.out_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, ppg: torch.Tensor):
        # ppg: (B, 1, T)

        s1, x = self.enc1(ppg)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck_conv(x)
        x = self.transformer_bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        ecg_hat = self.final(x)
        return ecg_hat


# ============================================================
# Discriminator / Critic modules
# ============================================================

class CNNFeatureEncoder1D(nn.Module):
    """
    Three Conv1D layers:
    kernels = (65, 33, 17)
    channels = (64, 128, 256)
    stride = 1
    LayerNorm at first layer and encoder output
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c1, c2, c3 = cfg.disc_conv_channels
        k1, k2, k3 = cfg.disc_conv_kernels

        self.conv1 = nn.Conv1d(1, c1, kernel_size=k1, stride=cfg.disc_conv_stride, padding=k1 // 2)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k2, stride=cfg.disc_conv_stride, padding=k2 // 2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k3, stride=cfg.disc_conv_stride, padding=k3 // 2)

        self.ln1 = nn.LayerNorm(c1)
        self.ln_out = nn.LayerNorm(c3)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        # x: (B, 1, T)

        x = self.conv1(x)            # (B, 64, T)
        x = x.transpose(1, 2)        # (B, T, 64)
        x = self.ln1(x)
        x = x.transpose(1, 2)
        x = self.act(x)

        x = self.act(self.conv2(x))  # (B, 128, T)
        x = self.act(self.conv3(x))  # (B, 256, T)

        x = x.transpose(1, 2)        # (B, T, 256)
        x = self.ln_out(x)
        return x


class ECGTransformerDiscriminator(nn.Module):
    """
    Transformer critic with CLS token.
    Supports:
    - critic score output for WGAN-GP
    - optional SSL head for transformation classification
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_encoder = CNNFeatureEncoder1D(cfg)

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.disc_model_dim) * 0.02)
        self.pos_embed = LearnedPositionalEmbedding(cfg.disc_max_tokens, cfg.disc_model_dim)

        self.transformer = TransformerEncoder(
            dim=cfg.disc_model_dim,
            depth=cfg.disc_transformer_layers,
            num_heads=cfg.disc_transformer_heads,
            mlp_ratio=cfg.disc_ffn_dim // cfg.disc_model_dim,
            dropout=cfg.disc_dropout,
        )

        # WGAN critic head -> scalar
        self.critic_head = nn.Linear(cfg.disc_model_dim, 1)

        # self-supervised pretraining head
        self.ssl_head = nn.Sequential(
            nn.Linear(cfg.disc_model_dim, cfg.ssl_hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.ssl_hidden_dim1, cfg.ssl_hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.ssl_hidden_dim2, cfg.ssl_num_classes),
        )

        # fine-tuning discriminator auxiliary head
        self.finetune_head = nn.Sequential(
            nn.Linear(cfg.disc_model_dim, cfg.finetune_hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.finetune_hidden_dim1, cfg.finetune_hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.finetune_hidden_dim2, 1),
        )

    def forward_features(self, ecg: torch.Tensor):
        # ecg: (B, 1, T)
        x = self.feature_encoder(ecg)    # (B, T, 256)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)   # (B,1,256)
        x = torch.cat([cls, x], dim=1)           # (B,T+1,256)
        x = self.pos_embed(x)
        x = self.transformer(x, causal=False)

        e_cls = x[:, 0]                          # (B,256)
        return e_cls, x

    def forward(self, ecg: torch.Tensor, mode: str = "critic"):
        e_cls, contextual = self.forward_features(ecg)

        if mode == "critic":
            return self.critic_head(e_cls).squeeze(-1)

        if mode == "ssl":
            return self.ssl_head(e_cls)

        if mode == "finetune":
            return self.finetune_head(e_cls).squeeze(-1)

        raise ValueError(f"Unknown mode: {mode}")

    def freeze_transformer_backbone(self):
        for p in self.feature_encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.pos_embed.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ============================================================
# Datasets
# ============================================================

class PairedPPGECGDataset(Dataset):
    """
    For GAN training:
    returns (ppg, ecg)
    shapes:
        ppg -> (1, T)
        ecg -> (1, T)
    """

    def __init__(self, ppg_segments: np.ndarray, ecg_segments: np.ndarray):
        assert ppg_segments.shape == ecg_segments.shape
        assert ppg_segments.ndim == 2
        self.ppg = ppg_segments.astype(np.float32)
        self.ecg = ecg_segments.astype(np.float32)

    def __len__(self):
        return len(self.ppg)

    def __getitem__(self, idx):
        ppg = torch.tensor(self.ppg[idx], dtype=torch.float32).unsqueeze(0)
        ecg = torch.tensor(self.ecg[idx], dtype=torch.float32).unsqueeze(0)
        return ppg, ecg


class SSLTransformationDataset(Dataset):
    """
    For discriminator self-supervised pretraining.
    Expects transformed ECG segments and pseudo-labels.
    """

    def __init__(self, ecg_segments: np.ndarray, labels: np.ndarray):
        assert len(ecg_segments) == len(labels)
        self.x = ecg_segments.astype(np.float32)
        self.y = labels.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


# ============================================================
# Losses
# ============================================================

def generator_wgan_loss(fake_scores: torch.Tensor):
    # LG = - E[D(G(p))]
    return -fake_scores.mean()


def discriminator_wgan_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor):
    # LD = -E[D(real)] + E[D(fake)]
    return -real_scores.mean() + fake_scores.mean()


def compute_gradient_penalty(
    discriminator: ECGTransformerDiscriminator,
    real_ecg: torch.Tensor,
    fake_ecg: torch.Tensor,
    lambda_gp: float,
    mode: str = "critic",
):
    """
    WGAN-GP gradient penalty:
        lambda * E[(||grad D(x_hat)||_2 - 1)^2]
    """
    B = real_ecg.size(0)
    device = real_ecg.device

    eps = torch.rand(B, 1, 1, device=device)
    x_hat = eps * real_ecg + (1.0 - eps) * fake_ecg
    x_hat.requires_grad_(True)

    d_hat = discriminator(x_hat, mode=mode)

    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(B, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1.0) ** 2).mean()
    return gp


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best_value = None
        self.best_state = None
        self.counter = 0

    def step(self, value: float, model: nn.Module):
        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == "min" and value < self.best_value:
            improved = True
        elif self.mode == "max" and value > self.best_value:
            improved = True

        if improved:
            self.best_value = value
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


# ============================================================
# Training: self-supervised pretraining
# ============================================================

def train_ssl_pretraining(
    discriminator: ECGTransformerDiscriminator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ModelConfig,
):
    device = cfg.device
    discriminator.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=cfg.lr_discriminator,
        betas=cfg.betas,
    )
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="min")

    for epoch in range(cfg.epochs_pretrain):
        discriminator.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = discriminator(x, mode="ssl")
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        discriminator.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = discriminator(x, mode="ssl")
                loss = criterion(logits, y)

                val_loss += loss.item() * x.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[SSL] Epoch {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        should_stop = stopper.step(val_loss, discriminator)
        if should_stop:
            print("Early stopping in SSL pretraining.")
            break

    if stopper.best_state is not None:
        discriminator.load_state_dict(stopper.best_state)

    return discriminator


# ============================================================
# Training: WGAN-GP
# ============================================================

def train_wgan_gp(
    generator: PPGtoECGGenerator,
    discriminator: ECGTransformerDiscriminator,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    cfg: ModelConfig,
):
    device = cfg.device
    generator.to(device)
    discriminator.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=cfg.lr_generator, betas=cfg.betas)
    opt_d = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=cfg.lr_discriminator,
        betas=cfg.betas,
    )

    stopper = EarlyStopping(patience=cfg.early_stopping_patience, mode="min")

    history = {
        "d_loss": [],
        "g_loss": [],
        "val_l1": [],
    }

    for epoch in range(cfg.epochs_gan):
        generator.train()
        discriminator.train()

        running_d = 0.0
        running_g = 0.0
        total_batches = 0

        for i, (ppg, real_ecg) in enumerate(train_loader):
            ppg = ppg.to(device)
            real_ecg = real_ecg.to(device)

            # --------------------------------------------------
            # Train critic n_critic times
            # --------------------------------------------------
            for _ in range(cfg.n_critic):
                fake_ecg = generator(ppg).detach()

                real_scores = discriminator(real_ecg, mode="critic")
                fake_scores = discriminator(fake_ecg, mode="critic")

                d_loss_base = discriminator_wgan_loss(real_scores, fake_scores)
                gp = compute_gradient_penalty(
                    discriminator=discriminator,
                    real_ecg=real_ecg,
                    fake_ecg=fake_ecg,
                    lambda_gp=cfg.lambda_gp,
                    mode="critic",
                )
                d_loss = d_loss_base + gp

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            # --------------------------------------------------
            # Train generator
            # --------------------------------------------------
            fake_ecg = generator(ppg)
            fake_scores = discriminator(fake_ecg, mode="critic")
            g_loss = generator_wgan_loss(fake_scores)

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            running_d += d_loss.item()
            running_g += g_loss.item()
            total_batches += 1

        epoch_d = running_d / max(1, total_batches)
        epoch_g = running_g / max(1, total_batches)

        history["d_loss"].append(epoch_d)
        history["g_loss"].append(epoch_g)

        msg = f"[GAN] Epoch {epoch+1:03d} | D Loss: {epoch_d:.4f} | G Loss: {epoch_g:.4f}"

        # Optional validation: use L1 reconstruction only as monitoring
        if val_loader is not None:
            generator.eval()
            val_l1 = 0.0
            count = 0
            with torch.no_grad():
                for ppg, real_ecg in val_loader:
                    ppg = ppg.to(device)
                    real_ecg = real_ecg.to(device)
                    fake_ecg = generator(ppg)
                    loss_l1 = F.l1_loss(fake_ecg, real_ecg, reduction="mean")
                    val_l1 += loss_l1.item() * ppg.size(0)
                    count += ppg.size(0)
            val_l1 /= max(1, count)
            history["val_l1"].append(val_l1)
            msg += f" | Val L1: {val_l1:.4f}"

            should_stop = stopper.step(val_l1, generator)
            print(msg)

            if should_stop:
                print("Early stopping in GAN training.")
                break
        else:
            print(msg)

    if stopper.best_state is not None:
        generator.load_state_dict(stopper.best_state)

    return generator, discriminator, history


# ============================================================
# Transfer learning utility
# ============================================================

def transfer_pretrained_discriminator_backbone(
    pretrained_discriminator: ECGTransformerDiscriminator,
    gan_discriminator: ECGTransformerDiscriminator,
    freeze_backbone: bool = True,
):
    gan_discriminator.feature_encoder.load_state_dict(pretrained_discriminator.feature_encoder.state_dict())
    gan_discriminator.transformer.load_state_dict(pretrained_discriminator.transformer.state_dict())
    gan_discriminator.cls_token.data.copy_(pretrained_discriminator.cls_token.data)
    gan_discriminator.pos_embed.pos_embed.data.copy_(pretrained_discriminator.pos_embed.pos_embed.data)

    if freeze_backbone:
        gan_discriminator.freeze_transformer_backbone()

    return gan_discriminator


# ============================================================
# Example LOSO split helper
# ============================================================

def loso_split(subject_ids: np.ndarray, held_out_subject):
    train_idx = np.where(subject_ids != held_out_subject)[0]
    test_idx = np.where(subject_ids == held_out_subject)[0]
    return train_idx, test_idx


# ============================================================
# Example main
# ============================================================

def main():


    np.random.seed(42)
    ppg_segments = np.random.randn(N, T).astype(np.float32)
    ecg_segments = np.random.randn(N, T).astype(np.float32)
    subject_ids = np.repeat(np.arange(34), N // 34 + 1)[:N]

    # --------------------------------------------------------
    # Example LOSO split
    # --------------------------------------------------------
    held_out_subject = subject_ids[0]
    train_idx, test_idx = loso_split(subject_ids, held_out_subject)

    # validation split: 10% of training
    n_train = len(train_idx)
    perm = np.random.permutation(train_idx)
    n_val = max(1, int(0.1 * n_train))
    val_idx = perm[:n_val]
    final_train_idx = perm[n_val:]

    train_dataset = PairedPPGECGDataset(ppg_segments[final_train_idx], ecg_segments[final_train_idx])
    val_dataset = PairedPPGECGDataset(ppg_segments[val_idx], ecg_segments[val_idx])
    test_dataset = PairedPPGECGDataset(ppg_segments[test_idx], ecg_segments[test_idx])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # --------------------------------------------------------
    # Optional self-supervised pretraining example
    # Here we simulate transformed ECG data and pseudo-labels
    # --------------------------------------------------------
    ssl_x = np.random.randn(1024, T).astype(np.float32)
    ssl_y = np.random.randint(0, cfg.ssl_num_classes, size=(1024,))

    ssl_train = SSLTransformationDataset(ssl_x[:900], ssl_y[:900])
    ssl_val = SSLTransformationDataset(ssl_x[900:], ssl_y[900:])

    ssl_train_loader = DataLoader(ssl_train, batch_size=cfg.batch_size, shuffle=True)
    ssl_val_loader = DataLoader(ssl_val, batch_size=cfg.batch_size, shuffle=False)

    pretrained_discriminator = ECGTransformerDiscriminator(cfg)
    pretrained_discriminator = train_ssl_pretraining(
        discriminator=pretrained_discriminator,
        train_loader=ssl_train_loader,
        val_loader=ssl_val_loader,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # Build GAN model
    # --------------------------------------------------------
    generator = PPGtoECGGenerator(cfg)
    discriminator = ECGTransformerDiscriminator(cfg)

    discriminator = transfer_pretrained_discriminator_backbone(
        pretrained_discriminator=pretrained_discriminator,
        gan_discriminator=discriminator,
        freeze_backbone=True,   # as described in fine-tuning
    )

    generator, discriminator, history = train_wgan_gp(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
    )

    # --------------------------------------------------------
    # Quick test forward pass
    # --------------------------------------------------------
    generator.eval()
    with torch.no_grad():
        sample_ppg, sample_ecg = next(iter(test_loader))
        sample_ppg = sample_ppg.to(device)
        fake_ecg = generator(sample_ppg)
        print("Sample PPG shape:", sample_ppg.shape)
        print("Generated ECG shape:", fake_ecg.shape)

    print("Training complete.")


if __name__ == "__main__":
    main()