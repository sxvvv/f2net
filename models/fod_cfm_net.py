# models/fod_cfm_net.py

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import enum


# ============================================================================
# Schedule utilities
# ============================================================================
class ModelType(enum.Enum):
    """Prediction target of the flow network."""
    FINAL_X = enum.auto()   # predict the terminal state x_T (i.e. μ)
    FLOW = enum.auto()      # predict the full flow x_T − x_0
    SFLOW = enum.auto()     # predict the stochastic flow x_T − x_t (recommended)


def get_cosine_schedule(num_timesteps, s=0.008):
    """Cosine noise schedule (used for the mean-reversion rate θ)."""
    steps = num_timesteps + 1
    t = np.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def get_linear_schedule(num_timesteps):
    """Linear noise schedule (used for the diffusion coefficient σ)."""
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def mean_flat(tensor):
    """Average over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """Index a numpy array by batch timesteps and broadcast to spatial dims."""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


# ============================================================================
# FoD diffusion schedule
# ============================================================================
class FoDSchedule:
    """Forward-only Diffusion (FoD) schedule.

    Implements the state-dependent SDE:
        dx_t = θ_t(μ − x_t)dt + σ_t(x_t − μ)dw_t

    whose closed-form solution is:
        x_t = (x_s − μ) exp(−∫_s^t (θ_z + σ_z²/2)dz + ∫_s^t σ_z dw_z) + μ

    Args:
        num_timesteps: Number of discrete transport steps T.
        theta_schedule: Schedule type for mean-reversion rate θ.
        sigma_schedule: Schedule type for diffusion coefficient σ.
        sigmas_scale: Normalisation constant for σ² (ensures Σσ²=1).
    """
    def __init__(
        self,
        num_timesteps: int = 100,
        theta_schedule: str = 'cosine',
        sigma_schedule: str = 'linear',
        sigmas_scale: float = 1.0,
    ):
        self.num_timesteps = num_timesteps

        if theta_schedule == 'cosine':
            thetas = get_cosine_schedule(num_timesteps)
        elif theta_schedule == 'linear':
            thetas = get_linear_schedule(num_timesteps)
        else:
            thetas = np.ones(num_timesteps) * 0.01

        if sigma_schedule == 'linear':
            sigma2s = get_linear_schedule(num_timesteps)
        elif sigma_schedule == 'cosine':
            sigma2s = get_cosine_schedule(num_timesteps)
        else:
            sigma2s = np.ones(num_timesteps) * 0.01

        thetas = np.array(thetas, dtype=np.float64)
        sigma2s = np.array(sigma2s, dtype=np.float64)

        # Prepend zero for the t=0 boundary
        self.thetas = np.append(0.0, thetas)

        # Normalise σ² so that Σσ² = sigmas_scale (numerical stability)
        if np.sum(sigma2s) > 0:
            sigma2s = sigmas_scale * sigma2s / np.sum(sigma2s)
        self.sigma2s = np.append(0.0, sigma2s)

        # Cumulative sums
        self.thetas_cumsum = np.cumsum(self.thetas)
        self.sigma2s_cumsum = np.cumsum(self.sigma2s)

        # Exponent of the mean: m̄_t = −∫_0^t (θ_z + σ_z²/2) dz
        expo_mean = -(self.thetas + 0.5 * self.sigma2s)
        expo_mean_cumsum = -(self.thetas_cumsum + 0.5 * self.sigma2s_cumsum)

        # Choose dt such that the terminal deviation is ~0.001
        self.dt = math.log(0.001) / expo_mean_cumsum[-1]

        # Scaled schedule arrays
        self.expo_mean = expo_mean * self.dt
        self.sqrt_expo_variance = np.sqrt(self.sigma2s * self.dt)
        self.expo_mean_cumsum = expo_mean_cumsum * self.dt
        self.sqrt_expo_variance_cumsum = np.sqrt(self.sigma2s_cumsum * self.dt)

    def expo_normal_cumsum(self, t, noise):
        """Compute exp(m̄_t + σ̄_t · ε) where ε ~ N(0, I)."""
        # Force float32 to avoid bf16 overflow (bf16 max ≈ 65504 ≈ e^11)
        noise_f = noise.float()
        exponent = (
            _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) +
            _extract_into_tensor(self.sqrt_expo_variance_cumsum, t, noise.shape) * noise_f
        )
        return torch.exp(exponent.clamp(-20.0, 20.0))

    def expo_normal_transition(self, s, t, noise):
        """Compute exp(m̄_{s:t} + σ̄_{s:t} · ε) for step s → t."""
        noise_f = noise.float()
        expo_mean_cumsum = (
            _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) -
            _extract_into_tensor(self.expo_mean_cumsum, s, noise.shape)
        )
        expo_variance_cumsum = (
            _extract_into_tensor(self.sigma2s_cumsum * self.dt, t, noise.shape) -
            _extract_into_tensor(self.sigma2s_cumsum * self.dt, s, noise.shape)
        )
        exponent = expo_mean_cumsum + torch.sqrt(expo_variance_cumsum + 1e-8) * noise_f
        return torch.exp(exponent.clamp(-20.0, 20.0))

    def get_xt(self, x_start, x_final, t, noise):
        """Sample the intermediate state x_t from the closed-form transition.

        x_t = (x_start − μ) · exp(m̄_t + σ̄_t · ε) + μ

        Args:
            x_start: Source state (degraded image).
            x_final: Target state μ (clean image).
            t: Discrete timestep indices.
            noise: Gaussian noise ε ~ N(0, I).
        """
        transition = self.expo_normal_cumsum(t, noise)
        return (x_start - x_final) * transition + x_final

    def sde_step(self, x, x_final, t, noise):
        """Single Euler–Maruyama step of the forward SDE.

        dx = θ_t(μ − x)dt + σ_t(x − μ)dw
        """
        x_f = x.float()
        x_final_f = x_final.float()
        noise_f = noise.float()
        drift = _extract_into_tensor(self.thetas, t, x.shape) * (x_final_f - x_f)
        diffusion = _extract_into_tensor(np.sqrt(self.sigma2s), t, x.shape) * (x_f - x_final_f)
        return x_f + drift * self.dt + diffusion * math.sqrt(self.dt) * noise_f


# ============================================================================
# Time embedding
# ============================================================================
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    if timesteps.dtype in [torch.int32, torch.int64]:
        timesteps = timesteps.float()
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ============================================================================
# Building blocks
# ============================================================================
class ResBlock(nn.Module):
    """Residual block with FiLM-style time conditioning (scale + shift)."""
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch * 2),
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.time_mlp(temb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(F.silu(h))
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv(x)


# ============================================================================
# Squeeze-and-Excitation & expert adapters
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""
    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ComplexityExpert(nn.Module):
    """Lightweight expert adapter with varying dilation rates.

    Each expert uses a different dilation to capture degradation patterns
    at different spatial scales. The output projection is zero-initialised
    to preserve the residual path at the start of training.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expert_idx: int,
        num_experts: int = 4,
        base_adapter_dim: int = 64,
    ):
        super().__init__()
        self.expert_idx = expert_idx
        self.adapter_dim = max(16, base_adapter_dim // (2 ** expert_idx))

        dilation = 1 + expert_idx
        padding = dilation

        self.down_proj = nn.Conv2d(in_ch, self.adapter_dim, 1)
        self.spatial_conv = nn.Conv2d(
            self.adapter_dim, self.adapter_dim,
            kernel_size=3, padding=padding, dilation=dilation
        )
        self.norm = nn.GroupNorm(min(8, self.adapter_dim), self.adapter_dim)
        self.se = SEBlock(self.adapter_dim)
        self.up_proj = nn.Conv2d(self.adapter_dim, out_ch, 1)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = F.silu(self.norm(self.spatial_conv(h)))
        h = self.se(h)
        return self.up_proj(h)


class EnhancedExpert(nn.Module):
    """Enhanced expert adapter with dual convolutions and SE attention.

    Compared to ComplexityExpert:
    - Uniform bottleneck dimension across experts (differentiated by dilation).
    - Two 3×3 conv layers: dilated (multi-scale) followed by standard (refinement).
    - Total ~1.25M params across M=4 experts.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expert_idx: int,
        num_experts: int = 4,
        base_adapter_dim: int = 128,
    ):
        super().__init__()
        self.expert_idx = expert_idx
        self.adapter_dim = base_adapter_dim

        dilation = 1 + expert_idx
        padding = dilation

        self.down_proj = nn.Conv2d(in_ch, self.adapter_dim, 1)

        # Layer 1: dilated conv (different experts see different receptive fields)
        self.conv1 = nn.Conv2d(
            self.adapter_dim, self.adapter_dim,
            kernel_size=3, padding=padding, dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(min(8, self.adapter_dim), self.adapter_dim)

        # Layer 2: standard conv (feature refinement)
        self.conv2 = nn.Conv2d(
            self.adapter_dim, self.adapter_dim,
            kernel_size=3, padding=1,
        )
        self.norm2 = nn.GroupNorm(min(8, self.adapter_dim), self.adapter_dim)

        self.se = SEBlock(self.adapter_dim, reduction=4)

        # Zero-init output projection to preserve residual at init
        self.up_proj = nn.Conv2d(self.adapter_dim, out_ch, 1)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = F.silu(self.norm1(self.conv1(h)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = self.se(h)
        return self.up_proj(h)


# ============================================================================
# Bottleneck attention (Restormer-style transposed attention)
# ============================================================================
class BottleneckAttention(nn.Module):
    """Channel self-attention + FFN at the U-Net bottleneck.

    Uses transposed (C×C) attention instead of spatial (HW×HW) attention,
    following the Restormer design (Zamir et al., CVPR 2022). This is
    significantly cheaper than spatial attention at high resolutions.
    """
    def __init__(self, ch: int, num_heads: int = 4, ffn_expansion: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.qkv_dw = nn.Conv2d(ch * 3, ch * 3, 3, padding=1, groups=ch * 3)
        self.proj = nn.Conv2d(ch, ch, 1)

        self.norm2 = nn.GroupNorm(8, ch)
        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * ffn_expansion, 1),
            nn.GELU(),
            nn.Conv2d(ch * ffn_expansion, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # --- Channel attention ---
        h = self.norm1(x)
        qkv = self.qkv_dw(self.qkv(h))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # (B, heads, C/h, C/h)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        x = x + self.proj(out)

        # --- Feed-forward ---
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Frequency embedding
# ============================================================================
class FrequencyEmbedding(nn.Module):
    """Extract high-frequency cues from bottleneck features for expert routing.

    A fixed Laplacian high-pass filter captures texture/edge statistics,
    which are then mapped to an embedding via a small MLP.
    """
    def __init__(self, ch: int, out_dim: int = 128):
        super().__init__()
        self.high_pass = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        kernel = torch.tensor([[[[-1, -1, -1],
                                 [-1,  8, -1],
                                 [-1, -1, -1]]]], dtype=torch.float32)
        self.high_pass.weight.data = kernel.repeat(ch, 1, 1, 1)
        self.high_pass.weight.requires_grad = False

        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch * 2),
            nn.GELU(),
            nn.Linear(ch * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.high_pass(x))
        h = self.pool(h).flatten(1)
        return self.mlp(h)


# ============================================================================
# Brightness pre-enhancer
# ============================================================================
class BrightnessPreEnhancer(nn.Module):
    """Lightweight brightness pre-enhancement module.

    Shortens the transport distance for low-light inputs by learning a
    brightness residual Δ = f(LQ), gated by the input luminance:
        output = LQ + gate · Δ
    The gate is near 1 for dark images and near 0 for well-lit ones,
    so the module is effectively bypassed for non-low-light inputs.

    Adds ~0.5M parameters to the model.
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        num_blocks: int = 4,
        brightness_threshold: float = 0.3,
    ):
        super().__init__()
        self.brightness_threshold = brightness_threshold

        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
            ))

        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
            nn.Tanh(),
        )

        # Learned gating network
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Zero-init the tail so the initial residual is near zero
        nn.init.zeros_(self.tail[-2].weight)
        nn.init.zeros_(self.tail[-2].bias)

    def compute_brightness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample mean luminance (ITU-R BT.601)."""
        x_01 = (x + 1) / 2  # [-1,1] → [0,1]
        luminance = 0.299 * x_01[:, 0] + 0.587 * x_01[:, 1] + 0.114 * x_01[:, 2]
        return luminance.mean(dim=[1, 2])  # (B,)

    def forward(
        self,
        x: torch.Tensor,
        force_enhance: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input image (B, 3, H, W), range [-1, 1].
            force_enhance: Force enhancement (used during training).
            return_gate: Also return the gating values.
        """
        brightness = self.compute_brightness(x)

        # Brightness-adaptive gate: dark → ~1, bright → ~0
        gate_brightness = torch.sigmoid(
            (self.brightness_threshold - brightness) * 10.0
        ).view(-1, 1, 1, 1)

        gate_learned = self.gate_net(x).view(-1, 1, 1, 1)
        gate = gate_brightness * gate_learned

        # Compute brightness residual
        h = self.head(x)
        for block in self.blocks:
            h = h + block(h)
        delta = self.tail(h)

        out = (x + gate * delta).clamp(-1, 1)

        if return_gate:
            return out, gate
        return out


# ============================================================================
# Degradation parser
# ============================================================================
class DegradationParser(nn.Module):
    """
    Estimate degradation composition from the input image.
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        emb_dim: int = 128,
        num_factors: int = 4,
    ):
        super().__init__()
        self.num_factors = num_factors

        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = base_ch * 4

        self.w_mlp = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, num_factors),
        )

        self.cls_mlp = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(emb_dim, num_factors),
        )

        self.spatial_decoder = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, num_factors, 3, padding=1),
        )

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.encoder(y)
        g = self.global_pool(feat).flatten(1)

        logits = self.cls_mlp(g)
        w = torch.sigmoid(self.w_mlp(g))

        m = torch.sigmoid(self.spatial_decoder(feat))
        if m.shape[2:] != y.shape[2:]:
            m = F.interpolate(m, size=y.shape[2:], mode='bilinear', align_corners=False)

        return w, m, logits


# ============================================================================
# Shared backbone
# ============================================================================
class SharedBackbone(nn.Module):
    """4-stage U-Net backbone with time conditioning and bottleneck attention."""
    def __init__(
        self,
        in_ch: int = 6,
        out_ch: int = 3,
        base_ch: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        emb_dim: int = 256,
        freq_emb_dim: int = 128,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.base_ch = base_ch

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch_in = base_ch
        for i, mult in enumerate(ch_mult):
            ch_out = base_ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(ch_in, ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
            ]))
            ch_in = ch_out
            if i < len(ch_mult) - 1:
                self.downsamples.append(Downsample(ch_out))

        # Bottleneck
        self.mid1 = ResBlock(ch_in, ch_in, emb_dim)
        self.mid2 = ResBlock(ch_in, ch_in, emb_dim)
        self.mid3 = ResBlock(ch_in, ch_in, emb_dim)
        self.mid_attn = BottleneckAttention(ch_in, num_heads=4, ffn_expansion=4)
        self.freq_embed = FrequencyEmbedding(ch_in, out_dim=freq_emb_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in enumerate(reversed(ch_mult)):
            ch_out = base_ch * mult
            skip_ch = base_ch * ch_mult[len(ch_mult) - 1 - i]
            self.up_blocks.append(nn.ModuleList([
                ResBlock(ch_in + skip_ch, ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
            ]))
            ch_in = ch_out
            if i < len(ch_mult) - 1:
                self.upsamples.append(Upsample(ch_out))

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        return_features: bool = False,
    ):
        temb = sinusoidal_time_embedding(t, self.emb_dim)
        temb = self.time_proj(temb)

        h = torch.cat([x_t, x_cond], dim=1)
        h = self.in_conv(h)

        skips = []
        for i, blocks in enumerate(self.down_blocks):
            h = blocks[0](h, temb)
            h = blocks[1](h, temb)
            skips.append(h)
            if i < len(self.downsamples):
                h = self.downsamples[i](h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)
        h = self.mid3(h, temb)
        h = self.mid_attn(h)
        freq_emb = self.freq_embed(h)

        for i, blocks in enumerate(self.up_blocks):
            h = torch.cat([h, skips[-(i+1)]], dim=1)
            h = blocks[0](h, temb)
            h = blocks[1](h, temb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        h_final = h
        out = self.out_conv(F.silu(self.out_norm(h_final)))

        if return_features:
            return out, h_final, temb, freq_emb
        return out


# ============================================================================
# F²D-Net (main model)
# ============================================================================
class FoDAugmentedFlowNet(nn.Module):
    """F²D-Net: Factorized forward-only diffusion with expert adapters.

    The factorized flow field is:
        v(x_t, t) = v_shared(x_t, t) + Σ_i α_i(w, m, t) · m_i ⊙ Δv_i(h)

    where α_i are degradation- and time-conditioned gating weights, m_i are
    per-factor spatial intensity maps, and Δv_i are expert adapter outputs.
    An optional brightness pre-enhancer reduces the transport distance for
    low-light inputs.
    """
    def __init__(
        self,
        in_ch: int = 6,
        out_ch: int = 3,
        base_ch: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
        emb_dim: int = 256,
        num_experts: int = 4,
        adapter_dim: int = 128,
        num_timesteps: int = 100,
        model_type: ModelType = ModelType.SFLOW,
        use_brightness_enhancer: bool = True,
        brightness_enhancer_ch: int = 64,
        brightness_enhancer_blocks: int = 6,
        brightness_threshold: float = 0.3,
        freq_emb_dim: int = 128,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.emb_dim = emb_dim
        self.out_ch = out_ch
        self.num_timesteps = num_timesteps
        self.model_type = model_type
        self.use_brightness_enhancer = use_brightness_enhancer

        # FoD schedule
        self.schedule = FoDSchedule(
            num_timesteps=num_timesteps,
            theta_schedule='cosine',
            sigma_schedule='linear',
            sigmas_scale=1.0,
        )

        # Brightness pre-enhancer (optional)
        if use_brightness_enhancer:
            self.brightness_enhancer = BrightnessPreEnhancer(
                in_ch=3,
                base_ch=brightness_enhancer_ch,
                num_blocks=brightness_enhancer_blocks,
                brightness_threshold=brightness_threshold,
            )
        else:
            self.brightness_enhancer = None

        # Degradation parser
        self.parser = DegradationParser(
            in_ch=3, base_ch=32, emb_dim=128,
            num_factors=num_experts,
        )

        # Shared backbone
        self.backbone = SharedBackbone(
            in_ch=in_ch, out_ch=out_ch,
            base_ch=base_ch, ch_mult=ch_mult, emb_dim=emb_dim,
            freq_emb_dim=freq_emb_dim,
        )

        # Expert adapters
        self.experts = nn.ModuleList([
            EnhancedExpert(
                in_ch=base_ch, out_ch=out_ch,
                expert_idx=i, num_experts=num_experts,
                base_adapter_dim=adapter_dim,
            )
            for i in range(num_experts)
        ])

        # Complexity bias: proportional to each expert's parameter count
        expert_params = [e.num_params for e in self.experts]
        max_params = max(expert_params)
        self.register_buffer(
            'complexity_bias',
            torch.tensor([p / max_params for p in expert_params])
        )

        # Gating MLP (conditioned on time, degradation weights, and frequency)
        self.freq_emb_dim = freq_emb_dim
        alpha_input_dim = emb_dim + num_experts * 2 + freq_emb_dim
        self.alpha_mlp = nn.Sequential(
            nn.Linear(alpha_input_dim, alpha_input_dim // 2),
            nn.SiLU(),
            nn.Linear(alpha_input_dim // 2, num_experts),
        )

    def compute_alpha(
        self,
        temb: torch.Tensor,
        w: torch.Tensor,
        m: torch.Tensor,
        freq_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute degradation-aware gating weights α (with frequency cues)."""
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        m_mean = m.mean([2, 3])

        parts = [temb, w_norm, m_mean]
        if freq_emb is not None:
            parts.append(freq_emb)
        alpha_input = torch.cat(parts, dim=1)
        alpha_logits = self.alpha_mlp(alpha_input)
        alpha_logits = alpha_logits - self.complexity_bias.view(1, -1)

        if self.training:
            noise_std = 0.3 / self.num_experts
            noise = torch.randn_like(alpha_logits) * noise_std
            alpha_logits = alpha_logits + noise

        alpha = F.softmax(alpha_logits, dim=1)
        return alpha

    def pre_enhance(
        self,
        x: torch.Tensor,
        force_enhance: bool = False,
    ) -> torch.Tensor:
        """Apply brightness pre-enhancement (if enabled)."""
        if self.brightness_enhancer is not None:
            return self.brightness_enhancer(x, force_enhance=force_enhance)
        return x

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x_cond: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        m: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
        skip_pre_enhance: bool = False,
    ):
        """
        Args:
            x_t: (B, 3, H, W) current state along the transport path.
            t: (B,) discrete timestep indices.
            x_cond: (B, 3, H, W) conditioning image (degraded input).
            w: Pre-computed degradation weights (optional).
            m: Pre-computed spatial intensity maps (optional).
            return_alpha: Whether to also return gating weights.
            skip_pre_enhance: Skip the brightness enhancer (when already applied).

        Returns:
            Prediction whose semantics depend on ``model_type``
            (SFLOW: μ − x_t, FLOW: μ − x_0, FINAL_X: μ).
        """
        if self.brightness_enhancer is not None and not skip_pre_enhance:
            x_cond = self.brightness_enhancer(x_cond, force_enhance=self.training)

        if w is None or m is None:
            w_pred, m_pred, _ = self.parser(x_cond)
            w = w if w is not None else w_pred
            m = m if m is not None else m_pred

        v_share, h_final, temb, freq_emb = self.backbone(
            x_t, x_cond, t, return_features=True
        )

        alpha = self.compute_alpha(temb, w, m, freq_emb)

        # Factorized output: v = v_shared + Σ α_i · m_i ⊙ Δv_i
        output = v_share.clone()
        for i, expert in enumerate(self.experts):
            delta_v = expert(h_final)
            m_i = m[:, i:i+1]
            if m_i.shape[2:] != delta_v.shape[2:]:
                m_i = F.interpolate(m_i, size=delta_v.shape[2:], mode='bilinear', align_corners=False)
            alpha_i = alpha[:, i].view(-1, 1, 1, 1)
            output = output + alpha_i * m_i * delta_v

        if return_alpha:
            return output, alpha
        return output


# ============================================================================
# Model factory
# ============================================================================
def create_fod_model(
    base_ch: int = 64,
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
    emb_dim: int = 256,
    num_experts: int = 4,
    adapter_dim: int = 128,
    num_timesteps: int = 100,
    model_type: str = 'SFLOW',
    use_brightness_enhancer: bool = True,
    brightness_enhancer_ch: int = 64,
    brightness_enhancer_blocks: int = 6,
    brightness_threshold: float = 0.3,
    freq_emb_dim: int = 128,
) -> FoDAugmentedFlowNet:
    """Create an F²D-Net model.

    Args:
        base_ch: Base channel width of the U-Net backbone.
        ch_mult: Channel multipliers for each encoder stage.
        emb_dim: Dimension of the time embedding.
        num_experts: Number of degradation-specific expert adapters (M).
        adapter_dim: Bottleneck dimension of each expert adapter.
        num_timesteps: Number of discrete transport steps T.
        model_type: Prediction target ('SFLOW', 'FLOW', or 'FINAL_X').
        use_brightness_enhancer: Enable the brightness pre-enhancer.
        brightness_enhancer_ch: Channel width of the brightness pre-enhancer.
        brightness_enhancer_blocks: Number of residual blocks in the pre-enhancer.
        brightness_threshold: Luminance threshold for adaptive enhancement.
        freq_emb_dim: Dimension of the frequency embedding.

    Returns:
        Configured FoDAugmentedFlowNet instance.
    """
    type_map = {
        'FINAL_X': ModelType.FINAL_X,
        'FLOW': ModelType.FLOW,
        'SFLOW': ModelType.SFLOW,
    }
    return FoDAugmentedFlowNet(
        in_ch=6, out_ch=3,
        base_ch=base_ch, ch_mult=ch_mult, emb_dim=emb_dim,
        num_experts=num_experts, adapter_dim=adapter_dim,
        num_timesteps=num_timesteps,
        model_type=type_map.get(model_type, ModelType.SFLOW),
        use_brightness_enhancer=use_brightness_enhancer,
        brightness_enhancer_ch=brightness_enhancer_ch,
        brightness_enhancer_blocks=brightness_enhancer_blocks,
        brightness_threshold=brightness_threshold,
        freq_emb_dim=freq_emb_dim,
    )


# ============================================================================
# Training loss
# ============================================================================
def _pixel_loss(diff: torch.Tensor, loss_type: str = "l1",
                charbonnier_eps: float = 1e-3) -> torch.Tensor:
    """Per-sample pixel loss, averaged over spatial and channel dims."""
    if loss_type == "charbonnier":
        return mean_flat(torch.sqrt(diff ** 2 + charbonnier_eps ** 2))
    else:
        return mean_flat(torch.abs(diff))


def fod_training_loss(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,       # degraded image (source distribution)
    x_final: torch.Tensor,       # clean image (target μ)
    t: torch.Tensor,             # timestep indices
    noise: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    present: Optional[torch.Tensor] = None,
    m_gt: Optional[torch.Tensor] = None,
    lambda_cls: float = 0.01,
    lambda_balance: float = 0.01,
    lambda_w: float = 0.1,
    lambda_m: float = 0.05,
    lambda_recon: float = 0.1,
    lambda_alpha: float = 0.0,
    alpha_target: Optional[torch.Tensor] = None,
    use_hard_sample_weighting: bool = False,
    hard_sample_psnr_threshold: float = 26.0,
    hard_sample_weight_factor: float = 2.0,
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
    lambda_freq: float = 0.0,
    lambda_enhance: float = 0.1,
    is_lowlight: Optional[torch.Tensor] = None,
    lowlight_boost: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute the stochastic flow matching training loss.

    The core objective is:
        x_t = (x_start − μ) · exp(m̄_t + σ̄_t · ε) + μ
        target = μ − x_t  (for SFLOW)
        L_main = E[ℓ(target − f_φ(x_t, t))]

    Additional loss terms:
    - Brightness pre-enhancer luminance alignment (lambda_enhance).
    - Frequency-domain L1 on real/imag parts (lambda_freq).
    - Degradation factor classification BCE (lambda_cls).
    - Degradation weight supervision (lambda_w).
    - Spatial intensity map supervision (lambda_m).
    - Expert load-balancing via CV² penalty (lambda_balance).
    - Gating prior supervision (lambda_alpha).

    Returns:
        (total_loss, loss_dict) where loss_dict contains per-term scalars.
    """
    schedule = model.schedule
    device = x_start.device
    B = x_start.shape[0]

    if noise is None:
        noise = torch.randn_like(x_final)

    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    # === Brightness pre-enhancer loss ===
    if model.brightness_enhancer is not None and lambda_enhance > 0:
        x_enhanced, enhance_gate = model.brightness_enhancer(
            x_start, force_enhance=True, return_gate=True
        )

        if is_lowlight is not None:
            lowlight_mask = is_lowlight.float().view(-1, 1, 1, 1)
        else:
            # Auto-detect low-light samples by luminance
            brightness = model.brightness_enhancer.compute_brightness(x_start)
            lowlight_mask = (brightness < model.brightness_enhancer.brightness_threshold).float()
            lowlight_mask = lowlight_mask.view(-1, 1, 1, 1)

        # Luminance-focused loss (Y channel only; the enhancer is too small
        # for full-pixel reconstruction, so we only supervise brightness)
        lum_w = torch.tensor([0.299, 0.587, 0.114], device=device,
                             dtype=x_enhanced.dtype).view(1, 3, 1, 1)
        enhanced_lum = (x_enhanced * lum_w).sum(dim=1, keepdim=True)
        gt_lum = (x_final * lum_w).sum(dim=1, keepdim=True)

        lum_diff = (enhanced_lum - gt_lum) * lowlight_mask
        loss_enhance_lum = mean_flat(torch.abs(lum_diff)).mean()

        # Auxiliary RGB L1 to prevent colour drift
        rgb_diff = (x_enhanced - x_final) * lowlight_mask
        loss_enhance_rgb = _pixel_loss(rgb_diff, loss_type, charbonnier_eps).mean()

        loss_enhance = 0.7 * loss_enhance_lum + 0.3 * loss_enhance_rgb

        # Gate regularisation: encourage the gate to activate for dark inputs
        gate_reg = (lowlight_mask.squeeze() * (1 - enhance_gate.squeeze())).mean()

        # Penalise if the enhancer increases the transport distance
        with torch.no_grad():
            dist_before = mean_flat(torch.abs(x_start - x_final)).mean()
            dist_after = mean_flat(torch.abs(x_enhanced - x_final)).mean()
            dist_penalty_active = (dist_after > dist_before).float()
        loss_dist_penalty = mean_flat(torch.abs(rgb_diff)).mean() * dist_penalty_active

        loss_enhance_total = loss_enhance + 0.1 * gate_reg + 0.2 * loss_dist_penalty
        total_loss = total_loss + lambda_enhance * loss_enhance_total
        losses['enhance'] = loss_enhance.item()
        losses['enhance_gate'] = enhance_gate.mean().item()
        losses['enhance_lum'] = loss_enhance_lum.item()

        x_cond = x_enhanced
    else:
        x_cond = x_start

    # === Sample intermediate state x_t ===
    x_t = schedule.get_xt(x_cond, x_final, t, noise)

    # Parse degradation (on pre-enhanced image)
    w_pred, m_pred, logits = model.parser(x_cond)

    # Model prediction (skip pre-enhancer since it was already applied above)
    model_output, alpha = model(
        x_t, t, x_cond, w=w_pred, m=m_pred,
        return_alpha=True, skip_pre_enhance=True
    )

    # Compute target
    if model.model_type == ModelType.FINAL_X:
        target = x_final
    elif model.model_type == ModelType.FLOW:
        target = x_final - x_cond
    elif model.model_type == ModelType.SFLOW:
        target = x_final - x_t

    # Reconstruct x_final for auxiliary losses
    if model.model_type == ModelType.FINAL_X:
        x_final_pred = model_output
    elif model.model_type == ModelType.FLOW:
        x_final_pred = x_cond + model_output
    elif model.model_type == ModelType.SFLOW:
        x_final_pred = x_t + model_output

    # --- Main loss (L1 or Charbonnier, with optional hard-sample weighting) ---
    diff = target - model_output
    loss_per_sample = _pixel_loss(diff, loss_type, charbonnier_eps)
    sample_weights = torch.ones(B, device=device)

    if use_hard_sample_weighting:
        with torch.no_grad():
            mse_ps = mean_flat((x_final_pred - x_final) ** 2)
            psnr_ps = 10.0 * torch.log10(4.0 / (mse_ps + 1e-8))
            hard_mask = psnr_ps < hard_sample_psnr_threshold
            sample_weights[hard_mask] = hard_sample_weight_factor
        losses['hard_ratio'] = hard_mask.float().mean().item()

    if is_lowlight is not None and lowlight_boost > 1.0:
        sample_weights[is_lowlight] = sample_weights[is_lowlight] * lowlight_boost
        losses['lowlight_ratio'] = is_lowlight.float().mean().item()

    sample_weights = sample_weights / (sample_weights.mean() + 1e-8)
    loss_main = (loss_per_sample * sample_weights).mean()

    losses['main'] = loss_main.item()
    total_loss = total_loss + loss_main

    # --- Reconstruction loss ---
    if lambda_recon > 0:
        recon_diff = x_final_pred - x_final
        loss_recon = _pixel_loss(recon_diff, loss_type, charbonnier_eps).mean()
        total_loss = total_loss + lambda_recon * loss_recon
        losses['recon'] = loss_recon.item()

    # --- Frequency-domain loss (real/imag L1, following MoCE-IR) ---
    if lambda_freq > 0:
        pred_fft = torch.fft.rfft2(x_final_pred.float())
        gt_fft = torch.fft.rfft2(x_final.float())
        pred_ri = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        gt_ri = torch.stack([gt_fft.real, gt_fft.imag], dim=-1)
        loss_freq = F.l1_loss(pred_ri, gt_ri)
        total_loss = total_loss + lambda_freq * loss_freq
        losses['freq'] = loss_freq.item()

    # --- Degradation weight supervision ---
    if lambda_w > 0:
        w_target = present if present is not None else labels
        if w_target is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                loss_w = F.binary_cross_entropy(w_pred.float(), w_target.float())
            total_loss = total_loss + lambda_w * loss_w
            losses['w'] = loss_w.item()

    # --- Spatial intensity map supervision ---
    if lambda_m > 0 and m_gt is not None:
        if m_gt.shape != m_pred.shape:
            m_gt = F.interpolate(m_gt, size=m_pred.shape[2:], mode='bilinear', align_corners=False)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_m = F.binary_cross_entropy(m_pred.float(), m_gt.float())
        total_loss = total_loss + lambda_m * loss_m
        losses['m'] = loss_m.item()

    # --- Factor classification loss ---
    if lambda_cls > 0 and labels is not None:
        loss_cls = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss = total_loss + lambda_cls * loss_cls
        losses['cls'] = loss_cls.item()

    # --- Expert load-balancing (CV² of importance) ---
    if lambda_balance > 0:
        importance = alpha.sum(dim=0)
        w_balance = present if present is not None else w_pred
        w_sum = w_balance.sum(dim=0).clamp(min=1e-8)
        weighted_importance = importance / w_sum
        imp_mean = weighted_importance.mean()
        imp_std = weighted_importance.std()
        loss_balance = (imp_std / (imp_mean + 1e-8)) ** 2
        total_loss = total_loss + lambda_balance * loss_balance
        losses['balance'] = loss_balance.item()

    # --- Gating prior supervision ---
    if lambda_alpha > 0 and alpha_target is not None:
        loss_alpha = F.mse_loss(alpha, alpha_target)
        total_loss = total_loss + lambda_alpha * loss_alpha
        losses['alpha'] = loss_alpha.item()

    return total_loss, losses


# ============================================================================
# Inference 
# ============================================================================
@torch.no_grad()
def fod_inference(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,
    num_steps: int = -1,
    sample_type: str = "MC",
    clip_denoised: bool = True,
    use_tta: bool = False,
) -> torch.Tensor:
    """Multi-step inference: transport from degraded x_start towards clean μ.
    Args:
        model: Trained F²D-Net.
        x_start: Degraded input (B, 3, H, W), range [-1, 1].
        num_steps: Number of sampling steps (-1 = use all T steps).
            Use ``num_steps=1`` for the fastest inference that remains
            consistent with the training distribution.
        sample_type: Sampling strategy ('EM', 'MC', or 'NMC').
        clip_denoised: Clip intermediate predictions to [-1, 1].
        use_tta: Enable test-time augmentation (geometric self-ensemble).
    """
    model.eval()
    schedule = model.schedule

    if not use_tta:
        return _fod_inference_single(model, x_start, num_steps, sample_type, clip_denoised)

    # TTA: geometric self-ensemble (identity + 3 flips)
    transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
    ]
    inverse_transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
    ]

    results = []
    for trans, inv_trans in zip(transforms, inverse_transforms):
        x_aug = trans(x_start)
        x_hat = _fod_inference_single(model, x_aug, num_steps, sample_type, clip_denoised)
        results.append(inv_trans(x_hat))

    return torch.stack(results, dim=0).mean(dim=0).clamp(-1, 1)


def _fod_inference_single(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,
    num_steps: int,
    sample_type: str,
    clip_denoised: bool,
) -> torch.Tensor:
    """Single-pass multi-step FoD inference."""
    schedule = model.schedule
    device = x_start.device
    B = x_start.shape[0]

    if num_steps <= 0:
        num_steps = schedule.num_timesteps

    # Brightness pre-enhancement (if enabled)
    if model.brightness_enhancer is not None:
        x_cond = model.brightness_enhancer(x_start, force_enhance=False)
    else:
        x_cond = x_start

    img = x_cond.clone()

    indices = np.linspace(0, schedule.num_timesteps, num_steps + 1).astype(int)
    times = np.copy(indices)

    # Parse degradation once on the pre-enhanced image
    w, m, _ = model.parser(x_cond)

    for i, idx in enumerate(indices[:-1]):
        t = torch.tensor([idx] * B, device=device)
        t_next = torch.tensor([times[i+1]] * B, device=device)

        model_output = model(img, t, x_cond, w=w, m=m, skip_pre_enhance=True)

        if model.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif model.model_type == ModelType.FLOW:
            x_final = x_cond + model_output
        elif model.model_type == ModelType.SFLOW:
            x_final = img + model_output

        if clip_denoised:
            x_final = x_final.clamp(-1, 1)

        # Transition to next state (all in float32 to prevent bf16 overflow)
        noise = torch.randn(img.shape, device=device, dtype=torch.float32)
        img_f = img.float()
        x_final_f = x_final.float()
        x_cond_f = x_cond.float()
        if sample_type == "EM":
            img = schedule.sde_step(img_f, x_final_f, t_next, noise)
        elif sample_type == "MC":
            img = (img_f - x_final_f) * schedule.expo_normal_transition(t, t_next, noise) + x_final_f
        elif sample_type == "NMC":
            img = (x_cond_f - x_final_f) * schedule.expo_normal_cumsum(t_next, noise) + x_final_f
        img = img.clamp(-2, 2)

    return img.clamp(-1, 1)


# ============================================================================
# Single-step inference
# ============================================================================
@torch.no_grad()
def fod_one_step_inference(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    
    model.eval()
    device = x_start.device
    B = x_start.shape[0]

    if not use_tta:
        if model.brightness_enhancer is not None:
            x_cond = model.brightness_enhancer(x_start, force_enhance=False)
        else:
            x_cond = x_start

        t = torch.tensor([model.num_timesteps] * B, device=device)
        w, m, _ = model.parser(x_cond)
        model_output = model(x_cond, t, x_cond, w=w, m=m, skip_pre_enhance=True)

        if model.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif model.model_type == ModelType.FLOW:
            x_final = x_cond + model_output
        elif model.model_type == ModelType.SFLOW:
            x_final = x_cond + model_output

        return x_final.clamp(-1, 1)

    # TTA: geometric self-ensemble
    transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
    ]
    inverse_transforms = [
        lambda x: x,
        lambda x: x.flip(-1),
        lambda x: x.flip(-2),
        lambda x: x.flip(-1).flip(-2),
    ]

    results = []
    for trans, inv_trans in zip(transforms, inverse_transforms):
        x_aug = trans(x_start)

        if model.brightness_enhancer is not None:
            x_cond = model.brightness_enhancer(x_aug, force_enhance=False)
        else:
            x_cond = x_aug

        t = torch.tensor([model.num_timesteps] * B, device=device)
        w, m, _ = model.parser(x_cond)
        model_output = model(x_cond, t, x_cond, w=w, m=m, skip_pre_enhance=True)

        if model.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif model.model_type == ModelType.FLOW:
            x_final = x_cond + model_output
        elif model.model_type == ModelType.SFLOW:
            x_final = x_cond + model_output

        results.append(inv_trans(x_final))

    return torch.stack(results, dim=0).mean(dim=0).clamp(-1, 1)