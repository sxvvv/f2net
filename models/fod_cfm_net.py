# models/fod_cfm_net.py
# 基于FoD的增广流匹配网络 (Forward-only Diffusion + Augmented Flow Matching)
#
# 核心创新（融合FoD论文方法）：
# 1. 前向扩散过程：使用FoD的状态依赖SDE，无需反向扩散
#    - SDE: dx_t = θ_t(μ - x_t)dt + σ_t(x_t - μ)dw_t
#    - 解：x_t = (x_s - μ)exp(m̄_{s:t} + σ̄_{s:t}ε) + μ
#
# 2. 增广状态空间：保留辅助变量消歧机制
#    - z_t = [x_t; y_t]
#    - 辅助变量编码退化信息损失
#
# 3. 随机流匹配目标：
#    - L = E[||（μ - x_t）- f_φ(x_t, t)||²]
#
# 4. 灵活采样策略：EM, MC, NMC
#
# 参考: FoD论文 (arXiv:2505.16733)

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import enum


# ============================================================================
# FoD 调度和工具函数
# ============================================================================
class ModelType(enum.Enum):
    """模型预测类型"""
    FINAL_X = enum.auto()   # 预测最终状态 x_T (μ)
    FLOW = enum.auto()      # 预测流场 x_T - x_0
    SFLOW = enum.auto()     # 预测随机流场 x_T - x_t (FoD推荐)


def get_cosine_schedule(num_timesteps, s=0.008):
    """余弦调度 (用于θ)"""
    steps = num_timesteps + 1
    t = np.linspace(0, num_timesteps, steps) / num_timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def get_linear_schedule(num_timesteps):
    """线性调度 (用于σ)"""
    scale = 1000 / num_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def mean_flat(tensor):
    """对非batch维度取平均"""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """从numpy数组中提取批次索引对应的值"""
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


# ============================================================================
# FoD 扩散核心类
# ============================================================================
class FoDSchedule:
    """
    FoD (Forward-only Diffusion) 调度

    核心SDE: dx_t = θ_t(μ - x_t)dt + σ_t(x_t - μ)dw_t

    解: x_t = (x_s - μ)exp(-∫_s^t(θ_z + σ_z²/2)dz + ∫_s^t σ_z dw_z) + μ

    参数:
        thetas: θ调度 (均值回归速度)
        sigma2s: σ²调度 (扩散强度)
        sigmas_scale: σ²归一化系数
    """
    def __init__(
        self,
        num_timesteps: int = 100,
        theta_schedule: str = 'cosine',
        sigma_schedule: str = 'linear',
        sigmas_scale: float = 1.0,
    ):
        self.num_timesteps = num_timesteps

        # 获取调度
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

        # 在前面添加0（t=0时刻）
        self.thetas = np.append(0.0, thetas)

        # 归一化σ²使其和为1（保证数值稳定性）
        if np.sum(sigma2s) > 0:
            sigma2s = sigmas_scale * sigma2s / np.sum(sigma2s)
        self.sigma2s = np.append(0.0, sigma2s)

        # 计算累积和
        self.thetas_cumsum = np.cumsum(self.thetas)
        self.sigma2s_cumsum = np.cumsum(self.sigma2s)

        # 指数均值项: m̄_t = -∫_0^t(θ_z + σ_z²/2)dz
        expo_mean = -(self.thetas + 0.5 * self.sigma2s)
        expo_mean_cumsum = -(self.thetas_cumsum + 0.5 * self.sigma2s_cumsum)

        # 计算dt使得终态收敛到μ（设定终态偏离为0.001）
        self.dt = math.log(0.001) / expo_mean_cumsum[-1]

        # 缩放后的参数
        self.expo_mean = expo_mean * self.dt
        self.sqrt_expo_variance = np.sqrt(self.sigma2s * self.dt)
        self.expo_mean_cumsum = expo_mean_cumsum * self.dt
        self.sqrt_expo_variance_cumsum = np.sqrt(self.sigma2s_cumsum * self.dt)

    def expo_normal_cumsum(self, t, noise):
        """
        计算从0到t的指数正态变换

        exp(m̄_t + σ̄_t * ε), ε ~ N(0,I)
        """
        # 强制 float32 防止 bf16 溢出 (bf16 max ≈ 65504 ≈ e^11)
        noise_f = noise.float()
        exponent = (
            _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) +
            _extract_into_tensor(self.sqrt_expo_variance_cumsum, t, noise.shape) * noise_f
        )
        return torch.exp(exponent.clamp(-20.0, 20.0))

    def expo_normal_transition(self, s, t, noise):
        """
        计算从s到t的指数正态转移

        exp(m̄_{s:t} + σ̄_{s:t} * ε)
        """
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
        """
        计算t时刻的状态

        x_t = (x_start - x_final) * exp(m̄_t + σ̄_t * ε) + x_final

        Args:
            x_start: 起始状态 (LQ图像)
            x_final: 目标状态 (HQ图像, μ)
            t: 时间步 (整数索引)
            noise: 高斯噪声
        """
        transition = self.expo_normal_cumsum(t, noise)
        return (x_start - x_final) * transition + x_final

    def sde_step(self, x, x_final, t, noise):
        """
        SDE单步更新 (Euler-Maruyama)

        dx = θ_t(μ - x)dt + σ_t(x - μ)dw
        """
        x_f = x.float()
        x_final_f = x_final.float()
        noise_f = noise.float()
        drift = _extract_into_tensor(self.thetas, t, x.shape) * (x_final_f - x_f)
        diffusion = _extract_into_tensor(np.sqrt(self.sigma2s), t, x.shape) * (x_f - x_final_f)
        return x_f + drift * self.dt + diffusion * math.sqrt(self.dt) * noise_f


# ============================================================================
# 时间嵌入
# ============================================================================
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """正弦时间嵌入"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    # 支持整数和浮点时间步
    if timesteps.dtype in [torch.int32, torch.int64]:
        timesteps = timesteps.float()
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ============================================================================
# 基础模块
# ============================================================================
class ResBlock(nn.Module):
    """残差块，带FiLM时间调制"""
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
# SE注意力和复杂度专家
# ============================================================================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力"""
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
    """复杂度专家 - 不同膨胀率处理不同退化"""
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

        # 零初始化
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.down_proj(x)
        h = F.silu(self.norm(self.spatial_conv(h)))
        h = self.se(h)
        return self.up_proj(h)


class EnhancedExpert(nn.Module):
    """
    增强型复杂度专家 — 双层卷积 + SE注意力

    相比 ComplexityExpert：
    - adapter_dim 更大且衰减更平缓 (128→128→96→64)
    - 两层 3×3 卷积（第一层带膨胀，第二层标准）
    - 参数量从 ~0.06M 总计 → ~1.25M 总计
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
        # 所有专家保持同样大小，靠膨胀率区分
        self.adapter_dim = base_adapter_dim

        dilation = 1 + expert_idx
        padding = dilation

        # 下投影
        self.down_proj = nn.Conv2d(in_ch, self.adapter_dim, 1)

        # 第 1 层：膨胀卷积（不同专家捕获不同尺度）
        self.conv1 = nn.Conv2d(
            self.adapter_dim, self.adapter_dim,
            kernel_size=3, padding=padding, dilation=dilation,
        )
        self.norm1 = nn.GroupNorm(min(8, self.adapter_dim), self.adapter_dim)

        # 第 2 层：标准卷积（细化特征）
        self.conv2 = nn.Conv2d(
            self.adapter_dim, self.adapter_dim,
            kernel_size=3, padding=1,
        )
        self.norm2 = nn.GroupNorm(min(8, self.adapter_dim), self.adapter_dim)

        # 通道注意力
        self.se = SEBlock(self.adapter_dim, reduction=4)

        # 上投影（零初始化保持残差）
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
# 瓶颈层通道注意力（Restormer风格）
# ============================================================================
class BottleneckAttention(nn.Module):
    """
    在 U-Net 瓶颈处添加通道自注意力 + FFN。
    使用 Restormer 风格的转置注意力 (C×C 而非 HW×HW)，
    计算量远小于空间注意力，适合高分辨率特征。
    """
    def __init__(self, ch: int, num_heads: int = 4, ffn_expansion: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 注意力分支
        self.norm1 = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.qkv_dw = nn.Conv2d(ch * 3, ch * 3, 3, padding=1, groups=ch * 3)
        self.proj = nn.Conv2d(ch, ch, 1)

        # 前馈分支
        self.norm2 = nn.GroupNorm(8, ch)
        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * ffn_expansion, 1),
            nn.GELU(),
            nn.Conv2d(ch * ffn_expansion, ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # —— 通道注意力 ——
        h = self.norm1(x)
        qkv = self.qkv_dw(self.qkv(h))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # (B, heads, C//heads, C//heads) — 通道维度注意力
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        x = x + self.proj(out)

        # —— FFN ——
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# 频率嵌入模块（借鉴 MoCE-IR）
# ============================================================================
class FrequencyEmbedding(nn.Module):
    """
    从瓶颈特征中提取高频信息，用于引导专家路由。
    使用固定的高通滤波器捕获纹理/边缘频率特征，
    然后通过 MLP 映射到嵌入空间。
    """
    def __init__(self, ch: int, out_dim: int = 128):
        super().__init__()
        # 固定高通滤波核（拉普拉斯算子）
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
# 亮度预增强模块 (内置两阶段)
# ============================================================================
class BrightnessPreEnhancer(nn.Module):
    """
    轻量级亮度预增强模块

    针对低光图像进行预提亮，缩短 Flow Matching 的传输距离。
    设计原则：
    1. 轻量级：仅增加 ~0.5M 参数
    2. 自适应：根据图像亮度自动决定增强强度
    3. 内置：推理时无感，用户只看到单模型

    核心思路：学习一个亮度残差 Δ = f(LQ)，输出 = LQ + gate * Δ
    gate 由图像亮度自动计算，亮图像 gate≈0，暗图像 gate≈1
    """
    def __init__(
        self,
        in_ch: int = 3,
        base_ch: int = 32,
        num_blocks: int = 4,
        brightness_threshold: float = 0.3,  # 低于此亮度才激活
    ):
        super().__init__()
        self.brightness_threshold = brightness_threshold

        # 轻量级编码器-解码器
        self.head = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # 残差块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_ch, base_ch, 3, padding=1),
            ))

        # 输出层
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
            nn.Tanh(),  # 残差范围 [-1, 1]
        )

        # 亮度自适应门控
        self.gate_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # 初始化：让初始输出接近零
        nn.init.zeros_(self.tail[-2].weight)
        nn.init.zeros_(self.tail[-2].bias)

    def compute_brightness(self, x: torch.Tensor) -> torch.Tensor:
        """计算图像亮度 (per-sample)"""
        # x in [-1, 1], convert to [0, 1]
        x_01 = (x + 1) / 2
        # Y = 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * x_01[:, 0] + 0.587 * x_01[:, 1] + 0.114 * x_01[:, 2]
        return luminance.mean(dim=[1, 2])  # (B,)

    def forward(
        self,
        x: torch.Tensor,
        force_enhance: bool = False,
        return_gate: bool = False,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, H, W), 范围 [-1, 1]
            force_enhance: 强制应用增强（训练时使用）
            return_gate: 是否返回门控值
        """
        # 计算亮度自适应门控
        brightness = self.compute_brightness(x)  # (B,)

        # 自适应门控：暗图像 gate 接近 1，亮图像 gate 接近 0
        # gate = sigmoid((threshold - brightness) * scale)
        gate_brightness = torch.sigmoid(
            (self.brightness_threshold - brightness) * 10.0
        ).view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # 学习门控（让网络自己学习何时激活）
        gate_learned = self.gate_net(x).view(-1, 1, 1, 1)  # (B, 1, 1, 1)

        # 组合门控
        gate = gate_brightness * gate_learned

        # 计算亮度残差
        h = self.head(x)
        for block in self.blocks:
            h = h + block(h)
        delta = self.tail(h)

        # 应用增强
        if force_enhance:
            # 训练时：根据门控混合
            out = x + gate * delta
        else:
            # 推理时：同样根据门控混合
            out = x + gate * delta

        out = out.clamp(-1, 1)

        if return_gate:
            return out, gate
        return out


# ============================================================================
# 退化解析器
# ============================================================================
class DegradationParser(nn.Module):
    """退化解析器：估计全局权重w和空间强度图m"""
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
# 共享骨干网络
# ============================================================================
class SharedBackbone(nn.Module):
    """共享骨干网络（U-Net结构）"""
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

        # Middle
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
            skip = skips[-(i+1)]
            if h.shape[-2:] != skip.shape[-2:]:
                h = h[..., :skip.shape[-2], :skip.shape[-1]]
            h = torch.cat([h, skip], dim=1)
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
# FoD增广流匹配网络
# ============================================================================
class FoDAugmentedFlowNet(nn.Module):
    """
    FoD增广流匹配网络

    融合FoD的前向扩散过程和增广状态空间：
    - 使用FoD的SDE前向过程
    - 保留复杂度专家和退化解析
    - 预测随机流场 f_φ(x_t, t) ≈ μ - x_t
    - 内置亮度预增强模块（低光图像自动预提亮）
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

        # FoD调度
        self.schedule = FoDSchedule(
            num_timesteps=num_timesteps,
            theta_schedule='cosine',
            sigma_schedule='linear',
            sigmas_scale=1.0,
        )

        # 亮度预增强模块（内置两阶段）
        if use_brightness_enhancer:
            self.brightness_enhancer = BrightnessPreEnhancer(
                in_ch=3,
                base_ch=brightness_enhancer_ch,
                num_blocks=brightness_enhancer_blocks,
                brightness_threshold=brightness_threshold,
            )
        else:
            self.brightness_enhancer = None

        # 退化解析器
        self.parser = DegradationParser(
            in_ch=3, base_ch=32, emb_dim=128,
            num_factors=num_experts,
        )

        # 主骨干网络
        self.backbone = SharedBackbone(
            in_ch=in_ch, out_ch=out_ch,
            base_ch=base_ch, ch_mult=ch_mult, emb_dim=emb_dim,
            freq_emb_dim=freq_emb_dim,
        )

        # 增强型复杂度专家
        self.experts = nn.ModuleList([
            EnhancedExpert(
                in_ch=base_ch, out_ch=out_ch,
                expert_idx=i, num_experts=num_experts,
                base_adapter_dim=adapter_dim,
            )
            for i in range(num_experts)
        ])

        # 复杂度偏置
        expert_params = [e.num_params for e in self.experts]
        max_params = max(expert_params)
        self.register_buffer(
            'complexity_bias',
            torch.tensor([p / max_params for p in expert_params])
        )

        # 门控MLP（加入频率嵌入）
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
        """计算退化感知门控权重（融合频率嵌入）"""
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
        """
        对输入进行亮度预增强（如果启用）

        Args:
            x: 输入图像 (B, 3, H, W), 范围 [-1, 1]
            force_enhance: 强制增强（忽略亮度门控）
        """
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
        前向传播

        Args:
            x_t: (B, 3, H, W) 当前状态
            t: (B,) 时间步索引
            x_cond: (B, 3, H, W) 条件图像 (LQ)
            skip_pre_enhance: 跳过预增强（用于训练时已预增强的情况）

        Returns:
            output: 根据model_type返回不同预测
        """
        # 亮度预增强（内置两阶段）
        if self.brightness_enhancer is not None and not skip_pre_enhance:
            x_cond = self.brightness_enhancer(x_cond, force_enhance=self.training)

        # 解析退化信息（在预增强后的图像上）
        if w is None or m is None:
            w_pred, m_pred, _ = self.parser(x_cond)
            w = w if w is not None else w_pred
            m = m if m is not None else m_pred

        # 骨干网络
        v_share, h_final, temb, freq_emb = self.backbone(
            x_t, x_cond, t, return_features=True
        )

        # 门控权重（融合频率嵌入）
        alpha = self.compute_alpha(temb, w, m, freq_emb)

        # 因子化输出：out = v_share + Σ α_i * m_i ⊙ Δv_i
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
# 创建模型
# ============================================================================
def create_fod_model(
    base_ch: int = 64,
    ch_mult: Tuple[int, ...] = (1, 2, 4, 4),
    emb_dim: int = 256,
    adapter_dim: int = 128,
    num_timesteps: int = 100,
    model_type: str = 'SFLOW',
    num_experts: int = 4,
    use_brightness_enhancer: bool = True,
    brightness_enhancer_ch: int = 64,
    brightness_enhancer_blocks: int = 6,
    brightness_threshold: float = 0.3,
    freq_emb_dim: int = 128,
) -> FoDAugmentedFlowNet:
    """创建FoD增广流匹配网络"""
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
# FoD训练损失
# ============================================================================
def _pixel_loss(diff: torch.Tensor, loss_type: str = "l1",
                charbonnier_eps: float = 1e-3) -> torch.Tensor:
    """计算像素级损失 (per-sample, 对非batch维度取平均)"""
    if loss_type == "charbonnier":
        return mean_flat(torch.sqrt(diff ** 2 + charbonnier_eps ** 2))
    else:  # l1
        return mean_flat(torch.abs(diff))


def fod_training_loss(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,  # LQ图像 (源分布)
    x_final: torch.Tensor,  # HQ图像 (目标分布, μ)
    t: torch.Tensor,        # 时间步索引
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
    # v3 新增
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
    lambda_freq: float = 0.0,
    # v4 新增：预增强模块损失
    lambda_enhance: float = 0.1,
    is_lowlight: Optional[torch.Tensor] = None,  # (B,) bool tensor
    # v5 新增：低光显式加权
    lowlight_boost: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    FoD训练损失 (随机流匹配)

    核心公式：
    - x_t = (x_start - x_final) * exp(m̄_t + σ̄_t * ε) + x_final
    - 目标根据model_type:
        - FINAL_X: target = x_final
        - FLOW: target = x_final - x_start
        - SFLOW: target = x_final - x_t (推荐)
    - 损失: L = E[|target - model_output|]

    v3新增：
    - loss_type: "l1" 或 "charbonnier" (对PSNR更友好)
    - lambda_freq: 频域损失权重 (0=禁用)

    v4新增：
    - lambda_enhance: 预增强模块损失权重
    - is_lowlight: 标记哪些样本是低光图像
    """
    schedule = model.schedule
    device = x_start.device
    B = x_start.shape[0]

    if noise is None:
        noise = torch.randn_like(x_final)

    losses = {}
    total_loss = torch.tensor(0.0, device=device)

    # =========== 预增强模块损失 ===========
    if model.brightness_enhancer is not None and lambda_enhance > 0:
        # 对低光图像，预增强后应该接近 GT 的亮度
        x_enhanced, enhance_gate = model.brightness_enhancer(
            x_start, force_enhance=True, return_gate=True
        )

        # 亮度对齐损失：预增强后的亮度应该接近 GT 的亮度
        # 只对低光样本计算
        if is_lowlight is not None:
            lowlight_mask = is_lowlight.float().view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        else:
            # 自动检测低光样本
            brightness = model.brightness_enhancer.compute_brightness(x_start)  # (B,)
            lowlight_mask = (brightness < model.brightness_enhancer.brightness_threshold).float()
            lowlight_mask = lowlight_mask.view(-1, 1, 1, 1)

        # === 亮度聚焦损失（v5改进）===
        # 核心思路：enhancer 只有 ~100K 参数，不可能做全像素重建
        # 只监督亮度通道（Y），让它专注于把暗图提亮，不要分心搞纹理
        lum_w = torch.tensor([0.299, 0.587, 0.114], device=device,
                             dtype=x_enhanced.dtype).view(1, 3, 1, 1)
        enhanced_lum = (x_enhanced * lum_w).sum(dim=1, keepdim=True)  # (B,1,H,W)
        gt_lum = (x_final * lum_w).sum(dim=1, keepdim=True)

        # 亮度L1（主损失，权重 0.7）
        lum_diff = (enhanced_lum - gt_lum) * lowlight_mask
        loss_enhance_lum = mean_flat(torch.abs(lum_diff)).mean()

        # RGB L1（辅助，权重 0.3，保持颜色不偏移太远）
        rgb_diff = (x_enhanced - x_final) * lowlight_mask
        loss_enhance_rgb = _pixel_loss(rgb_diff, loss_type, charbonnier_eps).mean()

        loss_enhance = 0.7 * loss_enhance_lum + 0.3 * loss_enhance_rgb

        # 门控正则化：鼓励低光图像使用预增强
        gate_reg = (lowlight_mask.squeeze() * (1 - enhance_gate.squeeze())).mean()

        # 传输距离缩减奖励：enhancer 后离 GT 应该比 enhancer 前更近
        with torch.no_grad():
            dist_before = mean_flat(torch.abs(x_start - x_final)).mean()
            dist_after = mean_flat(torch.abs(x_enhanced - x_final)).mean()
            # 如果 enhancer 反而让距离变远了，额外惩罚
            dist_penalty_active = (dist_after > dist_before).float()
        loss_dist_penalty = mean_flat(torch.abs(rgb_diff)).mean() * dist_penalty_active

        loss_enhance_total = loss_enhance + 0.1 * gate_reg + 0.2 * loss_dist_penalty
        total_loss = total_loss + lambda_enhance * loss_enhance_total
        losses['enhance'] = loss_enhance.item()
        losses['enhance_gate'] = enhance_gate.mean().item()
        losses['enhance_lum'] = loss_enhance_lum.item()

        # 使用预增强后的图像作为 x_cond
        x_cond = x_enhanced
    else:
        x_cond = x_start

    # =========== 生成中间状态 x_t ===========
    # 使用预增强后的 x_cond 作为源分布
    x_t = schedule.get_xt(x_cond, x_final, t, noise)

    # 解析退化信息（在预增强后的图像上）
    w_pred, m_pred, logits = model.parser(x_cond)

    # 模型预测（跳过预增强，因为已经在上面做过了）
    model_output, alpha = model(
        x_t, t, x_cond, w=w_pred, m=m_pred,
        return_alpha=True, skip_pre_enhance=True
    )

    # 计算目标
    if model.model_type == ModelType.FINAL_X:
        target = x_final
    elif model.model_type == ModelType.FLOW:
        target = x_final - x_cond
    elif model.model_type == ModelType.SFLOW:
        target = x_final - x_t

    # 用于重建损失的 x_final 预测（从模型输出还原）
    if model.model_type == ModelType.FINAL_X:
        x_final_pred = model_output
    elif model.model_type == ModelType.FLOW:
        x_final_pred = x_cond + model_output
    elif model.model_type == ModelType.SFLOW:
        x_final_pred = x_t + model_output

    # 主损失，支持L1/Charbonnier + 困难样本加权 + 低光显式加权
    diff = target - model_output
    loss_per_sample = _pixel_loss(diff, loss_type, charbonnier_eps)  # (B,)
    sample_weights = torch.ones(B, device=device)

    # 困难样本加权
    if use_hard_sample_weighting:
        with torch.no_grad():
            mse_ps = mean_flat((x_final_pred - x_final) ** 2)  # (B,)
            psnr_ps = 10.0 * torch.log10(4.0 / (mse_ps + 1e-8))  # data_range=2 → max_val²=4
            hard_mask = psnr_ps < hard_sample_psnr_threshold
            sample_weights[hard_mask] = hard_sample_weight_factor
        losses['hard_ratio'] = hard_mask.float().mean().item()

    # 低光样本显式加权（与困难加权叠加）
    if is_lowlight is not None and lowlight_boost > 1.0:
        sample_weights[is_lowlight] = sample_weights[is_lowlight] * lowlight_boost
        losses['lowlight_ratio'] = is_lowlight.float().mean().item()

    # 归一化保持总量不变，再加权
    sample_weights = sample_weights / (sample_weights.mean() + 1e-8)
    loss_main = (loss_per_sample * sample_weights).mean()

    losses['main'] = loss_main.item()
    total_loss = total_loss + loss_main

    # 重建损失
    if lambda_recon > 0:
        recon_diff = x_final_pred - x_final
        loss_recon = _pixel_loss(recon_diff, loss_type, charbonnier_eps).mean()
        total_loss = total_loss + lambda_recon * loss_recon
        losses['recon'] = loss_recon.item()

    # 频域损失（MoCE-IR 风格：分别约束 real/imag，对相位更敏感）
    if lambda_freq > 0:
        pred_fft = torch.fft.rfft2(x_final_pred.float())
        gt_fft = torch.fft.rfft2(x_final.float())
        pred_ri = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        gt_ri = torch.stack([gt_fft.real, gt_fft.imag], dim=-1)
        loss_freq = F.l1_loss(pred_ri, gt_ri)
        total_loss = total_loss + lambda_freq * loss_freq
        losses['freq'] = loss_freq.item()

    # w 监督损失
    if lambda_w > 0:
        w_target = present if present is not None else labels
        if w_target is not None:
            with torch.autocast(device_type="cuda", enabled=False):
                loss_w = F.binary_cross_entropy(w_pred.float(), w_target.float())
            total_loss = total_loss + lambda_w * loss_w
            losses['w'] = loss_w.item()

    # m 监督损失
    if lambda_m > 0 and m_gt is not None:
        if m_gt.shape != m_pred.shape:
            m_gt = F.interpolate(m_gt, size=m_pred.shape[2:], mode='bilinear', align_corners=False)
        with torch.autocast(device_type="cuda", enabled=False):
            loss_m = F.binary_cross_entropy(m_pred.float(), m_gt.float())
        total_loss = total_loss + lambda_m * loss_m
        losses['m'] = loss_m.item()

    # 分类损失
    if lambda_cls > 0 and labels is not None:
        loss_cls = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss = total_loss + lambda_cls * loss_cls
        losses['cls'] = loss_cls.item()

    # 专家平衡损失
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

    # 门控时间先验监督
    if lambda_alpha > 0 and alpha_target is not None:
        loss_alpha = F.mse_loss(alpha, alpha_target)
        total_loss = total_loss + lambda_alpha * loss_alpha
        losses['alpha'] = loss_alpha.item()

    return total_loss, losses


# ============================================================================
# FoD推理 (前向采样)
# ============================================================================
@torch.no_grad()
def fod_inference(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,  # LQ图像
    num_steps: int = -1,
    sample_type: str = "MC",  # EM, MC, NMC
    clip_denoised: bool = True,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    FoD推理：从LQ前向采样到HQ

    采样策略:
    - EM: Euler-Maruyama (标准SDE求解)
    - MC: Markov Chain (x_t -> x_{t+k} 转移)
    - NMC: Non-Markov Chain (x_0 -> x_{t+k} 转移, 推荐)

    Args:
        model: FoD模型
        x_start: LQ图像 (B, 3, H, W)
        num_steps: 采样步数 (-1表示使用全部)
        sample_type: 采样策略
        clip_denoised: 是否裁剪到[-1, 1]
        use_tta: 是否使用测试时增强
    """
    model.eval()
    schedule = model.schedule

    if not use_tta:
        return _fod_inference_single(model, x_start, num_steps, sample_type, clip_denoised)

    # TTA: 几何自集成
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
    """单次FoD推理"""
    schedule = model.schedule
    device = x_start.device
    B = x_start.shape[0]

    if num_steps <= 0:
        num_steps = schedule.num_timesteps

    # 亮度预增强（内置两阶段，推理时自动激活）
    if model.brightness_enhancer is not None:
        x_cond = model.brightness_enhancer(x_start, force_enhance=False)
    else:
        x_cond = x_start

    # 初始状态（从预增强后的图像开始）
    img = x_cond.clone()

    # 时间步索引
    indices = np.linspace(0, schedule.num_timesteps, num_steps + 1).astype(int)
    times = np.copy(indices)

    # 获取退化信息 (在预增强后的图像上计算，只计算一次)
    w, m, _ = model.parser(x_cond)

    for i, idx in enumerate(indices[:-1]):
        t = torch.tensor([idx] * B, device=device)
        t_next = torch.tensor([times[i+1]] * B, device=device)

        # 模型预测（跳过预增强，因为已经在外面做过了）
        model_output = model(img, t, x_cond, w=w, m=m, skip_pre_enhance=True)

        # 根据预测类型计算x_final
        if model.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif model.model_type == ModelType.FLOW:
            x_final = x_cond + model_output
        elif model.model_type == ModelType.SFLOW:
            x_final = img + model_output

        if clip_denoised:
            x_final = x_final.clamp(-1, 1)

        # 采样下一状态 (schedule 计算在 float32 中进行，防止 bf16 溢出)
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
        # 每步 clamp 防止数值漂移
        img = img.clamp(-2, 2)

    return img.clamp(-1, 1)


# ============================================================================
# 快速单步推理
# ============================================================================
@torch.no_grad()
def fod_one_step_inference(
    model: FoDAugmentedFlowNet,
    x_start: torch.Tensor,
    use_tta: bool = False,
) -> torch.Tensor:
    """
    单步快速推理

    使用最后一个时间步直接预测x_final
    """
    model.eval()
    device = x_start.device
    B = x_start.shape[0]

    if not use_tta:
        # 亮度预增强
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

    # TTA
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

        # 亮度预增强
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
