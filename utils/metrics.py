# utils/metrics.py
# 图像质量评估指标：PSNR、SSIM、MS-SSIM

import torch
import torch.nn.functional as F
import numpy as np

try:
    from pytorch_msssim import ms_ssim, ssim
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False
    print("Warning: pytorch_msssim not installed. Install with: pip install pytorch-msssim")


def psnr_y_torch(img1, img2, data_range=2.0, per_sample=False):
    """
    计算 PSNR（在 Y 通道，即灰度图）。
    
    Args:
        img1: (B, 3, H, W) 或 (B, 1, H, W) tensor，值域 [-1, 1]
        img2: (B, 3, H, W) 或 (B, 1, H, W) tensor，值域 [-1, 1]
        data_range: 数据范围（默认 2.0，因为值域是 [-1, 1]）
        per_sample: 如果为True，返回每个样本的PSNR (B,) tensor；否则返回平均PSNR标量
    
    Returns:
        psnr: 标量或 (B,) tensor
    """
    # 转换为 Y 通道（灰度）
    if img1.shape[1] == 3:
        # RGB to Y: Y = 0.299*R + 0.587*G + 0.114*B
        weights = torch.tensor([0.299, 0.587, 0.114], device=img1.device, dtype=img1.dtype)
        weights = weights.view(1, 3, 1, 1)
        img1_y = (img1 * weights).sum(dim=1, keepdim=True)
        img2_y = (img2 * weights).sum(dim=1, keepdim=True)
    else:
        img1_y = img1
        img2_y = img2
    
    # 计算 MSE
    mse = torch.mean((img1_y - img2_y) ** 2, dim=[1, 2, 3])
    
    # 计算 PSNR
    psnr = 20 * torch.log10(data_range / (torch.sqrt(mse) + 1e-10))
    
    if per_sample:
        return psnr
    else:
        return psnr.mean()


def psnr_rgb_torch(img1, img2, data_range=2.0, per_sample=False):
    """
    计算 PSNR（在 RGB 全图）。
    
    Args:
        img1: (B, 3, H, W) 或 (B, 1, H, W) tensor，值域 [-1, 1]
        img2: (B, 3, H, W) 或 (B, 1, H, W) tensor，值域 [-1, 1]
        data_range: 数据范围（默认 2.0，因为值域是 [-1, 1]）
        per_sample: 如果为True，返回每个样本的PSNR (B,) tensor；否则返回平均PSNR标量
    
    Returns:
        psnr: 标量或 (B,) tensor
    """
    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(data_range / (torch.sqrt(mse) + 1e-10))
    if per_sample:
        return psnr
    return psnr.mean()


def ssim_torch_eval(img1, img2, data_range=2.0, size_average=True):
    """
    计算 SSIM。
    
    Args:
        img1: (B, 3, H, W) tensor，值域 [-1, 1]
        img2: (B, 3, H, W) tensor，值域 [-1, 1]
        data_range: 数据范围
        size_average: 是否平均
    
    Returns:
        ssim: 标量或 (B,) tensor
    """
    if not MSSSIM_AVAILABLE:
        # 简化版 SSIM（如果 pytorch_msssim 不可用）
        return _simple_ssim(img1, img2, data_range, size_average)
    
    # 归一化到 [0, 1]
    img1_norm = (img1 + 1.0) / 2.0
    img2_norm = (img2 + 1.0) / 2.0
    
    # 使用 pytorch_msssim
    ssim_val = ssim(img1_norm, img2_norm, data_range=1.0, size_average=size_average)
    return ssim_val


def _simple_ssim(img1, img2, data_range=2.0, size_average=True):
    """简化版 SSIM（如果 pytorch_msssim 不可用）"""
    # 简化实现：只计算结构相似性
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    
    sigma1_sq = ((img1 - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = ((img2 - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-10
    )
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def ms_ssim_loss(img1, img2, data_range=2.0, size_average=True):
    """
    计算 MS-SSIM loss（1 - MS-SSIM）。
    
    Args:
        img1: (B, 3, H, W) tensor
        img2: (B, 3, H, W) tensor
        data_range: 数据范围
        size_average: 是否平均
    
    Returns:
        loss: 标量（保证非负）
    """
    if not MSSSIM_AVAILABLE:
        # 如果不可用，使用简化版
        ssim_val = _simple_ssim(img1, img2, data_range, size_average)
        # 确保SSIM值在[0, 1]范围内，避免负数loss
        ssim_val = torch.clamp(ssim_val, 0.0, 1.0)
        return 1.0 - ssim_val
    
    # 归一化到 [0, 1]
    img1_norm = (img1 + 1.0) / 2.0
    img2_norm = (img2 + 1.0) / 2.0
    
    # 计算 MS-SSIM
    ms_ssim_val = ms_ssim(img1_norm, img2_norm, data_range=1.0, size_average=size_average)
    
    # 确保MS-SSIM值在[0, 1]范围内，避免负数loss（由于数值误差可能导致>1）
    ms_ssim_val = torch.clamp(ms_ssim_val, 0.0, 1.0)
    
    # 返回 loss（1 - MS-SSIM），保证非负
    return torch.clamp(1.0 - ms_ssim_val, min=0.0)
