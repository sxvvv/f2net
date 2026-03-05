# utils/misc.py
# 杂项工具函数

import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """设置随机种子，确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_id=None):
    """
    获取设备（GPU 或 CPU）。
    
    Args:
        device_id: GPU ID（int）或 None（自动选择）
    
    Returns:
        torch.device 对象
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        else:
            return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def makedirs(path):
    """创建目录（如果不存在）"""
    os.makedirs(path, exist_ok=True)

