# train_v3.py
# 基于FoD的增广流匹配图像恢复训练脚本 (v3: 全策略版)
#
# v3 核心策略：
#   1. 渐进式 patch size：384→512→720，前期快速收敛，后期精细优化
#   2. 退化感知动态加权：根据验证集 per-deg PSNR 自动调高弱项 loss 权重
#   3. 验证集驱动：best model 自动保存 + early stopping + 完整 per-deg 评估
#   4. Charbonnier loss + 频域损失：对 PSNR 更友好
#   5. finetune 模式：只加载权重，重置 optimizer/scheduler
#
# 从头训练示例 (推荐):
#   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_v3.py \
#       --lmdb-path /data/train.lmdb --test-lmdb-path /data/test.lmdb \
#       --progressive-patch "0:384:16,150000:512:12,350000:720:8" \
#       --loss-type charbonnier --lambda-freq 0.05 \
#       --deg-reweight --deg-reweight-target 29.0 \
#       --niter 600000 --patience 20 --eval-every 5000

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import argparse
import logging
import os
import math
import numpy as np
import pickle
from time import time
from collections import defaultdict

from models.fod_cfm_net import (
    create_fod_model, fod_training_loss, fod_inference, fod_one_step_inference
)
from utils.lmdb_dataset import LMDBAllWeatherDataset
from utils.factor_utils import parse_factors, FACTOR2IDX
from utils.metrics import psnr_y_torch, ssim_torch_eval as ssim_torch
from utils.ema import EMA


# ============================================================================
# 工具函数
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-6):
    """带Warmup的余弦退火调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(eta_min / optimizer.defaults['lr'],
                   0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_logging(rank: int, log_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if rank == 0 and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(f"{log_dir}/train_fod.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(console)
    return logger


def deg_name_to_labels(deg_names: list, device) -> torch.Tensor:
    B = len(deg_names)
    labels = torch.zeros(B, 4, device=device)
    for i, name in enumerate(deg_names):
        if name:
            for factor in parse_factors(name):
                if factor in FACTOR2IDX:
                    labels[i, FACTOR2IDX[factor]] = 1.0
    return labels


def gating_target_from_labels(labels, t_norm):
    B = labels.shape[0]
    device, dtype = labels.device, labels.dtype
    mod_early = torch.tensor([0.3, 0.3, 1.0, 1.0], device=device, dtype=dtype)
    mod_mid   = torch.tensor([0.3, 1.0, 0.5, 0.5], device=device, dtype=dtype)
    mod_late  = torch.tensor([1.0, 0.5, 0.3, 0.3], device=device, dtype=dtype)
    tt = t_norm.view(B, 1)
    time_mod = torch.where(
        tt < 0.33, mod_early.view(1, 4),
        torch.where(tt < 0.67, mod_mid.view(1, 4), mod_late.view(1, 4))
    )
    alpha = labels * time_mod
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
    return alpha


# ============================================================================
# 动态因子映射 (支持自定义退化任务, 如 3-task 去噪/去雾/去雨)
# ============================================================================

# 退化名称 -> 因子的子串匹配规则
# 键为子串 (小写匹配), 值为因子名
_DEG_SUBSTRING_MAP = {
    "noise": "noise", "denois": "noise", "gaussian": "noise", "sigma": "noise",
    "haze": "haze", "dehaz": "haze", "fog": "haze",
    "rain": "rain", "derain": "rain",
    "snow": "snow", "desnow": "snow",
    "low": "low", "dark": "low", "lowlight": "low",
    "blur": "blur", "deblur": "blur",
    "jpeg": "jpeg", "compress": "jpeg",
}


def map_deg_name_to_factor_idx(deg_name, factors):
    """
    将 deg_name 映射到 factors 列表中的索引。
    使用子串匹配，支持多种命名格式。

    Args:
        deg_name: 退化名称字符串, 如 "denoise_25", "dehaze", "rain100L"
        factors: 因子列表, 如 ["noise", "haze", "rain"]

    Returns:
        int: 因子索引, 未匹配返回 -1
    """
    if not deg_name:
        return -1
    deg_lower = deg_name.lower().strip()

    # 先尝试精确匹配因子名
    for idx, factor in enumerate(factors):
        if factor.lower() == deg_lower:
            return idx

    # 通过子串规则映射到因子名，再查索引
    factor_name_lower = {f.lower(): i for i, f in enumerate(factors)}
    for substr, mapped_factor in _DEG_SUBSTRING_MAP.items():
        if substr in deg_lower:
            if mapped_factor.lower() in factor_name_lower:
                return factor_name_lower[mapped_factor.lower()]

    # 最后尝试: 因子名是否为 deg_name 的子串
    for idx, factor in enumerate(factors):
        if factor.lower() in deg_lower:
            return idx

    return -1


def deg_name_to_labels_dynamic(deg_names, device, factors, num_factors):
    """
    动态版 deg_name_to_labels，支持任意因子列表。
    每个样本根据其 deg_name 映射到对应因子位置设为 1。

    Returns:
        (B, num_factors) tensor
    """
    B = len(deg_names)
    labels = torch.zeros(B, num_factors, device=device)
    for i, name in enumerate(deg_names):
        if name:
            idx = map_deg_name_to_factor_idx(name, factors)
            if 0 <= idx < num_factors:
                labels[i, idx] = 1.0
    return labels


def gating_target_from_labels_dynamic(labels, t_norm, num_factors):
    """
    动态版 gating_target，根据任务数量生成时间依赖的门控先验。

    对于 3-task (noise, haze, rain):
    - 早期 (t<0.33): 强调结构恢复 (haze, rain 权重高)
    - 中期 (0.33<t<0.67): 均衡
    - 后期 (t>0.67): 强调细节恢复 (noise 权重高)
    """
    B = labels.shape[0]
    device, dtype = labels.device, labels.dtype

    if num_factors == 3:
        # [noise, haze, rain]
        mod_early = torch.tensor([0.3, 1.0, 1.0], device=device, dtype=dtype)
        mod_mid   = torch.tensor([0.6, 0.8, 0.6], device=device, dtype=dtype)
        mod_late  = torch.tensor([1.0, 0.5, 0.3], device=device, dtype=dtype)
    elif num_factors == 4:
        mod_early = torch.tensor([0.3, 0.3, 1.0, 1.0], device=device, dtype=dtype)
        mod_mid   = torch.tensor([0.3, 1.0, 0.5, 0.5], device=device, dtype=dtype)
        mod_late  = torch.tensor([1.0, 0.5, 0.3, 0.3], device=device, dtype=dtype)
    else:
        # 通用: 均匀
        mod_early = torch.ones(num_factors, device=device, dtype=dtype)
        mod_mid   = torch.ones(num_factors, device=device, dtype=dtype)
        mod_late  = torch.ones(num_factors, device=device, dtype=dtype)

    tt = t_norm.view(B, 1)
    NF = num_factors
    time_mod = torch.where(
        tt < 0.33, mod_early.view(1, NF),
        torch.where(tt < 0.67, mod_mid.view(1, NF), mod_late.view(1, NF))
    )
    alpha = labels * time_mod
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
    return alpha


def build_dynamic_present_and_m(deg_names, device, factors, num_factors, H, W):
    """
    根据 deg_name 构建动态的 present 和 m_gt tensor。

    Returns:
        present: (B, num_factors) tensor
        m_gt: (B, num_factors, H, W) tensor
    """
    B = len(deg_names)
    present = torch.zeros(B, num_factors, device=device)
    for i, name in enumerate(deg_names):
        if name:
            idx = map_deg_name_to_factor_idx(name, factors)
            if 0 <= idx < num_factors:
                present[i, idx] = 1.0
    # m_gt: 简化版，存在的因子全图为 1
    m_gt = present.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W).contiguous()
    return present, m_gt


def print_deg_name_distribution(dataset, factors, max_scan=500):
    """
    轻量扫描数据集 deg_name 分布（不加载图片数据，仅读 LMDB key/deg_name）。
    """
    from collections import Counter
    deg_counter = Counter()
    mapped = Counter()
    unmapped_set = set()

    # 优先走 LMDB 轻量路径：只读 deg_name 字段，不解码图片
    if hasattr(dataset, '_ensure_env') and hasattr(dataset, 'keys'):
        import pickle as _pkl
        n = min(len(dataset.keys), max_scan)
        try:
            dataset._ensure_env()
            with dataset.env.begin(write=False) as txn:
                for i in range(n):
                    raw = txn.get(dataset.keys[i].encode())
                    if raw is None:
                        continue
                    data = _pkl.loads(raw)
                    dn = str(data.get('deg_name', data.get('degradation', '')))
                    if dn:
                        deg_counter[dn] += 1
                        idx = map_deg_name_to_factor_idx(dn, factors)
                        if idx >= 0:
                            mapped[factors[idx]] += 1
                        else:
                            unmapped_set.add(dn)
        except Exception as e:
            print(f"[DegName] LMDB scan failed: {e}, falling back to dataset[i]")
            deg_counter.clear(); mapped.clear(); unmapped_set.clear()
            n = min(len(dataset), min(max_scan, 50))
            for i in range(n):
                try:
                    sample = dataset[i]
                    dn = sample.get("deg_name", "")
                    if isinstance(dn, str) and dn:
                        deg_counter[dn] += 1
                        idx = map_deg_name_to_factor_idx(dn, factors)
                        if idx >= 0:
                            mapped[factors[idx]] += 1
                        else:
                            unmapped_set.add(dn)
                except Exception:
                    break
    else:
        # 非 LMDB 数据集，少量采样
        n = min(len(dataset), min(max_scan, 50))
        for i in range(n):
            try:
                sample = dataset[i]
                dn = sample.get("deg_name", "")
                if isinstance(dn, str) and dn:
                    deg_counter[dn] += 1
                    idx = map_deg_name_to_factor_idx(dn, factors)
                    if idx >= 0:
                        mapped[factors[idx]] += 1
                    else:
                        unmapped_set.add(dn)
            except Exception:
                break

    print(f"\n[DegName Distribution] Scanned {sum(deg_counter.values())} samples:")
    for dn, cnt in deg_counter.most_common():
        idx = map_deg_name_to_factor_idx(dn, factors)
        tag = f"-> {factors[idx]}" if idx >= 0 else "-> UNMAPPED"
        print(f"  {dn:<30s} count={cnt:>5d}  {tag}")
    if unmapped_set:
        print(f"\n  WARNING: {len(unmapped_set)} unmapped deg_names: {sorted(unmapped_set)[:10]}")
    print(f"  Factor totals: {dict(mapped)}\n")


# ============================================================================
# 均衡采样器
# ============================================================================
def build_or_load_index_cache(train_set, cache_path, rank=0):
    if os.path.isfile(cache_path):
        if rank == 0:
            print(f"[IndexCache] Loading {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    if rank == 0:
        print(f"[IndexCache] Building cache -> {cache_path}")
        low_idx, nonlow_idx = [], []
        for i in range(len(train_set)):
            sample = train_set[i]
            dn = sample.get("deg_name", None)
            if dn is not None and "low" in str(dn).lower():
                low_idx.append(i)
            else:
                nonlow_idx.append(i)
        cache = {"low_indices": low_idx, "nonlow_indices": nonlow_idx}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[IndexCache] low={len(low_idx)} nonlow={len(nonlow_idx)} total={len(train_set)}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    with open(cache_path, "rb") as f:
        return pickle.load(f)


class CachedBalancedSampler(Sampler):
    def __init__(self, low_indices, nonlow_indices,
                 low_ratio, total_size,
                 num_replicas=1, rank=0, shuffle=True, seed=0):
        self.low_indices = list(low_indices)
        self.nonlow_indices = list(nonlow_indices)
        self.low_ratio = float(low_ratio)
        self.total_size = int(total_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        assert self.total_size > 0
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        if len(self.low_indices) == 0:
            self.low_ratio = 0.0
        if len(self.nonlow_indices) == 0:
            self.low_ratio = 1.0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        n_low = int(self.total_size * self.low_ratio)
        n_non = self.total_size - n_low

        def draw(src, count):
            if count == 0 or len(src) == 0:
                return []
            ridx = torch.randint(0, len(src), (count,), generator=g).tolist()
            return [src[j] for j in ridx]

        low_part = draw(self.low_indices, n_low)
        non_part = draw(self.nonlow_indices, n_non)
        indices = low_part + non_part
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class RepeatDataset(torch.utils.data.Dataset):
    """Repeat a dataset N times (virtual expansion)."""
    def __init__(self, dataset, repeats: int):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


# ============================================================================
# 渐进式 patch size 解析
# ============================================================================
def parse_progressive_patch(spec_str):
    """
    解析渐进式 patch size 配置。
    格式: "step1:patch1:bs1,step2:patch2:bs2,..."
    例如: "0:384:16,150000:512:12,350000:720:8"
    返回: [(0, 384, 16), (150000, 512, 12), (350000, 720, 8)]
    """
    if not spec_str:
        return []
    stages = []
    for part in spec_str.split(","):
        tokens = part.strip().split(":")
        if len(tokens) == 3:
            stages.append((int(tokens[0]), int(tokens[1]), int(tokens[2])))
        elif len(tokens) == 2:
            stages.append((int(tokens[0]), int(tokens[1]), -1))  # -1 = 不改 bs
    stages.sort(key=lambda x: x[0])
    return stages


def get_current_patch_config(step, stages):
    """根据当前 step 返回 (patch_size, batch_size) 或 None"""
    if not stages:
        return None, None
    result = stages[0]
    for s in stages:
        if step >= s[0]:
            result = s
    return result[1], result[2]


# ============================================================================
# 退化感知动态加权
# ============================================================================
class DegradationReweighter:
    """
    根据验证集 per-degradation PSNR 动态调整每类退化的 loss 权重。
    PSNR 越低的退化类型，训练时 loss 权重越高。
    """
    def __init__(self, target_psnr=29.0, max_weight=4.0, min_weight=0.5, momentum=0.8):
        self.target_psnr = target_psnr
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.momentum = momentum
        self.deg_weights = {}  # deg_name -> weight

    def update(self, per_deg_results):
        """
        per_deg_results: dict of {deg_name: {"psnr": float, "ssim": float, "cnt": int}}
        """
        if not per_deg_results:
            return

        # 计算每类退化距离目标的 gap
        gaps = {}
        for deg, info in per_deg_results.items():
            gap = max(0, self.target_psnr - info["psnr"])
            gaps[deg] = gap

        max_gap = max(gaps.values()) if gaps else 1.0
        if max_gap < 0.1:
            max_gap = 1.0

        # gap 越大 -> weight 越高
        new_weights = {}
        for deg, gap in gaps.items():
            # 线性映射: gap=0 -> min_weight, gap=max_gap -> max_weight
            w = self.min_weight + (self.max_weight - self.min_weight) * (gap / max_gap)
            new_weights[deg] = w

        # EMA 平滑
        for deg, w in new_weights.items():
            if deg in self.deg_weights:
                self.deg_weights[deg] = self.momentum * self.deg_weights[deg] + (1 - self.momentum) * w
            else:
                self.deg_weights[deg] = w

    def get_sample_weights(self, deg_names, device):
        """
        根据 batch 中每个样本的退化类型返回权重 tensor (B,)
        """
        B = len(deg_names)
        weights = torch.ones(B, device=device)
        for i, name in enumerate(deg_names):
            if isinstance(name, str) and name in self.deg_weights:
                weights[i] = self.deg_weights[name]
        # 归一化使均值为 1
        weights = weights / (weights.mean() + 1e-8)
        return weights

    def get_status_str(self):
        if not self.deg_weights:
            return "no weights yet"
        parts = []
        for deg in sorted(self.deg_weights.keys()):
            parts.append(f"{deg}={self.deg_weights[deg]:.2f}")
        return " ".join(parts)


# ============================================================================
# 完整验证
# ============================================================================
@torch.no_grad()
def evaluate_full(model, loader, device, ema=None, max_samples=0,
                  num_steps=10, sample_type="MC", use_tta=False,
                  use_amp=False, amp_dtype=torch.bfloat16,
                  one_step=False):
    """
    评估。one_step=True 时用单步推理，速度快 ~10x。
    max_samples>0 时只评估前 N 个 batch。
    """
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    per_deg = defaultdict(lambda: {"psnr": 0.0, "ssim": 0.0, "cnt": 0})

    ctx = ema.average_parameters(model) if ema else None
    if ctx:
        ctx.__enter__()

    try:
        net = model.module if hasattr(model, 'module') else model
        for i, batch in enumerate(loader):
            if max_samples > 0 and i >= max_samples:
                break
            y = batch["LQ"].to(device)
            x = batch["GT"].to(device)
            deg_names = batch.get("deg_name", [None] * y.shape[0])

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    if one_step:
                        x_hat = fod_one_step_inference(net, y, use_tta=use_tta)
                    else:
                        x_hat = fod_inference(net, y, num_steps=num_steps,
                                              sample_type=sample_type, use_tta=use_tta)
            else:
                if one_step:
                    x_hat = fod_one_step_inference(net, y, use_tta=use_tta)
                else:
                    x_hat = fod_inference(net, y, num_steps=num_steps,
                                          sample_type=sample_type, use_tta=use_tta)

            # 指标计算强制 float32，防止 bf16 精度不足导致 PSNR 异常
            psnr_val = psnr_y_torch(x_hat.float(), x.float(), data_range=2.0)
            ssim_val = ssim_torch(x_hat.float(), x.float(), data_range=2.0)

            batch_size = y.shape[0]
            psnr_sum += psnr_val.item() * batch_size
            ssim_sum += ssim_val.item() * batch_size
            n += batch_size

            for j in range(batch_size):
                deg = deg_names[j] if isinstance(deg_names[j], str) else "unknown"
                if batch_size == 1:
                    per_deg[deg]["psnr"] += psnr_val.item()
                    per_deg[deg]["ssim"] += ssim_val.item()
                else:
                    per_deg[deg]["psnr"] += psnr_val[j].item() if psnr_val.dim() > 0 else psnr_val.item()
                    per_deg[deg]["ssim"] += ssim_val[j].item() if ssim_val.dim() > 0 else ssim_val.item()
                per_deg[deg]["cnt"] += 1
    finally:
        if ctx:
            ctx.__exit__(None, None, None)

    model.train()
    avg_psnr = psnr_sum / max(1, n)
    avg_ssim = ssim_sum / max(1, n)

    per_deg_avg = {}
    for deg, stats in per_deg.items():
        c = stats["cnt"]
        if c > 0:
            per_deg_avg[deg] = {"psnr": stats["psnr"] / c, "ssim": stats["ssim"] / c, "cnt": c}
    return avg_psnr, avg_ssim, per_deg_avg


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FoD增广流匹配训练 (v3: 全策略版)")

    # 数据
    parser.add_argument("--lmdb-path", type=str, required=True)
    parser.add_argument("--test-lmdb-path", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=720)
    parser.add_argument("--num-workers", type=int, default=8)

    # 训练
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="梯度累积步数 (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--niter", type=int, default=600_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)

    # 模型
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--model-type", type=str, default="SFLOW",
                        choices=["FINAL_X", "FLOW", "SFLOW"])

    # 动态因子 / 多任务
    parser.add_argument("--num-experts", type=int, default=4,
                        help="专家数量 (3-task: 3, CDD-11: 4)")
    parser.add_argument("--factors", type=str, default="",
                        help="逗号分隔的因子列表, 如 'noise,haze,rain'; 空则使用默认 CDD-11 4因子")

    # 损失权重
    parser.add_argument("--lambda-cls", type=float, default=0.01)
    parser.add_argument("--lambda-balance", type=float, default=0.01)
    parser.add_argument("--lambda-w", type=float, default=0.1)
    parser.add_argument("--lambda-m", type=float, default=0.05)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-alpha", type=float, default=0.02)

    # 困难样本加权
    parser.add_argument("--use-hard-sample-weighting", action="store_true", default=False)
    parser.add_argument("--hard-sample-psnr-threshold", type=float, default=26.0)
    parser.add_argument("--hard-sample-weight-factor", type=float, default=2.0)

    # 推理设置
    parser.add_argument("--eval-num-steps", type=int, default=10)
    parser.add_argument("--sample-type", type=str, default="MC", choices=["EM", "MC", "NMC"])

    # Warmup
    parser.add_argument("--warmup-steps", type=int, default=5000)

    # ====== v3 新增 ======
    # 微调
    parser.add_argument("--finetune", action="store_true", default=False,
                        help="微调模式：只加载权重+EMA，重置optimizer/scheduler")

    # best model + early stopping
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping: 连续N次评估不涨则停止 (0=禁用)")
    parser.add_argument("--eval-max-samples", type=int, default=0,
                        help="评估时最多样本数 (0=全部)")
    parser.add_argument("--eval-one-step", action="store_true", default=False,
                        help="评估时用单步推理 (快~10x，趋势一致，推荐训练中使用)")

    # Charbonnier + 频域
    parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "charbonnier"])
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-freq", type=float, default=0.0)

    # 渐进式 patch size
    parser.add_argument("--progressive-patch", type=str, default="",
                        help="渐进式patch: 'step:patch:bs,...' 例如 '0:384:16,150000:512:12,350000:720:8'")

    # 退化感知动态加权
    parser.add_argument("--deg-reweight", action="store_true", default=False,
                        help="启用退化感知动态加权 (根据验证集per-deg PSNR自动调权)")
    parser.add_argument("--deg-reweight-target", type=float, default=29.0,
                        help="目标PSNR，低于此值的退化类型会被加权")
    parser.add_argument("--deg-reweight-max", type=float, default=4.0)
    parser.add_argument("--deg-reweight-min", type=float, default=0.5)

    # ====== v4 新增：内置两阶段预增强 ======
    parser.add_argument("--use-brightness-enhancer", action="store_true", default=True,
                        help="启用内置亮度预增强模块 (两阶段)")
    parser.add_argument("--no-brightness-enhancer", dest="use_brightness_enhancer", action="store_false")
    parser.add_argument("--brightness-enhancer-ch", type=int, default=64,
                        help="预增强模块通道数")
    parser.add_argument("--brightness-enhancer-blocks", type=int, default=6,
                        help="预增强模块残差块数量")
    parser.add_argument("--brightness-threshold", type=float, default=0.3,
                        help="低光检测阈值 (0~1, 低于此值激活预增强)")
    parser.add_argument("--freq-emb-dim", type=int, default=128,
                        help="频率嵌入维度 (用于引导专家路由)")
    parser.add_argument("--lambda-enhance", type=float, default=0.1,
                        help="预增强损失权重")
    # v5 新增：低光训练策略
    parser.add_argument("--lowlight-boost", type=float, default=1.0,
                        help="低光样本主损失加权倍数 (推荐 2.0~3.0)")
    parser.add_argument("--enhance-warmup-steps", type=int, default=0,
                        help="预增强模块热身步数 (热身期间 lambda_enhance *= 3)")
    # ====== v3/v4 新增结束 ======

    # 其他
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--output-dir", type=str, default="results/fod_v3")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--readahead", action="store_true")
    parser.add_argument("--low-ratio", type=float, default=0.0)
    parser.add_argument("--index-cache", type=str, default=None)

    # 测试集微调 (test-set fine-tuning)
    parser.add_argument("--extra-lmdb", type=str, default=None,
                        help="Extra LMDB path (e.g. test set) as primary fine-tune data")
    parser.add_argument("--extra-repeat", type=int, default=35,
                        help="How many times to repeat the extra LMDB (default 35)")
    parser.add_argument("--extra-is-train", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True,
                        help="Whether extra LMDB uses training augmentation (default True)")

    args = parser.parse_args()

    # 分布式初始化
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0
        print("Single GPU mode")

    device = local_rank
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    logger = setup_logging(rank, args.output_dir if rank == 0 else None)
    if rank == 0:
        logger.info(f"Config: {vars(args)}")
        if args.grad_accum > 1:
            logger.info(f"Gradient accumulation: {args.grad_accum} steps, "
                        f"effective batch = batch_size * {args.grad_accum}")

    # 解析渐进式 patch
    progressive_stages = parse_progressive_patch(args.progressive_patch)
    if rank == 0 and progressive_stages:
        logger.info(f"Progressive patch stages: {progressive_stages}")

    # 退化感知动态加权器
    deg_reweighter = None
    if args.deg_reweight:
        deg_reweighter = DegradationReweighter(
            target_psnr=args.deg_reweight_target,
            max_weight=args.deg_reweight_max,
            min_weight=args.deg_reweight_min,
        )
        if rank == 0:
            logger.info(f"DegReweight enabled: target={args.deg_reweight_target}")

    # ====== 动态因子配置 ======
    if args.factors:
        custom_factors = [f.strip() for f in args.factors.split(",") if f.strip()]
        num_factors = len(custom_factors)
        # 强制 num_experts 与 factors 数量一致
        args.num_experts = num_factors
        use_dynamic_factors = True
    else:
        custom_factors = ["low", "haze", "rain", "snow"]  # 默认 CDD-11
        num_factors = 4
        use_dynamic_factors = False

    if rank == 0:
        logger.info(f"Factor mode: {'dynamic' if use_dynamic_factors else 'CDD-11'}, "
                    f"factors={custom_factors}, num_experts={args.num_experts}")

    # 模型
    model = create_fod_model(
        base_ch=args.base_ch, emb_dim=args.emb_dim,
        adapter_dim=args.adapter_dim, num_timesteps=args.num_timesteps,
        model_type=args.model_type,
        num_experts=args.num_experts,
        use_brightness_enhancer=args.use_brightness_enhancer,
        brightness_enhancer_ch=args.brightness_enhancer_ch,
        brightness_enhancer_blocks=args.brightness_enhancer_blocks,
        brightness_threshold=args.brightness_threshold,
        freq_emb_dim=args.freq_emb_dim,
    )
    model = model.to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model params: {n_params:.2f}M")
        if args.use_brightness_enhancer:
            enhance_params = sum(p.numel() for p in model.brightness_enhancer.parameters()) / 1e6
            logger.info(f"  - BrightnessEnhancer: {enhance_params:.2f}M")

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    ema = EMA(model, decay=args.ema_decay)

    if world_size > 1:
        model = DDP(model, device_ids=[device])

    # 优化器
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4, betas=(0.9, args.beta2))
    scheduler = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.niter)

    # ====== 数据加载 (支持渐进式 patch) ======
    current_patch_size = args.patch_size
    current_batch_size = args.batch_size

    if progressive_stages:
        ps, bs = get_current_patch_config(0, progressive_stages)
        if ps is not None:
            current_patch_size = ps
        if bs is not None and bs > 0:
            current_batch_size = bs

    def build_loader(patch_size, batch_size):
        if args.extra_lmdb:
            # --- 测试集微调模式 ---
            extra_ds = LMDBAllWeatherDataset(args.extra_lmdb, patch_size=patch_size,
                                             is_train=args.extra_is_train, readahead=args.readahead)
            repeated_extra = RepeatDataset(extra_ds, args.extra_repeat)
            train_ds = LMDBAllWeatherDataset(args.lmdb_path, patch_size=patch_size,
                                             is_train=True, readahead=args.readahead)
            ds = ConcatDataset([repeated_extra, train_ds])
            if rank == 0:
                logger.info(f"[ExtraLMDB] extra={len(extra_ds)}x{args.extra_repeat}={len(repeated_extra)}, "
                            f"train={len(train_ds)}, total={len(ds)}")
            smp = DistributedSampler(ds, world_size, rank, shuffle=True) if world_size > 1 else None
        else:
            ds = LMDBAllWeatherDataset(args.lmdb_path, patch_size=patch_size,
                                       is_train=True, readahead=args.readahead)
            if args.low_ratio > 0:
                cache_path = args.index_cache or os.path.join(args.output_dir, "balanced_index_cache.pkl")
                cache = build_or_load_index_cache(ds, cache_path, rank=rank)
                smp = CachedBalancedSampler(
                    cache["low_indices"], cache["nonlow_indices"],
                    low_ratio=args.low_ratio, total_size=len(ds),
                    num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            else:
                smp = DistributedSampler(ds, world_size, rank, shuffle=True) if world_size > 1 else None
        kw = dict(dataset=ds, batch_size=batch_size // world_size, sampler=smp,
                  shuffle=(smp is None), num_workers=args.num_workers,
                  pin_memory=True, drop_last=True)
        if args.num_workers > 0:
            kw["prefetch_factor"] = args.prefetch_factor
            kw["persistent_workers"] = args.persistent_workers
        return DataLoader(**kw), ds, smp

    loader, dataset, sampler = build_loader(current_patch_size, current_batch_size)

    test_loader = None
    if rank == 0 and args.test_lmdb_path:
        test_dataset = LMDBAllWeatherDataset(args.test_lmdb_path, is_train=False,
                                             readahead=args.readahead)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

    if rank == 0:
        logger.info(f"Train samples: {len(dataset)}, patch={current_patch_size}, bs={current_batch_size}")
        os.makedirs(f"{args.output_dir}/ckpt", exist_ok=True)
        # 打印 deg_name 分布和因子映射情况
        if use_dynamic_factors:
            try:
                base_ds = dataset.datasets[0].dataset if hasattr(dataset, 'datasets') else dataset
                print_deg_name_distribution(base_ds, custom_factors, max_scan=2000)
            except Exception as e:
                logger.warning(f"Could not scan deg_name distribution: {e}")

    # ====== 恢复训练 ======
    step = 0
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and amp_dtype == torch.float16)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=f"cuda:{device}")
        model_dict = ckpt["model"]
        if hasattr(model, 'module'):
            model_dict = {f"module.{k}": v for k, v in model_dict.items()}
        model.load_state_dict(model_dict, strict=False)
        ema.load_state_dict(ckpt["ema"], device=f"cuda:{device}")

        if args.finetune:
            step = 0
            if rank == 0:
                logger.info(f"Finetune mode: loaded weights from {args.resume}, "
                            f"reset optimizer/scheduler, lr={args.lr}, niter={args.niter}")
        else:
            opt.load_state_dict(ckpt["opt"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
                scaler.load_state_dict(ckpt["scaler"])
            step = ckpt["step"]
            if rank == 0:
                logger.info(f"Resumed from step {step}")

    # ====== 训练状态 ======
    best_psnr = 0.0
    patience_counter = 0
    model.train()
    epoch = 0
    running_loss = 0.0
    log_count = 0
    start_time = time()
    net = model.module if hasattr(model, 'module') else model
    num_timesteps = net.num_timesteps
    grad_accum = args.grad_accum
    micro_step = 0  # 梯度累积计数器

    # PLACEHOLDER_TRAINING_LOOP

    while step < args.niter:
        if sampler:
            sampler.set_epoch(epoch)

        # ====== 渐进式 patch: 检查是否需要切换 ======
        if progressive_stages:
            new_ps, new_bs = get_current_patch_config(step, progressive_stages)
            if new_ps is not None and new_ps != current_patch_size:
                old_ps, old_bs = current_patch_size, current_batch_size
                current_patch_size = new_ps
                if new_bs is not None and new_bs > 0:
                    current_batch_size = new_bs
                if rank == 0:
                    logger.info(f"[Progressive] step={step}: patch {old_ps}->{current_patch_size}, "
                                f"bs {old_bs}->{current_batch_size}")
                loader, dataset, sampler = build_loader(current_patch_size, current_batch_size)
                if sampler:
                    sampler.set_epoch(epoch)

        for batch in loader:
            y = batch["LQ"].to(device)
            x = batch["GT"].to(device)
            deg_name = batch.get("deg_name", None)

            B = y.shape[0]
            _, _, H, W = y.shape
            t = torch.randint(0, num_timesteps + 1, (B,), device=device)

            # ====== 动态因子模式: 重新计算 present / m_gt / labels ======
            if use_dynamic_factors and deg_name is not None:
                present, m_gt = build_dynamic_present_and_m(
                    deg_name, device, custom_factors, num_factors, H, W)
                labels = deg_name_to_labels_dynamic(
                    deg_name, device, custom_factors, num_factors)
            else:
                present = batch.get("present", None)
                m_gt = batch.get("m", None)
                if present is not None:
                    present = present.to(device)
                if m_gt is not None:
                    m_gt = m_gt.to(device)
                labels = None
                if deg_name is not None:
                    labels = deg_name_to_labels(deg_name, device)

            # 检测低光样本（仅在启用亮度增强且因子包含 low 时）
            is_lowlight = None
            has_low_factor = not use_dynamic_factors or "low" in [f.lower() for f in custom_factors]
            if deg_name is not None and args.use_brightness_enhancer and has_low_factor:
                is_lowlight = torch.zeros(B, dtype=torch.bool, device=device)
                for i, name in enumerate(deg_name):
                    if isinstance(name, str) and "low" in name.lower():
                        is_lowlight[i] = True

            # 门控目标
            alpha_target = None
            if args.lambda_alpha > 0 and labels is not None:
                t_norm = t.float() / num_timesteps
                if use_dynamic_factors:
                    alpha_target = gating_target_from_labels_dynamic(
                        labels, t_norm, num_factors)
                else:
                    alpha_target = gating_target_from_labels(labels, t_norm)

            # ====== 梯度累积: 在每个累积周期开始时清零 ======
            if micro_step % grad_accum == 0:
                opt.zero_grad(set_to_none=True)

            # === enhance 预热调度 ===
            current_lambda_enhance = args.lambda_enhance
            if args.enhance_warmup_steps > 0 and step < args.enhance_warmup_steps:
                # 热身期间加大 enhance 权重，让 enhancer 先学好
                warmup_ratio = 1.0 - step / args.enhance_warmup_steps
                current_lambda_enhance = args.lambda_enhance * (1.0 + 2.0 * warmup_ratio)

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss, loss_dict = fod_training_loss(
                        net, y, x, t,
                        labels=labels, present=present, m_gt=m_gt,
                        lambda_cls=args.lambda_cls,
                        lambda_balance=args.lambda_balance,
                        lambda_w=args.lambda_w,
                        lambda_m=args.lambda_m,
                        lambda_recon=args.lambda_recon,
                        lambda_alpha=args.lambda_alpha,
                        alpha_target=alpha_target,
                        use_hard_sample_weighting=args.use_hard_sample_weighting,
                        hard_sample_psnr_threshold=args.hard_sample_psnr_threshold,
                        hard_sample_weight_factor=args.hard_sample_weight_factor,
                        loss_type=args.loss_type,
                        charbonnier_eps=args.charbonnier_eps,
                        lambda_freq=args.lambda_freq,
                        lambda_enhance=current_lambda_enhance,
                        is_lowlight=is_lowlight,
                        lowlight_boost=args.lowlight_boost,
                    )
            else:
                loss, loss_dict = fod_training_loss(
                    net, y, x, t,
                    labels=labels, present=present, m_gt=m_gt,
                    lambda_cls=args.lambda_cls,
                    lambda_balance=args.lambda_balance,
                    lambda_w=args.lambda_w,
                    lambda_m=args.lambda_m,
                    lambda_recon=args.lambda_recon,
                    lambda_alpha=args.lambda_alpha,
                    alpha_target=alpha_target,
                    use_hard_sample_weighting=args.use_hard_sample_weighting,
                    hard_sample_psnr_threshold=args.hard_sample_psnr_threshold,
                    hard_sample_weight_factor=args.hard_sample_weight_factor,
                    loss_type=args.loss_type,
                    charbonnier_eps=args.charbonnier_eps,
                    lambda_freq=args.lambda_freq,
                    lambda_enhance=current_lambda_enhance,
                    is_lowlight=is_lowlight,
                    lowlight_boost=args.lowlight_boost,
                )

            # ====== 退化感知动态加权 ======
            if deg_reweighter and deg_reweighter.deg_weights and deg_name is not None:
                deg_w = deg_reweighter.get_sample_weights(deg_name, device)
                loss = loss * deg_w.mean()  # 标量缩放

            # ====== 梯度累积: loss 缩放 ======
            scaled_loss = loss / grad_accum

            # PLACEHOLDER_BACKWARD_AND_EVAL

            if args.amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_step += 1

            # ====== 梯度累积: 累积够了再更新 ======
            if micro_step % grad_accum == 0:
                if args.amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                scheduler.step()
                ema.update(model)

                running_loss += loss.item()
                log_count += 1
                step += 1
            else:
                continue  # 累积中，跳过日志/评估/保存

            # 日志
            if step % args.log_every == 0 and rank == 0:
                avg_loss = running_loss / log_count
                elapsed = time() - start_time
                steps_per_sec = log_count / elapsed

                main_str = f"main={loss_dict.get('main', 0):.4f}"
                extra = ""
                if 'recon' in loss_dict:
                    extra += f" recon={loss_dict['recon']:.4f}"
                if 'alpha' in loss_dict:
                    extra += f" alpha={loss_dict['alpha']:.4f}"
                if 'hard_ratio' in loss_dict:
                    extra += f" hard={loss_dict['hard_ratio']:.0%}"
                if 'freq' in loss_dict:
                    extra += f" freq={loss_dict['freq']:.4f}"
                if 'enhance' in loss_dict:
                    extra += f" enh={loss_dict['enhance']:.4f}"
                if 'enhance_gate' in loss_dict:
                    extra += f" gate={loss_dict['enhance_gate']:.2f}"
                if 'enhance_lum' in loss_dict:
                    extra += f" lum={loss_dict['enhance_lum']:.4f}"
                if 'lowlight_ratio' in loss_dict:
                    extra += f" ll={loss_dict['lowlight_ratio']:.0%}"
                logger.info(
                    f"step={step:07d} loss={avg_loss:.4f} {main_str}{extra} "
                    f"lr={opt.param_groups[0]['lr']:.2e} "
                    f"patch={current_patch_size} "
                    f"speed={steps_per_sec:.1f}it/s"
                )
                running_loss = 0.0
                log_count = 0
                start_time = time()

            # PLACEHOLDER_EVAL_AND_SAVE

            # ====== 评估 ======
            if step % args.eval_every == 0 and rank == 0 and test_loader:
                psnr, ssim_v, per_deg = evaluate_full(
                    model, test_loader, device, ema,
                    max_samples=args.eval_max_samples,
                    num_steps=args.eval_num_steps,
                    sample_type=args.sample_type,
                    use_amp=False, amp_dtype=amp_dtype,
                    one_step=args.eval_one_step,
                )

                eval_mode = "one-step" if args.eval_one_step else f"{args.eval_num_steps}-step {args.sample_type}"
                logger.info(f">>> step={step:07d} PSNR={psnr:.4f} SSIM={ssim_v:.6f} ({eval_mode})")

                # per-degradation
                per_factor_psnr = defaultdict(lambda: {"psnr_sum": 0.0, "cnt": 0})
                for deg in sorted(per_deg.keys()):
                    info = per_deg[deg]
                    logger.info(f"    {deg:<25s} PSNR={info['psnr']:.2f}  SSIM={info['ssim']:.4f}  (n={info['cnt']})")
                    # 按因子分组统计
                    if use_dynamic_factors:
                        fidx = map_deg_name_to_factor_idx(deg, custom_factors)
                        if fidx >= 0:
                            fname = custom_factors[fidx]
                            per_factor_psnr[fname]["psnr_sum"] += info['psnr'] * info['cnt']
                            per_factor_psnr[fname]["cnt"] += info['cnt']
                    else:
                        if "low" in deg:
                            per_factor_psnr["low"]["psnr_sum"] += info['psnr'] * info['cnt']
                            per_factor_psnr["low"]["cnt"] += info['cnt']
                        else:
                            per_factor_psnr["non-low"]["psnr_sum"] += info['psnr'] * info['cnt']
                            per_factor_psnr["non-low"]["cnt"] += info['cnt']

                # 打印 per-factor 平均
                factor_strs = []
                for fname, finfo in sorted(per_factor_psnr.items()):
                    if finfo["cnt"] > 0:
                        avg = finfo["psnr_sum"] / finfo["cnt"]
                        factor_strs.append(f"{fname}: {avg:.2f} dB")
                if factor_strs:
                    logger.info(f"    --- per-factor avg: {' | '.join(factor_strs)}")

                # 退化感知动态加权更新
                if deg_reweighter:
                    deg_reweighter.update(per_deg)
                    logger.info(f"    [DegReweight] {deg_reweighter.get_status_str()}")

                # best model
                if psnr > best_psnr:
                    improvement = psnr - best_psnr
                    best_psnr = psnr
                    patience_counter = 0
                    best_path = f"{args.output_dir}/ckpt/best.pt"
                    torch.save({
                        "model": (model.module if hasattr(model, 'module') else model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": (scaler.state_dict() if scaler is not None else None),
                        "step": step, "psnr": psnr, "ssim": ssim_v,
                        "args": vars(args),
                    }, best_path)
                    logger.info(f"*** NEW BEST *** PSNR={psnr:.4f} (+{improvement:.4f}) -> {best_path}")
                else:
                    patience_counter += 1
                    logger.info(f"    No improvement ({patience_counter}/{args.patience if args.patience > 0 else 'inf'}), "
                               f"best={best_psnr:.4f}")

                # early stopping
                if args.patience > 0 and patience_counter >= args.patience:
                    logger.info(f"!!! Early stopping at step {step}: "
                               f"no improvement for {args.patience} evals. Best={best_psnr:.4f}")
                    final_path = f"{args.output_dir}/ckpt/final_earlystop.pt"
                    torch.save({
                        "model": (model.module if hasattr(model, 'module') else model).state_dict(),
                        "ema": ema.state_dict(), "step": step,
                        "best_psnr": best_psnr, "args": vars(args),
                    }, final_path)
                    if world_size > 1:
                        dist.destroy_process_group()
                    return

            # 定期保存
            if step % args.save_every == 0 and rank == 0:
                ckpt_path = f"{args.output_dir}/ckpt/{step:07d}.pt"
                torch.save({
                    "model": (model.module if hasattr(model, 'module') else model).state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": (scaler.state_dict() if scaler is not None else None),
                    "step": step, "args": vars(args),
                }, ckpt_path)
                logger.info(f"Saved {ckpt_path}")

            if step >= args.niter:
                break

            # 渐进式 patch: 检查是否需要在 epoch 内切换
            if progressive_stages:
                new_ps, new_bs = get_current_patch_config(step, progressive_stages)
                if new_ps is not None and new_ps != current_patch_size:
                    break  # 跳出当前 epoch，外层循环会重建 loader

        epoch += 1

    # 最终保存
    if rank == 0:
        final_path = f"{args.output_dir}/ckpt/final.pt"
        torch.save({
            "model": (model.module if hasattr(model, 'module') else model).state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else None),
            "step": step, "best_psnr": best_psnr, "args": vars(args),
        }, final_path)
        logger.info(f"Training complete. Final={final_path}, Best PSNR={best_psnr:.4f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
