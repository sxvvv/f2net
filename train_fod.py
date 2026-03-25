#!/usr/bin/env python3

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
# Utilities
# ============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-6):
    """Cosine-annealing LR schedule with linear warmup."""
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
    """Convert a batch of degradation name strings to multi-hot label vectors."""
    B = len(deg_names)
    labels = torch.zeros(B, 4, device=device)
    for i, name in enumerate(deg_names):
        if name:
            for factor in parse_factors(name):
                if factor in FACTOR2IDX:
                    labels[i, FACTOR2IDX[factor]] = 1.0
    return labels


def gating_target_from_labels(labels, t_norm):
    """Compute time-dependent gating targets for expert routing supervision.

    Early timesteps emphasise rain/snow experts; late timesteps emphasise
    low-light/haze experts, loosely following the degradation removal order.
    """
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
# Balanced samplers
# ============================================================================
def build_or_load_index_cache(train_set, cache_path, rank=0, target_degs=None):
    """Build (or load) an index cache that classifies samples by degradation type.

    The cache enables balanced sampling during training, ensuring adequate
    representation of underrepresented degradation categories.
    """
    if os.path.isfile(cache_path):
        if rank == 0:
            print(f"[IndexCache] Loading {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
            if target_degs:
                need_rebuild = False
                for deg in target_degs:
                    if f"{deg}_indices" not in cache:
                        need_rebuild = True
                        break
                if need_rebuild:
                    if rank == 0:
                        print(f"[IndexCache] Cache missing target degs, rebuilding...")
                else:
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    return cache
    if rank == 0:
        print(f"[IndexCache] Building cache -> {cache_path}")
        low_idx, nonlow_idx = [], []
        target_indices = {}
        if target_degs:
            for deg in target_degs:
                target_indices[deg] = []

        for i in range(len(train_set)):
            try:
                sample = train_set[i]
                dn = sample.get("deg_name", None) if isinstance(sample, dict) else None
                if dn is not None:
                    dn_str = str(dn).lower()
                    if target_degs:
                        for deg in target_degs:
                            deg_lower = deg.lower()
                            if deg_lower == dn_str or deg_lower in dn_str or dn_str in deg_lower:
                                if deg not in target_indices:
                                    target_indices[deg] = []
                                target_indices[deg].append(i)
                                break
                    if "low" in dn_str:
                        low_idx.append(i)
                    else:
                        nonlow_idx.append(i)
                else:
                    nonlow_idx.append(i)
            except Exception as e:
                if rank == 0 and i < 10:
                    print(f"Warning: Failed to read sample {i}: {e}")
                nonlow_idx.append(i)

        cache = {"low_indices": low_idx, "nonlow_indices": nonlow_idx}
        if target_degs:
            for deg in target_degs:
                cache[f"{deg}_indices"] = target_indices.get(deg, [])

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

        print(f"[IndexCache] low={len(low_idx)} nonlow={len(nonlow_idx)} total={len(train_set)}")
        if target_degs:
            for deg in target_degs:
                print(f"[IndexCache] {deg}={len(target_indices.get(deg, []))}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    with open(cache_path, "rb") as f:
        return pickle.load(f)


class CachedBalancedSampler(Sampler):
    """Balanced sampler with configurable low-light / non-low-light ratio."""
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


class MultiDegBalancedSampler(Sampler):
    """Weighted sampler supporting per-degradation-type sampling ratios.

    Args:
        cache: Index cache dict from ``build_or_load_index_cache``.
        deg_ratios: {deg_name: ratio} for targeted oversampling.
        total_size: Total samples per epoch.
    """
    def __init__(self, cache, deg_ratios, total_size,
                 num_replicas=1, rank=0, shuffle=True, seed=0):
        self.cache = cache
        self.deg_ratios = dict(deg_ratios) if deg_ratios else {}
        self.total_size = int(total_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        assert self.total_size > 0
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

        for deg in self.deg_ratios.keys():
            key = f"{deg}_indices"
            if key not in cache:
                print(f"Warning: {key} not found in cache, will use empty list")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        def draw(src, count):
            if count == 0 or len(src) == 0:
                return []
            ridx = torch.randint(0, len(src), (count,), generator=g).tolist()
            return [src[j] for j in ridx]

        # Normalise ratios if they exceed 1.0
        total_ratio = sum(self.deg_ratios.values())
        if total_ratio > 1.0:
            for deg in self.deg_ratios:
                self.deg_ratios[deg] /= total_ratio

        indices = []
        used_indices = set()

        # Priority sampling for target degradation types
        for deg, ratio in self.deg_ratios.items():
            key = f"{deg}_indices"
            deg_indices = self.cache.get(key, [])
            if len(deg_indices) > 0:
                n_samples = int(self.total_size * ratio)
                sampled = draw(deg_indices, n_samples)
                indices.extend(sampled)
                used_indices.update(sampled)

        # Fill the remainder from all samples
        remaining_ratio = 1.0 - sum(self.deg_ratios.values())
        if remaining_ratio > 0.01:
            all_indices = self.cache.get("low_indices", []) + self.cache.get("nonlow_indices", [])
            available_indices = [i for i in all_indices if i not in used_indices]
            n_remaining = self.total_size - len(indices)
            if n_remaining > 0 and len(available_indices) > 0:
                remaining_samples = draw(available_indices, min(n_remaining, len(available_indices)))
                indices.extend(remaining_samples)

        # Pad with random samples if still short
        if len(indices) < self.total_size:
            all_indices = self.cache.get("low_indices", []) + self.cache.get("nonlow_indices", [])
            if len(all_indices) > 0:
                n_needed = self.total_size - len(indices)
                indices.extend(draw(all_indices, n_needed))

        indices = indices[:self.total_size]
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        # DDP shard
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class RepeatDataset(torch.utils.data.Dataset):
    """Virtually repeat a dataset N times (useful for fine-tuning on small sets)."""
    def __init__(self, dataset, repeats: int):
        self.dataset = dataset
        self.repeats = max(1, int(repeats))

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]


# ============================================================================
# Progressive patch size
# ============================================================================
def parse_progressive_patch(spec_str):
    """Parse a progressive patch-size schedule string.

    Format: "step1:patch1:bs1,step2:patch2:bs2,..."
    Example: "0:384:16,150000:512:12,350000:720:8"
    Returns: [(0, 384, 16), (150000, 512, 12), (350000, 720, 8)]
    """
    if not spec_str:
        return []
    stages = []
    for part in spec_str.split(","):
        tokens = part.strip().split(":")
        if len(tokens) == 3:
            stages.append((int(tokens[0]), int(tokens[1]), int(tokens[2])))
        elif len(tokens) == 2:
            stages.append((int(tokens[0]), int(tokens[1]), -1))
    stages.sort(key=lambda x: x[0])
    return stages


def get_current_patch_config(step, stages):
    """Return (patch_size, batch_size) for the current training step."""
    if not stages:
        return None, None
    result = stages[0]
    for s in stages:
        if step >= s[0]:
            result = s
    return result[1], result[2]


# ============================================================================
# Degradation-aware dynamic loss reweighting
# ============================================================================
class DegradationReweighter:
    """Dynamically adjust per-degradation loss weights based on validation PSNR.

    Degradation types with lower PSNR receive higher training weights,
    smoothed with exponential moving average across evaluation rounds.
    """
    def __init__(self, target_psnr=29.0, max_weight=4.0, min_weight=0.5, momentum=0.8):
        self.target_psnr = target_psnr
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.momentum = momentum
        self.deg_weights = {}

    def update(self, per_deg_results):
        """Update weights from per-degradation evaluation results.

        Args:
            per_deg_results: {deg_name: {"psnr": float, "ssim": float, "cnt": int}}
        """
        if not per_deg_results:
            return

        gaps = {}
        for deg, info in per_deg_results.items():
            gaps[deg] = max(0, self.target_psnr - info["psnr"])

        max_gap = max(gaps.values()) if gaps else 1.0
        if max_gap < 0.1:
            max_gap = 1.0

        new_weights = {}
        for deg, gap in gaps.items():
            w = self.min_weight + (self.max_weight - self.min_weight) * (gap / max_gap)
            new_weights[deg] = w

        for deg, w in new_weights.items():
            if deg in self.deg_weights:
                self.deg_weights[deg] = self.momentum * self.deg_weights[deg] + (1 - self.momentum) * w
            else:
                self.deg_weights[deg] = w

    def get_sample_weights(self, deg_names, device):
        """Return per-sample loss weights (B,) based on degradation type."""
        B = len(deg_names)
        weights = torch.ones(B, device=device)
        for i, name in enumerate(deg_names):
            if isinstance(name, str) and name in self.deg_weights:
                weights[i] = self.deg_weights[name]
        weights = weights / (weights.mean() + 1e-8)
        return weights

    def get_status_str(self):
        if not self.deg_weights:
            return "no weights yet"
        parts = [f"{deg}={self.deg_weights[deg]:.2f}" for deg in sorted(self.deg_weights)]
        return " ".join(parts)


# ============================================================================
# Validation
# ============================================================================
@torch.no_grad()
def evaluate_full(model, loader, device, ema=None, max_samples=0,
                  num_steps=10, sample_type="MC", use_tta=False,
                  use_amp=False, amp_dtype=torch.bfloat16,
                  one_step=False):
    """Full validation loop. Returns (avg_psnr, avg_ssim, per_deg_dict)."""
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

            # Force float32 for metric computation (bf16 can cause PSNR anomalies)
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
    parser = argparse.ArgumentParser(description="F2D-Net training")

    # Data
    parser.add_argument("--lmdb-path", type=str, required=True)
    parser.add_argument("--test-lmdb-path", type=str, default=None)
    parser.add_argument("--patch-size", type=int, default=720)
    parser.add_argument("--num-workers", type=int, default=8)

    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--niter", type=int, default=600_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)

    # Model
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=4,
                        help="Number of degradation-specific expert adapters (M)")
    parser.add_argument("--adapter-dim", type=int, default=128)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--model-type", type=str, default="SFLOW",
                        choices=["FINAL_X", "FLOW", "SFLOW"])

    # Loss weights
    parser.add_argument("--lambda-cls", type=float, default=0.01)
    parser.add_argument("--lambda-balance", type=float, default=0.01)
    parser.add_argument("--lambda-w", type=float, default=0.1)
    parser.add_argument("--lambda-m", type=float, default=0.05)
    parser.add_argument("--lambda-recon", type=float, default=0.1)
    parser.add_argument("--lambda-alpha", type=float, default=0.02)

    # Hard-sample weighting
    parser.add_argument("--use-hard-sample-weighting", action="store_true", default=False)
    parser.add_argument("--hard-sample-psnr-threshold", type=float, default=26.0)
    parser.add_argument("--hard-sample-weight-factor", type=float, default=2.0)

    # Inference
    parser.add_argument("--eval-num-steps", type=int, default=10)
    parser.add_argument("--sample-type", type=str, default="MC", choices=["EM", "MC", "NMC"])

    # LR schedule
    parser.add_argument("--warmup-steps", type=int, default=5000)

    # Fine-tuning
    parser.add_argument("--finetune", action="store_true", default=False,
                        help="Fine-tune mode: load weights + EMA, reset optimiser/scheduler")

    # Early stopping
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--eval-max-samples", type=int, default=0,
                        help="Max samples per evaluation (0 = all)")
    parser.add_argument("--eval-one-step", action="store_true", default=False,
                        help="Use single-step inference during eval (~10× faster)")

    # Charbonnier + frequency loss
    parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "charbonnier"])
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--lambda-freq", type=float, default=0.0)

    # Progressive patch size
    parser.add_argument("--progressive-patch", type=str, default="",
                        help="Progressive schedule: 'step:patch:bs,...' "
                             "e.g. '0:384:16,150000:512:12,350000:720:8'")

    # Degradation-aware reweighting
    parser.add_argument("--deg-reweight", action="store_true", default=False,
                        help="Enable degradation-aware dynamic loss reweighting")
    parser.add_argument("--deg-reweight-target", type=float, default=29.0)
    parser.add_argument("--deg-reweight-max", type=float, default=4.0)
    parser.add_argument("--deg-reweight-min", type=float, default=0.5)

    # Brightness pre-enhancer
    parser.add_argument("--use-brightness-enhancer", action="store_true", default=True)
    parser.add_argument("--no-brightness-enhancer", dest="use_brightness_enhancer", action="store_false")
    parser.add_argument("--brightness-enhancer-ch", type=int, default=64)
    parser.add_argument("--brightness-enhancer-blocks", type=int, default=6)
    parser.add_argument("--brightness-threshold", type=float, default=0.3,
                        help="Luminance threshold for pre-enhancement activation (0..1)")
    parser.add_argument("--freq-emb-dim", type=int, default=128)
    parser.add_argument("--lambda-enhance", type=float, default=0.1)

    # Low-light training strategy
    parser.add_argument("--lowlight-boost", type=float, default=1.0,
                        help="Loss weight multiplier for low-light samples (e.g. 2.0)")
    parser.add_argument("--enhance-warmup-steps", type=int, default=0,
                        help="Pre-enhancer warmup steps (higher lambda_enhance during warmup)")

    # General
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

    # Targeted degradation oversampling
    parser.add_argument("--target-degs", type=str, default=None,
                        help="Comma-separated degradation types to oversample, "
                             "e.g. 'low_haze_snow,low_haze_rain'")
    parser.add_argument("--target-deg-ratios", type=str, default=None,
                        help="Comma-separated sampling ratios (matching --target-degs)")

    # Extra LMDB for fine-tuning
    parser.add_argument("--extra-lmdb", type=str, default=None,
                        help="Additional LMDB path (e.g. test set) for fine-tuning")
    parser.add_argument("--extra-repeat", type=int, default=35,
                        help="How many times to repeat the extra LMDB")
    parser.add_argument("--extra-is-train", type=lambda x: x.lower() in ('true', '1', 'yes'),
                        default=True)

    args = parser.parse_args()

    # ====== Distributed setup ======
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

    progressive_stages = parse_progressive_patch(args.progressive_patch)
    if rank == 0 and progressive_stages:
        logger.info(f"Progressive patch stages: {progressive_stages}")

    # Degradation-aware reweighter
    deg_reweighter = None
    if args.deg_reweight:
        deg_reweighter = DegradationReweighter(
            target_psnr=args.deg_reweight_target,
            max_weight=args.deg_reweight_max,
            min_weight=args.deg_reweight_min,
        )
        if rank == 0:
            logger.info(f"DegReweight enabled: target={args.deg_reweight_target}")

    # ====== Model ======
    model = create_fod_model(
        base_ch=args.base_ch, emb_dim=args.emb_dim,
        num_experts=args.num_experts,
        adapter_dim=args.adapter_dim, num_timesteps=args.num_timesteps,
        model_type=args.model_type,
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

    # ====== Optimiser ======
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=1e-4, betas=(0.9, args.beta2))
    scheduler = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.niter)

    # ====== Data (supports progressive patch sizes) ======
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

            target_degs = None
            deg_ratios = {}
            if args.target_degs:
                target_degs = [d.strip() for d in args.target_degs.split(",") if d.strip()]
                if args.target_deg_ratios:
                    ratios = [float(r.strip()) for r in args.target_deg_ratios.split(",")]
                    if len(ratios) == len(target_degs):
                        deg_ratios = {deg: ratio for deg, ratio in zip(target_degs, ratios)}
                    else:
                        if rank == 0:
                            logger.warning(f"target_deg_ratios length ({len(ratios)}) != "
                                           f"target_degs length ({len(target_degs)}), using equal ratios")
                        equal_ratio = 1.0 / len(target_degs) if target_degs else 0.0
                        deg_ratios = {deg: equal_ratio for deg in target_degs}
                else:
                    equal_ratio = 1.0 / len(target_degs) if target_degs else 0.0
                    deg_ratios = {deg: equal_ratio for deg in target_degs}

            if target_degs and len(target_degs) > 0:
                cache_path = args.index_cache or os.path.join(args.output_dir, "balanced_index_cache.pkl")
                cache = build_or_load_index_cache(ds, cache_path, rank=rank, target_degs=target_degs)
                if rank == 0:
                    logger.info(f"[MultiDegSampler] target_degs={target_degs}, ratios={deg_ratios}")
                smp = MultiDegBalancedSampler(
                    cache, deg_ratios, total_size=len(ds),
                    num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            elif args.low_ratio > 0:
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

    # ====== Resume ======
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

    # ====== Training loop ======
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
    micro_step = 0

    while step < args.niter:
        if sampler:
            sampler.set_epoch(epoch)

        # Check for progressive patch-size transition
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
            present = batch.get("present", None)
            m_gt = batch.get("m", None)
            if present is not None:
                present = present.to(device)
            if m_gt is not None:
                m_gt = m_gt.to(device)

            B = y.shape[0]
            t = torch.randint(0, num_timesteps + 1, (B,), device=device)

            # Detect low-light samples from degradation name
            is_lowlight = None
            if deg_name is not None and args.use_brightness_enhancer:
                is_lowlight = torch.zeros(B, dtype=torch.bool, device=device)
                for i, name in enumerate(deg_name):
                    if isinstance(name, str) and "low" in name.lower():
                        is_lowlight[i] = True

            # Gating target (time-dependent expert prior)
            alpha_target = None
            if args.lambda_alpha > 0 and deg_name is not None:
                labels = deg_name_to_labels(deg_name, device)
                t_norm = t.float() / num_timesteps
                alpha_target = gating_target_from_labels(labels, t_norm)
            elif deg_name is not None:
                labels = deg_name_to_labels(deg_name, device)
            else:
                labels = None

            # Gradient accumulation: zero grads at start of each cycle
            if micro_step % grad_accum == 0:
                opt.zero_grad(set_to_none=True)

            # Pre-enhancer warmup: higher lambda_enhance during early training
            current_lambda_enhance = args.lambda_enhance
            if args.enhance_warmup_steps > 0 and step < args.enhance_warmup_steps:
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

            # Degradation-aware reweighting
            if deg_reweighter and deg_reweighter.deg_weights and deg_name is not None:
                deg_w = deg_reweighter.get_sample_weights(deg_name, device)
                loss = loss * deg_w.mean()

            scaled_loss = loss / grad_accum

            if args.amp:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            micro_step += 1

            # Accumulate, then update
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
                continue  # still accumulating

            # Logging
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

            # Evaluation
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

                low_psnr_sum, low_cnt = 0.0, 0
                nonlow_psnr_sum, nonlow_cnt = 0.0, 0
                for deg in sorted(per_deg.keys()):
                    info = per_deg[deg]
                    logger.info(f"    {deg:<25s} PSNR={info['psnr']:.2f}  SSIM={info['ssim']:.4f}  (n={info['cnt']})")
                    if "low" in deg:
                        low_psnr_sum += info['psnr'] * info['cnt']
                        low_cnt += info['cnt']
                    else:
                        nonlow_psnr_sum += info['psnr'] * info['cnt']
                        nonlow_cnt += info['cnt']

                if low_cnt > 0 and nonlow_cnt > 0:
                    logger.info(f"    --- low avg: {low_psnr_sum/low_cnt:.2f} dB  |  "
                               f"non-low avg: {nonlow_psnr_sum/nonlow_cnt:.2f} dB")

                if deg_reweighter:
                    deg_reweighter.update(per_deg)
                    logger.info(f"    [DegReweight] {deg_reweighter.get_status_str()}")

                # Best model
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

                # Early stopping
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

            # Periodic checkpoint
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

            # Progressive patch: check for mid-epoch transition
            if progressive_stages:
                new_ps, new_bs = get_current_patch_config(step, progressive_stages)
                if new_ps is not None and new_ps != current_patch_size:
                    break

        epoch += 1

    # Final checkpoint
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
