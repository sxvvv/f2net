#!/usr/bin/env python3
# eval_fod_testset.py
# FoD增广流匹配模型在LMDB测试集上的评估脚本

import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from models.fod_cfm_net import create_fod_model, fod_inference, fod_one_step_inference
from utils.lmdb_dataset import LMDBAllWeatherDataset
from utils.metrics import psnr_y_torch, ssim_torch_eval as ssim_torch
from utils.ema import EMA


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="FoD增广流匹配测试集评估")
    
    # 检查点
    parser.add_argument("--ckpt", type=str, required=True, help="检查点路径")
    parser.add_argument("--test-lmdb", type=str, required=True, help="测试集LMDB路径")
    
    # 模型参数（会从checkpoint自动读取，这里作为默认值）
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--num-timesteps", type=int, default=100)
    parser.add_argument("--num-experts", type=int, default=None,
                        help="专家数量（默认从checkpoint读取，若无则为4）")
    parser.add_argument("--model-type", type=str, default="SFLOW")
    
    # 推理参数
    parser.add_argument("--num-steps", type=int, default=10, help="采样步数")
    parser.add_argument("--sample-type", type=str, default="MC", 
                        choices=["EM", "MC", "NMC"], help="采样策略")
    parser.add_argument("--one-step", action="store_true", help="使用单步快速推理")
    parser.add_argument("--use-tta", action="store_true", help="使用测试时增强")
    # 默认使用 EMA；需要关闭时传 --no-ema
    parser.add_argument("--use-ema", dest="use_ema", action="store_true", default=True, help="使用EMA权重")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false", help="不使用EMA权重（用原始模型权重）")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, 
                        help=">0则只评估前N个样本（用于快速测试）")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载检查点
    print(f"Loading checkpoint from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # 从checkpoint获取模型参数
    if "args" in ckpt:
        ckpt_args = ckpt["args"]
        args.base_ch = ckpt_args.get("base_ch", args.base_ch)
        args.emb_dim = ckpt_args.get("emb_dim", args.emb_dim)
        args.adapter_dim = ckpt_args.get("adapter_dim", args.adapter_dim)
        args.num_timesteps = ckpt_args.get("num_timesteps", args.num_timesteps)
        args.model_type = ckpt_args.get("model_type", args.model_type)
        if args.num_experts is None:
            args.num_experts = ckpt_args.get("num_experts", 4)
        # 亮度预增强参数
        args.use_brightness_enhancer = ckpt_args.get("use_brightness_enhancer", False)
        args.brightness_enhancer_ch = ckpt_args.get("brightness_enhancer_ch", 32)
        args.brightness_threshold = ckpt_args.get("brightness_threshold", 0.3)
        print(f"Model config from checkpoint: base_ch={args.base_ch}, "
              f"emb_dim={args.emb_dim}, num_timesteps={args.num_timesteps}, "
              f"model_type={args.model_type}, brightness_enhancer={args.use_brightness_enhancer}")
    else:
        # 旧版检查点兼容
        if args.num_experts is None:
            args.num_experts = 4
        args.use_brightness_enhancer = False
        args.brightness_enhancer_ch = 32
        args.brightness_threshold = 0.3

    # 创建模型
    model = create_fod_model(
        base_ch=args.base_ch,
        emb_dim=args.emb_dim,
        adapter_dim=args.adapter_dim,
        num_timesteps=args.num_timesteps,
        model_type=args.model_type,
        num_experts=args.num_experts,
        use_brightness_enhancer=args.use_brightness_enhancer,
        brightness_enhancer_ch=args.brightness_enhancer_ch,
        brightness_threshold=args.brightness_threshold,
    ).to(device)
    
    # 加载权重
    model.load_state_dict(ckpt["model"])
    print("Loaded model weights")

    # EMA 权重（本项目 checkpoint 的 EMA 格式是 {"shadow": name->tensor, ...}）
    if args.use_ema and "ema" in ckpt:
        ema_state = ckpt["ema"]
        try:
            ema = EMA(
                model,
                decay=float(ema_state.get("decay", 0.9999)) if isinstance(ema_state, dict) else 0.9999,
                warmup=bool(ema_state.get("warmup", True)) if isinstance(ema_state, dict) else True,
                requires_grad_only=bool(ema_state.get("requires_grad_only", True)) if isinstance(ema_state, dict) else True,
                update_buffers=bool(ema_state.get("update_buffers", False)) if isinstance(ema_state, dict) else False,
                register_new_params=bool(ema_state.get("register_new_params", False)) if isinstance(ema_state, dict) else False,
            )
            ema.load_state_dict(ema_state, device=str(device))
            ema.apply_to(model)
            print("Loaded EMA weights")
        except Exception as e:
            print(f"Warning: failed to load EMA weights ({e}), using model weights")
    
    model.eval()
    
    # 加载测试集
    print(f"Loading test dataset from {args.test_lmdb}...")
    test_dataset = LMDBAllWeatherDataset(
        lmdb_path=args.test_lmdb,
        patch_size=None,  # 使用全分辨率
        is_train=False,
    )
    
    if args.max_samples > 0:
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, range(min(args.max_samples, len(test_dataset))))
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    if args.one_step:
        print(f"Sampling: one-step, TTA={args.use_tta}")
    else:
        print(f"Sampling: steps={args.num_steps}, type={args.sample_type}, TTA={args.use_tta}")
    print("="*60)
    
    # 评估
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    per_deg = defaultdict(lambda: {"psnr": 0.0, "ssim": 0.0, "cnt": 0})
    
    pbar = tqdm(test_loader, desc="Evaluating", dynamic_ncols=True)
    for batch in pbar:
        y = batch["LQ"].to(device, non_blocking=True)  # LQ作为起始
        x = batch["GT"].to(device, non_blocking=True)  # HQ作为目标
        deg_names = batch.get("deg_name", [None] * y.shape[0])
        
        # FoD推理
        if args.one_step:
            x_hat = fod_one_step_inference(model, y, use_tta=args.use_tta)
        else:
            x_hat = fod_inference(
                model, y,
                num_steps=args.num_steps,
                sample_type=args.sample_type,
                use_tta=args.use_tta
            )
        
        # 计算指标
        psnr = psnr_y_torch(x_hat, x, data_range=2.0)
        ssim = ssim_torch(x_hat, x, data_range=2.0)
        
        batch_size = y.shape[0]
        psnr_sum += psnr.item() * batch_size
        ssim_sum += ssim.item() * batch_size
        n += batch_size
        
        # 按退化类型统计
        for i in range(batch_size):
            deg = deg_names[i] if isinstance(deg_names[i], str) else "unknown"
            if batch_size == 1:
                per_deg[deg]["psnr"] += psnr.item()
                per_deg[deg]["ssim"] += ssim.item()
            else:
                per_deg[deg]["psnr"] += psnr[i].item() if psnr.dim() > 0 else psnr.item()
                per_deg[deg]["ssim"] += ssim[i].item() if ssim.dim() > 0 else ssim.item()
            per_deg[deg]["cnt"] += 1
        
        # 更新进度条
        pbar.set_postfix({
            "PSNR": f"{psnr_sum/n:.2f}",
            "SSIM": f"{ssim_sum/n:.4f}",
        })
    
    pbar.close()
    
    # 输出结果
    avg_psnr = psnr_sum / max(1, n)
    avg_ssim = ssim_sum / max(1, n)
    
    print("\n" + "="*80)
    print("=== Evaluation Results ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Total samples: {n}")
    if args.one_step:
        print(f"Sampling: one-step, TTA={args.use_tta}")
    else:
        print(f"Sampling: {args.num_steps} steps, type={args.sample_type}, TTA={args.use_tta}")
    print("="*80)
    print(f"\n{'Overall Results':^80}")
    print(f"{'PSNR(Y)':>20} {avg_psnr:>15.4f} dB")
    print(f"{'SSIM':>20} {avg_ssim:>15.6f}")
    print("="*80)
    
    # 按退化类型输出
    if len(per_deg) > 1:
        print(f"\n{'Per-Degradation Results':^80}")
        print(f"{'Degradation':<30} {'Count':>10} {'PSNR(Y)':>15} {'SSIM':>15}")
        print("-"*80)
        for deg in sorted(per_deg.keys()):
            c = per_deg[deg]["cnt"]
            if c == 0:
                continue
            deg_psnr = per_deg[deg]["psnr"] / c
            deg_ssim = per_deg[deg]["ssim"] / c
            print(f"{deg:<30} {c:>10} {deg_psnr:>15.4f} {deg_ssim:>15.6f}")
        print("="*80)


if __name__ == "__main__":
    main()
