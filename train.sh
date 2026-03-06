#!/bin/bash
# =================================================================
# 3-Task All-in-One 图像恢复训练脚本 (去噪 / 去雾 / 去雨)
#
# 硬件: RTX 50 系列 (32GB VRAM), 单卡, AMP bf16
# 数据: PromptIR 风格合并 LMDB (BSD400+WED 去噪, RESIDE 去雾, Rain13K 去雨)
#
# 训练策略:
#   - 3 专家 MoE: 每个专家对应一个退化类型
#   - patch_size=256, effective_batch=32 (bs=16 x grad_accum=2)
#   - lr=2.5e-4, Charbonnier + FFT 频域损失 + 重建损失
#   - 困难样本加权: PSNR < 28 dB 样本权重 x2
#   - 余弦退火 + warmup 5K 步, EMA 0.9999
#   - 300K 步 (~170 epochs on ~28K samples)
#
# 模型: ~23M 参数 (3 专家, adapter_dim=128, BottleneckAttn, FreqEmb)
#       无亮度预增强模块 (不含低光任务)
#
# 显存估算: ~29 GB (bs=16, patch=256, bf16 AMP)
#
# 注意:
#   - 不使用 --channels-last (Blackwell 架构兼容性)
#   - 不使用 --readahead (避免 LMDB SIGBUS)
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ===== 路径配置 (请根据实际环境修改为绝对路径) =====
TRAIN_LMDB="/root/data/merged.lmdb"
RAIN_TRAIN_LMDB="/root/raintrainl.lmdb"
# 如果有单独的测试集 LMDB, 取消注释:
# TEST_LMDB="${SCRIPT_DIR}/promptir_combined_test.lmdb"
TEST_LMDB="/root/f2net/rain100l.lmdb"

OUTPUT_DIR="results/fod_3task_v1"

# ===== GPU 配置 =====
export CUDA_VISIBLE_DEVICES=0

# ===== 构建命令 =====
CMD="python ${SCRIPT_DIR}/train_v3.py"

# 数据路径
CMD="$CMD --lmdb-path ${TRAIN_LMDB}"
if [ -n "$TEST_LMDB" ] && [ -d "$TEST_LMDB" ]; then
    CMD="$CMD --test-lmdb-path ${TEST_LMDB}"
fi
CMD="$CMD --output-dir ${OUTPUT_DIR}"

# ===== Rain 训练数据 (通过 extra-lmdb 加入) =====
if [ -n "$RAIN_TRAIN_LMDB" ] && [ -d "$RAIN_TRAIN_LMDB" ]; then
    CMD="$CMD --extra-lmdb ${RAIN_TRAIN_LMDB}"
    CMD="$CMD --extra-repeat 100"
    CMD="$CMD --extra-is-train true"
fi

# ===== 核心: 3-task 动态因子配置 =====
CMD="$CMD --num-experts 3"
CMD="$CMD --factors noise,haze,rain"

# ===== 训练超参 =====
# bs=14 x grad_accum=2 = effective batch 28
# 32GB VRAM bf16 实测: bs=14 @ patch=256 → ~25GB
CMD="$CMD --patch-size 256"
CMD="$CMD --batch-size 16"
CMD="$CMD --grad-accum 2"
CMD="$CMD --lr 2.5e-4"
CMD="$CMD --beta2 0.99"
CMD="$CMD --niter 300000"
CMD="$CMD --warmup-steps 5000"

# ===== 模型架构 =====
CMD="$CMD --base-ch 64"
CMD="$CMD --emb-dim 256"
CMD="$CMD --adapter-dim 128"
CMD="$CMD --num-timesteps 100"
CMD="$CMD --model-type SFLOW"
CMD="$CMD --freq-emb-dim 128"
CMD="$CMD --no-brightness-enhancer"

# ===== 损失函数 =====
CMD="$CMD --loss-type charbonnier"
CMD="$CMD --charbonnier-eps 1e-3"
CMD="$CMD --lambda-freq 0.05"
CMD="$CMD --lambda-recon 0.1"
CMD="$CMD --lambda-cls 0.01"
CMD="$CMD --lambda-w 0.1"
CMD="$CMD --lambda-m 0.05"
CMD="$CMD --lambda-balance 0.01"
CMD="$CMD --lambda-alpha 0.02"

# ===== 困难样本加权 =====
CMD="$CMD --use-hard-sample-weighting"
CMD="$CMD --hard-sample-psnr-threshold 35.0"
CMD="$CMD --hard-sample-weight-factor 2.0"

# ===== 退化感知动态加权 =====
CMD="$CMD --deg-reweight"
CMD="$CMD --deg-reweight-target 30.0"
CMD="$CMD --deg-reweight-max 3.0"
CMD="$CMD --deg-reweight-min 0.5"

# ===== 评估 =====
CMD="$CMD --eval-every 10000"
CMD="$CMD --eval-one-step"
CMD="$CMD --eval-max-samples 100"
CMD="$CMD --eval-num-steps 10"
CMD="$CMD --sample-type MC"
CMD="$CMD --save-every 20000"
CMD="$CMD --patience 40"

# ===== AMP + 性能优化 =====
# bf16 AMP (RTX 50 Blackwell 原生支持)
CMD="$CMD --amp --amp-dtype bf16"
CMD="$CMD --tf32"
CMD="$CMD --cudnn-benchmark"
# 注意: 不用 --channels-last (Blackwell 新架构可能 SIGBUS)
# 注意: 不用 --readahead (LMDB mmap 在某些文件系统上会 SIGBUS)

# ===== EMA =====
CMD="$CMD --ema-decay 0.9999"

# ===== 数据加载 =====
CMD="$CMD --num-workers 4"
CMD="$CMD --prefetch-factor 2"

# ===== 随机种子 =====
CMD="$CMD --seed 42"

# ===== 日志 =====
CMD="$CMD --log-every 100"

# ===== 恢复训练 =====
CMD="$CMD --resume /root/f2net/results/fod_3task_v1/ckpt/0080000.pt"

# ===== 执行 =====
echo "============================================"
echo "3-Task Training (Denoise / Dehaze / Derain)"
echo "============================================"
echo "Output:     ${OUTPUT_DIR}"
echo "Train LMDB: ${TRAIN_LMDB}"
echo "Rain LMDB:  ${RAIN_TRAIN_LMDB}"
echo "Test LMDB:  ${TEST_LMDB:-'(none)'}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Batch:      16 x grad_accum=2 = effective 32"
echo "Patch:      256"
echo "LR:         2.5e-4"
echo "AMP:        bf16"
echo "============================================"
echo ""
echo "Command:"
echo "$CMD"
echo ""

mkdir -p "${OUTPUT_DIR}/ckpt"
eval $CMD
