<div align="center">

# F²D-Net

### Factorized Stochastic Transport for Composite Degradation Image Restoration

[Xin Su](mailto:suxin4726@gmail.com)<sup>1,2,3,4</sup>, [Jianshu Chao](mailto:jchao@fjirsm.ac.cn)<sup>2,3,4*</sup>, Huifang Shen<sup>2,3,4</sup>, Anqi Chen<sup>2,3,4,5</sup>, Yuting Gao<sup>2,3,4,6</sup>

<sup>1</sup>Fuzhou University &nbsp; <sup>2</sup>Quanzhou Institute of Equipment Manufacturing, Haixi Institutes, CAS &nbsp; <sup>3</sup>FJIRSM, CAS &nbsp; <sup>4</sup>Fujian College, UCAS &nbsp; <sup>5</sup>FAFU &nbsp; <sup>6</sup>FJNU

<sup>*</sup>Corresponding author

**Pattern Recognition** (Under Review)

[[Paper]](https://arxiv.org/) [[Project Page]](https://sxvvv.github.io/f2net/)

<img src="assets/model.png" width="90%">
</div>

## Overview

**F²D-Net** is a unified image restoration framework that handles **composite degradations** (e.g., simultaneous low-light + haze + rain) through factorized stochastic flow transport.

**Key Features:**
- **Stochastic Transport**: State-dependent multiplicative noise whose magnitude scales with the local residual — large far from the solution to explore diverse reconstructions, vanishing upon convergence to preserve fine details.
- **Closed-Form Transitions**: Log-normal structure enables single-step inference at feed-forward speed.
- **Factorized Flow Field**: Shared backbone + degradation-specific expert increments assembled at every pixel via learned spatial gating.
- **10.3x Expert Selectivity**: Pixel-wise gating achieves 10.3x selectivity on CDD-11, versus 1.2x for image-level routing.

## Highlights

- Two failure modes of existing all-in-one restoration on composite degradations are identified.
- A factorized velocity field with spatially gated mixture-of-experts is proposed.
- State-dependent multiplicative noise enables closed-form single-step inference.
- Best structural and perceptual quality on CDD-11 with 10.3x expert selectivity.

## Installation

```bash
git clone https://github.com/sxvvv/f2net.git
cd f2net

conda create -n f2dnet python=3.10
conda activate f2dnet

pip install torch>=2.0 torchvision
pip install -r requirements.txt
```

**Tested environment:** 2x NVIDIA A100 80GB, PyTorch 2.0+.

## Dataset Preparation

We use **LMDB** format for efficient I/O. Each entry is a pickled dict:

```python
{"LQ": np.ndarray,  # degraded image, uint8, HxWx3
 "GT": np.ndarray,  # clean image, uint8, HxWx3
 "deg_name": str}   # degradation label, e.g. "low_haze_rain"
```

### CDD-11 (Composite Degradation Dataset)

CDD-11 covers 11 degradation configurations composed from 4 atomic types: **low-light (L)**, **haze (H)**, **rain (R)**, and **snow (S)**.

| Tier   | Configurations          |
| ------ | ----------------------- |
| Single | L, H, R, S              |
| Double | L+H, L+R, L+S, H+R, H+S |
| Triple | L+H+R, L+H+S            |

Organize your data as:

```
data/
├── CDD11/
│   ├── train.lmdb
│   └── test.lmdb
├── 3task/
│   ├── train.lmdb
│   └── test.lmdb
└── 5task/
    ├── train.lmdb
    └── test.lmdb
```

## Model Architecture

| Component             | Description                                  | Params     |
| --------------------- | -------------------------------------------- | ---------- |
| SharedBackbone        | 4-stage U-Net, base_ch=64, ch_mult=(1,2,4,4) | 21.49M     |
| BrightnessPreEnhancer | 6 residual blocks, adaptive gating           | 0.48M      |
| DegradationParser     | 4-layer CNN + spatial decoder                | 0.47M      |
| Expert Adapters (x4)  | Dual-conv, d=128, dilation 1-4, SE           | 1.25M      |
| Gating MLP            | Time-conditioned routing                     | 0.08M      |
| **Total**             |                                              | **23.77M** |

## Training

```bash
# CDD-11
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_fod.py \
    --lmdb-path data/CDD11/train.lmdb \
    --test-lmdb-path data/CDD11/test.lmdb \
    --batch-size 16 \
    --patch-size 512 \
    --niter 500000 \
    --lr 3e-4 \
    --loss-type charbonnier \
    --lambda-freq 0.05 \
    --output-dir results/cdd11
```

## Evaluation

Evaluation code and pretrained models will be released upon paper acceptance.

## Results

### CDD-11 (Composite Degradation)

| Method | Single PSNR | Pairwise PSNR | Triple PSNR | Avg PSNR | Avg SSIM |
|--------|------------|---------------|-------------|----------|----------|
| AirNet | 25.60 | 22.97 | 22.02 | 23.75 | 0.814 |
| PromptIR | 28.88 | 24.46 | 23.54 | 25.90 | 0.850 |
| OneRestore | 31.68 | 27.35 | 24.84 | 28.47 | 0.878 |
| MoCE-IR | **32.54** | **27.73** | **25.40** | **29.05** | 0.881 |
| **F2D-Net** | 32.08 | 27.72 | 25.18 | 28.72 | **0.884** |

### Three-Task All-in-One

| Method | Dehaze | Derain | Denoise s=15 | Denoise s=25 | Denoise s=50 |
|--------|--------|--------|-------------|-------------|-------------|
| PromptIR | 30.58 | 36.37 | 33.98 | 31.31 | 28.06 |
| MoCE-IR | 31.34 | **38.57** | 34.11 | 31.45 | 28.18 |
| **F2D-Net** | **31.40** | 37.62 | **34.48** | **31.74** | **28.35** |

### Five-Task All-in-One

| Method | Params | Dehaze | Derain | Denoise | Deblur | Low-light | Avg PSNR |
|--------|--------|--------|--------|---------|--------|-----------|----------|
| MoCE-IR | 25M | **30.48** | 38.04 | 31.34 | **30.05** | 23.00 | 30.58 |
| **F2D-Net** | 23.8M | 30.35 | **38.42** | **31.48** | 29.87 | **23.32** | **30.69** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{su2025f2dnet,
  title={Factorized Stochastic Transport for Composite Degradation Image Restoration},
  author={Su, Xin and Chao, Jianshu and Shen, Huifang and Chen, Anqi and Gao, Yuting},
  journal={Pattern Recognition},
  year={2025}
}
```

## Acknowledgements

This work was supported by Fuzhou University and the Chinese Academy of Sciences.

## License

This project is released under the [MIT License](LICENSE).
