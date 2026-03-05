# utils/ema.py
# Exponential Moving Average (EMA) - DDP-safe, in-place, checkpoint-friendly

import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional


def _unwrap_model(model):
    """兼容 DDP / DP：取到真实 module"""
    return model.module if hasattr(model, "module") else model


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.

    关键特性：
    - DDP-safe：内部自动 unwrap，不会出现 'module.' 前缀不匹配
    - in-place 更新：避免每步创建新 tensor
    - 支持 state_dict / load_state_dict（并兼容旧格式：直接是 shadow dict）
    - 支持评估时上下文切换：with ema.average_parameters(model): ...
    """

    def __init__(
        self,
        model,
        decay: float = 0.9999,
        warmup: bool = True,
        requires_grad_only: bool = True,
        update_buffers: bool = False,
        register_new_params: bool = False,
    ):
        """
        Args:
            model: PyTorch 模型（可为 DDP 包装）
            decay: EMA 衰减率
            warmup: 是否对前期 EMA 做 warmup（避免初期 EMA 过"迟钝"）
            requires_grad_only: 只跟踪 requires_grad=True 的参数（一般够用）
            update_buffers: 是否跟踪/复制 buffers（BN running_mean/var 等），默认 False
            register_new_params: 如果训练中出现"新增参数"（不推荐的写法），是否自动纳入 EMA
        """
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.requires_grad_only = bool(requires_grad_only)
        self.update_buffers = bool(update_buffers)
        self.register_new_params = bool(register_new_params)

        self.num_updates = 0

        self.shadow: Dict[str, torch.Tensor] = {}
        self.shadow_buffers: Dict[str, torch.Tensor] = {}

        self.backup: Dict[str, torch.Tensor] = {}
        self.backup_buffers: Dict[str, torch.Tensor] = {}

        m = _unwrap_model(model)
        with torch.no_grad():
            for name, p in m.named_parameters():
                if self.requires_grad_only and (not p.requires_grad):
                    continue
                self.shadow[name] = p.detach().clone()

            if self.update_buffers:
                for name, b in m.named_buffers():
                    # 只处理浮点 buffer（BN 的 running_mean/var 是 float）
                    if torch.is_floating_point(b):
                        self.shadow_buffers[name] = b.detach().clone()

    def _get_decay(self) -> float:
        """可选 warmup：让 EMA 在最初一段时间更快跟上参数"""
        if not self.warmup:
            return self.decay
        # 常见 warmup：前期用更小 decay，num_updates 增大后趋近 self.decay
        # (1+n)/(10+n) 约在 n=0 时 0.09，n=100 时 0.91，之后逐渐接近 1
        self.num_updates += 1
        d = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, d)

    @torch.no_grad()
    def update(self, model):
        """更新 EMA 参数（推荐每步调用一次）"""
        m = _unwrap_model(model)
        decay = self._get_decay()
        one_minus = 1.0 - decay

        for name, p in m.named_parameters():
            if self.requires_grad_only and (not p.requires_grad):
                continue

            if name not in self.shadow:
                if self.register_new_params:
                    self.shadow[name] = p.detach().clone()
                continue

            ema_p = self.shadow[name]
            src = p.detach()

            # 若 dtype/device 不一致，尽量对齐到 EMA tensor（避免每步额外分配，通常不会触发）
            if src.dtype != ema_p.dtype:
                src = src.to(dtype=ema_p.dtype)
            if src.device != ema_p.device:
                src = src.to(device=ema_p.device)

            # in-place: ema = decay*ema + (1-decay)*src
            # 用 lerp_ 等价且更简洁：ema = ema + (src-ema)*one_minus
            ema_p.lerp_(src, one_minus)

        if self.update_buffers:
            # buffers 通常不再做 EMA（BN 本身就是 moving average），更常见做法是"直接复制"
            for name, b in m.named_buffers():
                if not torch.is_floating_point(b):
                    continue
                if name not in self.shadow_buffers:
                    if self.register_new_params:
                        self.shadow_buffers[name] = b.detach().clone()
                    continue
                dst = self.shadow_buffers[name]
                src = b.detach()
                if src.dtype != dst.dtype:
                    src = src.to(dtype=dst.dtype)
                if src.device != dst.device:
                    src = src.to(device=dst.device)
                dst.copy_(src)

    @torch.no_grad()
    def apply_to(self, model):
        """把 EMA 权重覆盖到模型上（会备份当前权重，以便 restore）"""
        m = _unwrap_model(model)
        self.backup = {}
        self.backup_buffers = {}

        for name, p in m.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            src = self.shadow[name]
            if src.dtype != p.dtype or src.device != p.device:
                src = src.to(device=p.device, dtype=p.dtype)
            p.copy_(src)

        if self.update_buffers:
            for name, b in m.named_buffers():
                if name not in self.shadow_buffers:
                    continue
                self.backup_buffers[name] = b.detach().clone()
                src = self.shadow_buffers[name]
                if src.dtype != b.dtype or src.device != b.device:
                    src = src.to(device=b.device, dtype=b.dtype)
                b.copy_(src)

    @torch.no_grad()
    def restore(self, model):
        """恢复 apply_to 之前的原始权重"""
        m = _unwrap_model(model)

        for name, p in m.named_parameters():
            if name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}

        if self.update_buffers:
            for name, b in m.named_buffers():
                if name in self.backup_buffers:
                    b.copy_(self.backup_buffers[name])
            self.backup_buffers = {}

    @contextmanager
    def average_parameters(self, model):
        """评估时推荐用：with ema.average_parameters(model): ..."""
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict[str, Any]:
        """用于 checkpoint 保存"""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "requires_grad_only": self.requires_grad_only,
            "update_buffers": self.update_buffers,
            "register_new_params": self.register_new_params,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
            "shadow_buffers": self.shadow_buffers,
        }

    def load_state_dict(self, state: Dict[str, Any], device: Optional[str] = None):
        """
        兼容两种格式：
        1) 新格式：{"shadow": ..., "decay": ..., ...}
        2) 旧格式：直接就是 shadow dict（你现在 checkpoint["ema"] 就是这种）
        """
        # 旧格式兼容：state 本身就是 name->tensor
        if "shadow" not in state:
            shadow = state
            meta = {}
            shadow_buffers = {}
        else:
            meta = state
            shadow = meta.get("shadow", {})
            shadow_buffers = meta.get("shadow_buffers", {})
            self.decay = float(meta.get("decay", self.decay))
            self.warmup = bool(meta.get("warmup", self.warmup))
            self.requires_grad_only = bool(meta.get("requires_grad_only", self.requires_grad_only))
            self.update_buffers = bool(meta.get("update_buffers", self.update_buffers))
            self.register_new_params = bool(meta.get("register_new_params", self.register_new_params))
            self.num_updates = int(meta.get("num_updates", 0))

        self.shadow = {}
        for k, v in shadow.items():
            vv = v.detach().clone()
            if device is not None:
                vv = vv.to(device)
            self.shadow[k] = vv

        self.shadow_buffers = {}
        for k, v in shadow_buffers.items():
            vv = v.detach().clone()
            if device is not None:
                vv = vv.to(device)
            self.shadow_buffers[k] = vv
