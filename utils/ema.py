# utils/ema.py

import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional


def _unwrap_model(model):
    """Unwrap DDP/DP wrapper to access the underlying module."""
    return model.module if hasattr(model, "module") else model


class EMA:
    """Exponential Moving Average of model parameters.

    Features:
    - DDP-safe: automatically unwraps ``DistributedDataParallel``.
    - In-place updates via ``lerp_`` (no extra allocations per step).
    - ``state_dict`` / ``load_state_dict`` for checkpointing.
    - Context manager ``average_parameters`` for evaluation.

    Args:
        model: PyTorch model (may be DDP-wrapped).
        decay: EMA decay rate (e.g. 0.9999).
        warmup: Ramp up the effective decay during early training.
        requires_grad_only: Only track parameters with ``requires_grad=True``.
        update_buffers: Also track floating-point buffers (e.g. BN stats).
        register_new_params: Auto-register parameters that appear after init.
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
                    if torch.is_floating_point(b):
                        self.shadow_buffers[name] = b.detach().clone()

    def _get_decay(self) -> float:
        """Compute effective decay with optional warmup."""
        if not self.warmup:
            return self.decay
        self.num_updates += 1
        d = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, d)

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters (call once per training step)."""
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
            if src.dtype != ema_p.dtype:
                src = src.to(dtype=ema_p.dtype)
            if src.device != ema_p.device:
                src = src.to(device=ema_p.device)
            ema_p.lerp_(src, one_minus)

        if self.update_buffers:
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
        """Copy EMA weights into the model (backs up originals for restore)."""
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
        """Restore the original (non-EMA) weights after ``apply_to``."""
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
        """Context manager: temporarily apply EMA weights for evaluation."""
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def state_dict(self) -> Dict[str, Any]:
        """Serialise for checkpointing."""
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
        """Load from checkpoint.

        Supports two formats:
        1. New: ``{"shadow": ..., "decay": ..., ...}``
        2. Legacy: the dict itself is the shadow (name → tensor).
        """
        if "shadow" not in state:
            shadow, meta, shadow_buffers = state, {}, {}
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