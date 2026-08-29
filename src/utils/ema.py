"""
src/utils/ema.py

Model Exponential Moving Average (EMA) for PyTorch.

Maintains a smoothed shadow copy of model weights that reduces batch-to-batch
variance and improves generalization, particularly when training detectors from scratch.

Formula:
    θ_ema = β * θ_ema + (1 - β) * θ
where β is dynamically ramped up in early iterations (decay * (1 - exp(-step / tau))).
"""

import copy
import math
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).

    Keeps a moving average of model parameters for validation and testing.

    Args:
        model: PyTorch module whose parameters to track.
        decay: Maximum decay factor (momentum). Default: 0.9999.
        tau  : Warmup constant for dynamic decay. Default: 2000.0.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000.0,
    ) -> None:
        # Unwrap DataParallel if wrapped
        base_model = model.module if hasattr(model, "module") else model
        self.ema = copy.deepcopy(base_model).eval()
        self.decay = decay
        self.tau = tau
        self.updates = 0

        # Disable gradients for shadow parameters
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters with current model weights."""
        with torch.no_grad():
            self.updates += 1
            # Dynamic decay during early iterations to avoid cold start lag
            d = self.decay * (1.0 - math.exp(-self.updates / self.tau))

            base_model = model.module if hasattr(model, "module") else model
            model_state = base_model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * d + model_state[k].detach() * (1.0 - d))

    def update_attr(self, model: nn.Module, include: Optional[list] = None) -> None:
        """Copy specific attributes from model to EMA model."""
        base_model = model.module if hasattr(model, "module") else model
        if include:
            for k in include:
                if hasattr(base_model, k):
                    setattr(self.ema, k, getattr(base_model, k))
