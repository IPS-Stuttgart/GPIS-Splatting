from __future__ import annotations

import math

import torch


SQRT_2 = math.sqrt(2.0)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / SQRT_2))


def clamp_positive(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.clamp(x, min=eps)

