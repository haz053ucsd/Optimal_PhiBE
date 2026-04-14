from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LQR1DProblem:
    A: float
    B: float
    sig: float
    R: float
    Q: float


@dataclass(frozen=True)
class LQR2DProblem:
    A: torch.Tensor
    B: torch.Tensor
    sig: float
    R: torch.Tensor
    Q: torch.Tensor


@dataclass(frozen=True)
class MertonProblem:
    r: float
    r_b: float
    mu: float
    sig: float
    gamma: float
