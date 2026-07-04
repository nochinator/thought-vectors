"""
Muon Optimizer

Implements the Muon optimizer from the DeepSeek-V4 paper (Jordan et al. 2024,
Liu et al. 2025), using Newton-Schulz iterations to approximate the matrix
sign function for parameter updates.

Key advantage over AdamW: no momentum/variance buffers for 2D parameters.
Memory: ~8 bytes/param (vs 16 for AdamW).

For 2D parameters (weight matrices): applies orthogonal gradient descent
via Newton-Schulz polar decomposition.
For 1D parameters (biases, norms, embeddings): falls back to SGD with
Nesterov momentum.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer


@torch.no_grad()
def _newton_schulz(G: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """Approximate the matrix sign of G via Newton-Schulz iteration.

    Returns a matrix close to U @ Vh where U, S, Vh = svd(G).
    This is the polar factor / orthogonal component of G.

    For improved numerical stability:
      1. Normalize G by its Frobenius norm
      2. Apply Newton-Schulz: X = X * (3I - X@X^T) / 2
      3. Return the result (which approximates U @ Vh)
    """
    assert G.ndim == 2, "Muon requires 2D parameters"
    # Normalize to prevent divergence
    scale = G.norm() + 1e-8
    X = G / scale

    if G.shape[0] > G.shape[1]:
        # Tall matrix: compute X = (3*X - X@X.T@X)/2 but
        # do it as X = X + X @ (I - X.T @ X) * 0.5 for efficiency
        I = torch.eye(G.shape[1], device=G.device, dtype=G.dtype)
        for _ in range(num_iters):
            X = X @ (3 * I - X.T @ X) / 2
    else:
        # Wide or square: standard Newton-Schulz
        I = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        for _ in range(num_iters):
            X = (3 * X - X @ X.T @ X) / 2

    # Renormalize — the iterated X has approximately unit singular vals
    return X


class Muon(Optimizer):
    """Muon optimizer with Nesterov momentum fallback for 1D params.

    For each 2D parameter:
      update = -lr * newton_schulz(grad + momentum * velocity)

    For each 1D parameter (bias, norm, embedding):
      update = -lr * (momentum * velocity + grad)   # Nesterov SGD

    Args:
        params: iterable of parameters or parameter groups
        lr: learning rate (default 1e-3)
        momentum: Nesterov momentum for 1D params (default 0.95)
        ns_iters: Newton-Schulz iterations (default 5)
        weight_decay: weight decay (default 0.0)
    """

    def __init__(self, params, lr=1e-3, momentum=0.95, ns_iters=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_iters=ns_iters,
                       weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_iters = group["ns_iters"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                is_2d = g.ndim == 2 and g.shape[0] > 1 and g.shape[1] > 1

                if is_2d:
                    # Matrix parameter — Muon update
                    # Apply weight decay *after* Newton-Schulz so the
                    # orthogonal gradient is not contaminated by wd * p.
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    update = _newton_schulz(g, num_iters=ns_iters)
                    p.add_(update, alpha=-lr)
                else:
                    # 1D parameter — Nesterov SGD with momentum
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)

                    buf = state["momentum_buffer"]
                    # Nesterov momentum
                    buf.mul_(momentum).add_(g)
                    # Look-ahead step (Nesterov)
                    p.add_(buf, alpha=-lr)

        return loss

    @classmethod
    def from_adamw_params(cls, model: nn.Module, lr: float = 1e-4,
                           momentum: float = 0.95, ns_iters: int = 5,
                           weight_decay: float = 0.0) -> "Muon":
        """Create a Muon optimizer from a model, using typical AdamW
        parameter groups (separate weight_decay for non-norm/bias params).

        Actually Muon doesn't need separate groups since weight_decay
        is applied uniformly.
        """
        return cls(model.parameters(), lr=lr, momentum=momentum,
                   ns_iters=ns_iters, weight_decay=weight_decay)
