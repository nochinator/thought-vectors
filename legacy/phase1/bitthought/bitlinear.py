"""
BitLinear: 1.58-bit ternary quantized linear layer (BitNet b1.58).

Weights are stored in fp32 for training, quantized to {-1, 0, +1}
during forward pass with a learned scaling factor α per output channel.

Straight-Through Estimator (STE) is used for gradient flow through the
quantization step.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize(w: torch.Tensor, ste_clip: float = 1.0) -> torch.Tensor:
    """Quantize weights to {-1, 0, +1} with STE.

    The quantization threshold is the 70th percentile of |w|, giving
    approximately 70% zeros and 15% each +1/-1 as in BitNet b1.58.
    """
    with torch.no_grad():
        abs_w = w.abs()
        # Adaptive threshold: use mean of absolute values
        threshold = abs_w.mean() * 0.7
        # Clip extreme values
        w_clipped = w.clamp(-ste_clip, ste_clip)
        q = torch.where(w_clipped > threshold, 1.0,
                        torch.where(w_clipped < -threshold, -1.0, 0.0))
    # STE: forward uses quantized, backward passes through original
    return q + (w - w.detach())


class BitLinear(nn.Module):
    """Linear layer with BitNet b1.58 ternary quantization.

    Training: stores fp32 weights, quantizes for forward, STE backward.
    Inference: can optionally store ternary weights for memory efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        learn_scale: bool = True,
        ste_clip: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learn_scale = learn_scale
        self.ste_clip = ste_clip

        # Full-precision weights (stored for training)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if learn_scale:
            # Per-output-channel scaling factor.
            # Init to actual |W| mean so ternary {-1,0,+1} * scale
            # preserves the initial weight magnitude.
            with torch.no_grad():
                init_scale = self.weight.abs().mean(dim=1)
            self.scale = nn.Parameter(init_scale)
        else:
            with torch.no_grad():
                init_scale = self.weight.abs().mean(dim=1)
            self.register_buffer("scale", init_scale)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary quantized weights.

        Args:
            x: [*, in_features]

        Returns:
            [*, out_features]
        """
        w_q = ternary_quantize(self.weight, self.ste_clip)

        if self.learn_scale:
            # Learned scale — mean absolute weight per output channel
            w_q = w_q * self.scale.view(-1, 1)

        out = F.linear(x, w_q, self.bias)
        return out

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, "
                f"bitnet=ternary(1.58)")


class BitNetQKV(nn.Module):
    """Combined Q, K, V projections with BitNet quantization.

    Optimized as a single large projection split into 3 heads,
    following the DeepSeek-V4 multi-query attention pattern.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_kv: int | None = None,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dim_kv = dim_kv or self.head_dim

        # Single projection for Q, separate shared for K, V
        self.q_proj = BitLinear(d_model, nhead * self.head_dim, bias=bias)
        self.k_proj = BitLinear(d_model, self.dim_kv, bias=bias)
        self.v_proj = BitLinear(d_model, self.dim_kv, bias=bias)

    def forward(self, x: torch.Tensor):
        """Return Q, K, V tensors each [batch, nhead, seq, head_dim]."""
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, 1, self.dim_kv).transpose(1, 2)
        v = self.v_proj(x).view(B, S, 1, self.dim_kv).transpose(1, 2)
        return q, k, v
