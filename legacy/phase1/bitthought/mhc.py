"""
Manifold-Constrained Hyper-Connections (mHC).

Adapted from DeepSeek-V4: expands the residual stream by a factor of n_mhc
and constrains the residual mapping to the Birkhoff polytope of doubly
stochastic matrices via Sinkhorn-Knopp iteration.

This stabilizes signal propagation across deep stacks — especially
important when using BitNet quantization that adds quantization noise.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinkhorn_knopp(M: torch.Tensor, num_iters: int = 8) -> torch.Tensor:
    """Project a square matrix onto the Birkhoff polytope.

    Args:
        M: [n, n] matrix (pre-softplus to ensure positivity)
        num_iters: Sinkhorn iterations (default 8)

    Returns:
        Doubly stochastic matrix (rows and columns sum to 1)
    """
    # Ensure positivity via softplus
    M = F.softplus(M) + 1e-8

    for _ in range(num_iters):
        # Row normalize
        M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        # Column normalize
        M = M / M.sum(dim=-2, keepdim=True).clamp(min=1e-8)

    return M


class ManifoldHyperConnection(nn.Module):
    """mHC residual connection block.

    Expands residual stream from [d_model] to [n_mhc, d_model],
    applies learned input/residual/output mappings constrained to
    preserve signal norm.

    The residual mapping B is constrained to the Birkhoff polytope
    (doubly stochastic) via Sinkhorn-Knopp iteration.
    """

    def __init__(
        self,
        d_model: int,
        n_mhc: int = 4,
        sinkhorn_iters: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_mhc = n_mhc
        self.sinkhorn_iters = sinkhorn_iters

        # Dynamic parameter generators — input-dependent components
        self.W_pre = nn.Linear(n_mhc * d_model, n_mhc, bias=False)
        self.W_post = nn.Linear(n_mhc * d_model, n_mhc, bias=False)
        self.W_res = nn.Linear(n_mhc * d_model, n_mhc * n_mhc, bias=False)

        # Static biases
        self.S_pre = nn.Parameter(torch.zeros(1, n_mhc))
        self.S_post = nn.Parameter(torch.zeros(1, n_mhc))
        self.S_res = nn.Parameter(torch.zeros(n_mhc, n_mhc))

        # Learnable gating factors (initialized small)
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mHC residual connection.

        Args:
            x: Input to the sub-layer, [batch, seq, d_model]
            layer_output: Output of the sub-layer, [batch, seq, d_model]

        Returns:
            Updated residual state, [batch, seq, d_model]
        """
        B, S, D = x.shape
        N = self.n_mhc

        # Expand residual stream: [B, S, D] -> [B, S, N, D]
        # We tile the input N times along a new dimension
        x_expanded = x.unsqueeze(2).expand(-1, -1, N, -1)  # [B, S, N, D]

        # Flatten for dynamic parameter generation: [B, S, N*D]
        flat = x_expanded.reshape(B, S, N * D)
        flat_norm = F.rms_norm(flat, (N * D,))

        # Generate dynamic parameters
        A_tilde = self.alpha_pre * self.W_pre(flat_norm) + self.S_pre   # [B, S, N]
        C_tilde = self.alpha_post * self.W_post(flat_norm) + self.S_post  # [B, S, N]

        res_raw = self.alpha_res * self.W_res(flat_norm)  # [B, S, N*N]
        res_raw = res_raw.reshape(B, S, N, N) + self.S_res.unsqueeze(0).unsqueeze(0)  # [B, S, N, N]

        # Apply constraints
        A = torch.sigmoid(A_tilde)  # [B, S, N] — non-negative, bounded
        C = 2.0 * torch.sigmoid(C_tilde)  # [B, S, N] — non-negative, bounded in [0, 2]

        # B: doubly stochastic via Sinkhorn-Knopp
        # Apply per position in the batch*seq dimension
        B, S_, N, _ = res_raw.shape
        B = _sinkhorn_knopp(
            res_raw.reshape(-1, N, N),
            num_iters=self.sinkhorn_iters,
        ).reshape(B, S_, N, N)

        # A: [B, S, N] -> [B, S, N, 1] for broadcasting
        A = A.unsqueeze(-1)  # [B, S, N, 1]
        # Layer input: A * x_expanded -> sum over N -> [B, S, D]
        layer_in = (A * x_expanded).sum(dim=2)  # [B, S, D]

        # Apply the actual layer (done outside this module — we just compute
        # the residual update)
        # del layer_output is passed in

        # Residual transformation: B @ x_expanded -> sum to [B, S, D]
        # B: [B, S, N, N], x_expanded: [B, S, N, D]
        residual_update = torch.matmul(B, x_expanded)  # [B, S, N, D]

        # Output mapping: C -> [B, S, N, 1], sum over N
        C = C.unsqueeze(-1)  # [B, S, N, 1]
        layer_update = C * layer_output.unsqueeze(2)  # [B, S, N, D]

        # Combine: residual + layer contributions -> sum over N
        new_x = (residual_update + layer_update).sum(dim=2)  # [B, S, D]

        return self.dropout(new_x)


class BitMHCBlock(nn.Module):
    """A transformer block with mHC residual and BitNet linear layers.

    Combines:
      - Manifold-Constrained Hyper-Connection for the residual
      - BitLinear layers for projections
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        use_mhc: bool = True,
        mhc_width: int = 4,
        sinkhorn_iters: int = 8,
        window_size: int = 0,
    ):
        super().__init__()
        self.use_mhc = use_mhc
        self.d_model = d_model
        self.head_dim = d_model // nhead
        self.window_size = window_size

        from bitthought.bitlinear import BitLinear

        # Pre-attention normalization (critical for BitNet stability)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

        # Self-attention projections (BitNet quantized)
        self.q_proj = BitLinear(d_model, d_model, bias=False)
        self.k_proj = BitLinear(d_model, self.head_dim, bias=False)
        self.v_proj = BitLinear(d_model, self.head_dim, bias=False)
        self.out_proj = BitLinear(d_model, d_model, bias=False)

        # Feed-forward (BitNet quantized)
        self.ff_gate = BitLinear(d_model, dim_feedforward, bias=False)
        self.ff_up = BitLinear(d_model, dim_feedforward, bias=False)
        self.ff_down = BitLinear(dim_feedforward, d_model, bias=False)

        # mHC or LayerNorm
        if use_mhc:
            self.mhc_attn = ManifoldHyperConnection(
                d_model, mhc_width, sinkhorn_iters, dropout
            )
            self.mhc_ff = ManifoldHyperConnection(
                d_model, mhc_width, sinkhorn_iters, dropout
            )
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        self.nhead = nhead

    def _make_sliding_window_mask(self, S: int, device: torch.device) -> torch.Tensor | None:
        """Create a sliding window causal mask [S, S].

        Each position i can attend to positions [i-W+1, i] where W = window_size.
        When window_size <= 0, returns None (full attention).
        Falls back to None for S > 16K (full attention, memory constraint).
        """
        if self.window_size <= 0 or self.window_size >= S:
            return None
        if S > 16384:
            return None
        mask = torch.full((S, S), float("-inf"), device=device)
        for i in range(S):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i + 1] = 0.0
        return mask

    def _apply_attention(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        B, S, D = x.shape
        H = self.nhead
        hd = self.head_dim

        q = self.q_proj(x).view(B, S, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, S, 1, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, S, 1, hd).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(hd)

        # Sliding window mask (applied before attn_mask — masks from outside
        # take priority)
        sw_mask = self._make_sliding_window_mask(S, x.device)
        if sw_mask is not None:
            attn = attn + sw_mask

        if attn_mask is not None:
            attn = attn + attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn_weights = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # [B, H, S, hd]
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(attn_out)

    def _apply_ffn(self, x: torch.Tensor):
        gate = F.silu(self.ff_gate(x))
        up = self.ff_up(x)
        return self.ff_down(gate * up)

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None):
        if self.use_mhc:
            # Normalize before attention (critical for BitNet stability)
            x_norm = self.attn_norm(x)
            attn_out = self._apply_attention(x_norm, attn_mask, key_padding_mask)
            x = self.mhc_attn(x, attn_out)

            x_norm = self.ffn_norm(x)
            ff_out = self._apply_ffn(x_norm)
            x = self.mhc_ff(x, ff_out)
        else:
            attn_out = self._apply_attention(
                self.norm1(x), attn_mask, key_padding_mask
            )
            x = x + self.dropout1(attn_out)

            ff_out = self._apply_ffn(self.norm2(x))
            x = x + self.dropout2(ff_out)

        return x
