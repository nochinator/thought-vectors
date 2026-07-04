"""
BitThought — Custom transformer layers with modern architecture.

Feature changes since GRU version:
  - Parallel thought vector generation (learned seed → cross-attend once)
  - K-predictor head predicts number of vectors to keep (regression 1..num_thoughts)
  - Soft mask for differentiable K selection during fine-tuning
  - No GRU, no loss predictor, no stop classifier
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitthought.config import BitThoughtConfig


# ── Helpers ────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always run RMSNorm in FP32 — x.pow(2) overflows FP16 when x > 256
        x_f32 = x.float()
        rms = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_f32 * rms * self.weight.float()).type_as(x)


def precompute_rope_freqs(d_head: int, max_len: int, theta: float = 10000.0, device: torch.device = None):
    half = d_head // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def _rotate(x, freqs_c, freqs_s):
        xr = x.float().reshape(*x.shape[:-1], -1, 2)
        x_r, x_i = xr[..., 0], xr[..., 1]
        return torch.stack([x_r*freqs_c - x_i*freqs_s, x_r*freqs_s + x_i*freqs_c], dim=-1).flatten(-2).type_as(x)
    Tq, Tk = xq.size(2), xk.size(2)
    return (_rotate(xq, cos[:Tq].unsqueeze(0).unsqueeze(0), sin[:Tq].unsqueeze(0).unsqueeze(0)),
            _rotate(xk, cos[:Tk].unsqueeze(0).unsqueeze(0), sin[:Tk].unsqueeze(0).unsqueeze(0)))


# ── SwiGLU ─────────────────────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        h = int(2 * dim_feedforward / 3)
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(h, d_model, bias=False)
        self.w3 = nn.Linear(d_model, h, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, h3 = self.w1(x), self.w3(x)
        # Activation in FP32 to prevent FP16 overflow (silu × product is large)
        gate = F.silu(h1.float()) * h3.float()
        return self.w2(gate).type_as(x)


# ── Grouped-Query Attention ────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, n_kv_heads: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else nhead
        self.n_rep = nhead // self.n_kv_heads
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.wq = nn.Linear(d_model, nhead * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(nhead * self.head_dim, d_model, bias=False)

    def forward(self, query, key=None, value=None, mask=None, padding_mask=None,
                freqs_cos=None, freqs_sin=None):
        B, Tq, d = query.shape
        kv = key if key is not None else query
        val = value if value is not None else query
        Tk = kv.size(1)
        q = self.wq(query).view(B, Tq, self.nhead, self.head_dim).transpose(1, 2)
        k = self.wk(kv).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(val).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if freqs_cos is not None and freqs_sin is not None:
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        attn_mask = None
        if mask is not None or padding_mask is not None:
            attn_mask = query.new_zeros(B, 1, Tq, Tk)
            if mask is not None:
                attn_mask = attn_mask + mask[None, None, :, :]
            if padding_mask is not None:
                attn_mask = attn_mask.masked_fill(padding_mask[:, None, None, :], float('-inf'))
        # Attention in FP32 — softmax overflows FP16 on ROCm
        q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
        attn_mask_f32 = attn_mask.float() if attn_mask is not None else None
        out = F.scaled_dot_product_attention(q_f32, k_f32, v_f32, attn_mask=attn_mask_f32,
                                             dropout_p=self.dropout_p if self.training else 0.0)
        out = out.half().type_as(q)
        return self.wo(out.transpose(1, 2).contiguous().view(B, Tq, -1))


# ── Transformer Layers ─────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, nhead, dropout=dropout)
        self.ff = SwiGLU(d_model, dim_feedforward)
        self.n1 = RMSNorm(d_model); self.n2 = RMSNorm(d_model)
        self.do = nn.Dropout(dropout)
    def forward(self, x, mask=None, key_padding_mask=None, freqs_cos=None, freqs_sin=None):
        x = x + self.do(self.self_attn(self.n1(x), mask=mask, padding_mask=key_padding_mask,
                                        freqs_cos=freqs_cos, freqs_sin=freqs_sin))
        return x + self.do(self.ff(self.n2(x)))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = GroupedQueryAttention(d_model, nhead, dropout=dropout)
        self.ff = SwiGLU(d_model, dim_feedforward)
        self.n1 = RMSNorm(d_model); self.n2 = RMSNorm(d_model); self.n3 = RMSNorm(d_model)
        self.do = nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, freqs_cos=None, freqs_sin=None):
        x = x + self.do(self.self_attn(self.n1(x), mask=tgt_mask, padding_mask=tgt_key_padding_mask,
                                        freqs_cos=freqs_cos, freqs_sin=freqs_sin))
        x = x + self.do(self.cross_attn(self.n2(x), memory, memory, padding_mask=memory_key_padding_mask,
                                         freqs_cos=freqs_cos, freqs_sin=freqs_sin))
        return x + self.do(self.ff(self.n3(x)))


# ── Encoder ────────────────────────────────────────────────────────────

class BitThoughtEncoder(nn.Module):
    """Encoder: text → parallel thought vectors + K-predictor.

    All N=num_thoughts vectors are generated in parallel via a learned
    seed cross-attended to the encoded text. A K-predictor head reads
    the pooled encoded text and outputs a scalar in [1, num_thoughts]
    indicating how many vectors to pass forward.

    During training: K is provided externally (from compression scheduler).
    During inference: K comes from the K-predictor head.
    """

    def __init__(self, config: BitThoughtConfig, shared_embed: nn.Embedding | None = None):
        super().__init__()
        d = config.d_model
        td = config.effective_thought_dim
        self.num_thoughts = config.num_thoughts
        self.gradient_checkpointing = config.use_gradient_checkpointing

        self.embed = shared_embed if shared_embed is not None else nn.Embedding(config.vocab_size, d)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d, config.nhead, config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])

        # Parallel thought generation
        self.thought_seed = nn.Parameter(torch.randn(1, config.num_thoughts, d) * 0.02)
        self.cross_attn = GroupedQueryAttention(d, config.nhead, dropout=config.dropout)
        self.cross_norm = RMSNorm(d)
        self.thought_proj = nn.Linear(d, td, bias=False)

        # K-predictor: reads pooled encoded text → scalar K in [1, num_thoughts]
        if config.use_k_predictor:
            kh = config.k_hidden_dim
            self.k_predictor = nn.Sequential(
                nn.Linear(d, kh, bias=False),
                nn.ReLU(),
                nn.Linear(kh, 1, bias=False),
                nn.Sigmoid(),  # [0, 1], then scaled to [1, num_thoughts]
            )
        else:
            self.k_predictor = None

        self.k_temperature = config.k_temperature

        d_head = d // config.nhead
        cos, sin = precompute_rope_freqs(d_head, config.max_seq_len)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def _encode_text(self, input_ids, padding_mask=None):
        x = self.embed(input_ids) * math.sqrt(self.embed.weight.size(1))
        for l in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    l, x, None, padding_mask,
                    self.freqs_cos, self.freqs_sin,
                    use_reentrant=False)
            else:
                x = l(x, key_padding_mask=padding_mask,
                      freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
        return x

    def predict_K(self, input_ids, padding_mask=None):
        """Run K-predictor on the encoded text. Returns predicted K [B, 1] in [1, num_thoughts]."""
        x = self._encode_text(input_ids, padding_mask)
        pooled = x.mean(dim=1)  # [B, d]
        k_raw = self.k_predictor(pooled)  # [B, 1] in [0, 1]
        k_pred = 1 + (self.num_thoughts - 1) * k_raw  # [B, 1] in [1, num_thoughts]
        return k_pred

    def build_soft_mask(self, K, dtype=None):
        """Build soft mask weights from K values.
        
        Returns [B, num_thoughts] weights where position i is ~1 if K > i, ~0 if K < i.
        """
        if dtype is None:
            dtype = K.dtype
        positions = torch.arange(self.num_thoughts, device=K.device, dtype=dtype)  # [128]
        weights = torch.sigmoid(self.k_temperature * (K - positions - 0.5))         # [B, 128]
        return weights

    def forward(self, input_ids, padding_mask=None, K=None):
        """Forward pass.

        Args:
            input_ids: [B, T]
            padding_mask: [B, T] bool, True = pad
            K: int or None. If int, use that many vectors (hard selection).
               If None (inference), use K-predictor to determine K.
        Returns:
            thought_vectors: [B, N, d_model]  (all N vectors, always)
            k_pred: [B, 1]  (predicted K, only meaningful if K=None)
            weights: [B, N]  (soft mask weights — all 1s if K was provided as int)
        """
        B = input_ids.size(0)
        x = self._encode_text(input_ids, padding_mask)

        # Generate all N thought vectors in parallel
        seeds = self.thought_seed.expand(B, -1, -1)  # [B, N, d]
        attended = self.cross_attn(seeds, x, x, padding_mask=padding_mask,
                                    freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
        thoughts = self.cross_norm(seeds + attended)  # [B, N, d]
        thought_vectors = self.thought_proj(thoughts)  # [B, N, td]

        if K is not None:
            # Hard selection — build a hard mask
            if isinstance(K, int):
                K_t = torch.full((B, 1), K, dtype=torch.float, device=input_ids.device)
            else:
                K_t = K.float().view(B, 1)
            # weights = 1 for first K positions, 0 for rest
            weights = self.build_soft_mask(K_t)
            # During curriculum training, we hard-slice in the training step, 
            # but return soft weights for flexibility
            k_pred = K_t
        else:
            # Use K-predictor
            pooled = x.mean(dim=1)  # [B, d]
            k_raw = self.k_predictor(pooled)  # [B, 1]
            k_pred = 1 + (self.num_thoughts - 1) * k_raw  # [B, 1] in [1, N]
            weights = self.build_soft_mask(k_pred)

        return thought_vectors, k_pred, weights


# ── Decoder ────────────────────────────────────────────────────────────

class BitThoughtDecoder(nn.Module):
    """Decoder: thought vectors → token embeddings."""

    def __init__(self, config: BitThoughtConfig, shared_embed: nn.Embedding | None = None):
        super().__init__()
        d = config.d_model
        td = config.effective_thought_dim
        self.thought_to_hidden = nn.Linear(td, d, bias=False)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(config.vocab_size, d)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d, config.nhead, config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = RMSNorm(d)
        self.embed_pred = nn.Linear(d, d, bias=False)
        self.gradient_checkpointing = config.use_gradient_checkpointing
        d_head = d // config.nhead
        cos, sin = precompute_rope_freqs(d_head, config.max_seq_len)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, thought_vectors, target_ids, target_padding_mask=None, thought_mask=None):
        T = target_ids.size(1)
        memory = self.thought_to_hidden(thought_vectors)
        tgt = self.embed(target_ids) * math.sqrt(self.embed.weight.size(1))
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=target_ids.device), diagonal=1)
        for l in self.layers:
            if self.gradient_checkpointing and self.training:
                tgt = torch.utils.checkpoint.checkpoint(
                    l, tgt, memory, causal_mask, target_padding_mask, thought_mask,
                    self.freqs_cos, self.freqs_sin, use_reentrant=False)
            else:
                tgt = l(tgt, memory, tgt_mask=causal_mask,
                        tgt_key_padding_mask=target_padding_mask,
                        memory_key_padding_mask=thought_mask,
                        freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
        tgt = self.final_norm(tgt)
        return self.embed_pred(tgt)


# ── Full Model ─────────────────────────────────────────────────────────

class BitThoughtModel(nn.Module):
    """Full BitThought model with parallel thought vectors + K-predictor.

    Forward returns (pred_embeds, thoughts, k_pred, weights).
    """

    def __init__(self, config: BitThoughtConfig):
        super().__init__()
        self.config = config
        self.shared_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = BitThoughtEncoder(config, shared_embed=self.shared_embed)
        self.decoder = BitThoughtDecoder(config, shared_embed=self.shared_embed)

    def forward(self, input_ids, padding_mask=None, K=None):
        thought_vectors, k_pred, weights = self.encoder(input_ids, padding_mask, K=K)
        # Apply soft mask to thought vectors
        masked = thought_vectors * weights.unsqueeze(-1)
        pe = self.decoder(
            masked, input_ids[:, :-1],
            None if padding_mask is None else padding_mask[:, :-1],
        )
        return pe, thought_vectors, k_pred, weights

    def embed_to_logits(self, pred_embeds: torch.Tensor) -> torch.Tensor:
        return pred_embeds @ self.shared_embed.weight.T

    @torch.no_grad()
    def generate(self, thought_vectors, max_len=384, bos_token_id=1, eos_token_id=2, temperature=1.0):
        B = thought_vectors.size(0)
        device = thought_vectors.device
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            pred_embeds = self.decoder(thought_vectors, generated)
            logits = self.embed_to_logits(pred_embeds)
            next_logits = logits[:, -1, :]
            if temperature == 0 or temperature is None:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        tokens = generated[0, 1:].tolist()
        if eos_token_id in tokens:
            tokens = tokens[:tokens.index(eos_token_id) + 1]
        return tokens


# ── Checkpoint Migration ───────────────────────────────────────────────

def migrate_state_dict(state_dict: dict) -> dict:
    """Map old checkpoints to current architecture.

    Handles GRU-based → parallel + K-predictor migration.
    """
    mapping = {
        "encoder.token_embedding.weight": "shared_embed.weight",
        "decoder.token_embedding.weight": "shared_embed.weight",
        "decoder.lm_head.weight": "shared_embed.weight",
        "decoder.lm_head.bias": None,
        "encoder.positional_encoding.pe": None,
        "decoder.positional_encoding.pe": None,
        "encoder.thought_seed": None,  # old parallel seed, re-init
        "encoder.initial_state": None,  # GRU parameter → discarded
        "encoder.thought_gru.weight_ih": None,  # GRU → discarded
        "encoder.thought_gru.weight_hh": None,
        "encoder.thought_gru.bias_ih": None,
        "encoder.thought_gru.bias_hh": None,
        "encoder.stop_classifier.weight": None,  # stop classifier → K-predictor
        "encoder.loss_predictor.0.weight": None,  # loss predictor → K-predictor
        "encoder.loss_predictor.2.weight": None,
        "encoder.cross_attention.wq.weight": "encoder.cross_attn.wq.weight",
        "encoder.cross_attention.wk.weight": "encoder.cross_attn.wk.weight",
        "encoder.cross_attention.wv.weight": "encoder.cross_attn.wv.weight",
        "encoder.cross_attention.wo.weight": "encoder.cross_attn.wo.weight",
    }
    new_dict = {}
    for k, v in state_dict.items():
        if k in mapping:
            target = mapping[k]
            if target is None:
                continue
            if target not in new_dict:
                new_dict[target] = v
        elif k.endswith(".weight") and "positional_encoding" in k:
            continue
        elif "in_proj_weight" in k:
            d = v.size(1)
            k_base = k.replace("self_attn.in_proj_weight", "self_attn.wq.weight")
            if k_base not in new_dict:
                new_dict[k_base] = v[:d]
            kk = k.replace("self_attn.in_proj_weight", "self_attn.wk.weight")
            if kk not in new_dict:
                new_dict[kk] = v[d:2*d]
            kv = k.replace("self_attn.in_proj_weight", "self_attn.wv.weight")
            if kv not in new_dict:
                new_dict[kv] = v[2*d:]
        elif "in_proj_bias" in k:
            d = v.size(0) // 3
            k_base = k.replace("self_attn.in_proj_bias", "self_attn.wq.bias")
            if k_base not in new_dict:
                new_dict[k_base] = v[:d]
            kk = k.replace("self_attn.in_proj_bias", "self_attn.wk.bias")
            if kk not in new_dict:
                new_dict[kk] = v[d:2*d]
            kv = k.replace("self_attn.in_proj_bias", "self_attn.wv.bias")
            if kv not in new_dict:
                new_dict[kv] = v[2*d:]
        elif "linear1" in k:
            kn = k.replace("linear1", "feed_forward.w1")
            if kn not in new_dict:
                new_dict[kn] = v
        elif "linear2" in k:
            new_dict[k.replace("linear2", "feed_forward.w2")] = v
        elif "norm1" in k or "norm2" in k or "norm3" in k:
            new_dict[k] = v
        elif k == "decoder.lm_head.weight" and "shared_embed.weight" not in new_dict:
            new_dict["shared_embed.weight"] = v
        elif k == "encoder.cross_attention.wq.weight":
            new_dict["encoder.cross_attn.wq.weight"] = v
        elif k == "encoder.cross_attention.wk.weight":
            new_dict["encoder.cross_attn.wk.weight"] = v
        elif k == "encoder.cross_attention.wv.weight":
            new_dict["encoder.cross_attn.wv.weight"] = v
        elif k == "encoder.cross_attention.wo.weight":
            new_dict["encoder.cross_attn.wo.weight"] = v
        else:
            new_dict[k] = v
    return new_dict
