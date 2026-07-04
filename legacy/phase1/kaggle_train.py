#!/usr/bin/env python3
"""BitThought — standalone Kaggle training script.
Contains ALL code inline. No external package imports beyond torch.
Upload this + datasets to Kaggle, hit Run.

Fixes vs previous version:
  - Compression curriculum is threshold-based (not per-batch increment)
  - GRU thought loop breaks at K during training (not just inference)
  - stop_classifier target is stable (tied to current K, not a moving ratio)
  - target_ratio only advances when rolling accuracy >= acc_threshold
  - Checkpoints saved at each compression level
  - target_ratio_max added to config to bound compression
"""
import os, sys, math, random, time, json, csv, re, datetime
from pathlib import Path
from typing import Callable
from collections import OrderedDict, deque

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
class BitThoughtConfig:
    def __init__(self, d_model=768, nhead=8, num_encoder_layers=5, num_decoder_layers=5,
                 dim_feedforward=3072, num_thoughts=128, dropout=0.1,
                 max_seq_len=384, vocab_size=32000, tokenizer_name="llama2",
                 use_loss_predictor=True, use_gradient_checkpointing=False,
                 stop_threshold=0.5, min_thoughts=3,
                 use_stop_classifier=True,
                 # Compression curriculum
                 target_ratio_start=1.0,   # starting tokens-per-vector ratio (1.0 = ~1:1, easy)
                 target_ratio_inc=0.1,     # how much to increase ratio when threshold is met
                 target_ratio_max=20.0,    # hard ceiling on compression ratio
                 acc_threshold=0.95,       # rolling accuracy required to advance compression
                 acc_window=1000,          # number of batches in rolling accuracy window
                 stop_weight=1.0):
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.num_thoughts = num_thoughts
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tokenizer_name = tokenizer_name
        self.use_loss_predictor = use_loss_predictor
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.stop_threshold = stop_threshold
        self.min_thoughts = min_thoughts
        self.use_stop_classifier = use_stop_classifier
        self.target_ratio_start = target_ratio_start
        self.target_ratio_inc = target_ratio_inc
        self.target_ratio_max = target_ratio_max
        self.acc_threshold = acc_threshold
        self.acc_window = acc_window
        self.stop_weight = stop_weight
        self.effective_thought_dim = d_model


# ─────────────────────────────────────────────────────────────────────
# MODEL — RMSNorm, RoPE, SwiGLU, GQA
# ─────────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_rope_freqs(d_head, max_len, theta=10000.0, device=None):
    half = d_head // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs.cos(), freqs.sin()

def apply_rotary_emb(xq, xk, cos, sin):
    def _rotate(x, fc, fs):
        xr = x.float().reshape(*x.shape[:-1], -1, 2)
        x_r, x_i = xr[..., 0], xr[..., 1]
        return torch.stack([x_r*fc - x_i*fs, x_r*fs + x_i*fc], dim=-1).flatten(-2).type_as(x)
    Tq, Tk = xq.size(2), xk.size(2)
    return (_rotate(xq, cos[:Tq].unsqueeze(0).unsqueeze(0), sin[:Tq].unsqueeze(0).unsqueeze(0)),
            _rotate(xk, cos[:Tk].unsqueeze(0).unsqueeze(0), sin[:Tk].unsqueeze(0).unsqueeze(0)))

class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_ff):
        super().__init__()
        h = int(2 * dim_ff / 3)
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(h, d_model, bias=False)
        self.w3 = nn.Linear(d_model, h, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, nhead, n_kv_heads=None, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads or nhead
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
        if freqs_cos is not None:
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
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                             dropout_p=self.dropout_p if self.training else 0.0)
        return self.wo(out.transpose(1, 2).contiguous().view(B, Tq, -1))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d, nh, ff, dp=0.1):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d, nh, dropout=dp)
        self.ff = SwiGLU(d, ff)
        self.n1 = RMSNorm(d); self.n2 = RMSNorm(d)
        self.do = nn.Dropout(dp)
    def forward(self, x, mask=None, padding_mask=None, fc=None, fs=None):
        x = x + self.do(self.self_attn(self.n1(x), mask=mask, padding_mask=padding_mask,
                                        freqs_cos=fc, freqs_sin=fs))
        return x + self.do(self.ff(self.n2(x)))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d, nh, ff, dp=0.1):
        super().__init__()
        self.self_attn = GroupedQueryAttention(d, nh, dropout=dp)
        self.cross_attn = GroupedQueryAttention(d, nh, dropout=dp)
        self.ff = SwiGLU(d, ff)
        self.n1 = RMSNorm(d); self.n2 = RMSNorm(d); self.n3 = RMSNorm(d)
        self.do = nn.Dropout(dp)
    def forward(self, x, mem, tm=None, tpm=None, mpm=None, fc=None, fs=None):
        x = x + self.do(self.self_attn(self.n1(x), mask=tm, padding_mask=tpm,
                                        freqs_cos=fc, freqs_sin=fs))
        x = x + self.do(self.cross_attn(self.n2(x), mem, mem, padding_mask=mpm,
                                         freqs_cos=fc, freqs_sin=fs))
        return x + self.do(self.ff(self.n3(x)))


class BitThoughtEncoder(nn.Module):
    """Encoder with autoregressive GRU-based thought generation and EOS-style stop.

    Thought vectors are generated one at a time. Each step:
      1. Cross-attend current GRU state to encoded text
      2. Project to thought vector
      3. Update GRU state
    Two stopping signals:
      - stop_classifier: BCE-trained binary "stop now" at position K (stable target)
      - loss_predictor:  MSE-trained quality estimate (fallback at inference)

    During training: runs exactly K steps (not all 128), saving compute.
    During inference: runs up to num_thoughts steps, stopping on classifier signal.
    """
    def __init__(self, cfg, shared_embed=None):
        super().__init__()
        d = cfg.d_model
        self.num_thoughts = cfg.num_thoughts
        self.min_thoughts = cfg.min_thoughts
        self.stop_threshold = cfg.stop_threshold
        self.embed = shared_embed or nn.Embedding(cfg.vocab_size, d)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_encoder_layers)
        ])
        # Autoregressive thought generation
        self.initial_state = nn.Parameter(torch.randn(1, d) * 0.02)
        self.thought_gru = nn.GRUCell(d, d)
        self.cross_attn = GroupedQueryAttention(d, cfg.nhead, dropout=cfg.dropout)
        self.cross_norm = RMSNorm(d)
        self.thought_proj = nn.Linear(d, cfg.effective_thought_dim, bias=False)
        # Loss predictor: estimates reconstruction loss if we stopped here
        self.loss_predictor = nn.Sequential(
            nn.Linear(cfg.effective_thought_dim, d // 2, bias=False),
            nn.ReLU(),
            nn.Linear(d // 2, 1, bias=False)
        ) if cfg.use_loss_predictor else None
        # Stop classifier: binary "should we stop?" trained with BCE
        self.stop_classifier = nn.Linear(cfg.effective_thought_dim, 1, bias=False) \
            if getattr(cfg, 'use_stop_classifier', True) else None
        self.gradient_checkpointing = cfg.use_gradient_checkpointing

        dh = d // cfg.nhead
        cos, sin = precompute_rope_freqs(dh, cfg.max_seq_len)
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
                x = l(x, padding_mask=padding_mask, fc=self.freqs_cos, fs=self.freqs_sin)
        return x

    def forward(self, input_ids, padding_mask=None, K=None):
        """
        Args:
            input_ids: [B, T]
            padding_mask: [B, T] bool, True = pad
            K: number of thought vectors to generate.
               - Training: pass the current curriculum K (runs exactly K steps)
               - Inference: pass None (runs up to num_thoughts, stops on classifier)
        Returns:
            thoughts:    [B, K, d]
            preds:       [B, K]   loss predictor outputs
            stop_logits: [B, K]   stop classifier logits (pre-sigmoid)
        """
        B = input_ids.size(0)
        x = self._encode_text(input_ids, padding_mask)

        state = self.initial_state.expand(B, -1)  # [B, d]
        thoughts, preds, stop_logits = [], [], []

        max_steps = K if (K is not None) else self.num_thoughts

        for i in range(max_steps):
            # Cross-attend state to encoded text
            state_3d = state.unsqueeze(1)  # [B, 1, d]
            att = self.cross_attn(state_3d, x, x, padding_mask=padding_mask,
                                  freqs_cos=self.freqs_cos, freqs_sin=self.freqs_sin)
            th = self.cross_norm(state_3d + att).squeeze(1)  # [B, d]
            tv = self.thought_proj(th)                        # [B, d_thought]

            pred = self.loss_predictor(tv).squeeze(-1) \
                if self.loss_predictor is not None \
                else torch.zeros(B, device=state.device)

            stop_logit = self.stop_classifier(tv).squeeze(-1) \
                if self.stop_classifier is not None \
                else torch.zeros(B, device=state.device)

            thoughts.append(tv)
            preds.append(pred)
            stop_logits.append(stop_logit)

            # Inference-time early stopping (not during training — K is fixed then)
            if K is None and (i + 1) >= self.min_thoughts:
                if self.stop_classifier is not None:
                    if torch.sigmoid(stop_logit).mean().item() > 0.5:
                        break
                elif self.loss_predictor is not None:
                    if pred.mean().item() < self.stop_threshold:
                        break

            # GRU state update
            state = self.thought_gru(th, state)  # [B, d]

        thoughts_t    = torch.stack(thoughts, dim=1)     # [B, K, d]
        preds_t       = torch.stack(preds, dim=1)        # [B, K]
        stop_logits_t = torch.stack(stop_logits, dim=1)  # [B, K]
        return thoughts_t, preds_t, stop_logits_t


class BitThoughtDecoder(nn.Module):
    def __init__(self, cfg, shared_embed=None):
        super().__init__()
        d = cfg.d_model
        self.to_hidden = nn.Linear(cfg.effective_thought_dim, d, bias=False)
        self.embed = shared_embed or nn.Embedding(cfg.vocab_size, d)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d, cfg.nhead, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.num_decoder_layers)
        ])
        self.final_norm = RMSNorm(d)
        self.embed_pred = nn.Linear(d, d, bias=False)
        self.gradient_checkpointing = cfg.use_gradient_checkpointing
        dh = d // cfg.nhead
        cos, sin = precompute_rope_freqs(dh, cfg.max_seq_len)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, tv, target_ids, target_padding_mask=None, thought_mask=None):
        T = target_ids.size(1)
        mem = self.to_hidden(tv)
        tgt = self.embed(target_ids) * math.sqrt(self.embed.weight.size(1))
        cm = torch.triu(torch.full((T, T), float('-inf'), device=target_ids.device), diagonal=1)
        for l in self.layers:
            if self.gradient_checkpointing and self.training:
                tgt = torch.utils.checkpoint.checkpoint(
                    l, tgt, mem, cm, target_padding_mask, thought_mask,
                    self.freqs_cos, self.freqs_sin,
                    use_reentrant=False)
            else:
                tgt = l(tgt, mem, tm=cm, tpm=target_padding_mask, mpm=thought_mask,
                        fc=self.freqs_cos, fs=self.freqs_sin)
        return self.embed_pred(self.final_norm(tgt))


class BitThoughtModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.shared_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = BitThoughtEncoder(cfg, self.shared_embed)
        self.decoder = BitThoughtDecoder(cfg, self.shared_embed)

    def forward(self, input_ids, padding_mask=None, K=None):
        tv, preds, stop_logits = self.encoder(input_ids, padding_mask, K=K)
        pe = self.decoder(tv, input_ids[:, :-1],
                          None if padding_mask is None else padding_mask[:, :-1])
        return pe, tv, preds, stop_logits

    def embed_to_logits(self, pe):
        return pe @ self.shared_embed.weight.T


# ─────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────
class ThoughtTokenizer:
    def __init__(self, vocab_path):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(vocab_path))
        self.vocab_size = self._tok.get_vocab_size()
        self.bos_id = 1; self.eos_id = 2; self.pad_id = 0

    def encode(self, text, add_special_tokens=True):
        ids = self._tok.encode(text).ids
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor): ids = ids.tolist()
        ids = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self._tok.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────
def load_csv(path):
    txts = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0].strip():
                txts.append(row[0].strip())
    return txts

def load_pairs(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                try:
                    pairs.append((row[0], row[1], float(row[2])))
                except ValueError:
                    pass
    return pairs

def tokenize_dataset(texts, tokenizer, max_len=384, cache=None):
    if cache and os.path.exists(cache + ".flat.pt"):
        d = torch.load(cache + ".flat.pt", weights_only=True)
        tok_t, lens = d["tokens"], d["lengths"]
        result = []; off = 0
        for l in lens.tolist():
            result.append(tok_t[off:off+l].tolist())
            off += l
        print(f"[cache] loaded {len(result)} seqs")
        return result
    result = [tokenizer.encode(t)[:max_len] for t in texts]
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        lengths = torch.tensor([len(s) for s in result], dtype=torch.int)
        tokens = torch.cat([torch.tensor(s, dtype=torch.int) for s in result])
        torch.save({"tokens": tokens, "lengths": lengths}, cache + ".flat.pt")
        print(f"[cache] saved {len(result)} seqs")
    return result

class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = [s for s in seqs if s]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]

def collate(batch, pad_id, max_len=384):
    seqs = [s[:max_len] for s in batch]
    ml = max(len(s) for s in seqs)
    out = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ─────────────────────────────────────────────────────────────────────
# COMPRESSION CURRICULUM STATE
# ─────────────────────────────────────────────────────────────────────
class CompressionScheduler:
    """Tracks rolling accuracy and advances compression ratio when threshold is met.

    Design:
      - Holds current target_ratio fixed until rolling accuracy >= acc_threshold
        over the last acc_window batches.
      - On threshold crossing: saves a checkpoint, increments ratio by target_ratio_inc.
      - Stops advancing when target_ratio > target_ratio_max.
      - K (number of thought vectors) = ceil(seq_len / target_ratio), clamped to [min_k, max_k].
    """
    def __init__(self, cfg, save_dir):
        self.ratio      = cfg.target_ratio_start
        self.ratio_inc  = cfg.target_ratio_inc
        self.ratio_max  = cfg.target_ratio_max
        self.threshold  = cfg.acc_threshold
        self.window     = cfg.acc_window
        self.min_k      = cfg.min_thoughts
        self.max_k      = cfg.num_thoughts
        self.save_dir   = Path(save_dir)
        self._acc_buf   = deque(maxlen=cfg.acc_window)
        self.at_ceiling = False

    def compute_K(self, seq_len):
        """Return K thought vectors for the current compression ratio."""
        return max(self.min_k, min(int(math.ceil(seq_len / max(self.ratio, 0.1))), self.max_k))

    def update(self, acc, model, history):
        """Call after each batch with that batch's accuracy.
        Returns True if the ratio just advanced (useful for logging)."""
        self._acc_buf.append(acc)
        if self.at_ceiling or len(self._acc_buf) < self.window:
            return False
        rolling_acc = sum(self._acc_buf) / len(self._acc_buf)
        if rolling_acc >= self.threshold:
            # Threshold met — checkpoint and advance
            ckpt = self.save_dir / f"ratio_{self.ratio:.1f}.pt"
            torch.save({"model": model.state_dict(),
                        "ratio": self.ratio,
                        "history": history}, str(ckpt))
            print(f"\n[curriculum] ratio {self.ratio:.1f} → {self.ratio + self.ratio_inc:.1f} "
                  f"(rolling_acc={rolling_acc:.4f} >= {self.threshold}) — saved {ckpt.name}")
            self.ratio += self.ratio_inc
            self._acc_buf.clear()  # reset window for the new level
            if self.ratio > self.ratio_max:
                self.ratio = self.ratio_max
                self.at_ceiling = True
                print(f"[curriculum] reached max ratio {self.ratio_max:.1f}, holding.")
            return True
        return False

    @property
    def rolling_acc(self):
        if not self._acc_buf: return 0.0
        return sum(self._acc_buf) / len(self._acc_buf)


# ─────────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────────
def training_step(model, input_ids, pad_id, cfg, K, *,
                  length_penalty=0.01, pred_weight=1.0, contrastive_weight=0.0,
                  exact_prob=0.0, repeat_penalty=0.0, input_ids_b=None, scores=None,
                  stop_weight=1.0):
    pm = input_ids.eq(pad_id)
    if pm[:, 0].any(): pm[:, 0] = False

    # Encoder runs exactly K steps during training
    thoughts, preds, stop_logits = model.encoder(input_ids, pm, K=K)

    tpm = pm[:, :-1].clone()
    if tpm[:, 0].any(): tpm[:, 0] = False

    pe = model.decoder(thoughts, input_ids[:, :-1], tpm)
    target = input_ids[:, 1:]
    t_emb = model.shared_embed(target)

    mse = F.mse_loss(pe, t_emb)
    logits = model.embed_to_logits(pe)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                         target.reshape(-1), ignore_index=pad_id)
    combined = (mse + ce) / 2
    recon = combined  # always use the averaged loss, no random switching

    # Repetition penalty
    rpt = torch.tensor(0.0, device=input_ids.device)
    if repeat_penalty > 0:
        with torch.no_grad():
            dec = logits.argmax(dim=-1)
            adj = (dec[:, 1:] == dec[:, :-1]).float()
            valid = (target[:, 1:] != pad_id).float()
            rpt = repeat_penalty * (adj * valid).sum() / valid.sum().clamp(min=1)

    # Contrastive
    ctr = torch.tensor(0.0, device=input_ids.device)
    if contrastive_weight > 0 and input_ids_b is not None:
        pb = input_ids_b.eq(pad_id)
        if pb[:, 0].any(): pb[:, 0] = False
        tb, _, _ = model.encoder(input_ids_b, pb, K=K)
        cs = F.cosine_similarity(thoughts.mean(1), tb.mean(1))
        ctr = F.mse_loss(cs, scores)

    # Loss predictor: train the LAST thought position to predict combined loss.
    # This is stable — it always targets the final position for the current K.
    pred_loss = F.mse_loss(preds[:, -1], combined.detach().expand(preds.size(0)))

    # Stop classifier: train position K-1 (last) to predict "stop=1", all others "stop=0".
    # This is stable — the stop position is always the last generated thought.
    stop_targets = torch.zeros_like(stop_logits)
    stop_targets[:, -1] = 1.0  # last thought = stop signal
    stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)

    len_p = recon.new_tensor(length_penalty * K)
    total = (recon + len_p
             + pred_weight * pred_loss
             + stop_weight * stop_loss
             + contrastive_weight * ctr
             + rpt)

    with torch.no_grad():
        dec = logits.argmax(dim=-1)
        valid = ~tpm
        acc = ((dec == target) & valid).float().sum() / valid.float().sum().clamp(min=1)

    return total, {
        "recon": float(recon.detach()),
        "pred":  float(pred_loss.detach()),
        "stop":  float(stop_loss.detach()),
        "ctr":   float(ctr.detach()),
        "rpt":   float(rpt.detach()),
        "total": float(total.detach()),
        "vecs":  float(K),
        "acc":   float(acc.detach()),
        "pred_l": float(preds[:, -1].mean().detach()),
        "act_l":  float(combined.detach()),
    }


# ─────────────────────────────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────────────────────────────
def train_model(model, cfg, tok, texts, *, device, epochs=1, batch_size=8, lr=3e-4,
                wd=1e-5, len_penalty=0.005, pred_weight=1.0, contrastive_weight=0.0,
                exact_prob=0.5, repeat_penalty=0.03, pred_lr=1e-3, warmup=100,
                log_every=1000, sample_every=1000, pairs=None, save_every=5000,
                save_path="checkpoint.pt", cache_name="cache",
                compression_scheduler=None):
    random.seed(42); torch.manual_seed(42)
    seqs = tokenize_dataset(texts, tok, cfg.max_seq_len, cache=cache_name)
    ds = SeqDataset(seqs)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_id, cfg.max_seq_len))

    paired_ld = None
    if pairs and contrastive_weight > 0:
        class PairDataset(Dataset):
            def __init__(self, pairs): self.pairs = [(a,b,s) for a,b,s in pairs if a and b]
            def __len__(self): return len(self.pairs)
            def __getitem__(self, i): return self.pairs[i]
        def collate_pair(batch):
            def tok_seqs(texts):
                seqs = [tok.encode(t)[:cfg.max_seq_len] for t in texts]
                ml = max(len(s) for s in seqs)
                out = torch.full((len(seqs), ml), tok.pad_id, dtype=torch.long)
                for i, s in enumerate(seqs):
                    out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                return out
            return tok_seqs([p[0] for p in batch]), \
                   tok_seqs([p[1] for p in batch]), \
                   torch.tensor([p[2] for p in batch], dtype=torch.float)
        paired_ds = PairDataset(pairs)
        paired_ld = DataLoader(paired_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_pair)
        print(f"[paired] {len(pairs)} pairs, weight={contrastive_weight}")

    pred_params  = [p for n, p in model.named_parameters() if "loss_predictor" in n]
    main_params  = [p for n, p in model.named_parameters() if "loss_predictor" not in n]
    try:
        opt = optim.AdamW([{"params": main_params, "lr": lr, "weight_decay": wd},
                           {"params": pred_params, "lr": pred_lr, "weight_decay": 0.0}],
                          fused=True)
    except Exception:
        opt = optim.AdamW([{"params": main_params, "lr": lr, "weight_decay": wd},
                           {"params": pred_params, "lr": pred_lr, "weight_decay": 0.0}])
    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])

    total_steps = epochs * len(ld)
    model.to(device)
    history = []
    paired_iter = iter(paired_ld) if paired_ld else None
    stop_weight = getattr(cfg, 'stop_weight', 1.0)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0; batches = 0; t0 = time.time()

        for bi, input_ids in enumerate(ld, 1):
            input_ids = input_ids.to(device)
            gs = epoch * len(ld) + bi - 1

            # Paired batch
            input_ids_b, sim_scores = None, None
            if paired_iter is not None:
                try:
                    _, input_ids_b, sim_scores = next(paired_iter)
                except StopIteration:
                    paired_iter = iter(paired_ld)
                    _, input_ids_b, sim_scores = next(paired_iter)
                input_ids_b = input_ids_b.to(device)
                sim_scores  = sim_scores.to(device)

            # LR schedule: warmup then cosine decay
            if warmup > 0 and gs < warmup:
                scale = gs / warmup
            else:
                p = (gs - warmup) / max(1, total_steps - warmup)
                scale = 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale

            # Compute K from current compression ratio
            seq_len = input_ids.size(1)
            if compression_scheduler is not None:
                K = compression_scheduler.compute_K(seq_len)
            else:
                K = cfg.num_thoughts

            opt.zero_grad(set_to_none=True)
            loss, stats = training_step(
                model, input_ids, tok.pad_id, cfg, K,
                length_penalty=len_penalty, pred_weight=pred_weight,
                contrastive_weight=contrastive_weight, exact_prob=exact_prob,
                repeat_penalty=repeat_penalty, input_ids_b=input_ids_b,
                scores=sim_scores, stop_weight=stop_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Update compression scheduler with this batch's accuracy
            advanced = False
            if compression_scheduler is not None:
                advanced = compression_scheduler.update(stats["acc"], model,
                                                        history + [epoch_loss / max(batches, 1)])

            epoch_loss += float(loss.detach()); batches += 1
            if torch.isnan(loss):
                raise RuntimeError(f"NaN at batch {bi}")

            if save_every > 0 and bi % save_every == 0:
                torch.save({"model": model.state_dict(),
                            "history": history + [epoch_loss / batches]}, save_path)

            if bi % log_every == 0 or advanced:
                elapsed = time.time() - t0
                avg = epoch_loss / batches
                ratio_str = f"{compression_scheduler.ratio:.1f}" \
                    if compression_scheduler else "fixed"
                roll_str  = f"{compression_scheduler.rolling_acc:.3f}" \
                    if compression_scheduler else "n/a"
                print(f"  batch {bi}/{len(ld)} | loss={stats['total']:.4f} "
                      f"recon={stats['recon']:.4f} pred={stats['pred']:.4f} "
                      f"stop={stats['stop']:.4f} vecs={int(stats['vecs'])} "
                      f"ratio={ratio_str} acc={stats['acc']:.3f} "
                      f"roll_acc={roll_str} avg={avg:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} [{elapsed:.1f}s]")

            if bi % sample_every == 0:
                model.eval()
                with torch.no_grad():
                    for idx in range(min(3, input_ids.size(0))):
                        seq_len_i = input_ids.size(1)
                        K_inf = compression_scheduler.compute_K(seq_len_i) \
                            if compression_scheduler else cfg.num_thoughts
                        tv, _, _ = model.encoder(input_ids[idx:idx+1], K=K_inf)
                        pe = model.decoder(
                            tv, input_ids[idx:idx+1, :-1],
                            input_ids[idx:idx+1, :-1].eq(tok.pad_id)
                            if input_ids.size(1) > 1 else None)
                        sm = model.embed_to_logits(pe).argmax(dim=-1)
                        orig   = tok.decode(input_ids[idx].tolist())
                        recon_t = tok.decode(sm[0].tolist())
                        print(f"  [s{idx}] orig={orig[:80]!r}")
                        print(f"  [s{idx}] recon={recon_t[:80]!r}")
                model.train()

        history.append(epoch_loss / batches)
        print(f"  epoch {epoch+1} done: avg_loss={history[-1]:.4f} "
              f"ratio={compression_scheduler.ratio:.1f}" if compression_scheduler
              else f"  epoch {epoch+1} done: avg_loss={history[-1]:.4f}")

    # Final save at end of training (avoids relying solely on save_every intervals)
    torch.save({"model": model.state_dict(), "history": history}, save_path)
    return history


# ─────────────────────────────────────────────────────────────────────
# KAGGLE HELPERS
# ─────────────────────────────────────────────────────────────────────
def kaggle_data(name):
    kaggle_input = Path("/kaggle/input/nochinator/thought-vectors-dataset")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            p = d / f"{name}.csv"
            if p.exists():
                return str(p)
    for p in [Path(f"{name}.csv"), Path(f"datasets/nochinator/thought-vectors-dataset/{name}.csv")]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"Can't find {name}.csv — upload as a Kaggle dataset or place in working dir")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"Mem: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # Tokenizer
    tok_path = None
    kaggle_input = Path("/kaggle/input/datasets/nochinator/thought-vectors-dataset/")
    if kaggle_input.exists():
        for d in kaggle_input.iterdir():
            p = d / "tokenizer.json"
            if p.exists():
                tok_path = str(p); break
    if tok_path is None:
        tok_path = "/kaggle/input/datasets/nochinator/thought-vectors-dataset/tokenizer.json"
    print(f"Loading tokenizer from {tok_path}")
    tok = ThoughtTokenizer(tok_path)

    cfg = BitThoughtConfig(
        vocab_size=tok.vocab_size,
        # Compression curriculum settings
        target_ratio_start=1.0,   # start at 1 token per vector (easy — no compression)
        target_ratio_inc=0.1,     # increment ratio by 0.1 when accuracy threshold met
        target_ratio_max=20.0,    # stop trying to compress beyond this ratio
        acc_threshold=0.95,       # 95% rolling accuracy required to advance
        acc_window=1000,          # rolling window size in batches
    )

    model = BitThoughtModel(cfg)
    enc_p = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_p = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"[BitThought] Parameters: encoder={enc_p:,} decoder={dec_p:,} total={enc_p+dec_p:,}")

    try:
        model = torch.compile(model)
        print("[compile] torch.compile enabled")
    except Exception as e:
        print(f"[compile] torch.compile not available ({e})")

    save_dir = Path("/kaggle/working/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: STSB (10 epochs, contrastive) ─────────────────
    print(f"\n{'='*60}\nStage 1: STSB (10 epochs, contrastive)\n{'='*60}")
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/STSB_train.csv")
    pairs = load_pairs("/kaggle/input/datasets/nochinator/thought-vectors-dataset/STSB_train.csv")
    print(f"Texts: {len(texts)}, Pairs: {len(pairs)}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=10, batch_size=8,
                    lr=5e-4, contrastive_weight=0.5, pairs=pairs,
                    exact_prob=0.5, repeat_penalty=0.03, pred_lr=2e-3,
                    warmup=50, log_every=500, sample_every=500,
                    save_every=3000, save_path=str(save_dir/"stsb.pt"),
                    cache_name=str(save_dir/"cache_stsb"))
    print(f"STSB done: loss={h[-1]:.4f}")

    # ── Stage 2: SNLI ──────────────────────────────────────────
    print(f"\n{'='*60}\nStage 2: SNLI\n{'='*60}")
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/SNLI_train.csv")
    print(f"Texts: {len(texts)}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=1, batch_size=8,
                    lr=3e-4, exact_prob=0.5, repeat_penalty=0.03, pred_lr=1e-3,
                    warmup=100, log_every=1000, sample_every=1000,
                    save_every=5000, save_path=str(save_dir/"snli.pt"),
                    cache_name=str(save_dir/"cache_snli"))
    print(f"SNLI done: loss={h[-1]:.4f}")

    # ── Stage 3: C4 (100K batches) ─────────────────────────────
    print(f"\n{'='*60}\nStage 3: C4\n{'='*60}")
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/C4subset-1.csv")[:800000]
    print(f"Texts: {len(texts)}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=1, batch_size=8,
                    lr=2e-4, exact_prob=0.5, repeat_penalty=0.03, pred_lr=8e-4,
                    warmup=100, log_every=1000, sample_every=1000,
                    save_every=10000, save_path=str(save_dir/"c4.pt"),
                    cache_name=str(save_dir/"cache_c4"))
    print(f"C4 done: loss={h[-1]:.4f}")

    # ── Stage 4: MiniPile (200K batches) ───────────────────────
    print(f"\n{'='*60}\nStage 4: MiniPile\n{'='*60}")
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/minipile.csv")[:1600000]
    print(f"Texts: {len(texts)}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=1, batch_size=8,
                    lr=1.5e-4, exact_prob=0.5, repeat_penalty=0.03, pred_lr=6e-4,
                    warmup=100, log_every=1000, sample_every=1000,
                    save_every=10000, save_path=str(save_dir/"minipile.pt"),
                    cache_name=str(save_dir/"cache_minipile"))
    print(f"MiniPile done: loss={h[-1]:.4f}")

    # ── Stage 5: C4 compression curriculum ─────────────────────
    print(f"\n{'='*60}\nStage 5: C4 compression curriculum\n{'='*60}")
    sched = CompressionScheduler(cfg, save_dir=save_dir)
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/C4subset-2.csv")[:800000]
    print(f"Texts: {len(texts)}")
    print(f"Compression: ratio={sched.ratio:.1f}, threshold={cfg.acc_threshold}, "
          f"window={cfg.acc_window}, max={cfg.target_ratio_max}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=1, batch_size=8,
                    lr=6e-5, exact_prob=0.5, repeat_penalty=0.03, pred_lr=4e-4,
                    warmup=50, log_every=1000, sample_every=1000,
                    save_every=5000, save_path=str(save_dir/"c4_comp.pt"),
                    cache_name=str(save_dir/"cache_c4_comp"),
                    compression_scheduler=sched)
    print(f"C4 compression done: loss={h[-1]:.4f}, final ratio={sched.ratio:.1f}")

    # ── Stage 6: MiniPile compression (continue same scheduler) ─
    print(f"\n{'='*60}\nStage 6: MiniPile compression\n{'='*60}")
    print(f"Continuing from ratio={sched.ratio:.1f}")
    texts = load_csv("/kaggle/input/datasets/nochinator/thought-vectors-dataset/minipile.csv")[1600000:]
    print(f"Texts: {len(texts)}")
    h = train_model(model, cfg, tok, texts, device=device, epochs=1, batch_size=8,
                    lr=5e-5, exact_prob=0.5, repeat_penalty=0.03, pred_lr=3e-4,
                    warmup=50, log_every=1000, sample_every=1000,
                    save_every=5000, save_path=str(save_dir/"minipile_comp.pt"),
                    cache_name=str(save_dir/"cache_minipile_comp"),
                    compression_scheduler=sched)
    print(f"MiniPile compression done: loss={h[-1]:.4f}, final ratio={sched.ratio:.1f}")

    print(f"\n{'='*60}")
    print(f"Training complete. Final compression ratio: {sched.ratio:.1f}x")
    print(f"Checkpoints per ratio level saved in {save_dir}")

if __name__ == "__main__":
    main()
