"""
BitThought — Training loop.

Supports:
  - Parallel thought vector generation with K-controlled selection
  - K-predictor training: regression head learning to predict optimal K
  - Compression curriculum: threshold-based advancement of target_ratio
  - ±noise on K during training for robustness
  - Soft mask for differentiable K during fine-tuning
"""

import math, os, random, time, threading
from contextlib import nullcontext
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

# Speed opts
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from bitthought.config import BitThoughtConfig
from bitthought.model import BitThoughtModel


# ── Compression Curriculum ─────────────────────────────────────────────

class CompressionScheduler:
    """Tracks rolling accuracy and advances compression ratio when threshold is met.

    K = ceil(seq_len / target_ratio), clamped to [1, num_thoughts].
    Starts at target_ratio_start (default 0.7 — no effective compression).
    Advances by target_ratio_inc when rolling accuracy >= acc_threshold.
    """
    def __init__(self, cfg: BitThoughtConfig, save_dir: str | Path):
        self.ratio      = cfg.target_ratio_start
        self.ratio_inc  = cfg.target_ratio_inc
        self.ratio_max  = cfg.target_ratio_max
        self.threshold  = cfg.acc_threshold
        self.window     = cfg.acc_window
        self.max_k      = cfg.num_thoughts
        self.save_dir   = Path(save_dir)
        self._acc_buf   = deque(maxlen=cfg.acc_window)
        self.at_ceiling = False

    def compute_K(self, seq_len: int) -> int:
        """Return K thought vectors for the current compression ratio."""
        return max(1, min(int(math.ceil(seq_len / max(self.ratio, 0.01))), self.max_k))

    def apply_noise(self, K: int, noise: float) -> int:
        """Apply ±noise% noise to K. Minimum 1."""
        if noise <= 0:
            return K
        noise_factor = 1.0 + random.uniform(-noise, noise)
        return max(1, int(round(K * noise_factor)))

    def update(self, acc: float, model: torch.nn.Module, history: list) -> bool:
        """Call after each batch. Returns True if ratio just advanced."""
        self._acc_buf.append(acc)
        if self.at_ceiling or len(self._acc_buf) < self.window:
            return False
        rolling_acc = sum(self._acc_buf) / len(self._acc_buf)
        if rolling_acc >= self.threshold:
            ckpt = self.save_dir / f"ratio_{self.ratio:.1f}.pt"
            torch.save({"model": model.state_dict(),
                        "ratio": self.ratio,
                        "history": history}, str(ckpt))
            print(f"\n[curriculum] ratio {self.ratio:.1f} -> {self.ratio + self.ratio_inc:.1f} "
                  f"(rolling_acc={rolling_acc:.4f} >= {self.threshold}) \u2014 saved {ckpt.name}")
            self.ratio += self.ratio_inc
            self._acc_buf.clear()
            if self.ratio > self.ratio_max:
                self.ratio = self.ratio_max
                self.at_ceiling = True
                print(f"[curriculum] reached max ratio {self.ratio_max:.1f}, holding.")
            return True
        return False

    @property
    def rolling_acc(self) -> float:
        if not self._acc_buf:
            return 0.0
        return sum(self._acc_buf) / len(self._acc_buf)


# ── Training Step ──────────────────────────────────────────────────────

def training_step(
    model: BitThoughtModel,
    input_ids: torch.Tensor,
    pad_token_id: int,
    config: BitThoughtConfig,
    K: int,
    *,
    length_penalty: float = 0.005,
    pred_weight: float = 1.0,
    contrastive_weight: float = 0.0,
    repeat_penalty: float = 0.03,
    input_ids_b: torch.Tensor | None = None,
    scores: torch.Tensor | None = None,
    train_k_predictor: bool = False,
    use_amp: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Single training step.

    - All 128 vectors generated in parallel.
    - First K vectors are used for decoding (hard selection).
    - If train_k_predictor: also compute K-prediction loss (MSE vs target K).
    - If use_amp: model forward runs in FP16, loss computation stays in FP32.
    """
    amp_ctx = torch.amp.autocast("cuda") if use_amp else nullcontext()

    padding_mask = input_ids.eq(pad_token_id)
    if padding_mask[:, 0].any():
        padding_mask[:, 0] = False

    with amp_ctx:
        # Generate all 128 vectors + K-predictor output
        thought_vectors, k_pred, weights = model.encoder(input_ids, padding_mask, K=None)
        if torch.isnan(thought_vectors).any() or torch.isinf(thought_vectors).any():
            print(f"[NaN/INF] primary thoughts: isnan={torch.isnan(thought_vectors).any()} isinf={torch.isinf(thought_vectors).any()}")

        # If contrastive, encode second text
        if contrastive_weight > 0 and input_ids_b is not None:
            pad_b = input_ids_b.eq(pad_token_id)
            if pad_b[:, 0].any():
                pad_b[:, 0] = False
            thoughts_b, _, _ = model.encoder(input_ids_b, pad_b, K=None)
            if torch.isnan(thoughts_b).any() or torch.isinf(thoughts_b).any():
                print(f"[NaN/INF] contrastive thoughts_b: isnan={torch.isnan(thoughts_b).any()} isinf={torch.isinf(thoughts_b).any()}")

        # Use K-hard slicing for decoder input
        thoughts_k = thought_vectors[:, :K, :]

        target_pad = padding_mask[:, :-1].clone()
        if target_pad[:, 0].any():
            target_pad[:, 0] = False

        pred_embeds = model.decoder(thoughts_k, input_ids[:, :-1], target_pad)
        target = input_ids[:, 1:]

    # Logit projection in FP32 (FP16 overflows on the 32K-vocab projection)
    pred_embeds_f32 = pred_embeds.float()
    target_embeds = model.shared_embed(target).float()
    logits = model.embed_to_logits(pred_embeds_f32)
    if torch.isnan(logits).any() or torch.isinf(logits).any() or torch.isnan(pred_embeds_f32).any() or torch.isinf(pred_embeds_f32).any():
        with torch.no_grad():
            print(f"[NaN/INF] logits: isnan={torch.isnan(logits).any()} isinf={torch.isinf(logits).any()}")
            print(f"  pred_embeds: isnan={torch.isnan(pred_embeds).any()} isinf={torch.isinf(pred_embeds).any()} range=[{pred_embeds.min():.4f}, {pred_embeds.max():.4f}]")
            print(f"  thought_vectors: isnan={torch.isnan(thought_vectors).any()} isinf={torch.isinf(thought_vectors).any()} range=[{thought_vectors.min():.4f}, {thought_vectors.max():.4f}]")
            # Check encoder text output
            x = model.encoder._encode_text(input_ids)
            print(f"  encoded_text: isnan={torch.isnan(x).any()} isinf={torch.isinf(x).any()} range=[{x.min():.4f}, {x.max():.4f}]")
            # Check each encoder layer
            for i, l in enumerate(model.encoder.layers):
                x = l(x, freqs_cos=model.encoder.freqs_cos, freqs_sin=model.encoder.freqs_sin)
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"  encoder layer {i} output: isnan={torch.isnan(x).any()} isinf={torch.isinf(x).any()} range=[{x.min():.4f}, {x.max():.4f}]")
                    break
            # Check decoder layers
            _, _, weights = model.encoder(input_ids, K=None)
            masked = thought_vectors * weights.unsqueeze(-1)
            tgt = model.decoder.embed(input_ids[:, :-1]) * math.sqrt(model.decoder.embed.weight.size(1))
            cm = torch.triu(torch.full((input_ids.size(1)-1, input_ids.size(1)-1), float('-inf'), device=input_ids.device), diagonal=1)
            for i, l in enumerate(model.decoder.layers):
                tgt = l(tgt, masked, tgt_mask=cm, freqs_cos=model.decoder.freqs_cos, freqs_sin=model.decoder.freqs_sin)
                if torch.isnan(tgt).any() or torch.isinf(tgt).any():
                    print(f"  decoder layer {i} output: isnan={torch.isnan(tgt).any()} isinf={torch.isinf(tgt).any()} range=[{tgt.min():.4f}, {tgt.max():.4f}]")
                    break

    # ── Loss computation in FP32 ─────────────────────────────────
    mse_loss = F.mse_loss(pred_embeds_f32, target_embeds)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
        ignore_index=pad_token_id,
    )

    recon_loss = (mse_loss + 5.0 * ce_loss) / 6.0  # weight CE 5× more than MSE for exact token prediction

    # Repetition penalty
    repeat_loss = torch.tensor(0.0, device=input_ids.device)
    if repeat_penalty > 0:
        with torch.no_grad():
            decoded = logits.float().argmax(dim=-1)
        adj = (decoded[:, 1:] == decoded[:, :-1]).float()
        valid = (target[:, 1:] != pad_token_id).float()
        repeat_rate = (adj * valid).sum() / valid.sum().clamp(min=1.0)
        repeat_loss = repeat_penalty * repeat_rate

    # Contrastive loss (thoughts_b from AMP context above)
    contrastive_loss = torch.tensor(0.0, device=input_ids.device)
    if contrastive_weight > 0 and input_ids_b is not None and scores is not None:
        cos_sim = F.cosine_similarity(thought_vectors.mean(1), thoughts_b.mean(1))
        contrastive_loss = F.mse_loss(cos_sim.float(), scores.float())

    # K-predictor loss (only applies during K-predictor training phase)
    k_loss = torch.tensor(0.0, device=input_ids.device)
    if train_k_predictor:
        k_target = torch.full_like(k_pred, K, dtype=torch.float)
        k_loss = F.mse_loss(k_pred, k_target)

    len_pen = recon_loss.new_tensor(length_penalty * K)
    total = (recon_loss + len_pen
             + contrastive_weight * contrastive_loss
             + repeat_loss
             + k_loss)

    # Token accuracy
    with torch.no_grad():
        decoded = logits.float().argmax(dim=-1)
        valid = ~target_pad
        token_acc = ((decoded == target) & valid).float().sum() / valid.float().sum().clamp(min=1)

    # K-predictor accuracy: predicted K within ±10% of target K
    with torch.no_grad():
        k_error = (k_pred.squeeze(-1) - K).abs()
        k_accurate = (k_error / max(K, 1) < 0.1).float().mean()

    stats = {
        "recon": float(recon_loss.detach()),
        "k_loss": float(k_loss.detach()),
        "ctr": float(contrastive_loss.detach()),
        "rpt": float(repeat_loss.detach()),
        "total": float(total.detach()),
        "vecs": float(K),
        "acc": float(token_acc.detach()),
        "k_pred": float(k_pred[0, 0].detach()),
        "k_acc": float(k_accurate.detach()),
    }
    return total, stats


# ── Data helpers ───────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = [s for s in seqs if s]
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return self.seqs[i]

def collate_seq_batch(batch, pad_id, max_len=384):
    seqs = [s[:max_len] for s in batch]
    ml = max(len(s) for s in seqs)
    out = torch.full((len(seqs), ml), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# ── Train Model ────────────────────────────────────────────────────────

def train_model(
    model,
    config,
    groups_or_tokenizer=None,
    texts_or_encode=None,
    pad_token_id_or_none=None,
    *,
    device,
    epochs=1,
    batch_size=8,
    learning_rate=3e-4,
    weight_decay=1e-5,
    length_penalty=0.005,
    prediction_weight=1.0,
    contrastive_weight=0.0,
    repeat_penalty=0.03,
    predictor_lr=1e-3,
    warmup_steps=100,
    log_every=1000,
    batch_pause_ms=0,  # ms to sleep after each batch (reduce GPU temp)
    sample_every=1000,
    tokenizer_decode=None,
    save_every=5000,
    save_path="checkpoint.pt",
    compression_scheduler=None,
    train_k_predictor=False,
    freeze_encoder_decoder=False,
    k_noise=0.0,
    # Legacy compat
    groups=None,
    tokenizer_encode=None,
    pad_token_id=0,
    tokenized_data=None,
    paired_pairs=None,
    seed=0,
    cache_name=None,
    # Kaggle compat
    tok=None,
    pairs=None,
    lr=None,
    wd=None,
    len_penalty=None,
    warmup=None,
):
    """Train BitThought model.

    Two calling conventions:
      1. (model, config, tokenizer, texts, *, ...)
      2. (model, config, groups, tokenizer_encode, pad_token_id, *, ...)

    Modes:
      - Normal: trains encoder+decoder with K from scheduler.
      - K-predictor: freezes encoder(except K-predictor)+decoder, trains only K-predictor.
    """
    # ── Normalise arguments ──────────────────────────────────────
    tokenizer = None
    texts = None
    local_groups = None
    if groups_or_tokenizer is not None:
        if isinstance(groups_or_tokenizer, list):
            local_groups = groups_or_tokenizer
        elif hasattr(groups_or_tokenizer, 'encode'):
            tokenizer = groups_or_tokenizer
        else:
            local_groups = groups_or_tokenizer
    if texts_or_encode is not None:
        if isinstance(texts_or_encode, list) and all(isinstance(t, str) for t in texts_or_encode):
            texts = texts_or_encode
        elif callable(texts_or_encode):
            tokenizer = tokenizer or texts_or_encode
        else:
            texts = texts_or_encode
    if pad_token_id_or_none is not None:
        pad_token_id = pad_token_id_or_none
    # Keyword `groups` overrides positional detection
    if groups is not None:
        local_groups = groups

    if lr is not None: learning_rate = lr
    if wd is not None: weight_decay = wd
    if len_penalty is not None: length_penalty = len_penalty
    if warmup is not None: warmup_steps = warmup
    if tok is not None: tokenizer = tok
    if pairs is not None: paired_pairs = pairs
    if tokenizer is None and tokenizer_encode is not None:
        tokenizer = tokenizer_encode

    if pad_token_id == 0 and tokenizer is not None:
        try: pad_token_id = tokenizer.pad_token_id
        except AttributeError:
            try: pad_token_id = tokenizer.pad_id
            except AttributeError: pass
    if pad_token_id == 0:
        pad_token_id = getattr(config, 'pad_token_id', 0)

    random.seed(seed)
    torch.manual_seed(seed)

    # ── Build dataset ────────────────────────────────────────────
    if tokenized_data is not None:
        seqs = tokenized_data
    elif texts is not None and tokenizer is not None:
        cache_path = Path(cache_name + ".flat.pt") if cache_name else None
        if cache_path and cache_path.exists():
            cached = torch.load(cache_path, weights_only=True)
            tok_t, lens = cached["tokens"], cached["lengths"]
            seqs = []; off = 0
            for l in lens.tolist():
                seqs.append(tok_t[off:off + l].tolist())
                off += l
            print(f"[cache] loaded {len(seqs)} seqs")
        else:
            _enc = tokenizer.encode if hasattr(tokenizer, 'encode') else tokenizer
            seqs = [_enc(t)[:config.max_seq_len] for t in texts if t]
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                lengths = torch.tensor([len(s) for s in seqs], dtype=torch.int)
                tokens = torch.cat([torch.tensor(s, dtype=torch.int) for s in seqs])
                torch.save({"tokens": tokens, "lengths": lengths}, str(cache_path))
                print(f"[cache] saved {len(seqs)} seqs")
    elif local_groups is not None:
        seqs = local_groups
    else:
        raise ValueError("Must provide texts, tokenized_data, or groups")

    dataset = SeqDataset(seqs) if isinstance(seqs, list) and seqs and isinstance(seqs[0], list) \
        else (lambda: None)()  # will use ThoughtDataset
    if dataset is None:
        from bitthought.data import ThoughtDataset
        dataset = ThoughtDataset(seqs)

    # DataLoader with speed opts
    num_cpus = os.cpu_count() or 2
    num_workers = min(8, num_cpus)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_seq_batch(b, pad_token_id, config.max_seq_len),
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Paired loader for contrastive
    paired_loader = None
    if paired_pairs is not None and contrastive_weight > 0 and tokenizer is not None:
        class PairDataset(Dataset):
            def __init__(self, ps): self.pairs = [(a, b, s) for a, b, s in ps if a and b]
            def __len__(self): return len(self.pairs)
            def __getitem__(self, i): return self.pairs[i]
        def collate_pair(batch):
            _enc = tokenizer.encode if hasattr(tokenizer, 'encode') else tokenizer
            def tok_seqs(ts):
                ss = [_enc(t)[:config.max_seq_len] for t in ts]
                ml = max(len(s) for s in ss)
                out = torch.full((len(ss), ml), pad_token_id, dtype=torch.long)
                for i, s in enumerate(ss):
                    out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
                return out
            return (tok_seqs([p[0] for p in batch]),
                    tok_seqs([p[1] for p in batch]),
                    torch.tensor([p[2] for p in batch], dtype=torch.float))
        paired_ds = PairDataset(paired_pairs)
        paired_loader = DataLoader(paired_ds, batch_size=batch_size, shuffle=True,
                                    collate_fn=collate_pair, num_workers=0)
        print(f"[paired] {len(paired_pairs)} pairs, weight={contrastive_weight}")

    # ── Optimizer ────────────────────────────────────────────────
    if train_k_predictor or freeze_encoder_decoder:
        # Only train K-predictor parameters
        k_params = [p for n, p in model.named_parameters() if "k_predictor" in n]
        other_params = [p for n, p in model.named_parameters() if "k_predictor" not in n]
        for p in other_params:
            p.requires_grad = False
        opt = optim.AdamW(k_params, lr=predictor_lr, weight_decay=0.0)
        print(f"[train] K-predictor mode: {sum(p.numel() for p in k_params):,} params trainable")
    else:
        # Train all — separate LR for K-predictor
        k_params = [p for n, p in model.named_parameters() if "k_predictor" in n]
        main_params = [p for n, p in model.named_parameters() if "k_predictor" not in n]
        try:
            opt = optim.AdamW([
                {"params": main_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": k_params, "lr": predictor_lr, "weight_decay": 0.0},
            ], fused=True)
        except Exception:
            opt = optim.AdamW([
                {"params": main_params, "lr": learning_rate, "weight_decay": weight_decay},
                {"params": k_params, "lr": predictor_lr, "weight_decay": 0.0},
            ])

    for g in opt.param_groups:
        g.setdefault("initial_lr", g["lr"])

    total_steps = max(1, epochs * len(loader))
    model.to(device)
    # Using autocast + GradScaler for FP16 matmuls while keeping model FP32.
    # RMSNorm, SDPA, and SwiGLU manually upcast to FP32 in model.py,
    # preventing NaN from ROCm autocast mishandling those ops.

    history = []
    paired_iter = iter(paired_loader) if paired_loader else None
    scaler = None  # Model is FP32; autocast handles FP16 forward, gradients are FP32 naturally

    enc_p = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    dec_p = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f"[BitThought] encoder={enc_p:,} decoder={dec_p:,} total={enc_p+dec_p:,} "
          f"k_predictor={sum(p.numel() for p in model.encoder.k_predictor.parameters()):,}"
          if hasattr(model.encoder, 'k_predictor') and model.encoder.k_predictor else "")

    # ── Training loop ────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0
        t0 = time.time()

        for batch_idx, input_ids in enumerate(loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            global_step = epoch * len(loader) + batch_idx - 1

            # LR schedule: warmup → constant 80% → cosine decay 20%
            if warmup_steps > 0 and global_step < warmup_steps:
                scale = global_step / warmup_steps
            else:
                remaining = max(1, total_steps - warmup_steps)
                progress = (global_step - warmup_steps) / remaining
                if progress < 0.8:
                    scale = 1.0  # constant plateau
                else:
                    d = (progress - 0.8) / 0.2  # decay phase [0, 1]
                    scale = 0.5 * (1 + math.cos(math.pi * min(1.0, d)))
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * scale

            # Paired batch
            ids_b, sim_scores = None, None
            if paired_iter is not None:
                try:
                    _, ids_b, sim_scores = next(paired_iter)
                except StopIteration:
                    paired_iter = iter(paired_loader)
                    _, ids_b, sim_scores = next(paired_iter)
                ids_b = ids_b.to(device, non_blocking=True)
                sim_scores = sim_scores.to(device)

            # Compute K from scheduler (with optional noise)
            seq_len = input_ids.size(1)
            if compression_scheduler is not None:
                K = compression_scheduler.compute_K(seq_len)
                if k_noise > 0 and not train_k_predictor:
                    K = compression_scheduler.apply_noise(K, k_noise)
            else:
                K = config.num_thoughts  # use all vectors

            opt.zero_grad(set_to_none=True)
            loss, stats = training_step(
                model, input_ids, pad_token_id, config, K,
                length_penalty=length_penalty,
                contrastive_weight=contrastive_weight,
                repeat_penalty=repeat_penalty,
                input_ids_b=ids_b,
                scores=sim_scores,
                train_k_predictor=train_k_predictor,
                use_amp=(scaler is not None),  # autocast + GradScaler
            )
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

            # Update compression scheduler
            advanced = False
            if compression_scheduler is not None and not train_k_predictor:
                advanced = compression_scheduler.update(
                    stats["acc"], model, history + [epoch_loss / max(batches, 1)])

            # Pacing: small sleep reduces GPU temp at minimal throughput cost
            if batch_pause_ms > 0:
                time.sleep(batch_pause_ms / 1000.0)

            epoch_loss += float(loss.detach())
            batches += 1
            if torch.isnan(loss):
                raise RuntimeError(f"NaN at batch {batch_idx}")

            if save_every > 0 and batch_idx % save_every == 0:
                sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                h = history + [epoch_loss / batches]
                threading.Thread(target=lambda: torch.save({"model": sd, "history": h}, save_path),
                                 daemon=True).start()

            if batch_idx % log_every == 0 or advanced:
                elapsed = time.time() - t0
                avg = epoch_loss / batches
                mode = "kp" if train_k_predictor else "normal"
                ratio_str = f"{compression_scheduler.ratio:.1f}" if compression_scheduler else "fixed"
                roll_str = f"{compression_scheduler.rolling_acc:.3f}" if compression_scheduler else "n/a"
                kp_str = f" k_loss={stats['k_loss']:.4f} k_pred={stats['k_pred']:.1f} k_acc={stats['k_acc']:.3f}" \
                    if train_k_predictor else ""
                print(f"  [{mode}] batch {batch_idx}/{len(loader)} | loss={stats['total']:.4f} "
                      f"recon={stats['recon']:.4f} vecs={int(stats['vecs'])} "
                      f"ratio={ratio_str} acc={stats['acc']:.3f} "
                      f"roll_acc={roll_str} avg={avg:.4f}"
                      f"{kp_str} lr={opt.param_groups[0]['lr']:.2e} [{elapsed:.1f}s]")

            if sample_every > 0 and batch_idx % sample_every == 0 and tokenizer_decode is not None:
                model.eval()
                with torch.no_grad():
                    for idx in range(min(3, input_ids.size(0))):
                        tv, _, _ = model.encoder(input_ids[idx:idx+1], K=K)
                        tv_k = tv[:, :K, :]
                        pe = model.decoder(tv_k, input_ids[idx:idx+1, :-1],
                                           input_ids[idx:idx+1, :-1].eq(pad_token_id)
                                           if input_ids.size(1) > 1 else None)
                        sm = model.embed_to_logits(pe).argmax(dim=-1)
                        orig = tokenizer_decode(input_ids[idx].tolist())
                        recon_t = tokenizer_decode(sm[0].tolist())
                        print(f"  [s{idx}] K={K} orig={orig[:80]!r}")
                        print(f"  [s{idx}] K={K} recon={recon_t[:80]!r}")
                model.train()

        history.append(epoch_loss / batches)
        ri = f"{compression_scheduler.ratio:.1f}" if compression_scheduler else ""
        print(f"  epoch {epoch+1} done: avg_loss={history[-1]:.4f} ratio={ri}")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    threading.Thread(target=lambda: torch.save({"model": sd, "history": history}, save_path),
                     daemon=True).start()
    return history
