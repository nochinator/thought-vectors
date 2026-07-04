#!/usr/bin/env python3
"""
Three-phase thinker training with multi-turn conversation context.

Phase 1 (1 epoch):  Fine-tune encoder + decoder on reconstruction of
                     conversation texts (domain adaptation of thought space).

Phase 2 (3-5 ep.):  Lock encoder + decoder. Train the thinker on multi-turn
                     conversation context:
                       [past_user, past_asst, past_user, past_asst, …, current_user]
                     → thinker → decoder generates only from current_user segment.
                     Optionally flips roles (acting as both "users").

Phase 3 (1 epoch):  Unlock everything (encoder, thinker, decoder) and
                     fine-tune the full pipeline with multi-turn contexts.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, LossPredictor, ThinkerModel


# ── helpers ──

def load_pairs(input_path: Path, output_path: Path):
    """Load all aligned (input, output) conversation pairs from single-column CSVs."""
    arts, sums = [], []
    with open(input_path, newline="") as fa, open(output_path, newline="") as fs:
        ra = csv.reader(fa)
        rs = csv.reader(fs)
        for ar, sr in zip(ra, rs):
            if ar and sr:
                arts.append(ar[0])
                sums.append(sr[0])
    return arts, sums


def filter_pairs(
    inputs: list[str], outputs: list[str],
    tokenizer, max_input_tokens: int, max_output_tokens: int,
) -> tuple[list[str], list[str]]:
    """Keep only pairs where both texts fit within token limits."""
    filtered_in, filtered_out = [], []
    n_skip_in = 0
    n_skip_out = 0
    for inp, out in zip(inputs, outputs):
        in_len = len(tokenizer.encode(inp, add_special_tokens=True))
        out_len = len(tokenizer.encode(out, add_special_tokens=True))
        if in_len > max_input_tokens:
            n_skip_in += 1
            continue
        if out_len > max_output_tokens:
            n_skip_out += 1
            continue
        filtered_in.append(inp)
        filtered_out.append(out)
    print(f"  filtered: {len(inputs)} → {len(filtered_in)} pairs "
          f"(skipped {n_skip_in} long inputs, {n_skip_out} long outputs)")
    return filtered_in, filtered_out


def encode_batch(texts: list[str], tokenizer, pad_id: int, device: torch.device) -> torch.Tensor:
    """Tokenize and pad a batch of texts → [B, T]."""
    encoded = [tokenizer.encode(t, add_special_tokens=True) for t in texts]
    max_len = max(len(e) for e in encoded)
    out = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        out[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return out.to(device)


def deduplicate_params(module: nn.Module) -> list[nn.Parameter]:
    """Return parameters with tied-weight duplicates removed (by data_ptr)."""
    seen: set[int] = set()
    params = []
    for p in module.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            params.append(p)
    return params


def _save_with_embeddings(path: Path, model: ThinkerModel, config: dict) -> None:
    """Save model state plus structural embeddings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "thinker_state": model.thinker.state_dict(),
        "predictor_state": model.predictor.state_dict(),
        "thinker_embeddings": {
            "turn_embedding": model.turn_embedding.state_dict(),
            "speaker_embedding": model.speaker_embedding.state_dict(),
            "decode_embedding": model.decode_embedding.data.clone(),
        },
        "config": config,
    }, str(path))


def _load_embeddings(model: ThinkerModel, ckpt: dict, device: torch.device) -> None:
    """Load structural embeddings from a checkpoint (if they exist)."""
    emb = ckpt.get("thinker_embeddings")
    if emb is None:
        return  # old checkpoint, embeddings stay at init values
    model.turn_embedding.load_state_dict(emb["turn_embedding"])
    model.speaker_embedding.load_state_dict(emb["speaker_embedding"])
    model.decode_embedding.data.copy_(emb["decode_embedding"].to(device))


def _pre_encode_and_memmap(
    texts_in: list[str], texts_out: list[str],
    encoder, tokenizer, pad_id: int,
    k: int, device: torch.device, save_path: str,
    batch_size: int = 64,
) -> str:
    """Pre-encode input+output texts into a SINGLE combined memmap.

    Combined shape: [2*n, k, d] — first half = inputs, second half = outputs.
    This lets _gather_pre read all rows with a single numpy batch index,
    eliminating the Python loop.
    """
    n = len(texts_in)
    d = encoder.d_model
    total = 2 * n
    # Single combined memmap file.
    mm = np.memmap(save_path, dtype="float32", mode="w+", shape=(total, k, d))
    encoder.eval()

    def encode_chunk(texts: list[str], offset: int):
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            ids = encode_batch(batch_texts, tokenizer, pad_id, device)
            pad_mask = ids.eq(pad_id)
            with torch.no_grad():
                thoughts = encoder(ids, pad_mask)
            mm[offset + i:offset + i + len(batch_texts)] = thoughts[:, :k, :].cpu().numpy()
            mm.flush()
            if ((offset + i) // batch_size) % 200 == 0:
                print(f"  pre-encode: {offset + min(i + batch_size, len(texts))}/{total}", flush=True)

    print(f"  pre-encoding inputs ({n})...", flush=True)
    encode_chunk(texts_in, 0)
    print(f"  pre-encoding outputs ({n})...", flush=True)
    encode_chunk(texts_out, n)

    del mm
    print(f"  saved combined to {save_path} ({total} × {k} × {d})")
    return save_path


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


# Module-level cache for pre-encoded lookups — avoids rebuilding 500K-entry
# dicts and reopening 32 GB memmap files on every batch call.
_pre_cache: dict = {}


def _gather_pre(
    combined_path: str,
    indices: list[tuple[int, bool]], k: int, device: torch.device,
    n_pairs: int,
) -> torch.Tensor:
    """Batch-read pre-encoded vectors from a single combined memmap.

    Combined file has shape [2*n_pairs, max_k, d_model] where first half
    is inputs and second half is outputs.  Uses numpy advanced indexing
    to read ALL requested rows in one call — no Python loop.
    """
    global _pre_cache
    if combined_path not in _pre_cache:
        n_entries = 2 * n_pairs
        # Infer stored_k from file size: stored_k = total_bytes / (n_entries * 4 * 256)
        stored_k = os.path.getsize(combined_path) // (n_entries * 4 * 256)
        _pre_cache[combined_path] = np.memmap(
            combined_path, dtype="float32", mode="r",
            shape=(n_entries, stored_k, 256),
        )
    mm = _pre_cache[combined_path]

    # Build a single index array: input rows are [0, n_pairs),
    # output rows are [n_pairs, 2*n_pairs).
    n = len(indices)
    idx_arr = np.empty(n, dtype=np.int64)
    for i, (idx_val, is_out) in enumerate(indices):
        idx_arr[i] = idx_val + (n_pairs if is_out else 0)

    # ONE numpy advanced-indexing call reads all rows.
    rows = mm[idx_arr]  # [n, max_k, d]
    return torch.from_numpy(rows[:, :k, :].copy()).to(device)


def build_multi_turn_batch(
    inputs: list[str],
    outputs: list[str],
    batch_indices: list[int],
    n_past: int,
    tokenizer,
    pad_id: int,
    device: torch.device,
    model: ThinkerModel,
    k: int,
    flip: bool,
    pre_in: torch.Tensor | None = None,
    pre_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build multi-turn thinker contexts with structural metadata.

    For each sample in the batch:
      - Sample ``n_past`` random past (user, assistant) pairs
      - Concatenate their thought vectors in turn order
      - Append the current user's thought vectors
      - Generate ``turn_ids``, ``speaker_ids``, ``decode_mask`` metadata.

    When ``flip=True`` the roles are reversed — past context becomes
    [asst, user, asst, user, …] and the "current input" is the output
    text with the original input as the target.

    Returns:
        contexts:    [B, (2*n_past+1)*k, d_model] — thinker input
        targets:     [B, T_out]                   — tokenised targets
        turn_ids:    [B, (2*n_past+1)*k] long     — which turn each vec belongs to
        speaker_ids: [B, (2*n_past+1)*k] long     — 0=user, 1=assistant
        decode_mask: [B, (2*n_past+1)*k] bool     — True for current-user segment
    """
    B = len(batch_indices)
    n_pairs = len(inputs)

    # For each sample, find n_past distinct past-pair indices.
    # Use random.sample on a range (O(k) with lazy range, not O(n)).
    # If idx happens to be sampled, swap with the last element to avoid
    # materializing a list of all n_pairs.
    past_idx_pairs: list[list[int]] = []
    for idx in batch_indices:
        k_needed = min(n_past, n_pairs - 1)
        chosen = random.sample(range(n_pairs), k=k_needed)
        for j in range(len(chosen)):
            if chosen[j] == idx:
                chosen[j] = n_pairs - 1 - j
        past_idx_pairs.append(chosen)

    # Collect texts in order: [past_u1, past_a1, …, past_uN, past_aN, current]
    # Store as (index, is_output) tuples — no string dict lookups.
    text_indices: list[tuple[int, bool]] = []
    for i, idx in enumerate(batch_indices):
        n_actual = len(past_idx_pairs[i])
        for pi in range(n_actual):
            pair_idx = past_idx_pairs[i][pi]
            if flip:
                text_indices.append((pair_idx, True))   # outputs[pair_idx]
                text_indices.append((pair_idx, False))  # inputs[pair_idx]
            else:
                text_indices.append((pair_idx, False))  # inputs[pair_idx]
                text_indices.append((pair_idx, True))   # outputs[pair_idx]
        text_indices.append((idx, flip))  # current = outputs[idx] if flip else inputs[idx]

    # Batch-encode → thought vectors → k-slice
    if pre_in is not None and pre_out is not None:
        all_thoughts_k = _gather_pre(pre_in, text_indices, k, device, len(inputs))
    else:
        # Convert back to strings for live encoding
        texts_to_encode: list[str] = [
            outputs[idx] if is_out else inputs[idx]
            for idx, is_out in text_indices
        ]
        all_ids = encode_batch(texts_to_encode, tokenizer, pad_id, device)
        with torch.set_grad_enabled(model.encoder.training):
            all_thoughts = model.encoder(all_ids, all_ids.eq(pad_id))  # [N, 256, D]
        all_thoughts_k = all_thoughts[:, :k, :]  # [N, k, D]

    # Reassemble per-sample contexts + metadata
    max_n_actual = max(len(p) for p in past_idx_pairs)
    total_k = (2 * max_n_actual + 1) * k
    d = all_thoughts_k.size(-1)
    contexts = torch.zeros(B, total_k, d, device=device)
    turn_ids = torch.zeros(B, total_k, dtype=torch.long, device=device)
    speaker_ids = torch.zeros(B, total_k, dtype=torch.long, device=device)
    decode_mask = torch.zeros(B, total_k, dtype=torch.bool, device=device)
    targets_list: list[torch.Tensor] = []

    ptr = 0
    for i, idx in enumerate(batch_indices):
        na = len(past_idx_pairs[i])
        nv = (2 * na + 1) * k

        # Vectors
        contexts[i, :nv] = all_thoughts_k[ptr:ptr + 2 * na + 1].reshape(nv, d)
        ptr += 2 * na + 1

        # Metadata: turn_ids, speaker_ids, decode_mask
        pos = 0
        for t in range(na):
            turn_ids[i, pos:pos + k] = t
            speaker_ids[i, pos:pos + k] = 0
            pos += k
            turn_ids[i, pos:pos + k] = t
            speaker_ids[i, pos:pos + k] = 1
            pos += k
        # Current user turn
        turn_ids[i, pos:pos + k] = na
        speaker_ids[i, pos:pos + k] = 0
        decode_mask[i, pos:pos + k] = 1

        # Target
        target_text = outputs[idx] if not flip else inputs[idx]
        target_ids = torch.tensor(
            [tokenizer.encode(target_text, add_special_tokens=True)],
            dtype=torch.long, device=device,
        )
        targets_list.append(target_ids)

    # Pad targets
    max_tgt = max(t.size(1) for t in targets_list)
    targets = torch.full((B, max_tgt), pad_id, dtype=torch.long, device=device)
    for i, t in enumerate(targets_list):
        targets[i, :t.size(1)] = t

    return contexts, targets, turn_ids, speaker_ids, decode_mask


# ── main ──

def main() -> None:
    parser = argparse.ArgumentParser(description="Three-phase thinker training")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts" / "thinker_big_s1.pt")
    parser.add_argument("--data-input", type=Path,
                        default=Path("/tmp/thinker_data_input.csv"))
    parser.add_argument("--data-output", type=Path,
                        default=Path("/tmp/thinker_data_output.csv"))
    parser.add_argument("--max-input-tokens", type=int, default=128,
                        help="Drop pairs where input exceeds this many tokens (+special)")
    parser.add_argument("--max-output-tokens", type=int, default=128,
                        help="Drop pairs where output exceeds this many tokens (+special)")
    parser.add_argument("--max-past-turns", type=int, default=3,
                        help="Max past conversation turns to include as context")
    parser.add_argument("--flip-prob", type=float, default=0.15,
                        help="Probability of swapping input↔output roles per batch")
    parser.add_argument("--max-k", type=int, default=64,
                        help="Maximum k (thought vectors per turn) during training. "
                             "Lower = less memory. Naive attention O(n²) on this ROCm stack.")
    parser.add_argument("--repetition-penalty-weight", type=float, default=0.1,
                        help="Penalize predicting the same token as the preceding context token")
    parser.add_argument("--diversity-weight", type=float, default=0.05,
                        help="Cosine diversity loss on thought vectors — push them apart")
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Gaussian noise stddev on thought vectors during training")
    parser.add_argument("--thinker-dim", type=int, default=384,
                        help="Internal dimension of the thinker transformer.  "
                             "Projections bridge encoder d_model (256) to this.")
    parser.add_argument("--thinker-layers", type=int, default=8,
                        help="Number of thinker transformer layers")
    parser.add_argument("--latent-consistency-weight", type=float, default=0.02,
                        help="MSE between thinker output vectors and target encoded vectors")
    parser.add_argument("--pre-encode-path", type=str, default="/tmp/pre_encoded",
                        help="Path prefix for pre-encoded thought vectors on disk")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-phase1", type=float, default=1e-5,
                        help="LR for encoder+decoder reconstruction fine-tune")
    parser.add_argument("--lr-phase2", type=float, default=5e-5,
                        help="LR for thinker training")
    parser.add_argument("--lr-phase3", type=float, default=1e-5,
                        help="LR for full fine-tune")
    parser.add_argument("--epochs-phase2", type=int, default=3)
    parser.add_argument("--output-phase1", type=Path,
                        default=ROOT / "artifacts" / "thinker_phase1.pt",
                        help="Save checkpoint after Phase 1")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1; load encoder+decoder from --output-phase1")
    parser.add_argument("--stop-after-phase1", action="store_true",
                        help="Save Phase 1 checkpoint and exit")
    parser.add_argument("--output-phase2", type=Path,
                        default=ROOT / "artifacts" / "thinker_phase2.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "thinker_three_phase.pt")
    parser.add_argument("--tokenizer-model", type=str,
                        default="/tmp/sp_c4_16k.model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer (needed before data loading for length filtering) ──
    tokenizer = SPTokenizer()
    tokenizer.load(args.tokenizer_model)
    pad_id = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: vocab_size={vocab_size}")

    # ── Data ──
    print("Loading data…")
    inputs_raw, outputs_raw = load_pairs(args.data_input, args.data_output)
    print(f"  {len(inputs_raw)} raw pairs")

    inputs, outputs = filter_pairs(
        inputs_raw, outputs_raw, tokenizer,
        args.max_input_tokens, args.max_output_tokens,
    )
    n_pairs = len(inputs)
    n_batches = n_pairs // args.batch_size
    print(f"  {n_pairs} usable pairs, {n_batches} batches/epoch")

    # ── Build model ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    enc = ThoughtEncoder(vocab_size, 256, 4, 4, 0.1, 8192, 256).to(device)
    dec = ThoughtDecoder(vocab_size, 256, 4, 4, 0.1, 8192).to(device)
    t_dim = args.thinker_dim
    t_heads = max(4, t_dim // 64)
    # Create thinker on CPU first — HIP kernel errors on init for larger
    # d_model sizes (384) on this gfx1031 GPU.
    thinker_net_cpu = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(t_dim, t_heads, dropout=0.1, batch_first=True,
                                   dim_feedforward=t_dim * 4),
        args.thinker_layers,
    )
    for p in thinker_net_cpu.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    thinker_net = thinker_net_cpu.to(device)
    pred = LossPredictor(256, 256).to(device)

    # TF32 matmul precision is deliberately NOT enabled here.
    # On this AMD ROCm stack the override causes numerical instability
    # (NaN divergence ~20K steps in).  Full FP32 is stable.

    enc.load_state_dict(ckpt["encoder_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    if "predictor_state" in ckpt:
        pred.load_state_dict(ckpt["predictor_state"])
    else:
        # No predictor in checkpoint (e.g. blend encoder+decoder).
        # Xavier init — will be trained from scratch.
        for p in pred.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    model = ThinkerModel(enc, dec, thinker_net, pred,
                         max_turns=args.max_past_turns + 1,
                         thinker_dim=t_dim)
    model.train()
    model.to(device)

    enc_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in dec.parameters() if p.requires_grad)
    thinker_params = sum(p.numel() for p in thinker_net.parameters() if p.requires_grad)
    print(f"Params: encoder={enc_params:,} decoder={dec_params:,} thinker={thinker_params:,}")

    # ════════════════════════════════════════════════════════════════
    # PHASE 1 — Encoder+Decoder reconstruction fine-tuning (1 epoch)
    # ════════════════════════════════════════════════════════════════
    if args.skip_phase1:
        print("\n" + "=" * 60)
        print("PHASE 1: skipped — loading from", args.output_phase1)
        print("=" * 60)
        p1_ckpt = torch.load(args.output_phase1, map_location=device, weights_only=True)
        enc.load_state_dict(p1_ckpt["encoder_state"])
        dec.load_state_dict(p1_ckpt["decoder_state"])
        # Thinker is NOT loaded — Xavier init avoids NaN on this ROCm stack.
        if "predictor_state" in p1_ckpt:
            pred.load_state_dict(p1_ckpt["predictor_state"])
        _load_embeddings(model, p1_ckpt, device)
    else:
        print("\n" + "=" * 60)
        print("PHASE 1: Encoder + Decoder reconstruction fine-tuning (1 epoch)")
        print("=" * 60)

        freeze(thinker_net)
        freeze(pred)
        unfreeze(enc)
        unfreeze(dec)

        opt_p1 = torch.optim.AdamW(
            [p for p in deduplicate_params(model) if p.requires_grad],
            lr=args.lr_phase1, weight_decay=1e-5,
        )

        # Pool all texts (inputs + outputs) for reconstruction
        all_texts = inputs + outputs  # 2× n_pairs
        random.shuffle(all_texts)
        n_p1 = len(all_texts) // args.batch_size
        t0 = time.time()

        for bi in range(n_p1):
            batch_texts = all_texts[bi * args.batch_size : (bi + 1) * args.batch_size]
            ids = encode_batch(batch_texts, tokenizer, pad_id, device)

            max_seq = model.encoder.positional_encoding.pe.size(1)
            if ids.size(1) > max_seq:
                ids = ids[:, :max_seq]

            pad_mask = ids.eq(pad_id)
            thoughts = model.encoder(ids, pad_mask)  # [B, 256, D]
            logits = model.decoder(thoughts, ids[:, :-1], pad_mask[:, :-1])

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                ids[:, 1:].reshape(-1),
                ignore_index=pad_id,
            )

            if loss.isnan() or loss.isinf():
                print(f"  [P1] batch {bi:>6}/{n_p1}  loss=NaN  (skipping)")
                opt_p1.zero_grad(set_to_none=True)
                continue

            opt_p1.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(opt_p1.param_groups[0]["params"], 1.0)

            # Save pre-step weights for NaN rollback (ROCm numerical
            # instability can corrupt weights even with NaN loss guard).
            if bi > 0 and bi % 100 == 0:
                pre_step = {k: v.data.clone() for k, v in model.encoder.state_dict().items()}
                pre_step.update({k: v.data.clone() for k, v in model.decoder.state_dict().items()})
            elif bi == 0:
                pre_step = {k: v.data.clone() for k, v in model.encoder.state_dict().items()}
                pre_step.update({k: v.data.clone() for k, v in model.decoder.state_dict().items()})

            opt_p1.step()

            # Post-step weight NaN check (ROCm gfx1031 can silently
            # corrupt parameters under the hipBLASLt fallback).
            if any(torch.isnan(p).any() for p in opt_p1.param_groups[0]["params"]):
                # Rollback to pre-step state
                model.encoder.load_state_dict({k: v for k, v in pre_step.items()
                                                if k in model.encoder.state_dict()}, strict=False)
                model.decoder.load_state_dict({k: v for k, v in pre_step.items()
                                                if k in model.decoder.state_dict()}, strict=False)
                print(f"  [P1] batch {bi:>6}/{n_p1}  weight=NaN  (rolled back)")
                continue

            if bi % 1000 == 0 or bi == n_p1 - 1:
                el = (time.time() - t0) / 60
                print(f"  [P1] batch {bi:>6}/{n_p1}  loss={loss.item():.4f}  {el:.1f}min")

        print(f"  [P1] done")

        # Save Phase 1 checkpoint
        print(f"\nSaving Phase 1 checkpoint → {args.output_phase1}")
        _save_with_embeddings(args.output_phase1, model, ckpt["config"])
        print("  done")

        if args.stop_after_phase1:
            print("--stop-after-phase1 set, exiting.")
            return

    # ════════════════════════════════════════════════════════════════
    # PHASE 2 — Thinker training (locked encoder + decoder, multi-turn)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"PHASE 2: Thinker training ({args.epochs_phase2} epochs)")
    print("       multi-turn: up to {}+1 turns, flip_prob={}".format(
        args.max_past_turns, args.flip_prob))
    print("=" * 60)

    freeze(enc)
    freeze(dec)
    unfreeze(thinker_net)
    freeze(pred)

    # Pre-encode all texts at max_k to a single combined memory-mapped file.
    # Combined shape: [2*n_pairs, max_k, d_model] — inputs then outputs.
    # The encoder is frozen, so we run it once and cache.
    combined_pre_path = f"{args.pre_encode_path}.mmap"
    if not Path(combined_pre_path).exists():
        print("\nPre-encoding texts via combined memmap (encoder is frozen, caching for speed)...")
        _pre_encode_and_memmap(inputs, outputs, enc, tokenizer, pad_id, args.max_k, device, combined_pre_path)
    else:
        print(f"\nUsing pre-encoded combined memmap: {combined_pre_path}")
    combined_pre_path_for_batches: str | None = combined_pre_path

    opt_p2 = torch.optim.AdamW(
        model.thinker.parameters(), lr=args.lr_phase2, weight_decay=1e-5,
    )
    t0 = time.time()

    for epoch in range(1, args.epochs_phase2 + 1):
        indices = list(range(n_pairs))
        random.shuffle(indices)
        epoch_loss = 0.0

        for bi in range(n_batches):
            batch_indices = indices[bi * args.batch_size : (bi + 1) * args.batch_size]
            B = len(batch_indices)

            # Sample n_past for this batch
            n_past = random.randint(0, args.max_past_turns)

            # Decide flip
            flip = random.random() < args.flip_prob

            # Sample k
            k = random.randint(4, args.max_k)

            # Build multi-turn contexts with metadata
            # Use pre-encoded vectors in Phase 2 (encoder frozen).
            contexts, targets, turn_ids, speaker_ids, decode_mask = build_multi_turn_batch(
                inputs, outputs, batch_indices, n_past,
                tokenizer, pad_id, device, model, k, flip,
                pre_in=combined_pre_path_for_batches, pre_out="",
            )

            # Noise perturbation on thought vectors (acts as regulariser
            # against brittle encoder outputs).
            if args.noise_std > 0 and model.training:
                contexts = contexts + torch.randn_like(contexts) * args.noise_std

            # Run thinker with structural embeddings
            thought_after = model.thinker_forward(
                contexts, turn_ids, speaker_ids, decode_mask,
            )  # [B, total_k, D]

            # Cosine diversity loss: push thought vectors apart within
            # each sample so they carry distinct information.
            if args.diversity_weight > 0 and k > 1:
                decode_vecs = thought_after[:, decode_mask[0], :]  # [B, k, D]
                normed = F.normalize(decode_vecs, dim=-1)
                sim = normed @ normed.transpose(-2, -1)
                eye = torch.eye(k, device=device, dtype=torch.bool)
                diversity = (sim[:, ~eye] ** 2).mean()

            # Decode from the current-user segment only, using
            # decode_mask to identify which vectors to read.
            # All samples in the batch share the same mask structure
            # (same n_past and k), so we can decode the whole batch
            # in one forward pass instead of looping per-sample.
            seg_batch = thought_after[:, decode_mask[0], :]  # [B, k, D]
            logits = model.decoder(seg_batch, targets[:, :-1])  # [B, T-1, V]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                ignore_index=pad_id,
            )

            # Repetition penalty: penalise predicting the same token
            # as the immediate preceding context token.
            if args.repetition_penalty_weight > 0:
                context_tokens = targets[:, :-1]  # [B, T-1]
                pred_probs = F.softmax(logits, dim=-1)
                rep_probs = pred_probs.gather(-1, context_tokens.unsqueeze(-1)).squeeze(-1)
                rep_loss = rep_probs.mean()
                loss = loss + args.repetition_penalty_weight * rep_loss

            # Add diversity loss
            if args.diversity_weight > 0 and k > 1:
                loss = loss + args.diversity_weight * diversity

            # Latent consistency loss: align thinker output vectors with
            # encoded target vectors from the pre-encoded memmap.
            # No live tokenization or encoder forward needed.
            if args.latent_consistency_weight > 0 and combined_pre_path_for_batches is not None:
                # Target indices: either outputs (offset=n) or inputs (offset=0)
                tgt_indices = [(idx, not flip) for idx in batch_indices]
                tgt_vecs = _gather_pre(combined_pre_path_for_batches, tgt_indices, k, device, n_pairs)
                decode_vecs = thought_after[:, decode_mask[0], :]  # [B, k, D]
                latent_loss = F.mse_loss(decode_vecs, tgt_vecs)
                loss = loss + args.latent_consistency_weight * latent_loss

            if loss.isnan() or loss.isinf():
                continue

            opt_p2.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.thinker.parameters(), 1.0)
            opt_p2.step()

            epoch_loss += float(loss.detach().cpu())

            if bi % 1000 == 0 or bi == n_batches - 1:
                el = (time.time() - t0) / 60
                print(f"  [P2] ep={epoch} batch={bi:>6}/{n_batches}  "
                      f"k={k:3d} past={n_past} flip={int(flip)}  "
                      f"loss={loss.item():.4f}  {el:.1f}min")

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"  [P2] epoch {epoch}/{args.epochs_phase2}  avg_loss={avg_loss:.4f}  "
              f"{el:.1f}min")

    # Save Phase 2 checkpoint
    print(f"\nSaving Phase 2 checkpoint → {args.output_phase2}")
    _save_with_embeddings(args.output_phase2, model, ckpt["config"])
    print("  done")

    # ════════════════════════════════════════════════════════════════
    # PHASE 3 — Full fine-tune (all unlocked, 1 epoch, multi-turn)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PHASE 3: Full fine-tune (all unlocked, 1 epoch)")
    print("       multi-turn: up to {}+1 turns, flip_prob={}".format(
        args.max_past_turns, args.flip_prob))
    print("=" * 60)

    unfreeze(enc)
    unfreeze(dec)
    unfreeze(thinker_net)
    unfreeze(pred)

    opt_p3 = torch.optim.AdamW(
        deduplicate_params(model), lr=args.lr_phase3, weight_decay=1e-5,
    )
    t0 = time.time()

    indices = list(range(n_pairs))
    random.shuffle(indices)

    for bi in range(n_batches):
        batch_indices = indices[bi * args.batch_size : (bi + 1) * args.batch_size]
        B = len(batch_indices)

        n_past = random.randint(0, args.max_past_turns)
        flip = random.random() < args.flip_prob
        k = random.randint(4, args.max_k)

        contexts, targets, turn_ids, speaker_ids, decode_mask = build_multi_turn_batch(
            inputs, outputs, batch_indices, n_past,
            tokenizer, pad_id, device, model, k, flip,
        )

        # Noise perturbation
        if args.noise_std > 0 and model.training:
            contexts = contexts + torch.randn_like(contexts) * args.noise_std

        thought_after = model.thinker_forward(
            contexts, turn_ids, speaker_ids, decode_mask,
        )

        # Cosine diversity loss
        if args.diversity_weight > 0 and k > 1:
            decode_vecs = thought_after[:, decode_mask[0], :]
            normed = F.normalize(decode_vecs, dim=-1)
            sim = normed @ normed.transpose(-2, -1)
            eye = torch.eye(k, device=device, dtype=torch.bool)
            diversity = (sim[:, ~eye] ** 2).mean()

        seg_batch = thought_after[:, decode_mask[0], :]
        logits = model.decoder(seg_batch, targets[:, :-1])

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )

        # Repetition penalty
        if args.repetition_penalty_weight > 0:
            context_tokens = targets[:, :-1]
            pred_probs = F.softmax(logits, dim=-1)
            rep_probs = pred_probs.gather(-1, context_tokens.unsqueeze(-1)).squeeze(-1)
            rep_loss = rep_probs.mean()
            loss = loss + args.repetition_penalty_weight * rep_loss

        # Add diversity loss
        if args.diversity_weight > 0 and k > 1:
            loss = loss + args.diversity_weight * diversity

        # Latent consistency loss
        if args.latent_consistency_weight > 0:
            target_texts_for_consistency = [outputs[idx] if not flip else inputs[idx] for idx in batch_indices]
            tgt_ids = encode_batch(target_texts_for_consistency, tokenizer, pad_id, device)
            tgt_thoughts = model.encoder(tgt_ids, tgt_ids.eq(pad_id))
            tgt_vecs = tgt_thoughts[:, :k, :]
            decode_vecs = thought_after[:, decode_mask[0], :]
            latent_loss = F.mse_loss(decode_vecs, tgt_vecs)
            loss = loss + args.latent_consistency_weight * latent_loss

        if loss.isnan() or loss.isinf():
            continue

        opt_p3.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(opt_p3.param_groups[0]["params"], 1.0)
        opt_p3.step()

        if bi % 1000 == 0 or bi == n_batches - 1:
            el = (time.time() - t0) / 60
            print(f"  [P3] batch={bi:>6}/{n_batches}  "
                  f"k={k:3d} past={n_past} flip={int(flip)}  "
                  f"loss={loss.item():.4f}  {el:.1f}min")

    # Save final checkpoint
    print(f"\nSaving final checkpoint → {args.output}")
    _save_with_embeddings(args.output, model, ckpt["config"])
    print("Done!")


if __name__ == "__main__":
    main()
