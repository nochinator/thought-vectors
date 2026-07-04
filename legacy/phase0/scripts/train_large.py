"""
Train a d_model=512, 256-thought-vector model on mixed C4+CNN+minipile data.

Uses 1.5× token-to-vector ratio: k = min(256, int(seq_len * 1.5)).
Skips texts longer than 512 tokens (no truncation).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from thought_vectors import SPTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel


def load_groups_from_path(path: Path) -> list[list[str]]:
    """Load each line as a single-text group (no CSV header handling needed)."""
    groups: list[list[str]] = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                groups.append([row[0]])
    return groups


def build_model(vocab_size: int, d_model: int, heads: int, layers: int,
                num_thoughts: int, dropout: float, max_seq_len: int) -> ThoughtVectorModel:
    encoder = ThoughtEncoder(vocab_size, d_model, heads, layers, dropout, max_seq_len, num_thoughts)
    decoder = ThoughtDecoder(vocab_size, d_model, heads, layers, dropout, max_seq_len)
    model = ThoughtVectorModel(encoder, decoder)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", required=True, help="CSV files with one text per line")
    parser.add_argument("--tokenizer-model", default="/tmp/sp_c4_16k.model")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--num-thoughts", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length for positional encoding")
    parser.add_argument("--filter-len", type=int, default=None,
                        help="Filter texts longer than this many tokens (defaults to max-seq-len)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from an existing checkpoint")
    parser.add_argument("--vector-ratio", type=float, default=1.5,
                        help="k = min(num_thoughts, int(seq_len * this_ratio))")
    parser.add_argument("--kl-beta", type=float, default=0.0,
                        help="Weight for KL divergence toward N(0,1).  0 = deterministic encoder.")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Gaussian noise stddev added to thought vectors during training")
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Probability of interpolating thought vectors between batch items")
    parser.add_argument("--non-autoregressive-recon", action="store_true",
                        help="Replace decoder inputs with blanks, disable causal mask. Forces decoder to use thought vectors.")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--sample-every", type=int, default=400)
    parser.add_argument("--output", default="artifacts/large_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---------- Load tokenizer ----------
    tok = SPTokenizer()
    tok.load(args.tokenizer_model)
    vocab_size = tok.vocab_size
    pad_id = tok.pad_token_id
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    print(f"Tokenizer: vocab_size={vocab_size}")

    # ---------- Filter data to ≤filter_len tokens ----------
    filter_len = args.filter_len if args.filter_len is not None else args.max_seq_len
    csv.field_size_limit(2 ** 31 - 1)
    texts: list[str] = []
    skipped = 0
    for data_path in args.data:
        path = Path(data_path)
        print(f"Loading {path.name}...")
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                ids = tok.encode(row[0], add_special_tokens=True)
                if len(ids) <= filter_len:
                    texts.append(row[0])
                else:
                    skipped += 1
                if len(texts) >= 300000:
                    break
        print(f"  -> {len(texts)} texts in memory ({skipped} skipped so far)")

    random.shuffle(texts)
    print(f"Total: {len(texts)} texts, skipped {skipped} long texts")

    # ---------- Build model ----------
    print(f"Building model: d_model={args.d_model}, layers={args.layers}, "
          f"heads={args.heads}, thoughts={args.num_thoughts}")
    model = build_model(vocab_size, args.d_model, args.heads, args.layers,
                        args.num_thoughts, args.dropout, args.max_seq_len)

    # Load existing checkpoint if available (for continued training)
    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}...")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            ckpt_state = ckpt["model_state"]
            # Filter to only load matching-shape keys (handles architecture changes)
            current_state = model.state_dict()
            filtered = {}
            for k, v in ckpt_state.items():
                if k in current_state and v.shape == current_state[k].shape:
                    filtered[k] = v
            model.load_state_dict(filtered, strict=False)
            n_loaded = len(filtered)
            n_total = len(current_state)
            print(f"  Loaded {n_loaded}/{n_total} weights ({n_total - n_loaded} randomly initialized)")
        else:
            print(f"  Checkpoint {ckpt_path} not found, starting fresh")

    model = model.to(device)

    # Count unique parameters (skip tied duplicates)
    seen_ptrs: set[int] = set()
    unique_params: list[nn.Parameter] = []
    for p in model.parameters():
        if p.data_ptr() not in seen_ptrs:
            seen_ptrs.add(p.data_ptr())
            unique_params.append(p)
    total_params = sum(p.numel() for p in unique_params)
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(unique_params, lr=args.lr)

    # ---------- Training loop ----------
    print(f"\nTraining: batch_size={args.batch_size}, epochs={args.epochs}, "
          f"vector_ratio={args.vector_ratio}")
    print(f"Total batches per epoch: {len(texts) // args.batch_size}")
    print(f"{'step':>6} {'epoch':>5} {'batch':>8} {'loss':>8} {'k_avg':>6} {'elapsed':>8}")
    print("-" * 50)

    losses: list[float] = []
    t_start = time.time()
    step = 0

    # Pre-compute token lengths for length-based bucketing
    print("Computing token lengths for bucketing...")
    text_lens: list[tuple[str, int]] = []
    for t in texts:
        ids_len = len(tok.encode(t, add_special_tokens=True))
        text_lens.append((t, ids_len))

    for epoch in range(1, args.epochs + 1):
        # Sort by token length so each batch has similar ks
        text_lens.sort(key=lambda x: x[1])
        num_batches = len(text_lens) // args.batch_size

        # Shuffle batch order for diversity across epochs
        batch_positions = list(range(num_batches))
        random.shuffle(batch_positions)

        for batch_idx, batch_pos in enumerate(batch_positions):
            step += 1
            start = batch_pos * args.batch_size
            batch_with_lens = text_lens[start:start + args.batch_size]
            batch_texts = [t for t, _ in batch_with_lens]

            # Tokenize and pad
            encoded = [tok.encode(t, add_special_tokens=True) for t in batch_texts]
            max_len = max(len(e) for e in encoded)
            ids = torch.full((args.batch_size, max_len), pad_id, dtype=torch.long)
            for i, e in enumerate(encoded):
                ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            ids = ids.to(device)

            # k = min(num_thoughts, int(seq_len * vector_ratio)).
            # Since texts are sorted by length within the batch, ks are all similar.
            seq_lens = (ids != pad_id).sum(dim=1)
            ks = [min(args.num_thoughts, max(4, int(l.item() * args.vector_ratio)))
                  for l in seq_lens]
            k = max(ks)

            padding_mask = ids.eq(pad_id)

            # VAE encoding (sampled) or deterministic encoding
            if args.kl_beta > 0:
                z, mu, logvar = model.encoder.encode_with_kl(ids, padding_mask)
                thoughts = z
                # KL divergence: sum over d_model, mean over batch × thoughts
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl.sum(dim=-1).mean()
            else:
                thoughts = model.encoder(ids, padding_mask)
                kl_loss = None

            thoughts = thoughts[:, :k, :]

            # Noise perturbation
            if args.noise_std > 0:
                thoughts = thoughts + torch.randn_like(thoughts) * args.noise_std

            # Mixup
            if args.mixup_alpha > 0 and random.random() < args.mixup_alpha:
                perm = torch.randperm(thoughts.size(0), device=thoughts.device)
                lam = torch.rand(1, device=thoughts.device).item()
                thoughts = lam * thoughts + (1 - lam) * thoughts[perm]

            # Non-autoregressive mode: blank out decoder inputs, disable causal masking.
            # The decoder must predict all target tokens from thought vectors alone.
            if args.non_autoregressive_recon:
                # Replace target input embeddings with pad tokens (position encoding only)
                blank_target = torch.full_like(ids[:, :-1], pad_id)
                logits = model.decoder(thoughts, blank_target, padding_mask[:, :-1], causal=False)
                # Predict all tokens in parallel — target is full sequence (shifted)
                target = ids[:, 1:]
            else:
                logits = model.decoder(thoughts, ids[:, :-1], padding_mask[:, :-1])
                target = ids[:, 1:]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=pad_id,
            )

            if kl_loss is not None:
                loss = loss + args.kl_beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if batch_idx % args.log_every == 0 or batch_idx == num_batches - 1:
                avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
                avg_k = sum(ks) // len(ks)
                elapsed = time.time() - t_start
                print(f"{step:>6} {epoch:>5} {batch_idx:>8}/{num_batches} "
                      f"{avg_loss:>8.4f} {avg_k:>6} {elapsed:>8.1f}s")

            # Sample reconstruction
            if (batch_idx % args.sample_every == 0 or batch_idx == num_batches - 1) \
                    and batch_idx > 0:
                with torch.no_grad():
                    sample_ids = ids[:1]
                    sample_k = ks[0]
                    thoughts_s = model.encoder(sample_ids, sample_ids.eq(pad_id))
                    thoughts_s = thoughts_s[:, :sample_k, :]
                    gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                    for _ in range(args.max_seq_len * 2):
                        logits_s = model.decoder(thoughts_s, gen,
                                                  None if gen.shape[1] == 1 else None)
                        nxt = logits_s[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nxt], dim=1)
                        if nxt.item() == eos_id:
                            break
                    input_text = tok.decode(sample_ids[0].tolist(), skip_special_tokens=True)
                    output_text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
                    print(f"  [sample] IN:  {input_text[:80]}")
                    print(f"  [sample] OUT: {output_text[:80]}")

    # ---------- Save ----------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": vocab_size, "d_model": args.d_model, "heads": args.heads,
            "encoder_layers": args.layers, "decoder_layers": args.layers,
            "dropout": args.dropout, "max_seq_len": args.max_seq_len,
            "num_thoughts": args.num_thoughts,
        },
        "tokenizer_model_path": args.tokenizer_model,
        "history": losses,
    }
    torch.save(checkpoint, output_path)
    total_time = time.time() - t_start
    print(f"\nSaved to {output_path}")
    print(f"Total training time: {total_time / 60:.1f}min")
    print(f"Final avg loss (last {min(200, len(losses))} steps): "
          f"{sum(losses[-200:]) / min(200, len(losses)):.4f}")


if __name__ == "__main__":
    main()
