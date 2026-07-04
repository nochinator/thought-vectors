"""
Train a Thinker transformer on CNN/DailyMail summarization.

Pipeline: article → [frozen encoder] → k-slice → [thinker] → [frozen decoder] → summary

The encoder, decoder, and loss predictor are loaded from a pre-trained
compression checkpoint.  Only the thinker transformer is trained.

Random k-slicing forces the thinker to handle compressed representations.
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
from thought_vectors import (
    SPTokenizer, ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel,
    LossPredictor, ThinkerModel,
)


def load_pretrained(path: str | Path, device: torch.device) -> tuple[ThoughtEncoder, ThoughtDecoder, LossPredictor]:
    """Load encoder, decoder, and predictor from a compression checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]

    encoder = ThoughtEncoder(
        cfg["vocab_size"], cfg["d_model"], cfg["heads"],
        cfg["encoder_layers"], cfg["dropout"], cfg["max_seq_len"], cfg["num_thoughts"],
    )
    decoder = ThoughtDecoder(
        cfg["vocab_size"], cfg["d_model"], cfg["heads"],
        cfg["decoder_layers"], cfg["dropout"], cfg["max_seq_len"],
    )
    predictor = LossPredictor(cfg["d_model"], cfg["num_thoughts"])

    # Load weights
    encoder.load_state_dict({k: v for k, v in ckpt["model_state"].items() if k.startswith("encoder.")}, strict=False)
    decoder.load_state_dict({k: v for k, v in ckpt["model_state"].items() if k.startswith("decoder.")}, strict=False)
    if "predictor_state" in ckpt:
        predictor.load_state_dict(ckpt["predictor_state"])

    # Freeze
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    return encoder.to(device), decoder.to(device), predictor.to(device)


def build_thinker(d_model: int, nhead: int, num_layers: int, dropout: float) -> nn.TransformerEncoder:
    """Build a transformer that operates on thought vectors."""
    layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="artifacts/vae_compressed_blend.pt",
                        help="Pre-trained compression checkpoint")
    parser.add_argument("--tokenizer-model", default="/tmp/sp_c4_16k.model")
    parser.add_argument("--article-data", type=str, required=True,
                        help="CSV with one article per line (for encoder input)")
    parser.add_argument("--summary-data", type=str, required=True,
                        help="CSV with one summary per line (for decoder target)")
    parser.add_argument("--thinker-layers", type=int, default=6,
                        help="Number of thinker transformer layers")
    parser.add_argument("--thinker-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-article-len", type=int, default=512)
    parser.add_argument("--max-summary-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--freeze-thinker", action="store_true",
                        help="Freeze thinker, only train decoder (Stage 0: bootstrap decoder)")
    parser.add_argument("--unfreeze-decoder", action="store_true",
                        help="Unfreeze decoder weights for summarization fine-tuning")
    parser.add_argument("--min-k", type=int, default=4,
                        help="Minimum k for random slicing during training")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--output", type=str, default="artifacts/thinker.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ──
    tok = SPTokenizer()
    tok.load(args.tokenizer_model)
    pad_id = tok.pad_token_id
    bos_id = tok.bos_token_id
    eos_id = tok.eos_token_id
    print(f"Tokenizer vocab={tok.vocab_size}")

    # ── Load pre-trained encoder + decoder + predictor ──
    print(f"Loading pre-trained from {args.checkpoint}...")
    encoder, decoder, predictor = load_pretrained(args.checkpoint, device)
    d_model = encoder.d_model
    num_thoughts = encoder.num_thoughts
    print(f"  Encoder/decoder loaded and frozen (d_model={d_model}, thoughts={num_thoughts})")

    # ── Build thinker ──
    thinker = build_thinker(d_model, args.thinker_heads, args.thinker_layers, args.dropout)
    print(f"  Thinker: {args.thinker_layers} layers, {args.thinker_heads} heads")
    print(f"  Thinker params: {sum(p.numel() for p in thinker.parameters()):,}")

    # ── Build full model ──
    model = ThinkerModel(encoder, decoder, thinker, predictor).to(device)

    # Optionally unfreeze decoder for summarization fine-tuning
    if args.unfreeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = True
        print("  Decoder unfrozen for fine-tuning")

    # Collect trainable parameters
    if args.freeze_thinker:
        for p in model.thinker.parameters():
            p.requires_grad = False
        trainable = [p for p in model.decoder.parameters()]
        print("  Thinker frozen, only training decoder")
    else:
        trainable = [p for p in model.thinker.parameters()]
        if args.unfreeze_decoder:
            trainable += [p for p in model.decoder.parameters()]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Load paired data ──
    csv.field_size_limit(2 ** 31 - 1)
    articles: list[str] = []
    summaries: list[str] = []
    print("Loading article-summary pairs...")

    with open(args.article_data) as fa, open(args.summary_data) as fs:
        ra = csv.reader(fa)
        rs = csv.reader(fs)
        for row_a, row_s in zip(ra, rs):
            if not row_a or not row_s:
                continue
            art_ids = tok.encode(row_a[0], add_special_tokens=True)
            sum_ids = tok.encode(row_s[0], add_special_tokens=True)
            if len(art_ids) <= args.max_article_len and len(sum_ids) <= args.max_summary_len:
                articles.append(row_a[0])
                summaries.append(row_s[0])
                if len(articles) >= 50000:
                    break

    print(f"  Loaded {len(articles)} article-summary pairs")

    # ── Optimiser ──
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-5)

    # Output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\nTraining: batch_size={args.batch_size}, epochs={args.epochs}")
    print(f"  Random k-slicing range: [{args.min_k}, {num_thoughts}]")
    print(f"{'step':>6} {'epoch':>5} {'k':>4} {'loss':>8} {'elapsed':>8}")
    print("-" * 40)

    losses: list[float] = []
    t_start = time.time()
    step = 0

    for epoch in range(1, args.epochs + 1):
        # Shuffle paired data
        combined = list(zip(articles, summaries))
        random.shuffle(combined)
        num_batches = len(combined) // args.batch_size

        for batch_idx in range(num_batches):
            step += 1
            batch = combined[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            batch_arts, batch_sums = zip(*batch)

            # Tokenize articles
            art_enc = [tok.encode(t, add_special_tokens=True) for t in batch_arts]
            max_a = max(len(e) for e in art_enc)
            art_ids = torch.full((args.batch_size, max_a), pad_id, dtype=torch.long)
            for i, e in enumerate(art_enc):
                art_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            art_ids = art_ids.to(device)

            # Tokenize summaries
            sum_enc = [tok.encode(t, add_special_tokens=True) for t in batch_sums]
            max_s = max(len(e) for e in sum_enc)
            sum_ids = torch.full((args.batch_size, max_s), pad_id, dtype=torch.long)
            for i, e in enumerate(sum_enc):
                sum_ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)
            sum_ids = sum_ids.to(device)

            # Random k-slicing or thinker bypass
            if args.freeze_thinker:
                k = -1  # Bypass thinker, use all vectors
            else:
                k = random.randint(args.min_k, num_thoughts)

            # Forward through thinker
            logits = model(art_ids, sum_ids, padding_mask=art_ids.eq(pad_id), k=k)

            # Loss: cross-entropy on summary tokens (shifted)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                sum_ids[:, 1:].reshape(-1),
                ignore_index=pad_id,
            )

            # Clamp loss to prevent NaN cascade
            if loss.isnan() or loss.isinf():
                print(f"  WARN: NaN/Inf loss at step {step}, skipping")
                continue

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()
            losses.append(loss.item())

            # Save partial checkpoint every 2000 batches
            if batch_idx > 0 and batch_idx % 2000 == 0:
                torch.save({
                    "encoder_state": model.encoder.state_dict(),
                    "decoder_state": model.decoder.state_dict(),
                    "thinker_state": model.thinker.state_dict(),
                    "predictor_state": model.predictor.state_dict(),
                    "config": {"d_model": d_model, "thoughts": num_thoughts,
                               "thinker_layers": args.thinker_layers, "thinker_heads": args.thinker_heads,
                               "vocab_size": tok.vocab_size},
                    "step": step, "losses": losses,
                }, str(output_path) + ".partial")

            if batch_idx % args.log_every == 0 or batch_idx == num_batches - 1:
                avg_loss = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
                elapsed = time.time() - t_start
                print(f"{step:>6} {epoch:>5} k={k:3d}  {avg_loss:>8.4f} {elapsed:>8.1f}s")

            # Sample generation
            if (batch_idx % args.sample_every == 0 or batch_idx == num_batches - 1) and batch_idx > 0:
                with torch.no_grad():
                    # Use first article in batch as sample
                    sample_art = art_ids[:1]
                    sample_sum = sum_ids[:1]
                    sample_k = k

                    thoughts = model.encoder(sample_art, sample_art.eq(pad_id))
                    if sample_k >= 0:
                        thoughts = thoughts[:, :sample_k, :]
                    if sample_k != -1:
                        thoughts = model.thinker(thoughts)

                    gen = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
                    for _ in range(args.max_summary_len * 2):
                        lg = model.decoder(thoughts, gen)
                        nx = lg[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nx], dim=1)
                        if nx.item() == eos_id:
                            break
                    input_text = tok.decode(sample_art[0].tolist(), skip_special_tokens=True)
                    target_text = tok.decode(sample_sum[0].tolist(), skip_special_tokens=True)
                    output_text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
                    print(f"  [sample] ARTICLE: {input_text[:80]}")
                    print(f"  [sample] TARGET:  {target_text[:80]}")
                    print(f"  [sample] OUTPUT:  {output_text[:80]}")

    # ── Save ──
    torch.save({
        "encoder_state": model.encoder.state_dict(),
        "decoder_state": model.decoder.state_dict(),
        "thinker_state": model.thinker.state_dict(),
        "predictor_state": model.predictor.state_dict(),
        "config": {
            "d_model": d_model, "thoughts": num_thoughts,
            "thinker_layers": args.thinker_layers, "thinker_heads": args.thinker_heads,
            "vocab_size": tok.vocab_size,
        },
        "history": losses,
    }, output_path)
    total_time = time.time() - t_start
    print(f"\nSaved to {output_path}")
    print(f"Total: {total_time / 60:.1f}min, final loss={sum(losses[-100:]) / min(100, len(losses)):.4f}")


if __name__ == "__main__":
    main()
