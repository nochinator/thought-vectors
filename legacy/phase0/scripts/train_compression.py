#!/usr/bin/env python3
"""Three-phase compression training for thought vectors."""
from __future__ import annotations

import argparse
import sys
import random as rnd
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel
from thought_vectors.data import GroupTextDataset, collate_group_batch
from thought_vectors.inference import decode_greedy


class LossPredictor(nn.Module):
    """Predicts reconstruction loss for each possible thought-vector prefix length."""

    def __init__(self, d_model: int, max_thoughts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, max_thoughts),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        # thoughts: (B, N, D) → pool → (B, D) → (B, N) loss predictions
        pooled = thoughts.mean(dim=1)
        return self.net(pooled)


def _compute_losses_for_prefixes(
    model: ThoughtVectorModel,
    thoughts: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
    max_count: int,
) -> list[float]:
    """Return reconstruction loss for each prefix length 1..max_count."""
    losses = []
    for k in range(1, max_count + 1):
        prefix = thoughts[:, :k, :]
        logits = model.decoder(prefix, input_ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
            ignore_index=pad_token_id,
        )
        losses.append(float(loss.detach().cpu()))
    return losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="C4 autoencoder checkpoint")
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/compressed.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--predictor-lr", type=float, default=1e-3)
    parser.add_argument("--phase1-steps", type=int, default=10000, help="Steps for random-count pretraining")
    parser.add_argument("--phase2-steps", type=int, default=5000, help="Steps for predictor training")
    parser.add_argument("--phase3-steps", type=int, default=20000, help="Steps for joint training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    rnd.seed(0)
    torch.manual_seed(0)

    # ── Load tokenizer ──
    tokenizer = SPTokenizer()
    tokenizer.load(args.tokenizer_model)
    print(f"Tokenizer vocab={tokenizer.vocab_size}")

    # ── Build + load base model ──
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    max_thoughts = cfg.get("num_thoughts", 32)
    encoder = ThoughtEncoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        nhead=cfg["heads"], num_layers=cfg["encoder_layers"],
        dropout=cfg["dropout"], max_seq_len=cfg["max_seq_len"],
        num_thoughts=max_thoughts,
    )
    decoder = ThoughtDecoder(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
        nhead=cfg["heads"], num_layers=cfg["decoder_layers"],
        dropout=cfg["dropout"], max_seq_len=cfg["max_seq_len"],
    )
    model = ThoughtVectorModel(encoder, decoder)

    # Transfer weights — skip shape mismatches (e.g. thought_seed when expanding thoughts)
    old_state = ckpt["model_state"]
    new_state = model.state_dict()
    transferred = 0
    for key, val in old_state.items():
        if key in new_state and val.shape == new_state[key].shape:
            new_state[key] = val
            transferred += 1
    model.load_state_dict(new_state)
    model.to(device)

    # Predictor
    predictor = LossPredictor(cfg["d_model"], max_thoughts).to(device)
    print(f"Base model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Predictor has {sum(p.numel() for p in predictor.parameters()):,} params")

    # ── Data ──
    from thought_vectors.data_loading import load_groups_from_path
    groups = load_groups_from_path(args.data)
    print(f"Loaded {len(groups)} texts")
    dataset = GroupTextDataset(groups)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_group_batch(batch, tokenizer.encode, tokenizer.pad_token_id),
        num_workers=0,
    )
    total_steps = args.phase1_steps + args.phase2_steps + args.phase3_steps
    steps_so_far = 0

    # ── Optimisers ──
    seen_ptrs: set[int] = set()
    unique_params = [p for p in model.parameters()
                     if p.data_ptr() not in seen_ptrs and not seen_ptrs.add(p.data_ptr())]
    opt = torch.optim.AdamW(unique_params, lr=args.lr, weight_decay=1e-5)
    opt_pred = torch.optim.AdamW(predictor.parameters(), lr=args.predictor_lr, weight_decay=0)

    # ═══════════════════════════════════════════
    # PHASE 1 — random-count pretraining
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}\nPHASE 1: Random-count pretraining ({args.phase1_steps} steps)\n{'='*60}")
    loader_iter = iter(loader)
    for step in range(args.phase1_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch.to(device)
        # Truncate to model's max sequence length
        max_seq = model.encoder.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_seq:
            input_ids = input_ids[:, :max_seq]
        pad_mask = input_ids.eq(tokenizer.pad_token_id)
        model.train()
        thoughts = model.encoder(input_ids, pad_mask)
        N = thoughts.size(1)

        # Blend two k-sampling strategies:
        #   50% uniform (original) — maintains robustness across all k
        #   50% skewed (compression-biased) — pushes performance at 2:1-4:1
        if rnd.random() < 0.5:
            # Uniform: original Phase 1 behavior
            k = rnd.randint(4, N)
        else:
            # Skewed toward higher compression ratios
            #   ratio 0.4-0.6 (2:1 zone): strong across all lengths
            #   ratio 0.25-0.4 (3:1 zone): works for short-medium texts
            #   ratio 0.15-0.25 (4:1+ zone): pushing the limit
            #   ratio 0.6-1.5 (easy zone): stability
            mean_seq = (input_ids != tokenizer.pad_token_id).sum(dim=1).float().mean().item()
            roll = rnd.random()
            if roll < 0.30:
                lo, hi = 0.4, 0.6
            elif roll < 0.60:
                lo, hi = 0.25, 0.4
            elif roll < 0.80:
                lo, hi = 0.15, 0.25
            else:
                lo, hi = 0.6, 1.5
            k = max(4, min(N, int(mean_seq * rnd.uniform(lo, hi))))

        # Encoder freeze at low counts — detach so no gradient flows back
        encoder_grad = k >= 8
        selected = thoughts[:, :k, :]
        if not encoder_grad:
            selected = selected.detach()

        logits = model.decoder(selected, input_ids[:, :-1])
        target = input_ids[:, 1:]
        recon = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                                ignore_index=tokenizer.pad_token_id)

        # Coherence weighting: at low counts, soften targets via KL vs smoothed distribution
        if k < 12:
            with torch.no_grad():
                smoothed = F.softmax(logits / 2.0, dim=-1).detach()  # temp=2 → softer target
            coherence = F.kl_div(F.log_softmax(logits, dim=-1), smoothed, reduction="batchmean")
            weight = (12 - k) / 8.0  # 0 at k=12, 1 at k=4
            loss = recon + weight * 0.1 * coherence
        else:
            loss = recon

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0 or step == args.phase1_steps - 1:
            print(f"  [{step:5d}] k={k:2d}  recon={recon.item():.4f}  {'grad' if encoder_grad else 'frozen'}")

    # ═══════════════════════════════════════════
    # PHASE 2 — predictor training
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}\nPHASE 2: Predictor training ({args.phase2_steps} steps)\n{'='*60}")
    model.eval()
    for step in range(args.phase2_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch.to(device)
        max_seq = model.encoder.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_seq:
            input_ids = input_ids[:, :max_seq]
        pad_mask = input_ids.eq(tokenizer.pad_token_id)
        with torch.no_grad():
            thoughts = model.encoder(input_ids, pad_mask)
            # Sample random prefix lengths instead of all 32 (much faster)
            sample_ks = rnd.sample(range(1, max_thoughts + 1), min(4, max_thoughts))
            targets = {}
            for k in sample_ks:
                prefix = thoughts[:, :k, :]
                logits = model.decoder(prefix, input_ids[:, :-1])
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                targets[k] = loss

        # Predict for all ks, but only compute loss for sampled ones
        preds = predictor(thoughts).mean(dim=0)  # (N,)
        p_loss = sum(F.mse_loss(preds[k - 1:k], t.unsqueeze(0)) for k, t in targets.items())
        p_loss = p_loss / len(targets)
        opt_pred.zero_grad()
        p_loss.backward()
        opt_pred.step()

        if step % 1000 == 0 or step == args.phase2_steps - 1:
            with torch.no_grad():
                err = p_loss.item()
            print(f"  [{step:5d}] pred_loss={p_loss.item():.4f}  mean_abs_err={err:.4f}")

    model.train()

    # ═══════════════════════════════════════════
    # PHASE 3 — joint training
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}\nPHASE 3: Joint training ({args.phase3_steps} steps)\n{'='*60}")
    train_predictor = True
    selected_k = max_thoughts  # default for logging
    for step in range(args.phase3_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch.to(device)
        # Truncate to model's max sequence length
        max_seq = model.encoder.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_seq:
            input_ids = input_ids[:, :max_seq]
        pad_mask = input_ids.eq(tokenizer.pad_token_id)
        model.train()
        thoughts = model.encoder(input_ids, pad_mask)

        # Decide: train predictor (40%) or train encoder-decoder at predicted K (60%)
        if train_predictor:
            model.eval()
            with torch.no_grad():
                sample_ks = rnd.sample(range(1, max_thoughts + 1), min(4, max_thoughts))
                targets = {}
                for k in sample_ks:
                    prefix = thoughts[:, :k, :]
                    logits = model.decoder(prefix, input_ids[:, :-1])
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                        ignore_index=tokenizer.pad_token_id,
                    )
                    targets[k] = loss
            preds = predictor(thoughts).mean(dim=0)
            p_loss = sum(F.mse_loss(preds[k - 1:k], t.unsqueeze(0)) for k, t in targets.items())
            p_loss = p_loss / len(targets)
            opt_pred.zero_grad()
            p_loss.backward()
            opt_pred.step()
            train_predictor = rnd.random() < 0.4
        else:
            model.train()
            with torch.no_grad():
                pred_losses = predictor(thoughts).mean(dim=0)  # (N,)
            target_loss = rnd.uniform(0.25, 2.5)
            # Find minimum k where predictor thinks loss <= target_loss
            selected_k = max_thoughts
            for k in range(1, max_thoughts + 1):
                if pred_losses[k - 1].item() <= target_loss:
                    selected_k = k
                    break

            encoder_grad = selected_k >= 8
            selected = thoughts[:, :selected_k, :]
            if not encoder_grad:
                selected = selected.detach()

            logits = model.decoder(selected, input_ids[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_predictor = rnd.random() < 0.4

        if step % 2000 == 0 or step == args.phase3_steps - 1 or step == 0:
            k_str = f"k={selected_k:2d}" if not train_predictor else "train_pred"
            print(f"  [{step:5d}] {k_str}  "
                  f"{'recon=' + f'{loss.item():.4f}' if not train_predictor else f'pred_loss={p_loss.item():.4f}'}")

    # ── Save ──
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "predictor_state": predictor.state_dict(),
        "config": cfg,
        "history": [],
        "tokenizer_model_path": args.tokenizer_model,
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
