#!/usr/bin/env python3
"""Expand model to 128 thoughts, transfer weights from 32-thought checkpoint, train on C4."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel
from thought_vectors.data import GroupTextDataset, collate_group_batch
from thought_vectors.data_loading import load_groups_from_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="32-thought checkpoint")
    parser.add_argument("--tokenizer-model", type=str, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/c4_256d_128t.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    # Load tokenizer
    tokenizer = SPTokenizer()
    tokenizer.load(args.tokenizer_model)
    print(f"Tokenizer vocab={tokenizer.vocab_size}")

    # Build 128-thought model
    enc = ThoughtEncoder(tokenizer.vocab_size, d_model=256, nhead=4, num_layers=4, num_thoughts=128)
    dec = ThoughtDecoder(tokenizer.vocab_size, d_model=256, nhead=4, num_layers=4)
    model = ThoughtVectorModel(enc, dec)

    # Transfer weights from 32-thought checkpoint — skip shape mismatches
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    old_state = ckpt["model_state"]
    new_state = model.state_dict()
    transferred = 0
    for key, val in old_state.items():
        if key in new_state and val.shape == new_state[key].shape:
            new_state[key] = val
            transferred += 1
    model.load_state_dict(new_state)
    print(f"Transferred {transferred}/{len(old_state)} weight tensors")
    model.to(device)
    model.train()

    # Load data
    groups = load_groups_from_path(args.data)
    print(f"Data: {len(groups)} texts")
    dataset = GroupTextDataset(groups)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda batch: collate_group_batch(batch, tokenizer.encode, tokenizer.pad_token_id),
        num_workers=0,
    )

    # Optimizer (deduplicate tied weights)
    seen: set[int] = set()
    params = [p for p in model.parameters() if p.data_ptr() not in seen and not seen.add(p.data_ptr())]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    # Train
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

        for batch_idx, batch in enumerate(loader, start=1):
            ids = batch.to(device)
            max_seq = model.encoder.positional_encoding.pe.size(1)
            if ids.size(1) > max_seq:
                ids = ids[:, :max_seq]
            pad = ids.eq(tokenizer.pad_token_id)

            thoughts = model.encoder(ids, pad)
            logits = model.decoder(thoughts, ids[:, :-1])
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.detach().cpu())
            batches += 1

            if batch_idx % args.log_every == 0 or batch_idx == len(loader):
                t1.record(); torch.cuda.synchronize()
                elapsed = t0.elapsed_time(t1) / 1000
                print(f"  epoch={epoch} batch={batch_idx}/{len(loader)} loss={epoch_loss/batches:.4f} elapsed={elapsed:.1f}s")

            if batch_idx % args.sample_every == 0 or batch_idx == len(loader):
                model.eval()
                with torch.no_grad():
                    gen = torch.full((1, 1), tokenizer.bos_token_id, dtype=torch.long, device=device)
                    for _ in range(64):
                        logits = model.decoder(thoughts[:1], gen)
                        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        gen = torch.cat([gen, nxt], dim=1)
                        if nxt.item() == tokenizer.eos_token_id: break
                    recon = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
                print(f"  [sample] {recon!r}")
                model.train()

        avg = epoch_loss / batches
        history.append(avg)
        print(f"  Epoch {epoch} done: avg_loss={avg:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {
        "vocab_size": tokenizer.vocab_size, "d_model": 256, "heads": 4,
        "encoder_layers": 4, "decoder_layers": 4, "dropout": 0.1,
        "max_seq_len": 512, "num_thoughts": 128,
    }, "history": history, "tokenizer_model_path": args.tokenizer_model}, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
