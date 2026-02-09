#!/usr/bin/env python3
from __future__ import annotations

import sys
import os

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SimpleTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel, train_model
from thought_vectors.data_loading import load_groups_from_path


def build_model_from_config(config: dict) -> ThoughtVectorModel:
    encoder = ThoughtEncoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
        num_thoughts=config["num_thoughts"],
    )
    decoder = ThoughtDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )
    return ThoughtVectorModel(encoder, decoder)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Thought Vector model on grouped text data.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset (.json, .jsonl, or .csv). CSV uses first column as text.")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable text normalization preprocessing.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Checkpoint path to resume model weights/history from.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--length-penalty", type=float, default=0.01)
    parser.add_argument("--num-thoughts", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-vectors", type=int, default=None)
    parser.add_argument("--selection-stride", type=int, default=2)
    parser.add_argument("--disable-dynamic-target", action="store_true")
    parser.add_argument("--target-start", type=float, default=1.8)
    parser.add_argument("--target-end", type=float, default=0.45)
    parser.add_argument("--target-length-weight", type=float, default=0.65)
    parser.add_argument("--target-noise-std", type=float, default=0.07)
    parser.add_argument("--target-extreme-prob", type=float, default=0.12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=8, help="Print reconstruction sample every N batches.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/thought_vectors.pt"))
    args = parser.parse_args()

    groups = load_groups_from_path(args.data, preprocess=not args.no_preprocess)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_history: list[float] = []

    if args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        config = payload["config"]
        tokenizer = SimpleTokenizer.from_token_to_id(payload["token_to_id"])
        model = build_model_from_config(config)
        model.load_state_dict(payload["model_state"])
        prior_history = [float(x) for x in payload.get("history", [])]
        print(f"[train] resumed from checkpoint: {args.resume_from}")
    else:
        tokenizer = SimpleTokenizer()
        tokenizer.fit(groups)
        config = {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "num_thoughts": args.num_thoughts,
        }
        model = build_model_from_config(config)

    history = train_model(
        model,
        groups,
        tokenizer.encode,
        tokenizer.pad_token_id,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        length_penalty=args.length_penalty,
        use_dynamic_loss_target=not args.disable_dynamic_target,
        target_start=args.target_start,
        target_end=args.target_end,
        target_length_weight=args.target_length_weight,
        target_noise_std=args.target_noise_std,
        target_extreme_prob=args.target_extreme_prob,
        max_vectors=args.max_vectors,
        selection_stride=args.selection_stride,
        log_every=args.log_every,
        sample_every_batches=args.sample_every,
        tokenizer_decode=tokenizer.decode,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    full_history = prior_history + history

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "token_to_id": tokenizer.token_to_id,
            "history": full_history,
        },
        args.output,
    )

    print(f"Device: {device}")
    print(f"Epoch losses (new): {history}")
    print(f"Epoch losses (full): {full_history}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
