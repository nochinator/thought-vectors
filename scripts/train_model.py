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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Thought Vector model on grouped text data.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset (.json, .jsonl, or .csv). CSV uses first column as text.")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable text normalization preprocessing.")
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
    parser.add_argument("--output", type=Path, default=Path("artifacts/thought_vectors.pt"))
    args = parser.parse_args()

    groups = load_groups_from_path(args.data, preprocess=not args.no_preprocess)
    tokenizer = SimpleTokenizer()
    tokenizer.fit(groups)

    encoder = ThoughtEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        num_thoughts=args.num_thoughts,
    )
    decoder = ThoughtDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
    )
    model = ThoughtVectorModel(encoder, decoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "d_model": args.d_model,
                "heads": args.heads,
                "layers": args.layers,
                "dropout": args.dropout,
                "max_seq_len": args.max_seq_len,
                "num_thoughts": args.num_thoughts,
            },
            "token_to_id": tokenizer.token_to_id,
            "history": history,
        },
        args.output,
    )

    print(f"Device: {device}")
    print(f"Epoch losses: {history}")
    print(f"Saved checkpoint: {args.output}")


if __name__ == "__main__":
    main()
