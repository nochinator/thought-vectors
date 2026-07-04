#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thought_vectors import SPTokenizer, SimpleTokenizer, ThoughtDecoder, ThoughtEncoder, ThoughtVectorModel, train_model
from thought_vectors.data_loading import load_groups_from_path


def build_model_from_config(config: dict) -> ThoughtVectorModel:
    encoder = ThoughtEncoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["encoder_layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
        num_thoughts=config["num_thoughts"],
    )
    decoder = ThoughtDecoder(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["heads"],
        num_layers=config["decoder_layers"],
        dropout=config["dropout"],
        max_seq_len=config["max_seq_len"],
    )
    return ThoughtVectorModel(encoder, decoder)


def _resize_embedding(old_embedding: nn.Embedding, new_vocab_size: int) -> nn.Embedding:
    old_vocab_size, d_model = old_embedding.weight.shape
    new_embedding = nn.Embedding(new_vocab_size, d_model)
    nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)
    copy_size = min(old_vocab_size, new_vocab_size)
    with torch.no_grad():
        new_embedding.weight[:copy_size] = old_embedding.weight[:copy_size]
    return new_embedding


def _resize_linear(old_linear: nn.Linear, new_out_features: int) -> nn.Linear:
    old_out, d_model = old_linear.weight.shape
    new_linear = nn.Linear(d_model, new_out_features)
    nn.init.normal_(new_linear.weight, mean=0.0, std=0.02)
    nn.init.zeros_(new_linear.bias)
    copy_size = min(old_out, new_out_features)
    with torch.no_grad():
        new_linear.weight[:copy_size] = old_linear.weight[:copy_size]
        new_linear.bias[:copy_size] = old_linear.bias[:copy_size]
    return new_linear


def maybe_expand_vocab(model: ThoughtVectorModel, new_vocab_size: int) -> None:
    old_vocab_size = model.encoder.token_embedding.num_embeddings
    if new_vocab_size <= old_vocab_size:
        return

    d_model = model.encoder.d_model
    # Resize the shared embedding once; both encoder and decoder reference it
    new_embedding = _resize_embedding(model.encoder.token_embedding, new_vocab_size)
    model.encoder.token_embedding = new_embedding
    model.decoder.token_embedding = new_embedding

    # Resize LM head, preserve bias, re-tie weight to the shared embedding
    old_lm_head = model.decoder.lm_head
    has_bias = old_lm_head.bias is not None
    new_lm_head = nn.Linear(d_model, new_vocab_size, bias=has_bias)
    if has_bias:
        with torch.no_grad():
            new_lm_head.bias[:old_vocab_size] = old_lm_head.bias
    new_lm_head.weight = new_embedding.weight  # re-tie
    model.decoder.lm_head = new_lm_head

    print(f"[train] expanded vocabulary: {old_vocab_size} -> {new_vocab_size}")


def save_checkpoint(path: Path, model: ThoughtVectorModel, config: dict, tokenizer, history: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "token_to_id": tokenizer.token_to_id if hasattr(tokenizer, "token_to_id") else tokenizer.to_dict(),
            "tokenizer_model_path": getattr(tokenizer, "_model_path", None),
            "history": history,
        },
        path,
    )


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
    parser.add_argument("--encoder-layers", type=int, default=6, help="Number of encoder transformer layers.")
    parser.add_argument("--decoder-layers", type=int, default=3, help="Number of decoder transformer layers.")
    parser.add_argument("--layers", type=int, default=None, help="Set both encoder and decoder layers (overrides --encoder-layers and --decoder-layers).")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-vectors", type=int, default=None)
    parser.add_argument("--selection-stride", type=int, default=2)
    parser.add_argument("--diversity-weight", type=float, default=0.1, help="Weight for thought-vector cosine diversity loss (0 to disable).")
    parser.add_argument("--contrastive-weight", type=float, default=0.0, help="Weight for group-contrastive InfoNCE loss (0 to disable).")
    parser.add_argument("--repetition-penalty", type=float, default=0.0, help="Weight for token-level repetition penalty (0 to disable).")
    parser.add_argument("--disable-dynamic-target", action="store_true")
    parser.add_argument("--target-start", type=float, default=1.8)
    parser.add_argument("--target-end", type=float, default=0.45)
    parser.add_argument("--target-length-weight", type=float, default=0.65)
    parser.add_argument("--target-noise-std", type=float, default=0.07)
    parser.add_argument("--target-extreme-prob", type=float, default=0.12)
    parser.add_argument("--save-every", action="store_true", help="Save checkpoint at the end of every epoch.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=8, help="Print reconstruction sample every N batches.")
    parser.add_argument("--tokenizer-min-frequency", type=int, default=1, help="Minimum token frequency retained in tokenizer vocab.")
    parser.add_argument("--tokenizer-max-vocab-size", type=int, default=None, help="Cap tokenizer vocab size.")
    parser.add_argument("--tokenizer-count-memory-limit", type=int, default=400_000, help="Bound intermediate token counter size to reduce RAM while fitting tokenizer.")
    parser.add_argument("--tokenizer-model", type=str, default=None, help="Path to a SentencePiece .model file (enables subword tokenizer).")
    parser.add_argument("--output", type=Path, default=Path("artifacts/thought_vectors.pt"))
    args = parser.parse_args()

    groups = load_groups_from_path(args.data, preprocess=not args.no_preprocess)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prior_history: list[float] = []

    # Resolve layer counts: --layers overrides both encoder and decoder
    if args.layers is not None:
        enc_layers = args.layers
        dec_layers = args.layers
    else:
        enc_layers = args.encoder_layers
        dec_layers = args.decoder_layers

    if args.tokenizer_model is not None:
        tokenizer = SPTokenizer()
        tokenizer.load(args.tokenizer_model)
        config = {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "heads": args.heads,
            "encoder_layers": enc_layers,
            "decoder_layers": dec_layers,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "num_thoughts": args.num_thoughts,
        }
        model = build_model_from_config(config)
    elif args.resume_from is not None:
        payload = torch.load(args.resume_from, map_location=device, weights_only=True)
        config = payload["config"]

        # Restore tokenizer — prefer saved .model path for SPTokenizer
        sp_model_path = payload.get("tokenizer_model_path")
        if sp_model_path and Path(sp_model_path).exists():
            tokenizer = SPTokenizer()
            tokenizer.load(sp_model_path)
        else:
            tokenizer = SimpleTokenizer.from_token_to_id(payload["token_to_id"])
            tokenizer.fit(
                groups,
                min_frequency=args.tokenizer_min_frequency,
                max_vocab_size=args.tokenizer_max_vocab_size,
                count_memory_limit=args.tokenizer_count_memory_limit,
            )

        model = build_model_from_config(config)
        model.load_state_dict(payload["model_state"])
        maybe_expand_vocab(model, tokenizer.vocab_size)
        config["vocab_size"] = tokenizer.vocab_size

        prior_history = [float(x) for x in payload.get("history", [])]
        print(f"[train] resumed from checkpoint: {args.resume_from}")
    else:
        tokenizer = SimpleTokenizer()
        tokenizer.fit(
            groups,
            min_frequency=args.tokenizer_min_frequency,
            max_vocab_size=args.tokenizer_max_vocab_size,
            count_memory_limit=args.tokenizer_count_memory_limit,
        )
        config = {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "heads": args.heads,
            "encoder_layers": enc_layers,
            "decoder_layers": dec_layers,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "num_thoughts": args.num_thoughts,
        }
        model = build_model_from_config(config)

    interrupted = False
    history: list[float] = []
    try:
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
            diversity_weight=args.diversity_weight,
            contrastive_weight=args.contrastive_weight,
            repetition_penalty_weight=args.repetition_penalty,
            log_every=args.log_every,
            sample_every_batches=args.sample_every,
            checkpoint_path=str(args.output) if args.save_every else None,
            tokenizer_decode=tokenizer.decode,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[train] keyboard interrupt received, saving checkpoint before exit...")

    full_history = prior_history + history
    save_checkpoint(args.output, model, config, tokenizer, full_history)

    print(f"Device: {device}")
    print(f"Epoch losses (new): {history}")
    print(f"Epoch losses (full): {full_history}")
    print(f"Saved checkpoint: {args.output}")
    if interrupted:
        print("[train] exited early due to keyboard interrupt.")


if __name__ == "__main__":
    main()

