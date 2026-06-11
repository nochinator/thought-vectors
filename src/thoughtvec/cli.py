"""CLI entry points: tv-train, tv-eval, tv-tokenizer, tv-pretokenize, tv-chat."""

from __future__ import annotations

import argparse
import os
import sys


def _rocm_env() -> None:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


def train_main() -> None:
    _rocm_env()
    parser = argparse.ArgumentParser(prog="tv-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--init-from", default=None, help="checkpoint to warm-start weights from (fresh schedule)"
    )
    parser.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = parser.parse_args()

    import torch

    from .config import load_config
    from .data import make_loader
    from .model import ThoughtAutoencoder
    from .tokenizer import Tokenizer
    from .train_loop import Trainer

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.train.seed)

    tokenizer = Tokenizer(cfg.run.tokenizer_path)
    assert tokenizer.vocab_size == cfg.model.vocab_size, (
        f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.model.vocab_size}"
    )
    model = ThoughtAutoencoder(cfg.model)
    print(f"model params: {model.param_count() / 1e6:.2f}M", flush=True)

    trainer = Trainer(cfg, model, tokenizer)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}", flush=True)
    elif args.init_from:
        trainer.load_checkpoint(args.init_from, reset_schedule=True)
        print(f"warm-started weights from {args.init_from}", flush=True)

    val_dir = cfg.data.val_shard_dir or cfg.data.shard_dir + "_val"
    train_loader = make_loader(
        cfg.data.shard_dir,
        cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        seed=cfg.train.seed,
    )
    val_loader = make_loader(val_dir, cfg.train.batch_size, shuffle=False, num_workers=0)
    trainer.fit(train_loader, val_loader)


def eval_main() -> None:
    _rocm_env()
    parser = argparse.ArgumentParser(prog="tv-eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--shard", required=True, help="val shard dir")
    parser.add_argument("--out", default=None, help="output dir (default: logs/<run>/eval)")
    parser.add_argument("--max-texts", type=int, default=2000)
    parser.add_argument("--decode-per-bucket", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    from .config import from_dict
    from .evaluate import evaluate
    from .model import ThoughtAutoencoder
    from .tokenizer import Tokenizer

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    model = ThoughtAutoencoder(cfg.model)
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device)
    tokenizer = Tokenizer(ckpt.get("tokenizer_path", cfg.run.tokenizer_path))
    out = args.out or os.path.join(cfg.run.log_dir, cfg.run.name, "eval")
    evaluate(
        model,
        tokenizer,
        args.shard,
        out,
        max_texts=args.max_texts,
        decode_per_bucket=args.decode_per_bucket,
        device=args.device,
    )


def tokenizer_main() -> None:
    parser = argparse.ArgumentParser(prog="tv-tokenizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--corpus", action="append", required=True, help="csv_path:every_nth")
    t.add_argument("--out", default="artifacts/tokenizer/spm16k_bpe")
    t.add_argument("--vocab-size", type=int, default=16384)
    i = sub.add_parser("inspect")
    i.add_argument("--model", default="artifacts/tokenizer/spm16k_bpe.model")
    i.add_argument("--csv", required=True)
    i.add_argument("--sample", type=int, default=2000)
    args = parser.parse_args()

    from .tokenizer import Tokenizer, iter_csv_texts, train_tokenizer

    if args.cmd == "train":
        files = []
        for spec in args.corpus:
            path, _, nth = spec.rpartition(":")
            files.append((path, int(nth)))
        out = train_tokenizer(files, args.out, vocab_size=args.vocab_size)
        print(f"tokenizer written to {out}")
    else:
        import numpy as np

        tok = Tokenizer(args.model)
        lengths = []
        mismatches = 0
        for i, text in enumerate(iter_csv_texts(args.csv)):
            if i >= args.sample:
                break
            ids = tok.encode(text)
            lengths.append(len(ids))
            if tok.decode(ids) != text and i < 200:
                mismatches += 1
        arr = np.array(lengths)
        print(f"vocab={tok.vocab_size} samples={len(arr)}")
        print(
            f"token lengths: mean={arr.mean():.1f} median={np.median(arr):.0f} "
            f"p10={np.percentile(arr, 10):.0f} p90={np.percentile(arr, 90):.0f} max={arr.max()}"
        )
        print(f"round-trip mismatches in first 200: {mismatches} "
              "(nmt_nfkc normalization makes some whitespace mismatches expected)")


def pretokenize_main() -> None:
    parser = argparse.ArgumentParser(prog="tv-pretokenize")
    parser.add_argument("--csv", required=True, action="append")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="artifacts/tokenizer/spm16k_bpe.model")
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=126)
    parser.add_argument("--val-frac", type=float, default=0.004)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--merge-rows", action="store_true")
    parser.add_argument("--no-chunk-long", action="store_true")
    parser.add_argument("--chunk-jitter", action="store_true")
    args = parser.parse_args()

    import json

    from .data import pretokenize
    from .tokenizer import Tokenizer

    meta = pretokenize(
        args.csv,
        args.out,
        Tokenizer(args.tokenizer),
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        val_frac=args.val_frac,
        max_rows=args.max_rows,
        merge_rows=args.merge_rows,
        chunk_long=not args.no_chunk_long,
        chunk_jitter=args.chunk_jitter,
    )
    print(json.dumps(meta, indent=2))


def chat_main() -> None:
    print("tv-chat arrives with milestone M4 (thinker).", file=sys.stderr)
    sys.exit(1)
