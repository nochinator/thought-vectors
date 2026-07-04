#!/usr/bin/env python3
"""Train a 16K SentencePiece tokenizer from Alpaca data."""
import sentencepiece as spm
from datasets import load_dataset
import os

ds = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
with open("/tmp/sp_train.txt", "w") as f:
    for i, ex in enumerate(ds):
        f.write(ex["instruction"] + "\n")
        f.write(ex["output"] + "\n")
        if i >= 500000:
            break
print(f"Wrote {i+1} lines")

spm.SentencePieceTrainer.train(
    input="/tmp/sp_train.txt",
    model_prefix="/tmp/sp_c4_16k",
    vocab_size=16384,
    character_coverage=1.0,
    model_type="unigram",
    bos_id=1, eos_id=2, pad_id=0, unk_id=3,
    num_threads=4,
)
print(f"Done: {os.path.getsize('/tmp/sp_c4_16k.model')} bytes")
os.unlink("/tmp/sp_train.txt")
