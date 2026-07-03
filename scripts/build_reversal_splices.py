"""Synthesize mid-conversation register-REVERSAL dialogues from EmpatheticDialogues.

R7 post-mortem (RESEARCH_LOG 2026-07-03): after an upbeat turn, ALL WTA
hypotheses reply positively — the thinker trunk conditions on conversation-level
mood, not the last turn. Root cause is a data absence: every training
conversation (SODA, personachat, ED alike) holds ONE mood throughout, so
register reversal within a conversation has zero support in the data.

This splices ED conversations by their emotion label:
    pos_conv[:2] + neg_conv[:4]   (default mix also emits some neg->pos)
Both halves are sharer-starts situation-sharing style, so parity (even = user)
and style survive the splice; only the mood flips — exactly the pattern the
model must learn to route on.

Usage: .venv/bin/python scripts/build_reversal_splices.py --dir <ed_csvs>/ \
           [--out data/conversations_reversal.jsonl] [--n 20000]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

POSITIVE = {
    "anticipating", "caring", "confident", "content", "excited", "faithful",
    "grateful", "hopeful", "impressed", "joyful", "prepared", "proud", "trusting",
}
NEGATIVE = {
    "afraid", "angry", "annoyed", "anxious", "apprehensive", "ashamed",
    "devastated", "disappointed", "disgusted", "embarrassed", "furious",
    "guilty", "jealous", "lonely", "sad", "terrified",
}
# ambiguous, excluded: surprised, nostalgic, sentimental


def clean(text: str) -> str:
    return " ".join(text.replace("_comma_", ",").split()).strip()


def load(dir_: Path) -> dict[str, list[list[str]]]:
    convs: dict[str, tuple[str, list[tuple[int, str]]]] = {}
    for split in ("train", "valid", "test"):
        with open(dir_ / f"{split}.csv", newline="") as f:
            for row in csv.DictReader(f):
                cid, utt = row.get("conv_id"), row.get("utterance")
                if not cid or not utt:
                    continue
                if cid not in convs:
                    convs[cid] = (row["context"], [])
                convs[cid][1].append((int(row["utterance_idx"]), clean(utt)))
    out: dict[str, list[list[str]]] = {"pos": [], "neg": []}
    for emotion, turns in convs.values():
        seq = [u for _, u in sorted(turns)]
        if len(seq) < 2 or any(not t or len(t) > 600 for t in seq):
            continue
        if emotion in POSITIVE:
            out["pos"].append(seq)
        elif emotion in NEGATIVE:
            out["neg"].append(seq)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with train/valid/test.csv")
    ap.add_argument("--out", default="data/conversations_reversal.jsonl")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--frac-neg-to-pos", type=float, default=0.25,
                    help="fraction spliced bad->good (teach the rule, not a bias)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pools = load(Path(args.dir))
    rng = random.Random(args.seed)
    print(f"pools: pos {len(pools['pos'])} neg {len(pools['neg'])}")

    with open(args.out, "w") as f:
        for i in range(args.n):
            flip = rng.random() < args.frac_neg_to_pos
            a = rng.choice(pools["neg" if flip else "pos"])
            b = rng.choice(pools["pos" if flip else "neg"])
            turns = a[:2] + b[:4]
            f.write(json.dumps({"source": "reversal", "turns": turns}) + "\n")
    print(f"wrote {args.n} spliced conversations -> {args.out}")


if __name__ == "__main__":
    main()
