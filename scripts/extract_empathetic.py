"""Convert facebook/empathetic_dialogues raw CSVs to our conversations JSONL.

R7 (RESEARCH_LOG 2026-07-03): the register disease — cheerful replies to bad
news, worst with an upbeat prior turn — is a data problem; dialogue_combined
is relentlessly upbeat smalltalk. EmpatheticDialogues is 24.8K conversations
grounded in labeled emotions (heavy on negative situations), purpose-built
bad-news -> commiseration data.

The HF dataset is script-based (dead in datasets 5.0), so this reads the raw
ParlAI tarball CSVs (https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/).
Row: conv_id, utterance_idx, context(emotion), prompt, speaker_idx, utterance,
selfeval, tags — commas inside utterances are escaped as "_comma_", so plain
csv parsing is safe. speaker parity already matches our convention: turn 0 =
sharer (user), turn 1 = listener (bot).

Usage: .venv/bin/python scripts/extract_empathetic.py --dir <extracted>/ \
           [--out data/conversations_empathetic.jsonl]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def clean(text: str) -> str:
    return " ".join(text.replace("_comma_", ",").split()).strip()


def conversations(csv_path: Path):
    convs: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cid, utt = row.get("conv_id"), row.get("utterance")
            if not cid or not utt:
                continue
            if cid not in convs:
                convs[cid] = []
                order.append(cid)
            convs[cid].append((int(row["utterance_idx"]), clean(utt)))
    for cid in order:
        yield [u for _, u in sorted(convs[cid])]


def usable(turns: list[str], min_turns: int = 2, max_chars: int = 600) -> bool:
    return len(turns) >= min_turns and all(0 < len(t) <= max_chars for t in turns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with train/valid/test.csv")
    ap.add_argument("--out", default="data/conversations_empathetic.jsonl")
    args = ap.parse_args()

    n = kept = 0
    with open(args.out, "w") as f:
        for split in ("train", "valid", "test"):
            for turns in conversations(Path(args.dir) / f"{split}.csv"):
                n += 1
                if not usable(turns):
                    continue
                f.write(json.dumps({"source": "empathetic", "turns": turns}) + "\n")
                kept += 1
    print(f"empathetic: kept {kept}/{n} conversations -> {args.out}")


if __name__ == "__main__":
    main()
