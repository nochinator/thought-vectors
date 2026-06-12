"""Download + normalize conversational corpora to a unified JSONL.

Output: data/conversations.jsonl, one conversation per line:
    {"source": "soda", "turns": ["...", "..."]}
Turns strictly alternate user/bot starting with turn 0 = user (parity defines
role downstream). Casual scope: SODA (1.5M social dialogues — the bulk),
PersonaChat (small talk), OASST1 English (light knowledge).

datasets 5.0 dropped script-based datasets, which killed daily_dialog and
blended_skill_talk; SODA covers the same ground at 100x the volume.

Usage: .venv/bin/python scripts/extract_conversations.py [--out data/conversations.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

HEDGES = re.compile(
    r"as an ai|i('| a)?m an ai|language model|i cannot|i can't assist", re.IGNORECASE
)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def usable(turns: list[str], min_turns: int = 2, max_chars: int = 600) -> bool:
    if len(turns) < min_turns:
        return False
    for t in turns:
        if not t or len(t) > max_chars or HEDGES.search(t):
            return False
    return True


def soda(cap: int = 400_000):
    # 1.19M train dialogues; "dialogue" is a list of utterances, two speakers
    # alternating. Stream so we never materialize the full set.
    ds = load_dataset("allenai/soda", split="train", streaming=True)
    n = 0
    for row in ds:
        if n >= cap:
            break
        yield [clean(u) for u in row["dialogue"]]
        n += 1


def personachat():
    # Rows are cumulative (one per utterance); the last row per conversation
    # holds the full history. Rows of one conversation are contiguous, so a
    # history-length drop marks a new conversation.
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    for row in ds:
        last = row["utterances"][-1]
        yield [clean(u) for u in last["history"]] + [clean(last["candidates"][-1])]


def oasst1_en():
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {r["message_id"]: r for r in ds if r["lang"] == "en"}
    children: dict[str | None, list[dict]] = {}
    for r in by_id.values():
        children.setdefault(r["parent_id"], []).append(r)
    # walk best-ranked path from each root
    for root in children.get(None, []):
        turns, node = [], root
        while node is not None:
            turns.append(clean(node["text"]))
            kids = [k for k in children.get(node["message_id"], []) if k["message_id"] in by_id]
            node = min(kids, key=lambda r: r.get("rank") or 0) if kids else None
        if root["role"] == "prompter":
            yield turns


SOURCES = {
    "soda": soda,
    "personachat": personachat,
    "oasst1": oasst1_en,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/conversations.jsonl")
    parser.add_argument("--sources", default=",".join(SOURCES))
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with open(args.out, "w") as f:
        for name in args.sources.split(","):
            n = 0
            try:
                for turns in SOURCES[name]():
                    if not usable(turns):
                        continue
                    f.write(json.dumps({"source": name, "turns": turns}) + "\n")
                    n += 1
            except Exception as e:  # a single source failing shouldn't kill the rest
                print(f"WARNING: {name} failed: {e}")
            counts[name] = n
            print(f"{name}: {n} conversations", flush=True)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
