"""Build conversations_combined.jsonl — the thinker flagship's training mix.

Recipe (RESEARCH_LOG 2026-06-15): the original data/dialogue was 96% SODA
(short formulaic replies = the filler attractor). The combined mix is simply
conversations_clean.jsonl (quality-filtered SODA + PersonaChat + best-path
OASST) plus the full multi-path OASST extract appended, cutting SODA's share
6x and boosting OASST 10x. The two OASST sets overlap slightly (~75 exact
duplicates out of 8.7K) — harmless, kept as-is to match the flagship's data.

Inputs (build these first):
    data/conversations_clean.jsonl   — scripts/filter_dialogue.py
    data/oasst1_conversations.jsonl  — scripts/extract_oasst.py

Usage: .venv/bin/python scripts/build_dialogue_combined.py
"""

import json
from collections import Counter

CLEAN = "data/conversations_clean.jsonl"
OASST = "data/oasst1_conversations.jsonl"
DST = "data/conversations_combined.jsonl"

counts = Counter()
with open(DST, "w") as fout:
    for src in (CLEAN, OASST):
        with open(src) as fin:
            for line in fin:
                fout.write(line)
                counts[json.loads(line)["source"]] += 1

total = sum(counts.values())
print(f"{DST}: {total} conversations {dict(counts)}")
# Flagship mix for reference: 70,866 = soda 61,464 + personachat 863 + oasst1 8,539
