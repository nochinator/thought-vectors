#!/usr/bin/env python3
"""Extract clean conversation pairs from multiple sources, no hedging."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from thought_vectors.tokenization import SPTokenizer

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
max_tok = 80

hedge = [
    "i am sorry", "i'm sorry", "i apologize", "i cannot", "i do not have",
    "i am not able", "i dont have", "i am an ai", "i'm an ai", "as an ai",
    "i understand you", "i appreciate your", "one key thing", "pleasure to help",
    "how can i assist", "is not possible", "i'm not able", "sorry, i cannot",
    "i cannot provide", "feel free to ask", "i am not capable",
    "i don't have feelings", "here are a few", "sure, here are",
    "sorry, but i", "i dont have the ability", "i do not have access",
    "i don't have access", "i am not able to provide",
    "i cannot provide information",
]

def clean(text: str) -> str:
    return text.strip()

def ok(text: str) -> bool:
    t = clean(text).lower()
    if len(t) < 4 or len(t) > 300:
        return False
    for p in hedge:
        if p in t[:120]:
            return False
    return True

def fits(text: str) -> bool:
    ids = tok.encode(clean(text), add_special_tokens=True)
    return 2 <= len(ids) <= max_tok

pairs: list[tuple[str, str]] = []

def try_add(u: str, a: str):
    if len(pairs) >= 300000:
        return True  # done
    if ok(u) and ok(a) and fits(u) and fits(a):
        pairs.append((u, a))
    return False

# 1. Dolly (human-written, clean)
print("Dolly...")
try:
    for ex in load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True):
        if try_add(ex.get("instruction", ""), ex.get("response", "")):
            break
except: pass
print(f"  {len(pairs)} pairs")

# 2. UltraChat 200K (filter hedged)
print("UltraChat...")
for ex in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
    msgs = ex["messages"]
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            if try_add(msgs[i]["content"].strip(), msgs[i + 1]["content"].strip()):
                break
    if len(pairs) >= 300000:
        break
print(f"  {len(pairs)} pairs")

# 3. LMSYS Arena
print("LMSYS...")
for ex in load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True):
    try:
        pl = json.loads(ex["prompt"]) if isinstance(ex["prompt"], str) and ex["prompt"].startswith("[") else [ex["prompt"]]
        ral = json.loads(ex["response_a"]) if isinstance(ex["response_a"], str) and ex["response_a"].startswith("[") else [ex["response_a"]]
        rbl = json.loads(ex["response_b"]) if isinstance(ex["response_b"], str) and ex["response_b"].startswith("[") else [ex["response_b"]]
    except Exception:
        continue
    prompt = (pl[0] if isinstance(pl, list) and pl else "").strip()
    for resp_raw in [ral[0] if isinstance(ral, list) and ral else "", rbl[0] if isinstance(rbl, list) and rbl else ""]:
        if not resp_raw or not isinstance(resp_raw, str):
            continue
        resp = resp_raw.strip()
        if try_add(prompt, resp):
            break
    if len(pairs) >= 300000:
        break
print(f"  {len(pairs)} pairs")

# 4. No Robots (curated, clean)
print("No Robots...")
try:
    for ex in load_dataset("HuggingFaceH4/no_robots", split="train_sft", streaming=True):
        msgs = ex["messages"]
        for i in range(len(msgs) - 1):
            if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
                if try_add(msgs[i]["content"].strip(), msgs[i + 1]["content"].strip()):
                    break
        if len(pairs) >= 300000:
            break
except: pass
print(f"  {len(pairs)} pairs")

# Save
print(f"Total: {len(pairs)} pairs")
with open("/tmp/thinker_data_input.csv", "w") as f:
    csv.writer(f).writerows([[t] for t, _ in pairs])
with open("/tmp/thinker_data_output.csv", "w") as f:
    csv.writer(f).writerows([[t] for _, t in pairs])
print("Saved to /tmp/thinker_data_input.csv / output.csv")
import statistics
ul = [len(tok.encode(t, add_special_tokens=True)) for t, _ in pairs]
al = [len(tok.encode(t, add_special_tokens=True)) for _, t in pairs]
print(f"User: mean={statistics.mean(ul):.0f}, max={max(ul)}")
print(f"Asst: mean={statistics.mean(al):.0f}, max={max(al)}")
for i in range(min(3, len(pairs))):
    print(f"  U: {pairs[i][0][:60]}")
    print(f"  A: {pairs[i][1][:60]}")
