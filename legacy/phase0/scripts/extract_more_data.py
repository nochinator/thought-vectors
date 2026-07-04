#!/usr/bin/env python3
"""Build large clean dataset — no Anthropic HH, relaxed hedge filter."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from thought_vectors.tokenization import SPTokenizer

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
max_tok = 80

# Only reject if hedge phrase is in the FIRST 40 chars (starts the response)
hedge = [
    "i am sorry", "i'm sorry", "i apologize", "i cannot", "i do not have",
    "i am not able", "i dont have", "i am an ai", "i'm an ai", "as an ai",
    "i understand you", "i appreciate your", "pleasure to help",
    "how can i assist", "is not possible", "i'm not able",
    "i cannot provide", "feel free to ask", "i am not capable",
    "sorry, but i", "i cannot provide information", "it is not possible",
    "i do not have the capability",
]

def ok(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 4 or len(t) > 300:
        return False
    # Only check START of response for hedging
    start = t[:40]
    for p in hedge:
        if p in start:
            return False
    return True

def fits(text: str) -> bool:
    ids = tok.encode(text.strip(), add_special_tokens=True)
    return 2 <= len(ids) <= max_tok

CAP = 1000000
pairs: list[tuple[str, str]] = []
t0 = time.time()

def add(u: str, a: str):
    if len(pairs) >= CAP:
        return True
    u, a = u.strip(), a.strip()
    if ok(u) and ok(a) and fits(u) and fits(a):
        pairs.append((u, a))
    return False

# 1. Dolly
print("Dolly...")
for ex in load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True):
    if add(ex.get("instruction", ""), ex.get("response", "")): break
print(f"  {len(pairs)}")

# 2. UltraChat (full 200K — underindexed before)
print("UltraChat...")
for ex in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
    msgs = ex["messages"]
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            if add(msgs[i]["content"], msgs[i + 1]["content"]): break
    if len(pairs) >= CAP: break
print(f"  {len(pairs)}")

# 3. LMSYS Arena
print("LMSYS...")
for ex in load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True):
    try:
        pl = json.loads(ex["prompt"]) if isinstance(ex["prompt"], str) and ex["prompt"].startswith("[") else [ex["prompt"]]
        ral = json.loads(ex["response_a"]) if isinstance(ex["response_a"], str) and ex["response_a"].startswith("[") else [ex["response_a"]]
        rbl = json.loads(ex["response_b"]) if isinstance(ex["response_b"], str) and ex["response_b"].startswith("[") else [ex["response_b"]]
    except: continue
    prompt = (pl[0] if isinstance(pl, list) and pl else "").strip()
    for r in [ral[0] if isinstance(ral, list) and ral else "", rbl[0] if isinstance(rbl, list) and rbl else ""]:
        r = r.strip() if r else ""
        if add(prompt, r): break
    if len(pairs) >= CAP: break
print(f"  {len(pairs)}")

# 4. No Robots
print("No Robots...")
try:
    for ex in load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True):
        msgs = ex.get("messages", [])
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                if add(msgs[i].get("content", ""), msgs[i + 1].get("content", "")): break
        if len(pairs) >= CAP: break
except: pass
print(f"  {len(pairs)}")

# 5. Alpaca
print("Alpaca...")
for ex in load_dataset("tatsu-lab/alpaca", split="train", streaming=True):
    if add(ex.get("instruction", ""), ex.get("output", "")): break
print(f"  {len(pairs)}")

# 6. OpenOrca (full scan until CAP)
print("OpenOrca...")
for i, ex in enumerate(load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)):
    if i > 0 and i % 500000 == 0: print(f"    scanned {i}, {len(pairs)} pairs")
    q = ex.get("question", "")
    a = ex.get("response", "")
    if not q or not a: continue
    a_clean = a.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
    if add(q, a_clean): break
print(f"  {len(pairs)}")

# 7. Anthropic HH (filtered, safe-only)
print("Anthropic HH (safe only)...")
try:
    seen = set()
    for ex in load_dataset("Anthropic/hh-rlhf", split="train", streaming=True):
        chosen = ex.get("chosen", "")
        if not chosen: continue
        lines = chosen.split("\n\n")
        for j in range(len(lines) - 1):
            if not lines[j].startswith("Human: "): continue
            if not lines[j + 1].startswith("Assistant: "): continue
            u = lines[j][7:].strip()
            a = lines[j + 1][11:].strip()
            # Skip safety-heavy conversations (contain common refusal patterns deeper in)
            al = a.lower()
            if any(p in al for p in ["i cannot ", "i'm sorry", "i apologize", "it is not possible"]):
                continue
            if (u, a) in seen: continue
            seen.add((u, a))
            if add(u, a): break
    if len(pairs) >= CAP: pass
except: pass
print(f"  {len(pairs)}")

# 8. LMSYS Chat 1M (real conversations)
print("LMSYS Chat 1M...")
try:
    for ex in load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True):
        conv = ex.get("conversation", [])
        for turn in conv:
            u = turn.get("user", "").strip()
            a = turn.get("assistant", "").strip()
            if u and a and add(u, a): break
        if len(pairs) >= CAP: break
except: pass
print(f"  {len(pairs)}")

# Save
print(f"Total: {len(pairs)} pairs ({time.time()-t0:.0f}s)")
with open("/tmp/thinker_data_input.csv", "w") as f:
    csv.writer(f).writerows([[t] for t, _ in pairs])
with open("/tmp/thinker_data_output.csv", "w") as f:
    csv.writer(f).writerows([[t] for _, t in pairs])
print("Saved")

import statistics
ul = [len(tok.encode(t, add_special_tokens=True)) for t, _ in pairs]
al = [len(tok.encode(t, add_special_tokens=True)) for _, t in pairs]
print(f"User: mean={statistics.mean(ul):.0f}, max={max(ul)}")
print(f"Asst: mean={statistics.mean(al):.0f}, max={max(al)}")
