#!/usr/bin/env python3
"""Build a cleaner conversation dataset — fewer canned/refusal responses."""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from thought_vectors.tokenization import SPTokenizer

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
MAX_TOK = 128  # up from 80 — matches our training cap

# ── Filters ──

# Responses that start with any of these are rejected outright
# (checked against lowercased first 80 chars of the response).
CANNED_STARTS = [
    "sure, here", "sure! here", "sure, i can", "sure! i can",
    "sure, what", "sure! what", "sure, i'll", "sure! i'll",
    "sure, i'd", "sure! i'd",
    "here's a", "here is a",
    "i am sorry", "i'm sorry", "i apologize",
    "i cannot", "i do not have", "i am not able",
    "i dont have", "i am an ai", "i'm an ai",
    "as an ai language model", "as an ai",
    "i understand your", "i appreciate your",
    "pleasure to help", "how can i assist",
    "is not possible", "it is not possible",
    "i cannot provide", "feel free to ask",
    "i am not capable", "sorry, but i",
    "i cannot provide information",
    "i do not have the capability",
    "i am a computer program",
    "as a language model",
    "i don't have feelings",
    "i don't have the ability",
    "i do not have access",
    "i don't have access",
    "i am not able to provide",
    "i'm just an ai",
    "i'm here to help",
    "that's a great question",
    "that is a great question",
    "great question",
    "thank you for your question",
    "thanks for your question",
    "thank you for reaching out",
    "i'd be happy to",
    "i would be happy to",
    "of course, i",
    "of course! i",
    "certainly, i",
    "certainly! i",
    "absolutely, i",
    "absolutely! i",
    "yes, i can",
    "yes, i'd",
    "yes, i will",
]

# Responses containing any of these patterns anywhere are rejected.
# These catch hedging and canned structures deeper in the text.
CANNED_ANYWHERE = [
    "i cannot ", "i can't ",
    "i'm sorry, but i cannot",
    "i apologize, but i cannot",
    "it is not possible for me",
    "i'm not able to",
    "i am not able to",
    "i do not have the capability",
    "as an ai language model",
    "as an ai, i",
]

# Heavily templated response prefixes that indicate classification
# or instruction-following data rather than natural conversation.
TEMPLATED_LABELS = [
    "positive", "negative", "neutral",
    "yes.", "no.", "maybe.", "false", "true.",
    "the review is", "the sentiment",
    "the option in line with common sense",
    "the answer is",
    "the correct option",
    "the sentence is",
    "from this sentence",
    "this sentence is",
    "the emotion",
    "context: imagine",
    "context: a company",
    "voici ", "je ne parle", "je suis",
    "¡hola", "lo siento",
]

# Common instruction templates on the input side that produce boilerplate outputs.
INSTRUCTION_TEMPLATES = [
    "write a sentence", "generate a sentence", "produce a sentence",
    "generate a short summary", "write a brief sentence",
    "pick the option", "choose the option",
    "add punctuation", "this text is missing",
    "generate a context", "generate a hypothesis",
    "write a sentence not in english",
    "write a sentence in spanish",
    "write a sentence in french",
    "write a sentence in german",
    "formulate an answer",
    "what would be the ★-rating",
    "generate a descriptive sentence",
    "produce a detailed sentence",
    "produce a long descriptive sentence",
    "here are some keywords",
    "here is some data",
    "here's a complex question",
    "please answer the following question",
    "given the task definition",
    "given those answer options",
    "answer the question",
    "generate a question",
    "create a question",
]


def ok_response(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 4 or len(t) > 500:
        return False
    # Check start of response for canned patterns
    start = t[:80]
    for p in CANNED_STARTS:
        if t.startswith(p) or start.startswith(p):
            return False
    # Check anywhere for deeper canned patterns
    for p in CANNED_ANYWHERE:
        if p in t:
            return False
    # Check for templated labels
    for p in TEMPLATED_LABELS:
        if t.startswith(p):
            return False
    return True


def ok_input(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 4 or len(t) > 500:
        return False
    # Skip instruction-templated inputs
    for p in INSTRUCTION_TEMPLATES:
        if t.startswith(p):
            return False
    return True


def fits(text: str) -> bool:
    ids = tok.encode(text.strip(), add_special_tokens=True)
    return 2 <= len(ids) <= MAX_TOK


CAP = 500000
pairs: list[tuple[str, str]] = []
t0 = time.time()


def add(u: str, a: str) -> bool:
    if len(pairs) >= CAP:
        return True
    u, a = u.strip(), a.strip()
    if ok_input(u) and ok_response(a) and fits(u) and fits(a):
        pairs.append((u, a))
    return False


# ── Sources (Anthropic HH removed — too safety-heavy) ──

# 1. Dolly (human-written, clean)
print("Dolly...")
for ex in load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True):
    if add(ex.get("instruction", ""), ex.get("response", "")):
        break
print(f"  {len(pairs)}")

# 2. UltraChat (synthetic instruct, conservative filter)
print("UltraChat...")
for ex in load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True):
    msgs = ex["messages"]
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            if add(msgs[i]["content"], msgs[i + 1]["content"]):
                break
    if len(pairs) >= CAP:
        break
print(f"  {len(pairs)}")

# 3. LMSYS Arena (real conversations — best source)
print("LMSYS Arena...")
for ex in load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True):
    try:
        pl = json.loads(ex["prompt"]) if isinstance(ex["prompt"], str) and ex["prompt"].startswith("[") else [ex["prompt"]]
        ral = json.loads(ex["response_a"]) if isinstance(ex["response_a"], str) and ex["response_a"].startswith("[") else [ex["response_a"]]
        rbl = json.loads(ex["response_b"]) if isinstance(ex["response_b"], str) and ex["response_b"].startswith("[") else [ex["response_b"]]
    except Exception:
        continue
    prompt = (pl[0] if isinstance(pl, list) and pl else "").strip()
    for r in [ral[0] if isinstance(ral, list) and ral else "", rbl[0] if isinstance(rbl, list) and rbl else ""]:
        r = r.strip() if r else ""
        if add(prompt, r):
            break
    if len(pairs) >= CAP:
        break
print(f"  {len(pairs)}")

# 4. No Robots (curated, clean conversations)
print("No Robots...")
try:
    for ex in load_dataset("HuggingFaceH4/no_robots", split="train_sft", streaming=True):
        msgs = ex.get("messages", [])
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                if add(msgs[i].get("content", ""), msgs[i + 1].get("content", "")):
                    break
        if len(pairs) >= CAP:
            break
except Exception:
    pass
print(f"  {len(pairs)}")

# 5. LMSYS Chat 1M (real conversations)
print("LMSYS Chat 1M...")
try:
    for ex in load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True):
        conv = ex.get("conversation", [])
        for turn in conv:
            u = turn.get("user", "").strip()
            a = turn.get("assistant", "").strip()
            if u and a and add(u, a):
                break
        if len(pairs) >= CAP:
            break
except Exception:
    pass
print(f"  {len(pairs)}")

# 6. OpenOrca (only if we haven't hit cap — heavily filtered)
if len(pairs) < CAP:
    print("OpenOrca...")
    for i, ex in enumerate(load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)):
        if i > 0 and i % 500000 == 0:
            print(f"    scanned {i}, {len(pairs)} pairs")
        q = ex.get("question", "")
        a = ex.get("response", "")
        if not q or not a:
            continue
        a_clean = a.replace("<|im_start|>assistant\n", "").replace("<|im_end|>", "").strip()
        if add(q, a_clean):
            break
        if len(pairs) >= CAP:
            break
    print(f"  {len(pairs)}")

# ── Save ──
print(f"\nTotal: {len(pairs)} pairs ({time.time() - t0:.0f}s)")
with open("/tmp/thinker_data_cleaner_input.csv", "w") as f:
    csv.writer(f).writerows([[t] for t, _ in pairs])
with open("/tmp/thinker_data_cleaner_output.csv", "w") as f:
    csv.writer(f).writerows([[t] for _, t in pairs])
print("Saved to /tmp/thinker_data_cleaner_input.csv / output.csv")

# Stats
ul = [len(tok.encode(t, add_special_tokens=True)) for t, _ in pairs]
al = [len(tok.encode(t, add_special_tokens=True)) for _, t in pairs]
print(f"User: mean={statistics.mean(ul):.0f}, max={max(ul)}")
print(f"Asst: mean={statistics.mean(al):.0f}, max={max(al)}")

# Show a sample
print("\nSample pairs (first 5):")
for i in range(min(5, len(pairs))):
    print(f"  U: {pairs[i][0][:80]}")
    print(f"  A: {pairs[i][1][:80]}")
    print()
