"""Extract OpenAssistant oasst1 into conversations.jsonl format.
Each conversation = linear path from a root message through its replies.
Filter: only assistant replies, >= 4 turns, substantive replies.
"""
import json
from collections import defaultdict
from datasets import load_dataset

print("Loading oasst1...")
ds = load_dataset("OpenAssistant/oasst1", split="train")
print(f"Total messages: {len(ds)}")

# Build message lookup and parent→children map
msgs = {}
children = defaultdict(list)
roots = []
for row in ds:
    mid = row["message_id"]
    parent = row["parent_id"]
    role = row["role"]
    text = row["text"]
    lang = row.get("lang", "en")
    if lang != "en":
        continue
    msgs[mid] = {"role": role, "text": text, "parent": parent}
    if parent is None:
        roots.append(mid)
    else:
        children[parent].append(mid)

# Extract linear conversation paths
def walk(msg_id, path):
    msg = msgs[msg_id]
    path.append(msg)
    kids = children[msg_id]
    if not kids:
        return [path]
    paths = []
    for kid in kids:
        paths.extend(walk(kid, list(path)))
    return paths

FILLER_STARTS = {
    'yeah', 'yea', 'oh', 'okay', 'ok', 'yes', 'no', 'nope', 'well',
    'hmm', 'uh', 'um', 'hey', 'hi', 'hello', 'mhm', 'haha', 'lol',
    'sorry', 'thanks', 'thank', 'sure', 'right', 'wow', 'cool',
}
MIN_TURNS = 4
MIN_WORDS = 8

all_convs = []
for root in roots:
    for path in walk(root, []):
        # Keep only assistant + prompter (user) messages
        roles = [m["role"] for m in path]
        if "assistant" not in roles:
            continue
        # Convert to user/bot alternating format
        turns = []
        for m in path:
            if m["role"] == "prompter":
                turns.append(m["text"])
            elif m["role"] == "assistant":
                turns.append(m["text"])
        if len(turns) < MIN_TURNS:
            continue
        # Filter: assistant replies (odd indices) must be substantive
        ok = True
        for i in range(1, len(turns), 2):
            reply = turns[i].strip().lower()
            words = reply.split()
            if len(words) < MIN_WORDS:
                ok = False
                break
            first = words[0].rstrip(',.!?;:')
            if first in FILLER_STARTS:
                ok = False
                break
        if ok:
            all_convs.append({"source": "oasst1", "turns": turns})

print(f"Extracted {len(all_convs)} conversations")

with open("data/oasst1_conversations.jsonl", "w") as f:
    for conv in all_convs:
        f.write(json.dumps(conv) + "\n")

print("Written to data/oasst1_conversations.jsonl")

# Print stats
total_turns = sum(len(c["turns"]) for c in all_convs)
reply_lens = [len(t.split()) for c in all_convs for i, t in enumerate(c["turns"]) if i % 2 == 1]
print(f"Total turns: {total_turns}")
print(f"Bot replies: {len(reply_lens)}")
print(f"Median reply words: {sorted(reply_lens)[len(reply_lens)//2]}")
print(f"Mean reply words: {sum(reply_lens)/len(reply_lens):.1f}")
