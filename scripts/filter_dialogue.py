"""Filter conversations.jsonl for quality: multi-turn, substantive replies, no filler starts."""
import json
import sys
from pathlib import Path

SRC = "data/conversations.jsonl"
DST = "data/conversations_clean.jsonl"

FILLER_STARTS = {
    'yeah', 'yea', 'oh', 'okay', 'ok', 'yes', 'no', 'nope', 'well',
    'hmm', 'uh', 'um', 'hey', 'hi', 'hello', 'mhm', 'haha', 'lol',
    'sorry', 'thanks', 'thank', 'sure', 'right', 'wow', 'cool',
}
COLLAPSE_PHRASES = {'a good thing', 'i have done', "i've done", 'i am doing',
                    'i will do', 'i can do', 'a lot of people'}

MIN_TURNS = 4
MIN_WORDS = 8

kept = 0
total = 0
with open(SRC) as fin, open(DST, 'w') as fout:
    for line in fin:
        total += 1
        conv = json.loads(line)
        turns = conv['turns']
        if len(turns) < MIN_TURNS:
            continue
        ok = True
        for i in range(1, len(turns), 2):  # bot replies
            reply = turns[i].strip().lower()
            words = reply.split()
            if len(words) < MIN_WORDS:
                ok = False
                break
            first = words[0].rstrip(',.!?;:')
            if first in FILLER_STARTS:
                ok = False
                break
            if any(phrase in reply for phrase in COLLAPSE_PHRASES):
                ok = False
                break
        if ok:
            fout.write(line)
            kept += 1

print(f"Filtered {kept} / {total} conversations ({100*kept/max(total,1):.1f}%)")
print(f"Output: {DST}")
