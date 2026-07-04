"""Extract short conversation pairs from LMSYS Chatbot Arena."""
import sys
sys.path.insert(0, "thought-vectors-main")
from thought_vectors.tokenization import SPTokenizer
from datasets import load_dataset
import csv, statistics

tok = SPTokenizer(); tok.load("/tmp/sp_c4_16k.model")
ds = load_dataset("lmsys/lmsys-arena-human-preference-55k", split="train", streaming=True)
print("LMSYS Arena loaded")

refuse = ["i cannot", "i do not have", "i am not able", "i dont have",
          "i am sorry", "im sorry", "i apologize", "im not able"]

pairs = []
for example in ds:
    conv = example.get("conversation", [])
    for i in range(len(conv) - 1):
        ut = conv[i].get("content", "").strip() if conv[i].get("role") == "user" else None
        at = conv[i+1].get("content", "").strip() if conv[i+1].get("role") == "assistant" else None
        if ut and at and 3 < len(ut) < 300 and 3 < len(at) < 300:
            u_ids = tok.encode(ut, add_special_tokens=True)
            a_ids = tok.encode(at, add_special_tokens=True)
            if 2 <= len(u_ids) <= 64 and 2 <= len(a_ids) <= 64:
                skip = any(at.lower().startswith(p) for p in refuse)
                if not skip:
                    pairs.append((ut, at))
                    if len(pairs) >= 20000:
                        break
    if len(pairs) >= 20000:
        break

print(f"Extracted {len(pairs)} pairs")
u_lens = [len(tok.encode(t, add_special_tokens=True)) for t,_ in pairs]
a_lens = [len(tok.encode(t, add_special_tokens=True)) for _,t in pairs]
print(f"User: mean={statistics.mean(u_lens):.0f}, max={max(u_lens)}")
print(f"Asst: mean={statistics.mean(a_lens):.0f}, max={max(a_lens)}")

with open("/tmp/lmsys_input.csv", "w") as f:
    w = csv.writer(f)
    for t,_ in pairs: w.writerow([t])
with open("/tmp/lmsys_output.csv", "w") as f:
    w = csv.writer(f)
    for _,t in pairs: w.writerow([t])

for i in range(min(5, len(pairs))):
    print(f'  U: {pairs[i][0][:60]}')
    print(f'  A: {pairs[i][1][:60]}')
    print()
