"""Chat probe for Round B LM checkpoints: fresh contexts, temp 0, CPU.

Served as the 2h hard-gate probe on B3 (logs/b3_lm_48m_24h/chatprobe_2h.txt).
Usage: .venv/bin/python scripts/probe_lm_chat.py [ckpt_path]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from thoughtvec.config import from_dict  # noqa: E402
from thoughtvec.lm import TokenLM  # noqa: E402
from thoughtvec.tokenizer import BOS_ID, EOS_ID, Tokenizer  # noqa: E402

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/b3_lm_48m_24h/last.pt"
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = from_dict(ckpt["cfg"])
model = TokenLM(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer(cfg.run.tokenizer_path)

PROBES = {
    "angry": [
        "You borrowed my car and returned it with an empty tank and a scratch on the door.",
        "I know, I know. I was going to tell you about it.",
        "Going to tell me? You should have told me the moment it happened. I trusted you with it.",
    ],
    "sad pivot": [
        "The garden has been wonderful this year, the tomatoes are finally coming in.",
        "That's great! You'll have more than you can eat by August.",
        "Actually I just got a call from the doctor. The test results came back and it isn't good news.",
    ],
    "happy": [
        "Guess what? I got the job! They called this morning and the offer is even better than I hoped.",
    ],
    "neutral": [
        "What time does the market open on Saturdays?",
    ],
}

with torch.no_grad():
    for name, turns in PROBES.items():
        ids = []
        for t in turns:
            ids += [BOS_ID] + tok.encode(t, add_special=False) + [EOS_ID]
        ids = ids[-(model.max_len - 49):]
        out = model.generate(torch.tensor(ids, dtype=torch.long), max_new=48, temperature=0.0)
        print(f"[{name}]\n  last> {turns[-1]}\n  pred> {tok.decode(out)}\n")
