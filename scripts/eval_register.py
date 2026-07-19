#!/usr/bin/env python
"""Sentiment-register eval: does the model reply cheerfully to bad news?

Motivation (RESEARCH_LOG 2026-07-03): FINAL_12H broke content collapse but the
mean-collapse survives in *sentiment* — bad news often draws "That's great!".
This measures it directly:

  reg_err  fraction of bad-news probes whose reply contains a
           positive-affect marker (lower is better)
  pos_ok   fraction of good-news probes whose reply contains one
           (sanity check — should stay HIGH; a model that never says
           "great" anywhere has just been lobotomized, not fixed)

Single-turn, fresh session per probe, temp 0, CPU.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402

BAD_NEWS = [
    "i twisted my ankle and can't play soccer for a month.",
    "my dog passed away last night.",
    "i just got laid off from my job.",
    "i failed my driving test again.",
    "my flight got cancelled and i'm stuck at the airport.",
    "i haven't slept well all week and i'm exhausted.",
    "my car broke down on the highway this morning.",
    "i had a big fight with my best friend.",
    "my laptop died and i lost all my files.",
    "i've been feeling really lonely since i moved.",
    "the doctor says i need surgery on my knee.",
    "i burned the dinner i spent all day cooking.",
]

# The register disease is context-conditional (RESEARCH_LOG 2026-07-03: single-turn
# bad-news replies are often apt; the "That's great!" errors appear after a positive
# earlier turn).  Each pair: upbeat turn 1, bad-news turn 2 — score turn-2 reply.
CTX_BAD = [
    ("we won our soccer match yesterday, i scored twice!",
     "but then i twisted my ankle and can't play for a month."),
    ("i've been loving my new job so far!",
     "but today my whole team got laid off."),
    ("my garden has been doing amazing this spring!",
     "unfortunately a storm destroyed most of it last night."),
    ("i adopted the sweetest puppy last month!",
     "she got really sick and we're at the vet now."),
    ("i finally saved up enough for my dream vacation!",
     "then my car broke down and the repairs cost all of it."),
    ("my sister's wedding planning has been so exciting!",
     "but our grandmother is too ill to attend now."),
]

GOOD_NEWS = [
    "i just got promoted at work!",
    "we won our soccer match yesterday!",
    "i passed my final exams with top marks!",
    "i just got engaged!",
    "my painting won first prize at the fair!",
    "i finally finished writing my novel!",
]

# Widened 2026-07-03 after R6_CYCLE50 audit: the model phrased cheer in words the
# first lexicon missed ("terrific", "beautiful", "nice day", bare "fun") and
# scored a fake reg_err_ctx win.  Widened again same day after R8_REV40 audit
# ("That is so sweet!" to a sick puppy — 4th miss).  Scores are NOT comparable
# across lexicon versions — rescore all checkpoints when this changes.
POSITIVE = re.compile(
    r"\b(great|glad|happy|amazing|awesome|wonderful|congrat\w*|proud|"
    r"exciting|excited|fantastic|love it|that's good|good to hear|"
    r"fun|terrific|beautiful|lovely|brilliant|perfect|delight\w*|enjoy\w*|"
    r"nice day|so nice|so sweet|cool|yay|well done)\b",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--hyp-select", default="decodable",
                    help="WTA winner rule: decodable (default) or affinity")
    ap.add_argument("--lm", action="store_true",
                    help="ckpt is a Round B token-LM baseline, not a thinker")
    args = ap.parse_args()

    if args.lm:
        from thoughtvec.lm import LMChatSession
        session = LMChatSession(args.ckpt, device=args.device)
    else:
        session = ChatSession(args.ckpt, device=args.device,
                              hyp_select=args.hyp_select)
    lines: list[str] = []

    def run(probes: list[str], label: str) -> float:
        hits = 0
        lines.append(f"=== {label} ===")
        for p in probes:
            session.reset()
            r = session.reply(p, temperature=args.temperature)
            hit = bool(POSITIVE.search(r))
            hits += hit
            lines.append(f"[{'POS' if hit else '   '}] user > {p}")
            lines.append(f"      bot  > {r}")
        lines.append("")
        return hits / len(probes)

    reg_err = run(BAD_NEWS, "bad news (positive marker = register error)")
    pos_ok = run(GOOD_NEWS, "good news (positive marker = correct)")

    # contextual bad news: upbeat turn 1, bad turn 2, score turn-2 reply
    hits = 0
    lines.append("=== contextual bad news (positive marker = register error) ===")
    for setup, bad in CTX_BAD:
        session.reset()
        session.reply(setup, temperature=args.temperature)
        r = session.reply(bad, temperature=args.temperature)
        hit = bool(POSITIVE.search(r))
        hits += hit
        lines.append(f"[{'POS' if hit else '   '}] user > {setup}  /  {bad}")
        lines.append(f"      bot  > {r}")
    lines.append("")
    reg_err_ctx = hits / len(CTX_BAD)

    summary = (f"register: reg_err {reg_err:.4f} reg_err_ctx {reg_err_ctx:.4f} "
               f"pos_ok {pos_ok:.4f} "
               f"(bad={len(BAD_NEWS)} ctx={len(CTX_BAD)} good={len(GOOD_NEWS)}, "
               f"temp={args.temperature})")
    lines.append(summary)
    print(summary)
    if args.dump:
        Path(args.dump).write_text("\n".join(lines) + "\n")
        print(f"transcripts -> {args.dump}")


if __name__ == "__main__":
    main()
