#!/usr/bin/env python
"""Multi-turn coherence eval for thinker checkpoints.

Motivation (RESEARCH_LOG 2026-06-25): single-turn distinct1 missed the
multi-turn hard-collapse where R4_UNFREEZE returned a near-identical reply on
every turn of a 4-turn chat.  This script drives scripted multi-turn
conversations through ChatSession and reports:

  self_rep   mean pairwise unigram-F1 between the model's OWN replies within a
             conversation (1.0 = identical reply every turn = hard collapse;
             lower is better)
  ctx_sens   unigram-F1 between the reply to the FINAL user turn with full
             history vs with history cleared (1.0 = context is ignored;
             lower = the history actually changes the reply)

Runs on CPU by default (GPU chat inference hit `HIP error: invalid device
function` on gfx1031 — see 2026-06-25 infra flag).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402

# Fixed scripts: smalltalk, emotional register, topic tracking, factual-ish,
# planning, self-reference.  4 user turns each.
SCRIPTS: list[list[str]] = [
    ["hey, how's it going?",
     "not bad. i just got a new puppy!",
     "she's a golden retriever named daisy.",
     "do you like dogs?"],
    ["i'm feeling really overwhelmed with work lately.",
     "my boss keeps piling on deadlines.",
     "i haven't slept well all week.",
     "what should i do?"],
    ["do you like cooking?",
     "i made pasta last night, it came out great.",
     "the secret was fresh basil from my garden.",
     "what's your favorite food?"],
    ["i'm planning a trip to the mountains next month.",
     "i want to go hiking and camping.",
     "i've never set up a tent before though.",
     "any advice for a first-time camper?"],
    ["my sister is getting married this summer!",
     "i'm the maid of honor so i have to give a speech.",
     "i'm pretty nervous about public speaking.",
     "how do i calm my nerves?"],
    ["what do you do for fun?",
     "i mostly play video games and read.",
     "right now i'm reading a mystery novel.",
     "can you recommend a book?"],
]


def unigrams(text: str) -> Counter:
    return Counter(text.lower().split())


def unigram_f1(a: str, b: str) -> float:
    ca, cb = unigrams(a), unigrams(b)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / max(sum(ca.values()), 1)
    r = overlap / max(sum(cb.values()), 1)
    return 2 * p * r / (p + r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dump", default=None, help="write transcripts here")
    ap.add_argument("--hyp-select", default="decodable",
                    help="WTA winner rule: decodable (default) or affinity")
    args = ap.parse_args()

    session = ChatSession(args.ckpt, device=args.device, hyp_select=args.hyp_select)
    lines: list[str] = []
    rep_scores: list[float] = []
    sens_scores: list[float] = []

    for si, script in enumerate(SCRIPTS):
        session.reset()
        replies: list[str] = []
        lines.append(f"=== conversation {si} ===")
        for user in script:
            bot = session.reply(user, temperature=args.temperature)
            replies.append(bot)
            lines.append(f"user > {user}")
            lines.append(f"bot  > {bot}")
        pair_f1 = [unigram_f1(a, b) for a, b in itertools.combinations(replies, 2)]
        rep = sum(pair_f1) / len(pair_f1)
        rep_scores.append(rep)

        # context sensitivity: same final user turn, no history
        session.reset()
        bare = session.reply(script[-1], temperature=args.temperature)
        sens = unigram_f1(replies[-1], bare)
        sens_scores.append(sens)
        lines.append(f"(no-context reply to final turn: {bare})")
        lines.append(f"self_rep={rep:.3f} ctx_sens={sens:.3f}")
        lines.append("")

    n = len(SCRIPTS)
    summary = (f"multiturn: self_rep {sum(rep_scores)/n:.4f} "
               f"ctx_sens {sum(sens_scores)/n:.4f} "
               f"(n={n} convs, temp={args.temperature})")
    lines.append(summary)
    print(summary)
    if args.dump:
        Path(args.dump).write_text("\n".join(lines) + "\n")
        print(f"transcripts -> {args.dump}")


if __name__ == "__main__":
    main()
