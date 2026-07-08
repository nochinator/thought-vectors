#!/usr/bin/env python
"""Novel-reversal register battery: 10 good-news->bad-news conversations plus
4 sustained-mood controls, all novel phrasing, pivot at turns 2-4, pivot cue
varied ("but" / "unfortunately" / none). Greedy decode (temp 0); the pivot-turn
reply is what gets hand-scored (commiserate / cheerful / neutral).

Built 2026-07-08 to audit the FINAL2_12H "template, not skill" verdict, which
rested on a 4-conversation chat probe (RESEARCH_LOG 2026-07-04). Results:
FINAL_12H 0/10 commiserations, FINAL2_12H 3/10 (4-5/10 with --hyp-select
affinity) -- partial, topic-gated routing (RESEARCH_LOG 2026-07-08).

Usage:
    scripts/register_battery.py --ckpt checkpoints/FINAL2_12H/best.pt \
        [--hyp-select decodable|affinity] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402

REVERSALS = {
    "R1 lottery->flood (but, t2)": [
        "we won a little money in the lottery last week!",
        "but our basement flooded and ruined everything we stored there."],
    "R2 baby->layoff (no cue, t2)": [
        "my wife and i are having a baby in the fall!",
        "i lost my job this morning."],
    "R3 marathon->diagnosis (but, t3)": [
        "i finally ran my first marathon on saturday!",
        "months of training really paid off.",
        "but my knee scans came back and i need surgery."],
    "R4 garden->hail (no cue, t3)": [
        "my vegetable garden is doing amazing this year!",
        "the tomatoes are the best i've ever grown.",
        "a hailstorm destroyed all of it last night."],
    "R5 concert->pickpocket (unfortunately, t2)": [
        "we got front row seats to see my favorite band!",
        "unfortunately someone stole my wallet at the show."],
    "R6 puppy->escape (no cue, t2)": [
        "we adopted the sweetest little puppy yesterday!",
        "he slipped out the gate this morning and we can't find him."],
    "R7 award->fire (but, t4)": [
        "i won an award at school for my science project!",
        "my parents were so proud of me.",
        "we went out for dinner to celebrate.",
        "but while we were out a fire started in our kitchen."],
    "R8 vacation->burglary (verbatim 2026-07-04 chat probe)": [
        "i just got back from an amazing beach vacation!",
        "the water was perfect and i learned to surf.",
        "but when i got home i found out my apartment was broken into."],
    "R9 date->crash (no cue, t3)": [
        "i had the best first date of my life on friday!",
        "we talked for hours and really hit it off.",
        "then on my drive home a truck rear-ended me."],
    "R10 house->mold (but, t2)": [
        "we finally closed on our first house!",
        "but the inspection missed black mold in the walls."],
}

CONTROLS = {
    "C1 sustained bad (bereavement)": [
        "my grandfather passed away over the weekend.",
        "we were really close, i visited him every sunday."],
    "C2 sustained bad (failure)": [
        "i failed my driving test again.",
        "that's the third time now, i feel hopeless."],
    "C3 sustained good (college)": [
        "i got accepted into my dream college!",
        "they even gave me a scholarship."],
    "C4 sustained good (sports)": [
        "our team won the championship last night!",
        "i scored the winning goal."],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hyp-select", default="decodable",
                    choices=["decodable", "affinity"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    session = ChatSession(args.ckpt, device=args.device,
                          hyp_select=args.hyp_select)
    print(f"# register battery — ckpt {args.ckpt} | hyp_select "
          f"{args.hyp_select} | temp 0")
    for name, turns in {**REVERSALS, **CONTROLS}.items():
        session.history = []
        print(f"\n--- {name} ---")
        for t in turns:
            reply = session.reply(t, temperature=0.0)
            print(f"user > {t}\nbot  > {reply}")


if __name__ == "__main__":
    main()
