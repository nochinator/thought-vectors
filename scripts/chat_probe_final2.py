"""Side-by-side chat probe: FINAL_12H vs FINAL2_12H, same scripted user turns."""
import sys
sys.path.insert(0, "src")
from thoughtvec.chat import ChatSession

CONVS = {
    "mood reversal (the target behavior)": [
        "i just got back from an amazing beach vacation!",
        "the water was perfect and i learned to surf.",
        "but when i got home i found out my apartment was broken into.",
        "they took my laptop and my grandmother's necklace.",
    ],
    "sustained bad news": [
        "i'm feeling really overwhelmed with work lately.",
        "my boss keeps piling on deadlines.",
        "i haven't slept well all week.",
        "what should i do?",
    ],
    "plain small talk": [
        "hey! what do you like to do on weekends?",
        "i've been getting into baking bread.",
        "my first sourdough came out pretty dense though.",
        "any tips?",
    ],
    "good news stays good": [
        "i just found out i got into my dream school!",
        "i'll be studying marine biology.",
        "i start in the fall and i can't wait.",
    ],
}

for name, ckpt in [("FINAL_12H", "checkpoints/FINAL_12H/best.pt"),
                   ("FINAL2_12H", "checkpoints/FINAL2_12H/best.pt")]:
    print(f"\n{'='*70}\n### {name}\n{'='*70}")
    for title, turns in CONVS.items():
        print(f"\n--- {title} ---")
        sess = ChatSession(ckpt, device="cpu")
        for t in turns:
            r = sess.reply(t)
            print(f"user > {t}")
            print(f"bot  > {r}")
