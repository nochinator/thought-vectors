"""Quick chat smoke test: feed the same multi-turn conversation to
r4_perslot and r4_C_control, compare replies side by side."""
import sys
sys.path.insert(0, "src")


from thoughtvec.chat import ChatSession

CONVERSATIONS = [
    # Short greeting
    ["hi there, how are you?"],
    # Two turns
    ["what do you think about artificial intelligence?",
     "do you think it could ever be truly creative?"],
    # Longer context
    ["hey, do you remember that project we talked about last week?",
     "the one about the thought vectors? yeah. how's it going?",
     "we got the compression working perfectly, but the conversation part is tricky"],
    # Emotional/social
    ["i'm feeling really overwhelmed today. everything is just too much.",
     "work has been insane, and i had a fight with my friend",
     "i just don't know how to handle all of this right now"],
]

def test_model(label: str, ckpt_path: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    try:
        session = ChatSession(ckpt_path, device="cuda")
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return

    for turns in CONVERSATIONS:
        session.reset()
        print(f"\n--- turns: {len(turns)} ---")
        for i, turn in enumerate(turns):
            who = "user" if i % 2 == 0 else "bot"
            if who == "user":
                print(f"  USER: {turn}")
                reply = session.reply(turn, temperature=0.0)
                print(f"  BOT : {reply}")
            else:
                # Feed bot turn into history so context builds
                session.history.append(turn)


if __name__ == "__main__":
    base = "/home/nochi/vault/projects/ThoughtVectors/thoughtvec"
    for label, ckpt in [
        ("r4_C_control (baseline)", f"{base}/checkpoints/r4_C_control/best.pt"),
        ("r4_perslot  (winner)",     f"{base}/checkpoints/r4_perslot/best.pt"),
    ]:
        test_model(label, ckpt)
