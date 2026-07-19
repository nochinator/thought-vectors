"""B5 — efficiency measurements (docs/BASELINE_ABLATIONS.md).

Per-reply wall time and measured FLOPs at history lengths {1,5,10,20} turns
(~40 tok/turn), temperature 0, CPU, for the codec+thinker system (phases
broken out: codec encode / thinker forward / codec decode) and the token-LM
baseline. Also reports the context representation each system carries per
turn (thought slots vs token KV).

Both sides are timed on their reference implementations, which share the
same naivety (no KV cache in TokenLM.generate, none in sample_decode); the
FLOP numbers are measured with torch's FlopCounterMode, not estimated.

Usage:
  .venv/bin/python scripts/bench_reply.py \
      --thinker checkpoints/FINAL_12H/best.pt \
      --lm checkpoints/b2_lm_B_640x8_lr6/best.pt \
      --out logs/bench_reply.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from thoughtvec.chat import ChatSession
from thoughtvec.config import from_dict
from thoughtvec.generate import sample_decode
from thoughtvec.lm import TokenLM
from thoughtvec.thinker_train import encode_turns
from thoughtvec.tokenizer import BOS_ID, EOS_ID, Tokenizer

TURNS = (1, 5, 10, 20)
REPEATS = 5
TURN_TOKENS = 40

BASE_TEXT = (
    "This morning I walked along the river before work and watched the fog "
    "lift off the water while two herons argued over the same fishing spot, "
    "which somehow made me feel better about my own week."
)


def make_history(tokenizer: Tokenizer, n_turns: int) -> list[str]:
    """n_turns alternating turns, each trimmed to ~TURN_TOKENS tokens."""
    ids = tokenizer.encode(BASE_TEXT, add_special=False)[: TURN_TOKENS - 3]
    base = tokenizer.decode(ids)
    return [f"{base} (turn {i})" for i in range(n_turns)]


# ── thinker side: mirrors ChatSession.reply with phase instrumentation ──

def thinker_phases(session: ChatSession, history: list[str]):
    tk = session.cfg.thinker
    turns = history[-tk.max_turns:]
    first_role = (len(history) - len(turns)) % 2
    seq_max = session.codec_cfg.model.max_seq_len
    tokenizer = session.tokenizer
    rows = [
        [BOS_ID] + tokenizer.encode(t, add_special=False)[: seq_max - 2] + [EOS_ID]
        for t in turns
    ]
    if tk.flat_context:
        flat: list[int] = []
        for r in rows:
            flat += r
        rows = [flat[-seq_max:]]
        first_role = 0
    max_t = max(len(r) for r in rows)
    ctx_ids = torch.zeros(1, len(rows), max_t, dtype=torch.long, device=session.device)
    for j, r in enumerate(rows):
        ctx_ids[0, j, : len(r)] = torch.tensor(r, dtype=torch.long)
    ctx_roles = torch.tensor(
        [[(first_role + j) % 2 for j in range(len(rows))]], device=session.device
    )
    ctx_turns = torch.tensor([len(rows)], device=session.device)
    resp_roles = torch.tensor([1], device=session.device)

    def enc():
        return encode_turns(session.codec, ctx_ids, tk.k_ctx, tau=tk.ctx_tau)

    def think(ctx_th, budgets):
        pred = session.thinker(ctx_th, ctx_roles, ctx_turns, resp_roles,
                               slot_budgets=budgets)
        if pred.dim() == 4:
            score = session.codec.predictor(pred[0])[:, tk.k_out - 1]
            pred = pred[:, int(score.argmin())]
        return pred

    def dec(pred):
        return sample_decode(session.codec, pred, seq_max,
                             temperature=0.0, no_repeat_ngram=3)

    return enc, think, dec, len(turns)


@torch.no_grad()
def bench_thinker(session: ChatSession, history: list[str]) -> dict:
    enc, think, dec, eff_turns = thinker_phases(session, history)
    times = {"encode": [], "thinker": [], "decode": []}
    n_out = 0
    for _ in range(REPEATS + 1):  # first iteration is warmup
        t0 = time.perf_counter()
        ctx_th, budgets = enc()
        t1 = time.perf_counter()
        pred = think(ctx_th, budgets)
        t2 = time.perf_counter()
        out = dec(pred)
        t3 = time.perf_counter()
        times["encode"].append(t1 - t0)
        times["thinker"].append(t2 - t1)
        times["decode"].append(t3 - t2)
        n_out = out.shape[-1]
    med = {k: statistics.median(v[1:]) for k, v in times.items()}
    flops = {}
    with FlopCounterMode(display=False) as fcm:
        ctx_th, budgets = enc()
    flops["encode"] = fcm.get_total_flops()
    with FlopCounterMode(display=False) as fcm:
        pred = think(ctx_th, budgets)
    flops["thinker"] = fcm.get_total_flops()
    with FlopCounterMode(display=False) as fcm:
        out = dec(pred)
    flops["decode"] = fcm.get_total_flops()
    # KV-equiv (same convention as the LM side): encode and thinker forward
    # are single-pass already; the token decoder's cached cost is one
    # teacher-forced pass over the reply it produced.
    with FlopCounterMode(display=False) as fcm:
        session.codec.decode(pred, out[:, :-1] if out.size(1) > 1 else out)
    kv_equiv = flops["encode"] + flops["thinker"] + fcm.get_total_flops()
    tk = session.cfg.thinker
    d = session.codec_cfg.model.d_model
    return {
        "effective_turns": eff_turns,
        "reply_tokens": int(n_out),
        "time_s": {**med, "total": sum(med.values())},
        "flops": {**flops, "total": sum(flops.values()), "kv_equiv": kv_equiv},
        "ctx_repr_floats": eff_turns * tk.k_ctx * d,
    }


# ── LM side ──

@torch.no_grad()
def bench_lm(model: TokenLM, tokenizer: Tokenizer, history: list[str],
             max_new: int = 48) -> dict:
    ids: list[int] = []
    for t in history:
        ids += [BOS_ID] + tokenizer.encode(t, add_special=False) + [EOS_ID]
    room = model.max_len - max_new - 1
    ids = ids[-room:]
    ctx = torch.tensor(ids, dtype=torch.long)
    times = []
    out: list[int] = []
    for _ in range(REPEATS + 1):
        t0 = time.perf_counter()
        out = model.generate(ctx, max_new=max_new, temperature=0.0)
        times.append(time.perf_counter() - t0)
    with FlopCounterMode(display=False) as fcm:
        model.generate(ctx, max_new=max_new, temperature=0.0)
    ref_flops = fcm.get_total_flops()
    # KV-cache-equivalent cost: one teacher-forced pass over ctx + reply is
    # what an optimized LM implementation pays in matmul FLOPs. Reported so
    # the naive reference decoder does not inflate the LM's numbers.
    full = torch.cat([ctx, torch.tensor([BOS_ID] + out, dtype=torch.long)])[None]
    with FlopCounterMode(display=False) as fcm:
        model(full)
    d_model = model.tok.embedding_dim
    n_layers = len(model.trunk.layers)
    kv_floats = (len(ids) + len(out)) * d_model * 2 * n_layers
    return {
        "ctx_tokens": len(ids),
        "reply_tokens": len(out),
        "time_s": {"total": statistics.median(times[1:])},
        "flops": {"total": ref_flops, "kv_equiv": fcm.get_total_flops()},
        "ctx_repr_floats": kv_floats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thinker", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--codec", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="logs/bench_reply.json")
    args = ap.parse_args()

    torch.set_num_threads(torch.get_num_threads())  # explicit default
    session = ChatSession(args.thinker, device=args.device, codec_ckpt=args.codec)
    tokenizer = session.tokenizer

    lm_ckpt = torch.load(args.lm, map_location="cpu", weights_only=False)
    lm_cfg = from_dict(lm_ckpt["cfg"])
    lm = TokenLM(lm_cfg)
    lm.load_state_dict(lm_ckpt["model"])
    lm.to(args.device).eval()

    results = {"turns": {}, "meta": {
        "thinker_ckpt": args.thinker, "lm_ckpt": args.lm,
        "repeats": REPEATS, "turn_tokens": TURN_TOKENS,
        "device": args.device,
        "note": "reference implementations on both sides (no KV cache either "
                "side); FLOPs measured with FlopCounterMode",
    }}
    for n in TURNS:
        history = make_history(tokenizer, n)
        th = bench_thinker(session, history)
        lm_r = bench_lm(lm, tokenizer, history)
        results["turns"][n] = {"thinker": th, "lm": lm_r}
        print(f"\n== {n} turns ==")
        print(f"  thinker: {th['time_s']['total']*1e3:8.1f} ms  "
              f"(enc {th['time_s']['encode']*1e3:.1f} / think "
              f"{th['time_s']['thinker']*1e3:.1f} / dec {th['time_s']['decode']*1e3:.1f})"
              f"  {th['flops']['total']/1e9:7.2f} GFLOP  "
              f"(kv-equiv {th['flops']['kv_equiv']/1e9:.2f})  "
              f"ctx {th['ctx_repr_floats']/1e3:.1f}k floats  "
              f"({th['effective_turns']} turns kept, {th['reply_tokens']} tok out)")
        print(f"  lm     : {lm_r['time_s']['total']*1e3:8.1f} ms"
              f"{'':>34}  {lm_r['flops']['total']/1e9:7.2f} GFLOP  "
              f"(kv-equiv {lm_r['flops']['kv_equiv']/1e9:.2f})  "
              f"ctx {lm_r['ctx_repr_floats']/1e3:.1f}k floats  "
              f"({lm_r['ctx_tokens']} ctx tok, {lm_r['reply_tokens']} tok out)")
        to, lo = th["reply_tokens"], lm_r["reply_tokens"]
        print(f"  per out tok: thinker {th['time_s']['total']/max(to,1)*1e3:.1f} ms"
              f" | lm {lm_r['time_s']['total']/max(lo,1)*1e3:.1f} ms"
              "   (reply lengths differ; totals are what a user experiences)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
