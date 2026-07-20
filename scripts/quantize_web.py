"""Precision study for the in-browser demo: fp16 and int8 variants of the
webdemo ONNX graphs, judged by full greedy-reply parity against torch.

For each variant, runs a multi-turn conversation plus the register probes
through the complete ONNX chat loop and compares replies verbatim with
ChatSession. Exact match on all probes is the bar for shipping a smaller
precision; the decoder (conditioned on lossy vectors, val_cos 0.428) is the
layer expected to break first. The lm graph (paper §6.5's matched token-LM
baseline) is judged the same way, independently, against LMChatSession.

Usage: .venv/bin/python scripts/quantize_web.py [--models webdemo/models]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402
from thoughtvec.lm import LMChatSession  # noqa: E402
from thoughtvec.tokenizer import BOS_ID, EOS_ID  # noqa: E402

TV_GRAPHS = ("encoder", "thinker", "decoder")
LM_GRAPHS = ("lm",)
GRAPHS = TV_GRAPHS + LM_GRAPHS

# One realistic conversation (each turn answered in sequence) + one-shot
# register probes with a fresh history each.
CONVERSATION = [
    "i'm feeling really overwhelmed with work lately.",
    "my boss keeps piling on deadlines.",
    "i haven't slept well all week.",
    "what should i do?",
]
ONESHOT = [
    "my dog passed away last night.",
    "i just got back from an amazing beach vacation!",
    "what do you like to do on weekends?",
    "i twisted my ankle and can't play soccer for a month.",
]


class OnnxChat:
    """The exact loop app.js will run, in python, against given sessions."""

    def __init__(self, sessions, tokenizer):
        self.s = sessions
        self.tok = tokenizer
        self.history: list[str] = []

    def reply(self, text: str) -> str:
        self.history.append(text.strip())
        turns = self.history[-6:]
        n = len(turns)
        first_role = (len(self.history) - n) % 2
        th, roles, dist = [], [], []
        for j, t in enumerate(turns):
            ids = [BOS_ID] + self.tok.encode(t, add_special=False)[:254] + [EOS_ID]
            th.append(self.s["encoder"].run(
                None, {"ids": np.array([ids], dtype=np.int64)})[0][0])
            roles.append((first_role + j) % 2)
            dist.append(min(n - j, 6))
        hyps, score = self.s["thinker"].run(None, {
            "ctx_th": np.stack(th)[None].astype(np.float32),
            "ctx_roles": np.array([roles], dtype=np.int64),
            "dist": np.array([dist], dtype=np.int64),
        })
        best = hyps[int(np.argmin(score))][None]
        out = [BOS_ID]
        for _ in range(255):
            fed = out + [0] if len(out) < 2 else out
            lg = self.s["decoder"].run(None, {
                "thoughts": best.astype(np.float32),
                "ids": np.array([fed], dtype=np.int64),
                "pos": np.array([len(out) - 1], dtype=np.int64),
            })[0][0].copy()
            if len(out) >= 3:  # no_repeat_ngram=3
                pre = tuple(out[-2:])
                for k in range(len(out) - 2):
                    if tuple(out[k:k + 2]) == pre:
                        lg[out[k + 2]] = -np.inf
            nxt = int(lg.argmax())
            out.append(nxt)
            if nxt == EOS_ID:
                break
        text_out = self.tok.decode(out)
        self.history.append(text_out)
        return text_out


class OnnxLmChat:
    """The exact loop app.js will run for the LM baseline: flat token
    history (no turn windowing), greedy, no repeat-ngram ban — mirrors
    TokenLM.generate / LMChatSession.reply exactly, warts included."""

    def __init__(self, session, tokenizer, max_len: int, max_new: int = 64):
        self.s = session
        self.tok = tokenizer
        self.max_len = max_len
        self.max_new = max_new
        self.history: list[str] = []

    def reply(self, text: str) -> str:
        self.history.append(text.strip())
        ids: list[int] = []
        for t in self.history:
            ids += [BOS_ID] + self.tok.encode(t, add_special=False) + [EOS_ID]
        room = self.max_len - self.max_new - 1
        out = ids[-room:] + [BOS_ID]
        gen: list[int] = []
        for _ in range(self.max_new):
            pos = np.array([len(out) - 1], dtype=np.int64)
            lg = self.s["lm"].run(None, {
                "ids": np.array([out], dtype=np.int64), "pos": pos})[0][0]
            nxt = int(lg.argmax())
            if nxt == EOS_ID:
                break
            gen.append(nxt)
            out.append(nxt)
        text_out = self.tok.decode(gen)
        self.history.append(text_out)
        return text_out


def torch_replies(session: ChatSession) -> list[str]:
    got = []
    session.history = []
    for t in CONVERSATION:
        got.append(session.reply(t, temperature=0.0))
    for t in ONESHOT:
        session.history = []
        got.append(session.reply(t, temperature=0.0))
    return got


def onnx_replies(sessions, tokenizer) -> list[str]:
    chat = OnnxChat(sessions, tokenizer)
    got = [chat.reply(t) for t in CONVERSATION]
    for t in ONESHOT:
        chat.history = []
        got.append(chat.reply(t))
    return got


def lm_torch_replies(session: LMChatSession) -> list[str]:
    got = []
    session.history = []
    for t in CONVERSATION:
        got.append(session.reply(t, temperature=0.0))
    for t in ONESHOT:
        session.history = []
        got.append(session.reply(t, temperature=0.0))
    return got


def lm_onnx_replies(sessions, tokenizer, max_len: int, max_new: int) -> list[str]:
    chat = OnnxLmChat(sessions, tokenizer, max_len, max_new)
    got = [chat.reply(t) for t in CONVERSATION]
    for t in ONESHOT:
        chat.history = []
        got.append(chat.reply(t))
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="webdemo/models")
    ap.add_argument("--ckpt", default="checkpoints/FINAL_12H/best.pt")
    ap.add_argument("--codec", default="checkpoints/m5_frontier/best.pt")
    ap.add_argument("--lm-ckpt", default="checkpoints/b3_lm_48m_24h/best.pt")
    args = ap.parse_args()
    mdir = Path(args.models)

    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for name in GRAPHS:
        src = mdir / f"{name}.onnx"
        m = onnx.load(str(src))
        del m.graph.value_info[:]  # dynamo leftovers break fp16/int8 tooling
        f16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(f16, str(mdir / f"{name}.fp16.onnx"),
                  save_as_external_data=False)
        # the dynamo exporter leaves value_info the quantizer's strict shape
        # inference rejects; preprocess (or strip it) first
        prep = mdir / f"{name}.prep.onnx"
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process
            quant_pre_process(str(src), str(prep), skip_symbolic_shape=False)
        except Exception:
            m2 = onnx.load(str(src))
            del m2.graph.value_info[:]
            onnx.save(m2, str(prep), save_as_external_data=False)
        quantize_dynamic(str(prep), str(mdir / f"{name}.int8.onnx"),
                         weight_type=QuantType.QInt8)
        prep.unlink(missing_ok=True)
    sizes = {}
    for tag in ("fp16", "int8"):
        sizes[tag] = sum((mdir / f"{n}.{tag}.onnx").stat().st_size
                         for n in GRAPHS) / 1e6
    fp32 = sum(f.stat().st_size for f in mdir.glob("*.onnx*")
               if ".fp16." not in f.name and ".int8." not in f.name) / 1e6
    print(f"sizes MB: fp32 {fp32:.0f}, fp16 {sizes['fp16']:.0f}, "
          f"int8 {sizes['int8']:.0f}")

    def path(n: str, tag: str) -> Path:
        return mdir / (f"{n}.onnx" if tag == "fp32" else f"{n}.{tag}.onnx")

    def size_of(chosen: dict) -> float:
        total = sum(path(g, chosen[g]).stat().st_size for g in chosen)
        total += sum((mdir / f"{g}.onnx.data").stat().st_size
                     for g in chosen if chosen[g] == "fp32"
                     and (mdir / f"{g}.onnx.data").exists())
        return total / 1e6

    def search(graphs, matches_fn, n_want: int) -> dict:
        for tag in ("fp32", "fp16", "int8"):
            n_ok = matches_fn({g: tag for g in graphs})
            print(f"uniform {tag}: {n_ok}/{n_want} replies match")
        chosen = {g: "fp32" for g in graphs}
        for g in graphs:
            for cand in ("int8", "fp16"):
                trial = dict(chosen, **{g: cand})
                if matches_fn(trial) == n_want:
                    chosen[g] = cand
                    break
        return chosen

    # ---- thought-vector pipeline ----
    session = ChatSession(args.ckpt, device="cpu", codec_ckpt=args.codec)
    want = torch_replies(session)

    def load(tag_map: dict[str, str]):
        return {n: ort.InferenceSession(str(path(n, tag_map[n])),
                                        providers=["CPUExecutionProvider"])
                for n in tag_map}

    def matches(tag_map) -> int:
        try:
            got = onnx_replies(load(tag_map), session.tokenizer)
        except Exception as e:
            print(f"   {tag_map}: FAILED — {str(e)[:100]}")
            return -1
        return sum(w == g for w, g in zip(want, got))

    print("-- thinker pipeline --")
    chosen = search(TV_GRAPHS, matches, len(want))
    print(f"chosen: {chosen} -> {size_of(chosen):.0f} MB, "
          f"{matches(chosen)}/{len(want)} exact")

    # ---- lm baseline (independent — no shared graphs with the pipeline above) ----
    lm_session = LMChatSession(args.lm_ckpt, device="cpu")
    lm_want = lm_torch_replies(lm_session)
    lm_max_len = lm_session.model.max_len
    lm_max_new = lm_session.max_new

    def lm_matches(tag_map) -> int:
        try:
            got = lm_onnx_replies(load(tag_map), lm_session.tokenizer,
                                  lm_max_len, lm_max_new)
        except Exception as e:
            print(f"   {tag_map}: FAILED — {str(e)[:100]}")
            return -1
        return sum(w == g for w, g in zip(lm_want, got))

    print("-- lm baseline --")
    lm_chosen = search(LM_GRAPHS, lm_matches, len(lm_want))
    print(f"chosen: {lm_chosen} -> {size_of(lm_chosen):.0f} MB, "
          f"{lm_matches(lm_chosen)}/{len(lm_want)} exact")


if __name__ == "__main__":
    main()
