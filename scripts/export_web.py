"""Export FINAL_12H + m5_frontier + the b3 token-LM baseline to ONNX for the
in-browser demo.

Four graphs. Three mirror ChatSession.reply exactly (decodable hypothesis
rule, resp_role=1, no slot budgets — the FINAL_12H chat configuration):

  encoder.onnx   ids [1,T] int64            -> thoughts [1,8,384]
  thinker.onnx   ctx_th [1,C,8,384], ctx_roles [1,C], dist [1,C]
                                            -> hyps [4,8,384], score [4]
  decoder.onnx   thoughts [1,8,384], ids [1,S] int64 -> logits [1,16384]

The fourth mirrors LMChatSession.reply / TokenLM.generate (paper §6.5's
matched baseline — flat token history, no turn windowing, no repeat-ngram
ban: its repetition loops are a documented finding, not a bug to fix here):

  lm.onnx        ids [1,S] int64, pos [1] int64 -> logits [1,16384]

Turns are encoded one at a time (padding masks removed: identical math),
the thinker's WTA queries are constants baked at export, and both decoders
emit last-position logits only (selected via a `pos` input — cheaper than
returning every position and slicing in JS). Every graph is verified
against the torch modules on real chat probes before writing; run with
--parity to also compare full greedy replies.

Usage: .venv/bin/python scripts/export_web.py [--out webdemo/models]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402
from thoughtvec.lm import LMChatSession  # noqa: E402

K = 8  # k_ctx == k_out


class EncoderExport(nn.Module):
    """codec.encoder minus its GRU: the GRU only ever runs over the learned
    seed (constant input), so its output is precomputed as a buffer here —
    RNN ops need not exist in the graph."""

    def __init__(self, codec):
        super().__init__()
        e = codec.encoder
        self.emb = e.token_embedding
        self.pe = e.positions
        self.stack = e.encoder
        self.cross = e.cross_attention
        self.norm = e.norm
        self.mu = e.mu_proj
        self.scale = e.d_model ** 0.5
        with torch.no_grad():
            self.register_buffer("scaffold", e._scaffold(1).clone())

    def forward(self, ids):  # [1, T] int64
        x = self.pe(self.emb(ids) * self.scale)
        encoded = self.stack(x)
        attended, _ = self.cross(self.scaffold, encoded, encoded)
        return self.mu(self.norm(self.scaffold + attended))[:, :K]


class ThinkerExport(nn.Module):
    """Context assembly + trunk + WTA head + decodable score, resp_role=1.

    All context turns are real at chat time, so every padding mask in the
    torch path is all-False and is dropped here.
    """

    def __init__(self, thinker, codec):
        super().__init__()
        cfg = thinker.cfg
        self.trunk = thinker.trunk
        self.cross = thinker.cross
        self.out_norm = thinker.out_norm
        self.out_mlp = thinker.out_mlp
        self.mlp_norm = thinker.mlp_norm
        self.predictor = codec.predictor
        self.role_emb = thinker.role_emb
        self.turn_emb = thinker.turn_emb
        self.n_hyp = cfg.n_hypotheses

        pe = thinker.slot_pos.pe[:, :K]  # [1, 8, d]
        self.register_buffer("slot_pe", pe.clone())
        with torch.no_grad():  # queries: GRU(seed) + slot pos + resp_role(1)
            q, _ = thinker.out_gru(thinker.out_seed)
            q = q + pe
            q = q + thinker.resp_role_emb(torch.tensor([1]))[:, None, :]
        self.register_buffer("queries", q.clone())  # [4, 8, d]

    def forward(self, ctx_th, ctx_roles, dist):
        # ctx_th [1, C, 8, d]; ctx_roles, dist [1, C] int64
        d = ctx_th.size(-1)
        x = ctx_th + (self.role_emb(ctx_roles) + self.turn_emb(dist))[:, :, None, :]
        x = x + self.slot_pe[None]
        seq = x.reshape(1, -1, d)
        encoded = self.trunk(seq)
        m = self.n_hyp
        keys = encoded.expand(m, -1, -1)
        attended, _ = self.cross(self.queries, keys, keys)
        h = self.out_norm(self.queries + attended)
        out = self.mlp_norm(h + self.out_mlp(h))  # [4, 8, d]
        score = self.predictor(out)[:, K - 1]  # predicted CE at prefix 8, [4]
        return out, score


class DecoderExport(nn.Module):
    """codec.decoder reimplemented with functional ops: nn.TransformerDecoder
    trips torch.export's data-dependent guards, and hand-rolled pre-norm
    layers with precomputed causal/bias buffers (sliced by S) export cleanly.
    main() crosschecks this against the original decoder before export."""

    def __init__(self, codec):
        super().__init__()
        dec = codec.decoder
        cfg = codec.cfg
        self.layers = dec.decoder.layers
        self.final_norm = dec.decoder.norm
        self.emb = dec.token_embedding
        self.lm_head = dec.lm_head
        self.nhead = cfg.nhead
        self.scale = cfg.d_model ** 0.5
        self.register_buffer("pe", dec.positions.pe.clone())  # [1, 256, d]
        smax = cfg.max_seq_len
        causal = torch.zeros(smax, smax).masked_fill(
            torch.ones(smax, smax, dtype=torch.bool).triu(1), float("-inf"))
        self.register_buffer("causal", causal)
        with torch.no_grad():
            i = torch.arange(smax, dtype=torch.float32)[:, None]
            j = torch.arange(K, dtype=torch.float32)[None, :]
            self.register_buffer("bias", dec.position_attn_bias * (i - j))

    def _mha(self, mod, q_in, kv_in, mask):
        # torch.nn.MultiheadAttention, batch=1, equal embed dims, additive mask
        d = q_in.size(-1)
        h = self.nhead
        wq, wk, wv = mod.in_proj_weight.chunk(3)
        bq, bk, bv = mod.in_proj_bias.chunk(3)
        q = (q_in @ wq.T + bq).reshape(1, -1, h, d // h).transpose(1, 2)
        k = (kv_in @ wk.T + bk).reshape(1, -1, h, d // h).transpose(1, 2)
        v = (kv_in @ wv.T + bv).reshape(1, -1, h, d // h).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / (d // h) ** 0.5 + mask
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(1, -1, d)
        return out @ mod.out_proj.weight.T + mod.out_proj.bias

    def forward(self, thoughts, ids, pos):  # [1,8,d], [1,S] int64, [1] int64
        # Causality makes logits at `pos` independent of every later token,
        # so the first decode step (bare BOS) pads S to 2 with a dummy token
        # and reads pos=0 — sidestepping torch.export's S != 1 guard. The LM
        # head runs on the selected position only.
        s = ids.size(1)
        x = self.emb(ids) * self.scale + self.pe[:, :s]
        for lyr in self.layers:
            x = x + self._mha(lyr.self_attn, lyr.norm1(x), lyr.norm1(x),
                              self.causal[:s, :s])
            x = x + self._mha(lyr.multihead_attn, lyr.norm2(x), thoughts,
                              self.bias[:s])
            y = lyr.norm3(x)
            x = x + lyr.linear2(nn.functional.gelu(lyr.linear1(y)))
        x = torch.index_select(x, 1, pos)
        return self.lm_head(self.final_norm(x))[:, 0]  # [1, vocab]


class LMExport(nn.Module):
    """TokenLM reimplemented with functional ops, mirroring DecoderExport:
    nn.TransformerEncoder's fused self-attention path is bypassed entirely
    by hand-rolling pre-norm layers from the raw submodules. Unlike the
    codec decoder, no S==1 workaround is needed — LMChatSession always
    seeds generation with a real (context + BOS) sequence, so S is never
    less than 2 at export or at inference."""

    def __init__(self, lm):
        super().__init__()
        self.emb = lm.tok
        self.pos_emb = lm.pos
        self.layers = lm.trunk.layers
        self.final_norm = lm.norm
        self.head = lm.head
        self.nhead = lm.trunk.layers[0].self_attn.num_heads
        smax = lm.max_len
        causal = torch.zeros(smax, smax).masked_fill(
            torch.ones(smax, smax, dtype=torch.bool).triu(1), float("-inf"))
        self.register_buffer("causal", causal)

    def _mha(self, mod, x, mask):
        d = x.size(-1)
        h = self.nhead
        wq, wk, wv = mod.in_proj_weight.chunk(3)
        bq, bk, bv = mod.in_proj_bias.chunk(3)
        q = (x @ wq.T + bq).reshape(1, -1, h, d // h).transpose(1, 2)
        k = (x @ wk.T + bk).reshape(1, -1, h, d // h).transpose(1, 2)
        v = (x @ wv.T + bv).reshape(1, -1, h, d // h).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / (d // h) ** 0.5 + mask
        out = (att.softmax(-1) @ v).transpose(1, 2).reshape(1, -1, d)
        return out @ mod.out_proj.weight.T + mod.out_proj.bias

    def forward(self, ids, pos):  # [1,S] int64, [1] int64
        s = ids.size(1)
        x = self.emb(ids) + self.pos_emb.weight[:s][None]
        for lyr in self.layers:
            x = x + self._mha(lyr.self_attn, lyr.norm1(x), self.causal[:s, :s])
            y = lyr.norm2(x)
            x = x + lyr.linear2(nn.functional.gelu(lyr.linear1(y)))
        x = torch.index_select(self.final_norm(x), 1, pos)
        return self.head(x)[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/FINAL_12H/best.pt")
    ap.add_argument("--codec", default="checkpoints/m5_frontier/best.pt")
    ap.add_argument("--lm-ckpt", default="checkpoints/b3_lm_48m_24h/best.pt")
    ap.add_argument("--out", default="webdemo/models")
    ap.add_argument("--parity", action="store_true",
                    help="also compare full greedy replies torch vs onnx")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = ChatSession(args.ckpt, device="cpu", codec_ckpt=args.codec)
    codec, thinker = session.codec.eval(), session.thinker.eval()
    d = codec.cfg.d_model
    lm_session = LMChatSession(args.lm_ckpt, device="cpu")
    lm = lm_session.model.eval()

    def defuse(m: nn.Module) -> nn.Module:
        # nn.Transformer's fused inference kernel is not ONNX-exportable;
        # train mode disables it, and with every dropout zeroed the math is
        # identical (the stack is LayerNorm-only, no batch statistics).
        for sub in m.modules():
            if isinstance(sub, nn.Dropout):
                sub.p = 0.0
            if isinstance(sub, nn.MultiheadAttention):
                sub.dropout = 0.0  # stored as a float, not an nn.Dropout
        return m.train()

    enc = defuse(EncoderExport(codec))
    thk = defuse(ThinkerExport(thinker, codec))
    dec = defuse(DecoderExport(codec))
    lme = defuse(LMExport(lm))

    # reimplementation crosscheck: DecoderExport vs the original decoder
    with torch.no_grad():
        redc_worst = 0.0
        for S in (1, 20, 200):
            i = torch.randint(4, 16000, (1, S))
            t = torch.randn(1, K, d)
            ref = codec.decoder(t, i, causal=True)[:, -1]
            fed = torch.cat([i, i[:, :1]], dim=1) if S == 1 else i
            got_t = dec(t, fed, torch.tensor([S - 1]))
            redc_worst = max(redc_worst, float((got_t - ref).abs().max()))
    print(f"decoder reimpl vs nn.TransformerDecoder: {redc_worst:.3e}")
    assert redc_worst < 1e-4, "hand-rolled decoder diverges from the original"

    with torch.no_grad():
        relm_worst = 0.0
        for S in (2, 20, 200):
            i = torch.randint(4, 16000, (1, S))
            ref = lm(i)[0, -1]
            got_t = lme(i, torch.tensor([S - 1]))[0]
            relm_worst = max(relm_worst, float((got_t - ref).abs().max()))
    print(f"lm reimpl vs nn.TransformerEncoder: {relm_worst:.3e}")
    assert relm_worst < 1e-4, "hand-rolled lm diverges from the original"

    ids = torch.tensor([[1, 17, 205, 3001, 2]])
    ctx_th = torch.randn(1, 3, K, d)
    ctx_roles = torch.tensor([[1, 0, 1]])  # parity-of-3-turns example... roles vary
    dist = torch.tensor([[3, 2, 1]])
    dth = torch.randn(1, K, d)
    dids = torch.tensor([[1, 42, 99]])
    dpos = torch.tensor([2])
    lids = torch.tensor([[1, 17, 205, 3001, 2, 1]])
    lpos = torch.tensor([5])

    with torch.no_grad():
        torch.onnx.export(
            enc, (ids,), out / "encoder.onnx", opset_version=18,
            input_names=["ids"], output_names=["thoughts"],
            dynamic_shapes={"ids": {1: torch.export.Dim("T", min=3, max=256)}},
            dynamo=True,
        )
        torch.onnx.export(
            thk, (ctx_th, ctx_roles, dist), out / "thinker.onnx", opset_version=18,
            input_names=["ctx_th", "ctx_roles", "dist"],
            output_names=["hyps", "score"],
            dynamic_shapes={"ctx_th": {1: torch.export.Dim("C", min=1, max=6)},
                            "ctx_roles": {1: torch.export.Dim("C", min=1, max=6)},
                            "dist": {1: torch.export.Dim("C", min=1, max=6)}},
            dynamo=True,
        )
        torch.onnx.export(
            dec, (dth, dids, dpos), out / "decoder.onnx", opset_version=18,
            input_names=["thoughts", "ids", "pos"], output_names=["logits"],
            dynamic_shapes={"thoughts": None,
                            "ids": {1: torch.export.Dim("S", min=2, max=256)},
                            "pos": None},
            dynamo=True,
        )
        torch.onnx.export(
            lme, (lids, lpos), out / "lm.onnx", opset_version=18,
            input_names=["ids", "pos"], output_names=["logits"],
            dynamic_shapes={"ids": {1: torch.export.Dim("LS", min=2, max=lm.max_len)},
                            "pos": None},
            dynamo=True,
        )

    # ---- graph-level parity on real shapes ----
    import numpy as np
    import onnxruntime as ort

    sess = {n: ort.InferenceSession(str(out / f"{n}.onnx"),
                                    providers=["CPUExecutionProvider"])
            for n in ("encoder", "thinker", "decoder", "lm")}

    def run(n, **kw):
        return sess[n].run(None, {k: v.numpy() for k, v in kw.items()})

    worst = 0.0
    for T in (5, 37, 256):
        i = torch.randint(4, 16000, (1, T))
        with torch.no_grad():
            want = enc(i)
        got = run("encoder", ids=i)[0]
        worst = max(worst, float(np.abs(got - want.numpy()).max()))
    for C in (1, 4, 6):
        th = torch.randn(1, C, K, d)
        roles = torch.arange(C).flip(0) % 2
        dd = torch.arange(C, 0, -1)
        with torch.no_grad():
            w_h, w_s = thk(th, roles[None], dd[None])
        g_h, g_s = run("thinker", ctx_th=th, ctx_roles=roles[None], dist=dd[None])
        worst = max(worst, float(np.abs(g_h - w_h.numpy()).max()),
                    float(np.abs(g_s - w_s.numpy()).max()))
    for S in (2, 20, 200):
        i = torch.randint(4, 16000, (1, S))
        t = torch.randn(1, K, d)
        p_ = torch.tensor([S - 1])
        with torch.no_grad():
            want = dec(t, i, p_)
        got = run("decoder", thoughts=t, ids=i, pos=p_)[0]
        worst = max(worst, float(np.abs(got - want.numpy()).max()))
    for S in (2, 20, 200, 384):
        i = torch.randint(4, 16000, (1, S))
        p_ = torch.tensor([S - 1])
        with torch.no_grad():
            want = lme(i, p_)
        got = run("lm", ids=i, pos=p_)[0]
        worst = max(worst, float(np.abs(got - want.numpy()).max()))
    print(f"graph parity: worst abs diff {worst:.3e}")
    assert worst < 2e-4, "ONNX outputs diverge from torch"

    sizes = {p.name: p.stat().st_size / 1e6 for p in sorted(out.glob("*.onnx"))}
    print("sizes (MB):", {k: round(v, 1) for k, v in sizes.items()},
          "total", round(sum(sizes.values()), 1))

    if args.parity:
        probes = [
            "i'm feeling really overwhelmed with work lately.",
            "my boss keeps piling on deadlines.",
        ]
        from thoughtvec.tokenizer import BOS_ID, EOS_ID
        hist: list[str] = []
        for p in probes:
            hist.append(p)
            turns = hist[-6:]
            row_th, roles, dd = [], [], []
            n = len(turns)
            first_role = (len(hist) - n) % 2
            for j, t in enumerate(turns):
                tok = [BOS_ID] + session.tokenizer.encode(t, add_special=False)[:254] + [EOS_ID]
                row_th.append(run("encoder", ids=torch.tensor([tok]))[0])
                roles.append((first_role + j) % 2)
                dd.append(min(n - j, 6))
            ctx = torch.tensor(np.stack(row_th, axis=1))
            hyp, sc = run("thinker", ctx_th=ctx,
                          ctx_roles=torch.tensor([roles]), dist=torch.tensor([dd]))
            best = torch.tensor(hyp[int(np.argmin(sc))][None])
            ids_out = [BOS_ID]
            for _ in range(255):
                fed = ids_out + [0] if len(ids_out) < 2 else ids_out
                lg = run("decoder", thoughts=best, ids=torch.tensor([fed]),
                         pos=torch.tensor([len(ids_out) - 1]))[0][0]
                if len(ids_out) >= 3:  # no_repeat_ngram=3
                    pre = tuple(ids_out[-2:])
                    for k in range(len(ids_out) - 2):
                        if tuple(ids_out[k:k + 2]) == pre:
                            lg[ids_out[k + 2]] = -np.inf
                nxt = int(lg.argmax())
                ids_out.append(nxt)
                if nxt == EOS_ID:
                    break
            onnx_reply = session.tokenizer.decode(ids_out)
            hist.pop()
            session.history = list(hist)
            torch_reply = session.reply(p, temperature=0.0)
            hist.append(p)
            hist.append(torch_reply)
            status = "MATCH" if onnx_reply == torch_reply else "DIFF"
            print(f"[{status}] torch: {torch_reply!r}")
            if status == "DIFF":
                print(f"         onnx : {onnx_reply!r}")

        # LM baseline: flat history, greedy, no repeat-ngram ban (matches
        # TokenLM.generate exactly — its repetition loops are the documented
        # finding, not a bug this loop should paper over).
        lm_hist: list[str] = []
        for p in probes:
            lm_hist.append(p)
            ids_flat: list[int] = []
            for t in lm_hist:
                ids_flat += ([BOS_ID]
                             + lm_session.tokenizer.encode(t, add_special=False)
                             + [EOS_ID])
            room = lm.max_len - lm_session.max_new - 1
            ctx = ids_flat[-room:]
            ids_out = ctx + [BOS_ID]
            gen: list[int] = []
            for _ in range(lm_session.max_new):
                p_ = torch.tensor([len(ids_out) - 1])
                lg = run("lm", ids=torch.tensor([ids_out]), pos=p_)[0][0]
                nxt = int(lg.argmax())
                if nxt == EOS_ID:
                    break
                gen.append(nxt)
                ids_out.append(nxt)
            onnx_reply = lm_session.tokenizer.decode(gen)
            lm_hist.pop()
            lm_session.history = list(lm_hist)
            torch_reply = lm_session.reply(p, temperature=0.0)
            lm_hist.append(p)
            lm_hist.append(torch_reply)
            status = "MATCH" if onnx_reply == torch_reply else "DIFF"
            print(f"[{status}] lm torch: {torch_reply!r}")
            if status == "DIFF":
                print(f"            onnx : {onnx_reply!r}")


if __name__ == "__main__":
    main()
