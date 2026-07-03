"""tv-chat: terminal REPL for the thinker.

Pipeline per turn: history texts -> codec encoder (k_ctx thoughts each) ->
thinker -> k_out predicted thoughts -> codec decoder -> reply text. The codec
is the one recorded in the thinker checkpoint (overridable); the user is role
0, the bot role 1, matching DialogueDataset parity.
"""

from __future__ import annotations

import torch

from .config import Config, from_dict
from .generate import sample_decode
from .model import ThoughtAutoencoder
from .thinker import Thinker
from .tokenizer import BOS_ID, EOS_ID, Tokenizer


class ChatSession:
    def __init__(self, ckpt_path: str, device: str = "cuda",
                 codec_ckpt: str | None = None,
                 hyp_select: str = "decodable") -> None:
        # hyp_select: WTA winner rule at temp 0. "decodable" = codec predictor
        # score (default; skews toward the compact positive-register attractor —
        # see RESEARCH_LOG 2026-07-03 R7 thought-space diagnostic). "affinity" =
        # highest pooled-thought cosine to the LAST user turn, so the reply is
        # routed by what was just said rather than by decode confidence.
        self.hyp_select = hyp_select
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg: Config = from_dict(ckpt["config"])
        tk = self.cfg.thinker

        codec_path = codec_ckpt or ckpt.get("codec_ckpt", tk.codec_ckpt)
        codec_state = torch.load(codec_path, map_location="cpu", weights_only=False)
        self.codec_cfg = from_dict(codec_state["config"])
        self.codec = ThoughtAutoencoder(self.codec_cfg.model)
        self.codec.load_state_dict(ckpt.get("codec", codec_state["model"]))
        self.codec.to(device).eval()

        self.thinker = Thinker(tk, self.codec_cfg.model.d_model)
        self.thinker.load_state_dict(ckpt["thinker"])
        self.thinker.to(device).eval()

        self.tokenizer = Tokenizer(self.cfg.run.tokenizer_path)
        self.device = torch.device(device)
        self.history: list[str] = []  # alternating, history[0] = user

    @torch.no_grad()
    def reply(self, user_text: str, temperature: float = 0.0,
              no_repeat_ngram: int = 3) -> str:
        tk = self.cfg.thinker
        self.history.append(user_text.strip())
        turns = self.history[-tk.max_turns :]
        first_role = (len(self.history) - len(turns)) % 2  # parity of turns[0]

        from .thinker_train import encode_turns

        seq_max = self.codec_cfg.model.max_seq_len
        rows = [
            [BOS_ID] + self.tokenizer.encode(t, add_special=False)[: seq_max - 2] + [EOS_ID]
            for t in turns
        ]
        if tk.flat_context:
            flat_row: list[int] = []
            for r in rows:
                flat_row += r
            rows = [flat_row[-seq_max:]]
            first_role = 0
        max_t = max(len(r) for r in rows)
        ctx_ids = torch.zeros(1, len(rows), max_t, dtype=torch.long, device=self.device)
        for j, r in enumerate(rows):
            ctx_ids[0, j, : len(r)] = torch.tensor(r, dtype=torch.long)

        ctx_th, budgets = encode_turns(self.codec, ctx_ids, tk.k_ctx, tau=tk.ctx_tau)
        ctx_roles = torch.tensor(
            [[(first_role + j) % 2 for j in range(len(rows))]], device=self.device
        )
        ctx_turns = torch.tensor([len(rows)], device=self.device)
        resp_roles = torch.tensor([1], device=self.device)  # bot replies

        pred = self.thinker(ctx_th, ctx_roles, ctx_turns, resp_roles, slot_budgets=budgets)
        if pred.dim() == 4:  # WTA head: greedy -> most decodable per the codec's
            # predictor; temperature>0 -> random hypothesis (varied replies)
            if temperature > 0:
                pred = pred[:, int(torch.randint(pred.size(1), (1,)))]
            elif self.hyp_select == "affinity":
                last = torch.nn.functional.normalize(ctx_th[0, -1].mean(0), dim=-1)
                hyp = torch.nn.functional.normalize(pred[0].mean(1), dim=-1)
                pred = pred[:, int((hyp @ last).argmax())]
            else:
                score = self.codec.predictor(pred[0])[:, tk.k_out - 1]
                pred = pred[:, int(score.argmin())]
        out = sample_decode(self.codec, pred, self.codec_cfg.model.max_seq_len,
                            temperature=temperature, no_repeat_ngram=no_repeat_ngram)
        text = self.tokenizer.decode(out[0].tolist())
        self.history.append(text)
        return text

    def reset(self) -> None:
        self.history.clear()


def repl(ckpt_path: str, device: str = "cuda", temperature: float = 0.0) -> None:
    session = ChatSession(ckpt_path, device=device)
    print(f"thoughtvec chat — ckpt {ckpt_path} | /reset clears history, /quit exits")
    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            session.reset()
            print("(history cleared)")
            continue
        print(f"bot > {session.reply(user, temperature=temperature)}")
