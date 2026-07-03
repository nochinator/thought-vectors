"""Thinker: context turn-thoughts -> predicted response thoughts.

Operates purely in the frozen codec's thought space. Context is a sequence of
turns, each compressed to k_ctx thought vectors; the thinker predicts the
k_out thought vectors of the reply, which the frozen decoder renders to text.

Two prediction modes (ablated):
  "query"  — learned response slots (GRU over a seed, the codec encoder's
             proven ordered-scaffold trick) cross-attend the trunk output and
             predict all k_out slots in parallel.
  "prefix" — response slots are appended to the context sequence with a
             causal mask over the response region (AR over thought slots,
             teacher-forced during training, iterative at inference).

Dropout MUST stay 0.0: nonzero thinker dropout NaN'd on this ROCm stack in
the prior project.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ThinkerCfg
from .modules import SinusoidalPositions


class Thinker(nn.Module):
    def __init__(self, cfg: ThinkerCfg, d_model: int) -> None:
        super().__init__()
        assert cfg.dropout == 0.0, "thinker dropout NaNs on ROCm gfx1031 (legacy M4)"
        self.cfg = cfg
        self.d_model = d_model

        self.role_emb = nn.Embedding(2, d_model)        # who spoke the turn
        self.resp_role_emb = nn.Embedding(2, d_model)   # who is replying
        self.turn_emb = nn.Embedding(cfg.max_turns + 1, d_model)  # distance from reply
        self.slot_pos = SinusoidalPositions(d_model, max(cfg.k_ctx, cfg.k_out) + 1, 0.0)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trunk = nn.TransformerEncoder(
            layer, num_layers=cfg.layers, norm=nn.LayerNorm(d_model)
        )

        assert cfg.n_hypotheses == 1 or cfg.mode == "query", "WTA head is query-mode only"
        self.out_seed = nn.Parameter(
            torch.randn(cfg.n_hypotheses, cfg.k_out, d_model) * 0.02
        )
        if cfg.mode == "query":
            self.out_gru = nn.GRU(d_model, d_model, batch_first=True)
            self.cross = nn.MultiheadAttention(d_model, cfg.nhead, batch_first=True)
            self.out_norm = nn.LayerNorm(d_model)
            self.out_mlp = nn.Sequential(
                nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model)
            )
            self.mlp_norm = nn.LayerNorm(d_model)
        else:  # prefix
            self.out_proj = nn.Linear(d_model, d_model)

        if cfg.w_reverse > 0:
            self.rev_seed = nn.Parameter(torch.randn(1, cfg.k_ctx, d_model) * 0.02)

    # ----- context assembly -----

    def _context_seq(
        self,
        ctx_thoughts: torch.Tensor,  # [B, C, k, d]
        ctx_roles: torch.Tensor,     # [B, C]
        ctx_turns: torch.Tensor,     # [B] real turn counts
        slot_budgets: torch.Tensor | None = None,  # [B, C] per-turn thought counts
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, c, k, d = ctx_thoughts.shape
        dist = (ctx_turns[:, None] - torch.arange(c, device=ctx_thoughts.device)[None, :]).clamp(
            min=0, max=self.cfg.max_turns
        )  # [B, C] distance from the reply (rightmost real turn = 1)
        x = ctx_thoughts + (self.role_emb(ctx_roles) + self.turn_emb(dist))[:, :, None, :]
        x = self.slot_pos(x.reshape(bsz * c, k, d)).reshape(bsz, c, k, d)
        seq = x.reshape(bsz, c * k, d)
        turn_real = torch.arange(c, device=seq.device)[None, :] < ctx_turns[:, None]  # [B, C]
        key_pad = ~turn_real[:, :, None].expand(bsz, c, k)
        if slot_budgets is None and self.cfg.k_ctx_schedule:
            sched = torch.as_tensor(self.cfg.k_ctx_schedule, device=seq.device)
            slot_budgets = sched[(dist - 1).clamp(min=0, max=sched.numel() - 1)]
        if slot_budgets is not None:
            # the codec orders thoughts by importance, so masking the tail of a
            # turn's slots IS decoding-equivalent to having used a smaller k
            slots = torch.arange(k, device=seq.device)
            key_pad = key_pad | (slots[None, None, :] >= slot_budgets[:, :, None])
        return seq, key_pad.reshape(bsz, c * k)

    def _queries(self, seed: torch.Tensor, resp_roles: torch.Tensor) -> torch.Tensor:
        """seed [M, k, d] -> queries [B*M, k, d] (hypotheses as extra batch rows)."""
        out, _ = self.out_gru(seed)
        q = self.slot_pos(out)  # [M, k, d]
        bsz, m = resp_roles.size(0), seed.size(0)
        q = q[None].expand(bsz, m, -1, -1).reshape(bsz * m, q.size(1), -1)
        role = self.resp_role_emb(resp_roles).repeat_interleave(m, dim=0)
        return q + role[:, None, :]

    # ----- forward -----

    def forward(
        self,
        ctx_thoughts: torch.Tensor,
        ctx_roles: torch.Tensor,
        ctx_turns: torch.Tensor,
        resp_roles: torch.Tensor,
        target_thoughts: torch.Tensor | None = None,  # [B, k_out, d] for prefix-mode TF
        slot_budgets: torch.Tensor | None = None,     # [B, C] per-turn thought counts
        ss_prob: float | None = None,                 # override cfg.ss_prob (for schedule)
    ) -> torch.Tensor:
        cfg = self.cfg
        seq, key_pad = self._context_seq(ctx_thoughts, ctx_roles, ctx_turns, slot_budgets)
        bsz = seq.size(0)

        if cfg.mode == "query":
            encoded = self.trunk(seq, src_key_padding_mask=key_pad)
            m = cfg.n_hypotheses
            q = self._queries(self.out_seed, resp_roles)
            attended, _ = self.cross(
                q,
                encoded.repeat_interleave(m, dim=0),
                encoded.repeat_interleave(m, dim=0),
                key_padding_mask=key_pad.repeat_interleave(m, dim=0),
            )
            h = self.out_norm(q + attended)
            out = self.mlp_norm(h + self.out_mlp(h))
            if m == 1:
                return out
            return out.reshape(bsz, m, cfg.k_out, -1)  # [B, M, k_out, d]

        # prefix mode: [ctx][response slots], causal over the response region
        if target_thoughts is None:
            return self._prefix_generate(seq, key_pad, resp_roles)
        # Scheduled sampling: with probability ss_prob, feed the model's own
        # prediction as the previous-slot input instead of the true target.
        # ss_prob > 0 enables the iterative path (one transformer pass per slot).
        _ss = ss_prob if ss_prob is not None else getattr(cfg, "ss_prob", 0.0)
        if self.training and _ss > 0:
            return self._prefix_scheduled(seq, key_pad, resp_roles, target_thoughts, _ss)
        slots = self.slot_pos(self.out_seed.expand(bsz, -1, -1).clone())
        slots = slots + self.resp_role_emb(resp_roles)[:, None, :]
        # teacher forcing: slot j also receives the true thought j-1
        tf = torch.cat(
            [torch.zeros_like(target_thoughts[:, :1]), target_thoughts[:, :-1]], dim=1
        )
        if self.training and cfg.tf_noise_std > 0:
            tf = tf + torch.randn_like(tf) * cfg.tf_noise_std
        slots = slots + tf
        full = torch.cat([seq, slots], dim=1)
        mask = self._prefix_mask(seq.size(1), cfg.k_out, seq.device)
        pad = torch.cat(
            [key_pad, torch.zeros(bsz, cfg.k_out, dtype=torch.bool, device=seq.device)], dim=1
        )
        out = self.trunk(full, mask=mask, src_key_padding_mask=pad)
        return self.out_proj(out[:, -cfg.k_out :])

    def _prefix_scheduled(
        self, seq, key_pad, resp_roles, target_thoughts, ss_prob
    ) -> torch.Tensor:
        """Prefix mode with scheduled sampling: iterative slot-by-slot with
        probability ss_prob of using own prediction as previous-slot input."""
        cfg = self.cfg
        bsz = seq.size(0)
        ctx_len = seq.size(1)
        outputs = []
        prev = torch.zeros(bsz, 1, self.d_model, device=seq.device)
        for j in range(cfg.k_out):
            slot = self.slot_pos(self.out_seed[:, j : j + 1].expand(bsz, -1, -1).clone())
            slot = slot + self.resp_role_emb(resp_roles)[:, None, :] + prev
            full = torch.cat([seq] + [o for o in outputs] + [slot], dim=1)
            total = full.size(1)
            mask = torch.zeros(total, total, dtype=torch.bool, device=seq.device)
            mask[:ctx_len, ctx_len:] = True
            if j > 0:
                resp_causal = torch.triu(
                    torch.ones(j + 1, j + 1, dtype=torch.bool, device=seq.device), diagonal=1
                )
                mask[- (j + 1) :, - (j + 1) :] = resp_causal
            new_pad = torch.cat(
                [key_pad, torch.zeros(bsz, j + 1, dtype=torch.bool, device=seq.device)], dim=1
            )
            out = self.trunk(full, mask=mask, src_key_padding_mask=new_pad)
            pred = self.out_proj(out[:, -1:])
            outputs.append(pred)
            if j + 1 < cfg.k_out:
                prev = pred.detach() if torch.rand(()).item() < ss_prob else target_thoughts[:, j : j + 1]
        return torch.cat(outputs, dim=1)

    @staticmethod
    def _prefix_mask(ctx_len: int, k_out: int, device) -> torch.Tensor:
        total = ctx_len + k_out
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:ctx_len, ctx_len:] = True  # context cannot see response slots
        resp = torch.triu(torch.ones(k_out, k_out, dtype=torch.bool, device=device), diagonal=1)
        mask[ctx_len:, ctx_len:] = resp  # causal within response
        return mask

    @torch.no_grad()
    def _prefix_generate(
        self, seq: torch.Tensor, key_pad: torch.Tensor, resp_roles: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg
        bsz = seq.size(0)
        generated = torch.zeros(bsz, 0, self.d_model, device=seq.device)
        for j in range(cfg.k_out):
            slots = self.slot_pos(self.out_seed.expand(bsz, -1, -1)[:, : j + 1].clone())
            slots = slots + self.resp_role_emb(resp_roles)[:, None, :]
            if j > 0:
                slots = slots + torch.cat(
                    [torch.zeros(bsz, 1, self.d_model, device=seq.device), generated], dim=1
                )
            full = torch.cat([seq, slots], dim=1)
            mask = self._prefix_mask(seq.size(1), j + 1, seq.device)
            pad = torch.cat(
                [key_pad, torch.zeros(bsz, j + 1, dtype=torch.bool, device=seq.device)], dim=1
            )
            out = self.trunk(full, mask=mask, src_key_padding_mask=pad)
            nxt = self.out_proj(out[:, -1:])
            generated = torch.cat([generated, nxt], dim=1)
        return generated

    def predict_reverse(
        self,
        ctx_thoughts: torch.Tensor,
        ctx_roles: torch.Tensor,
        ctx_turns: torch.Tensor,
        resp_roles: torch.Tensor,
        slot_budgets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aux head: predict the LAST context turn's thoughts (query mode only)."""
        seq, key_pad = self._context_seq(ctx_thoughts, ctx_roles, ctx_turns, slot_budgets)
        encoded = self.trunk(seq, src_key_padding_mask=key_pad)
        q = self.slot_pos(self.rev_seed).expand(seq.size(0), -1, -1)
        q = q + self.resp_role_emb(1 - resp_roles)[:, None, :]
        attended, _ = self.cross(q, encoded, encoded, key_padding_mask=key_pad)
        h = self.out_norm(q + attended)
        return self.mlp_norm(h + self.out_mlp(h))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
