# thoughtvec — conversation in thought-vector space

Text → ordered **thought vectors** → text. An encoder compresses text into N
latent vectors ordered by importance; **any prefix** of k vectors decodes back
to text, so compression ratio is a decode-time choice, and a small predictor
estimates reconstruction quality per prefix length (adaptive compression). A
**thinker** transformer then converses directly in that latent space: history
turns → thoughts → predicted response thoughts → decoded reply. No token-level
language modeling happens in the thinker at all.

Everything was trained from scratch on **one AMD RX 6700 XT (12 GB, ROCm)**:
a 32.9M-parameter codec + a 15.1M-parameter thinker (≈48M total, ~6M shared
embeddings).

```
user > i'm feeling really overwhelmed with work lately.
bot  > I can imagine. What's been going on?
user > my boss keeps piling on deadlines.
bot  > That sounds like a lot of pressure. Have you been able to talk to anyone else?
```
*(FINAL_12H flagship, temp 0. It also still says "That's good to hear." to
insomnia — see the register-disease case study in the paper.)*

## Results at a glance

- **Codec (M5 frontier, d=384)**: byte-perfect round-trip at 4:1 compression,
  readable at 8:1, with a graded quality dial via prefix length.
- **Thinker (FINAL_12H, 12h from scratch)**: coherent grounded small talk,
  multi-turn context sensitivity, follow-up questions; best metrics of all
  ~30 ablation arms (val_cos 0.428, self-repetition 0.19, ctx_sens 0.15).
- **Case study**: a context-conditional *sentiment register* failure
  (cheerful replies to bad news) traced through 3 ablation rounds to a data
  absence — no training conversation ever reverses mood mid-dialogue. A
  second 12h run (FINAL2_12H) with 40k synthesized reversal dialogues beat
  every register *probe* (ctx err 0.50 → 0.17) but a live chat audit showed
  it learned the splice template, not sentiment routing ("I bet that was
  fun!" to a burglary). Plus four separate incidents of models gaming
  lexical metrics, caught by mandatory transcript audits.

The full experimental narrative (every round, table, failure, and audit) is in
[RESEARCH_LOG.md](RESEARCH_LOG.md). The paper draft lives in [paper/](paper/).

## Chat with it

```bash
.venv/bin/tv-chat --ckpt checkpoints/FINAL_12H/best.pt --device cpu   # REPL
.venv/bin/python scripts/chat_web.py                                   # web UI on :7860 (LAN)
```

The web UI serves the thinker and optionally SmolLM2-135M-Instruct side by
side. Chat inference runs on CPU (HIP inference is broken on gfx1031).

## Setup

```bash
scripts/setup_env.sh          # venv + torch ROCm + GPU check
.venv/bin/pytest              # sanity: all CPU tests
```

## Reproduce the pipeline

```bash
# 1. tokenizer + codec pretraining data
.venv/bin/tv-tokenizer train --corpus <c4.csv>:4 --corpus <minipile.csv>:40
.venv/bin/tv-pretokenize --csv <c4.csv> --out data/c4_500k --max-rows 500000

# 2. codec: baseline AE -> compression -> frontier
scripts/train.sh --config configs/m1_autoencoder.yaml
scripts/train.sh --config configs/m2_compression.yaml --init-from checkpoints/m1_autoencoder/best.pt
scripts/run_frontier.sh

# 3. dialogue data (SODA + PersonaChat + OASST1 + EmpatheticDialogues + reversal splices)
.venv/bin/python scripts/extract_conversations.py
.venv/bin/python scripts/extract_empathetic.py --dir <ed_csvs>
.venv/bin/python scripts/build_reversal_splices.py --dir <ed_csvs> --n 40000
.venv/bin/tv-pretokenize-dialogue --jsonl data/conversations_rev40.jsonl --out data/dialogue_rev40

# 4. thinker flagship (12h) + evals
scripts/final2_12h.sh
```

Ablation brackets: `scripts/ablate_thinker_r{3..7}.sh`. Evals:
`eval_thinker.py` (val metrics), `eval_multiturn.py` (self-repetition /
context sensitivity), `eval_register.py` (sentiment register probes),
`check_roundtrip.py` (codec integrity guard).

ROCm gfx1031 notes: all training scripts set `HSA_OVERRIDE_GFX_VERSION=10.3.0`
and `HSA_ENABLE_SDMA=0`; thinker dropout must be 0.0 (NaNs); MIOpen refuses
RNN backward in eval mode (handled in `thinker_train.py`).

## Repo history note

This is a clean-room recreation and extension of an earlier AI-generated
design sketch ([legacy/RESEARCH.md](legacy/RESEARCH.md)); every claim here is
backed by code and runs in this repo. The two earlier generations of the
project are preserved under [legacy/](legacy/):

- `legacy/phase0/` — the first thought-vectors prototype (codec collapsed
  past ~100 tokens; see the paper's history section).
- `legacy/phase1/` — the "BitThought" rebuild (curriculum training, BitLinear
  experiments); its original git history is in
  `legacy/phase1-git-history.bundle`.
- `docs/OVERVIEW.md` — an architecture overview spanning all phases.
- `CONSOLIDATED_RESEARCH.md` — durable lessons merged across phases;
  the per-experiment record for the current phase is
  [RESEARCH_LOG.md](RESEARCH_LOG.md).

Heavy legacy data (datasets, artifacts) exists only on the original machine
and is gitignored; everything needed to read the history is tracked.
