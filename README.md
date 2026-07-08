# thoughtvec — conversation in thought-vector space

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21262842.svg)](https://doi.org/10.5281/zenodo.21262842)
[![Chat demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-thought--vectors--chat-blue)](https://huggingface.co/spaces/nochinator/thought-vectors-chat)

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
scripts/setup_env.sh --cpu    # CPU-only setup is enough for chat/eval
# from the GitHub Release (both files — the thinker checkpoint embeds the
# codec weights but reads the codec config from the codec file):
#   FINAL_12H-best.pt   -> checkpoints/FINAL_12H/best.pt
#   m5_frontier-best.pt -> checkpoints/m5_frontier/best.pt
.venv/bin/tv-chat --ckpt checkpoints/FINAL_12H/best.pt --device cpu   # REPL
.venv/bin/python scripts/chat_web.py                                   # web UI on :7860 (LAN)
```

The web UI serves the thinker and optionally SmolLM2-135M-Instruct side by
side. Chat inference runs on CPU (HIP inference is broken on gfx1031).

## Setup

```bash
scripts/setup_env.sh          # venv + torch ROCm + GPU check (--cpu for CPU-only)
.venv/bin/pytest              # sanity: 44 CPU tests
```

## Reproduce the pipeline

Full verified walkthrough (data sources, exact commands, expected numbers at
each stage, ROCm gotchas): **[docs/REPRODUCE.md](docs/REPRODUCE.md)**. The
short version — ~25 GPU-hours on one RX 6700 XT:

```bash
# 1. tokenizer (ships in artifacts/) + codec data: length-jittered C4+minipile mix
.venv/bin/tv-pretokenize --csv <c4.csv> --csv <minipile.csv> \
  --out data/mix_uni --max-tokens 254 --chunk-jitter

# 2. codec: 60-min warm base, then the crash-resilient 12h frontier run
scripts/train.sh --config configs/m5_frontier.yaml run.name=warm_base train.max_seconds=3600
scripts/run_frontier.sh

# 3. dialogue data (SODA + PersonaChat + OASST1)
.venv/bin/python scripts/extract_conversations.py
.venv/bin/python scripts/filter_dialogue.py
.venv/bin/python scripts/extract_oasst.py
.venv/bin/python scripts/build_dialogue_combined.py
.venv/bin/tv-pretokenize-dialogue --jsonl data/conversations_combined.jsonl --out data/dialogue_combined

# 4. thinker flagship (12h) + eval suite
scripts/final_12h.sh
```

`scripts/final2_12h.sh` reproduces the FINAL2 register case study (needs the
EmpatheticDialogues + reversal-splice data; see REPRODUCE.md §5). Ablation
brackets: `scripts/ablate_thinker_r{3..7}.sh`. Evals: `eval_thinker.py` (val
metrics), `eval_multiturn.py` (self-repetition / context sensitivity),
`eval_register.py` (sentiment register probes), `check_roundtrip.py` (codec
integrity guard).

ROCm gfx1031 notes: all training scripts set `HSA_OVERRIDE_GFX_VERSION=10.3.0`
and `HSA_ENABLE_SDMA=0`; thinker dropout must be 0.0 (NaNs); MIOpen refuses
RNN backward in eval mode (handled in `thinker_train.py`).

## Repo history note

This is a clean-room recreation and extension of an earlier AI-generated
design sketch ([legacy/phase0/RESEARCH.md](legacy/phase0/RESEARCH.md)); every
claim here is backed by code and runs in this repo. The two earlier
generations of the project are preserved under [legacy/](legacy/):

- `legacy/phase0/` — the first thought-vectors prototype and its full
  research log (codec collapsed past ~100 tokens; see the paper's history
  section).
- `legacy/phase1/` — the "BitThought" rebuild (curriculum training, BitLinear
  experiments); its original git history is in
  `legacy/phase1-git-history.bundle`.
- `docs/OVERVIEW.md` — an architecture overview spanning all phases.

The single source of truth for the experimental record is
[RESEARCH_LOG.md](RESEARCH_LOG.md) (current phase, plus an absorbed
cross-phase summary); heavy legacy data (datasets, artifacts) exists only on
the original machine and is gitignored.
