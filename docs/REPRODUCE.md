# Reproducing the models

Everything below was run on **one AMD RX 6700 XT (12 GB, ROCm, gfx1031)**, on
this exact codebase. Total budget for the two flagship models: ~25 GPU-hours
(1h codec warm-up + 12h codec + 12h thinker) plus data preparation.

If you only want to **talk to the released model**, skip to
[Run the released checkpoints](#run-the-released-checkpoints) — that works on
any CPU.

## 0. Environment

```bash
scripts/setup_env.sh          # ROCm torch (what training ran on)
scripts/setup_env.sh --cpu    # CPU-only torch (enough for chat/eval)
.venv/bin/pytest              # sanity: 44 CPU tests
```

ROCm notes for gfx1031 (RX 6700 XT), all already baked into the scripts:

- `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `HSA_ENABLE_SDMA=0` for every run.
- Thinker dropout must be 0.0 (NaNs otherwise).
- Never set `torch.set_float32_matmul_precision("high")` — NaN after ~20K
  steps; the trainer asserts `"highest"`.
- Long runs sporadically page-fault; `scripts/run_frontier.sh` handles this
  by resuming from the last checkpoint in a loop.
- Chat/eval inference runs on CPU (`--device cpu`); HIP inference is broken
  on this GPU.
- On NVIDIA/CUDA none of this should be needed, but nothing here has been
  tested on CUDA.

## Run the released checkpoints

Download `FINAL_12H` (thinker flagship; contains the codec + tokenizer path
inside the checkpoint) from the GitHub Release into `checkpoints/FINAL_12H/`,
then:

```bash
.venv/bin/tv-chat --ckpt checkpoints/FINAL_12H/best.pt --device cpu
.venv/bin/python scripts/chat_web.py    # web UI on :7860 (LAN)
```

The tokenizer (`artifacts/tokenizer/spm16k_bpe.model`) ships in the repo; all
checkpoints depend on it. Codec-only evaluation:

```bash
.venv/bin/tv-eval --ckpt checkpoints/m5_frontier/best.pt \
  --shard data/mix_uni_val --device cpu
```

## 1. Source data

Two plain-text corpora exported as single-column CSVs (header `text`, one
document per row):

| CSV | Source | Used for |
|---|---|---|
| `C4subset-1.csv` (~800 MB) | [allenai/c4](https://huggingface.co/datasets/allenai/c4) (en) | tokenizer + codec |
| `minipile.csv` (~5.9 GB) | [JeanKaddour/minipile](https://huggingface.co/datasets/JeanKaddour/minipile) | tokenizer + codec |

Dialogue corpora are downloaded by the extraction scripts themselves
(HuggingFace `datasets`; EmpatheticDialogues comes from the ParlAI tarball —
URL in `scripts/extract_empathetic.py`).

## 2. Tokenizer

Already shipped at `artifacts/tokenizer/spm16k_bpe.model` (16K SentencePiece
BPE) — **do not retrain it if you want to reuse any released checkpoint**; all
checkpoints are tied to it. To rebuild from scratch:

```bash
.venv/bin/tv-tokenizer train --corpus C4subset-1.csv:4 --corpus minipile.csv:40
```

(`path:N` = take every Nth row.)

## 3. Codec pretraining data (`data/mix_uni`)

Length-jittered mix of C4 + minipile. The jitter (`--chunk-jitter`: long
documents cut at uniform(16, 254) tokens instead of always max length) is
load-bearing — long-only data never engages the thought channel (RESEARCH_LOG
2026-06-10, E256 collapse).

```bash
.venv/bin/tv-pretokenize --csv C4subset-1.csv --csv minipile.csv \
  --out data/mix_uni --max-tokens 254 --chunk-jitter
```

Reference meta for the shard the flagship trained on (`data/mix_uni/meta.json`):
10,955,786 samples, 1,415,198,817 tokens, length mean 129.2 / median 126.

## 4. Codec (M5 frontier, 12h)

The 12h frontier run warm-starts from a 60-minute run of the *same* config:

```bash
# 60-min warm base (writes checkpoints/warm_base/final.pt)
scripts/train.sh --config configs/m5_frontier.yaml \
  run.name=warm_base train.max_seconds=3600

# 12h frontier (crash-resilient loop; picks up warm_base automatically)
scripts/run_frontier.sh
```

Expected result (see RESEARCH_LOG 2026-06-12 and `logs/m5_frontier/eval/`):
byte-perfect round-trip at 4:1 compression up to 257 tokens, readable at 8:1,
monotone CE-vs-k (the tau dial).

```bash
.venv/bin/tv-eval --ckpt checkpoints/m5_frontier/best.pt --shard data/mix_uni_val
.venv/bin/python scripts/check_roundtrip.py --ckpt checkpoints/m5_frontier/best.pt
```

## 5. Thinker dialogue data (`data/dialogue_combined`)

```bash
.venv/bin/python scripts/extract_conversations.py    # SODA + PersonaChat + OASST → conversations.jsonl
.venv/bin/python scripts/filter_dialogue.py          # quality filter → conversations_clean.jsonl
.venv/bin/python scripts/extract_oasst.py            # multi-path OASST → oasst1_conversations.jsonl
.venv/bin/python scripts/build_dialogue_combined.py  # → conversations_combined.jsonl
.venv/bin/tv-pretokenize-dialogue --jsonl data/conversations_combined.jsonl \
  --out data/dialogue_combined
```

Reference composition: 70,866 conversations = SODA 61,464 + PersonaChat 863 +
OASST 8,539 (the build script prints this and it should match exactly).

For the FINAL2 register case study (paper §case study), additionally:

```bash
.venv/bin/python scripts/extract_empathetic.py --dir <extracted_ed_csvs>
.venv/bin/python scripts/build_reversal_splices.py --dir <extracted_ed_csvs> --n 40000
.venv/bin/tv-pretokenize-dialogue --jsonl data/conversations_rev40.jsonl \
  --out data/dialogue_rev40
```

## 6. Thinker flagship (FINAL_12H, 12h)

The full recipe (WTA-4 hypotheses, k_ctx=8/k_out=8, cycle loss, decoder
co-training) is pinned in one script, which also runs the eval suite at the
end:

```bash
scripts/final_12h.sh     # needs checkpoints/m5_frontier/best.pt
```

`scripts/final2_12h.sh` is the same for the FINAL2 case-study run (on
`data/dialogue_rev40`).

Expected FINAL_12H metrics (RESEARCH_LOG 2026-07-03): val_cos 0.428,
ref_F1 0.297, self_rep 0.19, ctx_sens 0.146. Exact numbers vary with the
hardware-dependent step count; the qualitative behaviors (grounded small
talk, follow-up questions, the register disease) should reproduce.

## 7. Evals and audits

```bash
.venv/bin/python scripts/eval_thinker.py   --ckpt checkpoints/FINAL_12H/best.pt
.venv/bin/python scripts/eval_multiturn.py --ckpt checkpoints/FINAL_12H/best.pt --device cpu
.venv/bin/python scripts/eval_register.py  --ckpt checkpoints/FINAL_12H/best.pt
.venv/bin/python scripts/check_roundtrip.py --ckpt checkpoints/FINAL_12H/best.pt
```

Then **read transcripts** (`tv-chat`, temp 0 and 0.8) before believing any
metric — four separate incidents in this project of models gaming lexical
metrics were caught only by transcript audits (paper §honest record; and the
chat-probe gate in RESEARCH_LOG).

## Ablation brackets (optional)

The full ablation history is re-runnable: `scripts/ablate.sh` (codec W-series),
`scripts/ablate_thinker_r{3..7}.sh` (thinker rounds). Each arm is an
equal-wall-clock slot; see RESEARCH_LOG.md for what each round tested and
found. Throughput probe for sizing arms on your GPU: `scripts/probe.py`.
