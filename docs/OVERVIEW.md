# ThoughtVectors

**Text → ordered latent vectors → text.** An encoder-decoder transformer that
compresses text into a variable-length sequence of continuous thought vectors,
then reconstructs text from any prefix of those vectors.

The core idea: transform text into a compact, semantically-grounded latent
space where downstream reasoning, search, and manipulation are orders of
magnitude cheaper than operating on raw tokens.

---

## Why thought vectors?

Unlike standard autoencoders (which produce a single fixed-size vector) or
sequence models (which operate token-by-token), thought vectors give you:

- **Variable-rate compression** — choose your compression ratio at decode
  time by taking a longer or shorter prefix of the thought sequence.
- **Ordered information hierarchy** — the GRU forces early vectors to carry
  the most important information, later vectors refine it. This means the
  system degrades gracefully under aggressive compression.
- **Latent-space reasoning** — a "thinker" transformer operates directly on
  thought vectors (16-256 positions) instead of token sequences (512+
  positions), making reasoning roughly 10-100x cheaper in the attention
  mechanism.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      SHARED EMBEDDING                            │
│               nn.Embedding(V=50K, d=384)                         │
│          Used by: encoder input, decoder input, LM head          │
│          (weight-tied: one matrix, three roles)                  │
└──────────────────────────────────────────────────────────────────┘
        │                                              │
        ▼                                              ▼
┌─────────────────┐                      ┌──────────────────────────┐
│    ENCODER       │                      │        DECODER           │
│ Transformer × L  │                      │   Transformer × L        │
│  ┌────────────┐  │                      │  ┌──────────────────┐    │
│  │  GRU +     │  │                      │  │ Embed predictor  │    │
│  │ CrossAttn  │  │                      │  │ Linear(d, d)     │    │
│  └────────────┘  │                      │  └──────────────────┘    │
│        │         │                      │        │                 │
│  thought vecs    │                      │  pred_embeds → logits    │
│  [B, N, d]       │                      │      (W^T projection)   │
│  + loss_preds    │                      │                          │
│  [B, N] (0-1)    │                      └──────────────────────────┘
└─────────────────┘
```

### Encoder

Token embedding → sinusoidal positional encoding → stacked Transformer
encoder layers (full self-attention) → learned thought seed → GRU →
cross-attention to encoder output → LayerNorm → thought vectors.

The GRU is the critical component. It imposes a sequential hierarchy on the
thought vectors: the first vector carries the most important content, and
each subsequent vector refines or fills in detail. This ordering makes
**prefix-truncation compression** work — taking the first k vectors gives
you the best possible reconstruction for that budget.

A small **loss predictor** (2-layer MLP, sigmoid output) estimates
reconstruction quality for each prefix length. This enables the runtime to
choose k adaptively to meet a quality target (the `tau` knob).

### Decoder

Thought vectors → hidden projection → stacked Transformer decoder layers
(causal-masked self-attention + cross-attention to thought vectors) →
embedding predictor (Linear(d, d)) → logits via embedding transpose.

Unlike standard autoregressive language models, the decoder outputs
**token embeddings** not logits directly, and projects through the shared
embedding matrix (weight tying). This saves ~25M parameters vs. a separate
LM head.

### Thinker (latent-space reasoning)

A separate transformer that operates entirely in thought-vector space.
Given encoded thought vectors for a user input (and optionally past
conversation turns), the thinker predicts response thought vectors that the
frozen decoder renders to text:

```
text_user → Encoder → thoughts_user → Thinker → thoughts_response → Decoder → text_response
```

No token-by-token autoregressive generation happens in the reasoning step.
The thinker operates on 16-256 thought positions instead of 512+ token
positions, making it roughly 10-100x cheaper per attention operation.

The current thinker uses **Winner-Take-All (WTA)** training: 4 parallel
hypothesis heads produce candidate responses, and only the best (lowest
loss) receives gradient — preventing mode-averaging across plausible
responses.

---

## Compression Mechanics

**Key insight: compression is a decode-time choice.** The encoder always
produces all N thought vectors (e.g., 256). The decoder receives only k << N
(the first k vectors). This forces the encoder to pack the most important
information into the earliest positions, because the decoder might only see
a prefix.

The **tau knob** controls this tradeoff:

| tau | Compression ratio | CE loss | Use case |
|-----|------------------|---------|----------|
| 0.25 | ~4:1 | 0.08 | High fidelity |
| 0.5 | ~1.2:1 | 0.16 | Balanced |
| 1.0 | ~0.5:1 | 0.46 | Aggressive |
| 2.0 | ~0.2:1 | 1.5 | Maximum compression |

The monotone predictor head ensures tau is a clean graded dial with no
garbage regime.

---

## Project Layout

> This overview predates the 2026-07 restructure; layout updated, prose kept.

| Directory | Contents |
|---|---|
| `src/thoughtvec/` | Current codebase (Phase 2): encoder, decoder, thinker, data, cli. |
| `configs/` | YAML configs for m0 (smoke) through m5 (frontier codec). |
| `checkpoints/` | Trained models (gitignored; flagship ships via GitHub Release). |
| `data/` | Tokenized training data (gitignored). |
| `scripts/` | Shell entrypoints: `train.sh`, ablation brackets, eval scripts. |
| `logs/` | Training logs and eval transcripts (tracked; cited by the paper). |
| `paper/` | The arXiv paper (LaTeX + PDF). |
| `legacy/phase1/` | Phase 1: "BitThought" proof-of-concept (original implementation). |
| `legacy/phase0/` | Phase 0: earliest thought-vectors experiment scaffold. |
| `legacy/phase0/RESEARCH.md` | Phase 0/1 research log (original, kept for reference). |
| `RESEARCH_LOG.md` | Full research log (Phase 2 per-experiment + absorbed cross-phase summary). |

---

## Training Pipeline (Phase 2, current)

**Single-phase training** (replacing the original's three-phase schedule):

1. Train codec (m5_frontier) with blended k-sampling from step 0 +
   joint loss predictor. 12h on RX 6700 XT 12GB.
2. Freeze codec weights. Train thinker (m4_frontier) on dialogue data with
   WTA multi-hypothesis loss. 12h.

### Loss terms

| Term | Weight | Function |
|------|--------|----------|
| L_recon | 1.0 | Hybrid CE/MSE reconstruction |
| λ_len | 0.005 | Length penalty (k vectors used) |
| λ_pred | 1.0 | Loss predictor calibration |
| λ_contrast | 0.0–0.5 | STSB contrastive alignment |
| λ_repeat | 0.05 | Anti-repetition penalty |

### Training stages (legacy Phase 1)

| Stage | Data | Duration | Purpose |
|-------|------|----------|---------|
| STSB | 5.7K pairs | ~2 min | Learn basic structure |
| SNLI | 700K pairs | ~7 min/epoch | Scale + sharpen decoding |
| C4 | 8M texts | ~8h/epoch | Generalize to real text |

Current Phase 2 trains on a merged mix of C4 paragraphs + minipile (seq 256)
directly in a single phase, which produces a strictly better latent space
than the sequential fine-tune approach.

---

## Running

```bash
# Setup
scripts/setup_env.sh                    # venv + ROCm torch

# Train codec
scripts/train.sh --config configs/m5_frontier.yaml

# Train thinker
scripts/train.sh --config configs/m4_thinker.yaml \
  --init-from checkpoints/m5_frontier/best.pt

# Eval
.venv/bin/tv-eval --ckpt checkpoints/m5_frontier/best.pt --shard data/c4_500k_val

# Chat
.venv/bin/tv-chat --codec checkpoints/m5_frontier/best.pt \
  --thinker checkpoints/m4_frontier/best.pt
```

---

## Key Lessons

- **GRU > self-attention** for thought generation — the sequential bias is
  essential for prefix-based compression to work.
- **d384 5+5** is the optimal loss-per-hour point for the RX 6700 XT,
  not d256 or d512 as earlier experiments suggested.
- **Per-sample k training** (each sample uses its own ratio per step) is the
  single most impactful change in Phase 2 — CE at aggressive compression
  drops 2-3x.
- **Data quality trumps architecture** at this scale — SODA's formulaic
  dialogue patterns are the current quality ceiling for the thinker, not the
  model architecture itself.
- **d_model is the dominant bottleneck** for long-text quality — the 384-dim
  vectors represent a hard capacity ceiling; further scaling requires
  architectural changes (deeper, wider, or hierarchical).
