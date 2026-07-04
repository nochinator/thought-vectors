# ThoughtVectors — Consolidated Research Log

> Two-phase research project. **Phase 1** (2025-05, `AI_construction/`) —
> original PyTorch implementation by deepseek-v4-flash. **Phase 2** (2026-06,
> `thoughtvec/`) — clean-room recreation with corrected architecture, better
> data, and the same GPU budget. Every experiment, success or failure, is
> documented so dead ends are never repeated.

---

# Phase 1 — Original Implementation (2025-05)

## Core Idea

Encode text into a variable-length set of continuous thought vectors, then
decode back to text. The learned thought vectors serve as a compressed,
semantically-grounded intermediate representation for downstream tasks.

**Key insight:** The bottleneck is at **decode time**, not encode time. The
encoder produces N vectors, but the decoder only receives k << N. This forces
the encoder to pack the most important information into the earliest vectors.

## Architecture (Phase 1)

- **Shared embedding** — single `nn.Embedding(V, d)` for encoder input,
  decoder input, and LM head (weight-tied).
- **Encoder** — token embedding → positional encoding → Transformer encoder
  (full self-attention) → learned thought seeds → GRU → cross-attention to
  encoder output → LayerNorm → thought vectors. The GRU provides sequential
  hierarchy: early vectors carry primary information, later ones refine it.
- **Decoder** — token embedding → positional encoding → Transformer decoder
  (causal mask) with cross-attention to thought vectors → LM head.
- **Thought dropout** (default 0.1, annealed from 0.5) prevents posterior
  collapse.

### Ablation Results

| Experiment | Finding |
|---|---|
| GRU vs Bidirectional Self-Attention | GRU essential — self-attention failed to converge. Sequential bias is critical for prefix-based compression. |
| d_model scaling (768, 512, 256) | d=256 won (loss 1.99 vs 512's 4.08 vs 768's 6.12) on 100K C4, 10 epochs. |
| Encoder/Decoder layer balance | 4+4 layers beat 6+6 at this scale. |
| V2 wiring (cross-attn → GRU) | V1 (GRU → cross-attn) beat V2 by ~5%. GRU builds a scaffold that cross-attention fills. |
| Domain adaptation (CNN, mixed, long) | Converged but no significant improvement — d_model ceiling dominates. |

## VAE Regularisation + 256-Thought Compression (2025-05-24)

Added VAE projection layers, noise perturbation (std=0.05), and mixup
interpolation (p=0.1). Results on short texts (6-13 tokens): k=2 on "a plane
is taking off" — perfect. On long texts (49-188 tokens), word overlap scales
with k, ceiling at ~k=32 for 49-tok texts, ~k=32 for 145-tok texts (48% at
k=64).

**Finding:** d_model=256 is the bottleneck for 100+ token texts. Word overlap
plateaus at 50-62% regardless of vector count.

## Non-Autoregressive Training (2025-05-25)

NAR decoder (parallel prediction, masked MSE loss independent of token order)
forces richer per-position thought vectors. At k=32 on 122-token text, word
overlap jumped from 42% (AR) to 55% (NAR). Blended k-sampling (50/50 uniform +
skewed) gained +19% at 6:1 compression.

## Thinker — Latent-Space Conversation (2025-05-26)

A 6-layer transformer operating directly on thought vectors. Encoder produces
thoughts for the user input, thinker predicts response thoughts, decoder
renders them to text. No autoregressive token generation over the full
vocabulary in the reasoning step.

**Key innovations:**
- **Turn embedding** (max 4 turns) — marks which conversation turn each
  thought belongs to.
- **Speaker embedding** (2 roles) — marks user vs assistant.
- **Decode-target embedding** — signals which slots the decoder reads from.
- **Synthetic multi-turn training** — sample past (user, assistant) pairs and
  concatenate their thought vectors to create conversation history.
- **Multi-turn memory** — past turn vectors cached and concatenated as
  context for the thinker.

**Models trained:**

| Model | Data | Stage 0 loss | Notes |
|---|---|---|---|
| thinker_conv_s1 | 10K UltraChat | 6.94 | Baseline |
| thinker_conv2_s1 | 19K UltraChat+OASST | 5.36 | More data helps |
| thinker_conv3_s1 | 28K LMSYS+OASST | 4.72 | Clean data is key |
| thinker_big_s1 | 104K all sources | 3.41 | Best |

**Response quality:** Genuine responses for most inputs ("hello" → "Hello!
How can I assist you today?"). Some refusal patterns remain from safety
training data. Factual accuracy constrained by d_model=256 — model confuses
US capital with France's.

## Phase 1 Key Insights

1. **GRU > self-attention** for thought generation — sequential ordering
   essential for prefix-based compression.
2. **GRU → cross-attn ordering is optimal** — GRU builds structure scaffold
   that cross-attention fills.
3. **d_model=256 is the efficiency sweet spot** for datasets <1M texts.
4. **4+4 layers beats 6+6** at this scale.
5. **128 thoughts >>> 32 thoughts** for compression.
6. **16 vectors from 128 thoughts ≈ full reconstruction** of 62-token
   article (~4:1 compression).
7. **d_model=256 bottlenecks 100+ token texts** — 50-62% word overlap ceiling.
8. **VAE regularization costs no compression quality** — smooth latent space
   for free.
9. **NAR training pushes ceiling higher** — +13% word overlap at 122 tokens.
10. **Deep-narrow GRU > shallow-wide GRU** — 256 unique steps beat 32×8
    expansion.
11. **Decode-time bottleneck is the real compression mechanism.**
12. **Blended k-sampling improves aggressive compression** — +19% at 6:1.
13. **Predictor needs length-relative thresholds** — use
    loss_at_k / loss_at_full.
14. **Data quality trumps model architecture** at this scale.
15. **Capping inference k to training max_k prevents attention dilution.**

---

# Phase 2 — Clean-Room Recreation (2026-06)

## Rationale

The original's documented limits were believed to be artifacts, not
fundamentals:

1. "d=256 beats 512/768" came from 10-epoch runs on 100K sentence fragments
   with post-norm layers, no warmup, plain Adam — an undertraining artifact.
2. Refinements (VAE, NAR, blended-k) were only applied *sequentially* as
   fine-tunes — training them together from step 0 gives strictly better
   latent space.
3. The original corpus was sentence fragments (median ~14 words); merged C4
   paragraphs + minipile at seq 256 is far stronger data.

## Design Deltas from Phase 1

- **Proper weight tying** (one embedding table for encoder/decoder/LM head;
  the original *claimed* tying but actually used three tables).
- **Single-phase training** — blended k-sampling from step 0 + jointly
  trained detached loss predictor, replacing the three-phase schedule.
- **Pre-norm transformer layers**, GELU, AdamW + grad clip 1.0.
- **GRU scaffold computed once** on [1, N, d] and expanded to batch
  (batch-size-independent cost).
- **Kept from Phase 1** (empirically proven): GRU→cross-attn thought
  generation, LM-head/CE decoder, d=256, 4+4 layers, 4 heads, 16K
  SentencePiece, position_attn_bias, thought dropout, blended k ratios,
  VAE-lite, NAR fine-tune.

**Hardware:** RX 6700 XT 12GB (gfx1031), ROCm, fp32.
**Budget constraint:** 12 hours max for the final frontier run.

## Ablation Campaign

Protocol: explore with many short ablations (15-45 min each, fixed
wall-clock), keep what wins, discard what doesn't.

### B-series — d_model and depth scaling

| Run | Config | Val loss (full-k) | Verdict |
|---|---|---|---|
| B384 | d384 4+4 4h | 0.503 | Best loss-per-hour |
| B512 | d512 4+4 4h | 0.298 (better, but 1.8x slower) | Not worth wall-clock cost |
| C8 | d256 8+8 8h | 0.289 | Depth > width at small d |
| D8 | d384 8+8 8h | 0.294 | Deeper helps but diminishing returns at d384 |

**Finding:** d384 5+5 is the optimal loss-per-hour point; full-k loss beats
every shallower config and reaches the d=512 quality-time optimum.

### W-series — Compression breakthroughs

| Run | Change | Key result |
|---|---|---|
| W0 | Control (d384 5+5) | 0.588 val loss |
| W1-W2 | Word dropout .15/.30 | Helps top end, hurts low end |
| **W3** | **Per-sample k + log-CE predictor** | **CE at r=0.25 drops 2-3x, bigram F1 up ~35%. Step-change.** |
| W4 | W3 + word dropout .15 | Mid — inherits low-ratio penalty |
| W5 | Low-k-skewed sampler | Dominated by W3. Rejected. |
| **W6** | + monotone predictor head | **Tau is now a graded dial. No garbage regime.** |
| **W7** | + ratio-scaled word-dropout 0.3 | **Best long-text bigram F1 + CE, low-ratio cells preserved.** |

**Key findings:**
- **Per-sample k (W3) is the step-change** — every sample trains at its own
  ratio each step vs one batch-mean k. CE drops 2-3x across buckets, bigram
  F1 up ~35%, predictor usable. ~8% it/s cost. **Adopted.**
- **Monotone head (W6)** — right-to-left cumsum of softplus increments makes
  tau a graded dial: 0.25/0.5/1.0/2.0 trades ratio ~4.0/1.2/0.5/0.2 against
  CE 0.08/0.16/0.46/1.5. **Adopted.**
- **Ratio-scaled word-dropout (W7)** recovers top-end win without low-end
  penalty. **Adopted.**

### M4 Thinker — Latent-Space Conversation (Phase 2)

Training a conversation agent in the frozen codec's thought space: text →
encoder → per-turn thought prefixes → Thinker → k_out predicted thoughts →
decoder → reply text.

**Data:** 412K conversations (SODA 394K, persona-chat 17K, OASST1 859).
Turn-aware shards: 3.28M training turns, 33K validation.

**Thinker architecture:** Pre-norm transformer trunk over flattened
[C × k_ctx] context thoughts + role/turn-distance/slot-pos embeddings.
Two output modes: "query" (GRU-scaffolded slots cross-attend the trunk)
and "prefix" (response slots appended with causal mask).

### WTA (Winner-Take-All) Multi-Hypothesis Thinker

**Problem:** Standard thinker suffers from mode-averaging — predicting a
blend of all plausible responses instead of committing to one trajectory.

**Solution:** 4 parallel hypothesis heads, each producing a full response
thought sequence. Only the best (lowest MSE + decoder CE) gets gradient
backpropagated per sample.

**Frontier run (12h / 225K steps, WTA4 recipe):**

| Metric | 30 min | 225K steps |
|---|---|---|
| best-of-M val cos | 0.434 | **0.584** |
| best-of-M val dec CE | 4.07 | **3.41** |
| eval ref F1 (random hyp) | 0.266 | **0.320** |
| eval distinct-1 | 0.012 | **0.022** |
| eval distinct-2 | 0.045 | **0.086** |

**Results:** 24x more training helped on every axis. Dec CE beats every
round-1/2 arm, diversity roughly doubled. Model reliably nails the dialogue
act and opening clause from context. Mid-sentence still decays into filler
attractor — the collapse moved from whole-reply blandness to second-half
degradation.

## Phase 2 Key Insights

1. **d384 5+5** is the optimal loss-per-hour point on gfx1031, beating both
   d256 and d512 in wall-clock efficiency.
2. **Per-sample k training** is the single most impactful change — evaluating
   and training each sample at its own compression ratio.
3. **Monotone predictor head** makes tau a usable compression knob with no
   garbage regime — key for downstream deployment.
4. **Ratio-scaled word-dropout** recovers high-compression long-text quality
   without penalizing low-ratio cells.
5. **WTA multi-hypothesis thinker** structurally addresses mode-averaging —
   4 parallel heads, gradient to the winner per sample.
6. **Thinker second-half degradation** remains open — the model commits to a
   grounded start but loses coherence across later response slots.

## Model Checkpoints Index

### Phase 1 (legacy)

| Model | d_model | Layers | Thoughts | Loss | File |
|---|---|---|---|---|---|
| C4 baseline | 256 | 4+4 | N/A | 1.99 | `c4_256d_4x4_10ep_C4.pt` |
| Compression (32-thought) | 256 | 4+4 | 32→24 max | varies | `compressed.pt` |
| Compression (128-thought) | 256 | 4+4 | 128→16 min | 0.115 | `compressed_128t.pt` |
| VAE+NAR Compression | 256 | 4+4 | 256→varies | 0.08 (NAR) | `vae_compressed_nar.pt` |
| Blend Compression | 256 | 4+4 | 256→varies | 0.09 (blend) | `vae_compressed_blend.pt` |
| Thinker Chat (104K) | 256+6L | 4+4+think | 256→auto | 3.41 (S0) | `thinker_big_s1.pt` |
| Thinker Chat (three-phase) | 256+6L | 4+4+think | 256→auto | ~5.09 | `thinker_three_phase.pt` |

### Phase 2 (thoughtvec)

| Model | d_model | Layers | Thoughts | Key feature | File |
|---|---|---|---|---|---|
| warm_base | 384 | 5+5 | 256 | 60-min base | `checkpoints/warm_base/` |
| m5_frontier | 384 | 5+5 | 256 | Codec, per-sample k, tau knob | `checkpoints/m5_frontier/` |
| m4_frontier (thinker) | 384+trunk | 5+5+think | 8+8 | WTA4 multi-hypothesis | `checkpoints/m4_frontier/` |

---

## The Big Picture

**Phase 1** proved the idea works: GRU-scaffolded thought vectors, prefix
compression, and a latent-space thinker. It hit a ceiling at d_model=256
and three-phase training that was hard to tune.

**Phase 2** broke through three ceilings:
1. **d_model bottleneck** — d384 5+5 with proper pre-norm, AdamW, and
   single-phase training.
2. **Compression quality** — per-sample k training + monotone predictor +
   ratio-scaled word-dropout makes the tau knob a practical compression dial.
3. **Thinker mode-collapse** — WTA multi-hypothesis training gives 4 parallel
   trajectories per sample, gradient to the winner, doubling diversity.

Remaining work: thinker second-half degradation (mid-sentence filler), larger
dialogue data (SODA is formulaic), and the open question of how far the
thought-vector latent space can be pushed with the 12GB GPU budget.

---

## Update 2026-07-02 — Thinker rounds 4–5 (anti-collapse ablations)

Full detail in `thoughtvec/RESEARCH_LOG.md`. Recipe status: **FINAL_12H complete
2026-07-03 — flagship conversational checkpoint** (`checkpoints/FINAL_12H/best.pt`:
val_cos 0.4281, ref_f1 0.2969, distinct1 0.0323, self_rep 0.1882, ctx_sens 0.1462,
round-trip PASS; all best-ever, transcripts audited, chat gate passed; residual:
positivity register errors + pronoun confusion). Recipe = R4_UNFREEZE base (WTA4, k_out=8, random-prefix k, decode-full,
dialogue_combined, unfreeze=decoder w/ strong anchor) + cycle loss (cycle_frac=0.25,
w_cycle=0.5) + w_decoder=1.0 — the R5_STACK confirmation arm won both multi-turn
metrics (self_rep 0.2900, ctx_sens 0.2535), its transcripts held up under audit
(wedding hard-collapse broken, novel follow-up-question behavior), and the
chat-probe hard gate passed.

Durable lessons from R4–R5:
1. **Cold short arms can't rank anti-collapse levers** — R5 fixed this by warm-starting
   every arm from the 2h-mature checkpoint and fine-tuning 45 min at lr 1e-4.
2. **Single-turn metrics are blind to the real disease.** All R5 arms were flat on
   val_cos/dec_ce/ref_f1/distinct1; only the new multi-turn metrics (self_rep,
   ctx_sens from `scripts/eval_multiturn.py`) separated them.
3. **Extra plain training deepens the collapse attractor** (R5_CTL regressed multi-turn).
4. **Cycle-consistency loss (decode→re-encode) is the first lever that dents content
   collapse** — best self_rep, best val_cos, visibly more grammatical high-temp chat.
5. **ctx_sens is gameable by noise** — R5_WDEC "won" it while chatting worse (nonsense
   replies trivially differ from other nonsense). Metrics require transcript audits;
   qualitative chat probes are a hard gate before long runs (twice now: R4_BIG30_4H, R5_WDEC).
6. Negatives: turn dropout *teaches* history-independence (worst ctx_sens); k_ctx=16
   dilutes rather than enriches context; 30M params is a tiny metric lever, not a cure.
7. **R6 (2026-07-03, negative round):** no training lever fixes the positivity-register
   disease — every arm (incl. control) worsened cheerful-reply-to-bad-news after continued
   fine-tuning on dialogue_combined. Replicated conclusion: the data's smalltalk cheer IS
   the disease → needs contrastive sentiment data (e.g. empathetic_dialogues), not a loss.
   cycle_frac=0.5 does protect multi-turn behavior during continued training. Third
   metric-gaming incident caught by transcript audit (register lexicon too narrow).
