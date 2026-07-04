# Research Log — Thought Vectors

> **Procedure:** Every experiment — whether it succeeds or fails — is documented
> below in a dated entry.  Failed ideas are kept alongside successful ones so
> we don't repeat dead ends.

## Overview

A PyTorch implementation of a Thought Vector System: encode text into a
variable-length set of continuous thought vectors, then decode back to text
(or a different target).  The core hypothesis is that learned thought vectors
can serve as a compressed, semantically-grounded intermediate representation
for downstream tasks.

**Key insight:** The bottleneck is at **decode time**, not encode time.  The
encoder produces N vectors, but the decoder only receives k << N.  This means
the encoder can freely distribute information across many slots, while the
decoder must reconstruct from just a prefix — forcing the encoder to pack the
most important information into the earliest vectors.

---

## Architecture

### Encoder (text → thought vectors)
- Token embedding + positional encoding → Transformer encoder → GRU over learned
  thought seeds → cross-attention to encoded text → LayerNorm → thought vectors
- The GRU provides a natural sequential hierarchy: earlier vectors carry primary
  information, later vectors refine it.  This ordering is critical for the
  prefix-truncation compression strategy.

### Decoder (thought vectors → token logits)
- Token embedding + positional encoding → Transformer decoder (causal mask)
  with cross-attention to thought vectors → LM head
- Thought dropout (default 0.1, annealed from 0.5 during initial pretraining)
  prevents posterior collapse by forcing the decoder to read the thought vectors.

### Weight tying
- Encoder and decoder share a single token embedding table.
- Decoder LM head weight is tied to the shared embedding (standard practice).

---

## Experiments

### 2025-05-17: GRU vs. Bidirectional Self-Attention

**Hypothesis:** Replacing the GRU with bidirectional self-attention over thought
seeds would remove the GRU's sequential bias and produce better representations.

**Result: FAILED.** The self-attention model failed to converge on C4 —
reconstructions degenerated to repetitive generic phrases regardless of input.
Reverting to the GRU immediately restored convergence.

**Finding:** The GRU's sequential inductive bias is essential.  It forces a
natural information hierarchy across thought slots that bidirectional
self-attention couldn't replicate.  The prefix-truncation strategy relies on
earlier vectors being independently informative.

---

### 2025-05-17: d_model Scaling

Tested d_model = {768, 512, 256} on 100K C4 texts with 32 thoughts, 6 encoder
+ 6 decoder layers, 10 epochs:

| d_model | Params | Final loss | Batches/s | Notes |
|---------|--------|------------|-----------|-------|
| 768     | 123M   | 6.12       | 7.7       | Plateaued early, underfit |
| 512     | 55M    | 4.08       | 10.4      | Better, but still plateauing |
| 256     | 22M    | **1.99**   | **15.6**  | Best loss, fastest training |

**Finding:** Smaller models converged faster per batch.

---

### 2025-05-17: Encoder/Decoder Layer Balance

Tested layers = {6+6, 4+4} with 8 heads on 500K C4 texts, 10 epochs:

| Layers | Heads | Final loss | Batches/s |
|--------|-------|------------|-----------|
| 6 enc + 6 dec | 8 | 1.99 | 15.6 |
| **4 enc + 4 dec** | **4** | **0.62** | **21.9** |

**Finding:** The 4+4 model not only trained 40% faster but reached a dramatically
lower loss.  The deeper model was over-parameterised for 500K texts.

---

### 2025-05-18: Baseline C4 Reconstruction (proven)

**Config:** d_model=256, 4+4 layers, 4 heads, 32 vectors, SentencePiece 16K
vocab, 500K C4 texts, 10 epochs.

**Final loss:** 0.62 (94% below random baseline of 9.7).

**Reconstruction quality:**
- Short sentences (<20 tokens): 100% verbatim
- Medium sentences (20-50 tokens): ~95% word overlap
- Named entities preserved, sentence structure intact

**Checkpoint:** `artifacts/c4_256d_4x4_10ep_C4.pt`

---

### 2025-05-19 — Failed: Dynamic Search-Based Compression

**Hypothesis:** A `find_minimum_vectors_for_target` search over prefix lengths
with a dynamic loss target would teach the model to compress.

**Result: FAILED.** The search is expensive (dropping GPU utilization to ~40%),
the loss target bootstrapping causes NaN gradients, and vector selection was
too inconsistent for the model to learn from.

**Lesson:** Learned search is circular (need good search to train model, need
good model to run search).  A fixed schedule avoids this entirely.

---

### 2025-05-20: Three-Phase Compression Training

**Hypothesis:** Separating capacity-randomisation, predictor learning, and joint
optimisation into distinct phases avoids the circular dependency of online
search.

**Phase 1 — Random-Capacity Pretraining (10-15K steps):**
- Each batch: randomly choose k ∈ [4, max_thoughts] uniformly
- At low k (<8): detach encoder, decoder-only training
- At medium k (<12): add coherence weighting
- At high k: standard cross-entropy

Forces the decoder to handle any prefix length.

**Phase 2 — Predictor Training (5-8K steps):**
- Freeze encoder + decoder
- Small MLP (d_model → d_model/2 → max_thoughts) predicts reconstruction loss
  for each possible prefix length
- Sample random k values, compute actual loss, train predictor with MSE

**Phase 3 — Joint Training (20-30K steps):**
- 60% predictor selects minimal k for target loss 0.25–2.5 (train model at that k)
- 40% sample random k and train predictor

---

### 2025-05-20: 32-Thought Compression (first successful demonstration)

**Base:** 32-thought C4 autoencoder (d_model=256, 4+4 layers, loss=0.62).

**Training:** 10K Phase 1 + 5K Phase 2 + 20K Phase 3 on 500K C4 texts.
**Target:** max_vectors=24 (aggressive compression cap).

**Results:**
- Short texts (10-25 tokens): **5-8 vectors** sufficient for reconstruction.
  Compression ratio: ~4-8×.
- Simple sentences preserve core content at very low k (2-4 vectors).
- Complex sentences with named entities need more vectors (10-15).
- Word order degrades at high compression — the "bag of words" problem.

**Checkpoint:** `artifacts/compressed.pt`

---

### 2025-05-20: 128-Thought Compression (scaling up)

**Observation from 32-thought:** 32 vectors are barely enough for medium texts
(30-50 tokens) — at k=16 the 32-thought model garbles named entities that the
uncompressed 32-vector model reproduces perfectly.

**Approach:** Expand thought capacity from 32 to 128.  Same d_model=256, 4+4
layers architecture.  Weight-transfer from the C4 autoencoder for all shared
parameters (138/139 tensors transferred; thought_seed is new).

**Training:** 15K Phase 1 + 8K Phase 2 + 30K Phase 3 on 500K C4 texts.
**Target:** max_vectors=128 (full capacity during training).

**Results (tested on 62-token news article):**
| k | Ratio | Reconstruction quality |
|---|-------|----------------------|
| 4 | 15:1 | Fragments but key entities visible: "Potter ... Daniel ... $41.1 million" |
| 8 | 7:1 | Most content words present, word order scrambled |
| 12 | 5:1 | Near-complete content, some word order errors |
| **16** | **3:1** | **Perfect reconstruction with correct named entities** |
| 24+ | 2:1 | Identical to input |

**Key finding:** 16 vectors from a 128-thought model reconstruct a 62-token
text with correct named entities — something impossible with 32-thought model
at any k (even k=32 garbled entities).  The extra thought slots during encoding
allow the GRU to distribute fine-grained information across early positions
that can later be selected by the prefix.

**Comparison to 32-thought model (same 62-token text):**
- 32-thought at k=32: "LONDON, England (Reuters) -- Harry Potter star [...] gains access to £20 million" — correct up to "$41.1 million" which gets garbled
- 128-thought at k=16: Perfect reconstruction with everything correct
- 128-thought at k=8: Already gets "Harry" and "Daniel" and "£20 million"

**Checkpoint:** `artifacts/compressed_128t.pt`

---

### 2025-05-22: 100+ Token Compression Ceiling

**Test:** Evaluated the 128-thought compression model (compressed_128t.pt) on
C4 texts of 100-150 tokens with varying prefix sizes.

**Results:**

| Text length | k=8 | k=16 | k=24 | k=48 | k=64 |
|-------------|-----|------|------|------|------|
| 62 tokens   | 7:1 scrambled | 3:1 **perfect** | 2:1 perfect | — | — |
| 103 tokens  | 41% word overlap | 56% word overlap | 59% | 54% | 58% |
| 111 tokens  | 57% | 55% | 62% | 58% | 62% |
| 99 tokens   | 50% | 52% | 50% | 55% | 57% |

**Finding:** For texts up to ~60 tokens, 16-24 vectors suffice for near-perfect
reconstruction.  Beyond ~80 tokens, word overlap plateaus at 50-62% regardless
of k.  This is a **d_model capacity ceiling**, not a vector-count limit — each
vector has only 256 dimensions, which caps the total information that can be
carried regardless of how many vectors are passed.

---

### 2025-05-22: V1 vs V2 Architecture Comparison (GRU wiring)

**Hypothesis from user:** Maybe the GRU's hidden-state sequence is decaying and
we should wire it differently — pull encoder information *into* the GRU via
cross-attention first, then let the GRU process content-rich states instead of
learned seeds:

- **V1 (current):** learned seeds → GRU → cross-attn(encoder) → norm
- **V2 (proposed):** learned queries → cross-attn(encoder, pool to K vectors)
  → GRU → norm → decoder

Both with d_model=256, 4+4 layers, 4 heads, 128 thoughts, 16.5M params,
AdamW(lr=3e-4).  Trained on 200K C4 texts, 1000 steps.

**Results:**
| Variant | avg(901-1000) | Time |
|---------|---------------|------|
| V1 (current) | **9.20** | 52.9s |
| V2 (proposed) | 9.65 | 52.1s |

**V1 wins by ~5%.**  The current design is genuinely better.

**Why:** The GRU's job in V1 is to build a **structure scaffold** — a clean
ordered hierarchy where slot 1 carries the most information and slot 128 the
least.  It does this from learned seeds *before* seeing encoder content.
The cross-attention then fills each slot with targeted encoder information.

In V2, cross-attention first produces K vectors all *equally* derived from
encoder states (each learned query aggregates from all encoded tokens).
The GRU then tries to impose hierarchy retroactively, but the information
is already spread evenly.  The recurrence can't easily compress already-diffuse
content into a clean ordering.

**Conclusion:** The GRU → cross-attn ordering is optimal for the thought-vector
architecture.  The bottleneck remains d_model capacity, not GRU wiring.

---

### 2025-05-22: CNN Domain Adaptation / Mixed C4+CNN

**CNN adaptation:** Phase 1 only (10K steps) starting from compressed_128t on
42K CNN/DailyMail articles under 512 tokens.  **Completed.**
`artifacts/compressed_128t_cnn.pt`

**Mixed C4+CNN:** Phase 1 only (15K steps) on 542K texts (500K C4 + 42K CNN).
**Completed.**  `artifacts/compressed_128t_mixed.pt`

**Long-text adaptation:** Phase 1 only (10K steps) on 47K C4 texts with 32-128
tokens.  **Completed.**  `artifacts/compressed_128t_long.pt`

**Evaluation note:** All three adaptations converged but showed no significant
improvement over the base `compressed_128t.pt` in the 50-62% word overlap
range for long texts.  The d_model ceiling is the dominant constraint, not
domain-specific training data.

---

### 2025-05-24: VAE Regularisation + 256-Thought Compression

**Motivation:** The thought-vector space had no smoothness constraints —
similar texts could produce arbitrarily distant vectors.  A downstream model
needs a well-behaved space where interpolation works and every point produces
coherent output.

**Implementation:** Added VAE projection layers (`mu_proj`, `logvar_proj`) to
`ThoughtEncoder`.  `forward()` returns deterministic `mu` (backward compatible);
`encode_with_kl()` returns a sampled `z` with KL divergence toward N(0,1).

Also added during training:
- **Noise perturbation** (std=0.05) — Gaussian noise on thought vectors.
- **Mixup interpolation** (p=0.1) — convex blends of different batch items.

**Training:**
1. **VAE warmup** (1 epoch, 18750 steps): started from `compressed_128t.pt`
   (138/144 weights transferred).  `kl_beta=0.01`, `noise_std=0.05`,
   `mixup_alpha=0.1`.  Final loss: **0.5992** (19.8 min).
   `artifacts/vae_warmup.pt`
2. **Three-phase compression** (10K+5K+20K steps): standard random-capacity
   pretraining → predictor → joint training on 478K C4 texts.
   Final Phase 3: **k=8 at recon=0.046**.  `artifacts/vae_compressed.pt`

**Results on short texts (6-13 tokens):**
| Text | k=2 | k=4 | k=8 |
|---|---|---|---|
| "a plane is taking off." | **Perfect** | ✓ | ✓ |
| "the sky is dark at night." | scrambled | **Perfect** | ✓ |
| "The stock market crashed..." | scrambled | **Perfect** | ✓ |

**Results on long texts (49-188 tokens) — word overlap:**
| Length | k=4 | k=8 | k=16 | k=32 | k=64 | Ceiling |
|---|---|---|---|---|---|---|
| 49 tok | 50% | 73% | 81% | **96%** | 92% | ~k=32 |
| 69 tok | 62% | 77% | 85% | 89% | 85% | ~k=32 |
| 96 tok | 13% | 31% | 44% | 49% | **51%** | ~k=32 |
| 145 tok | 28% | 29% | 37% | 42% | **48%** | ~k=32 |

**Finding:** VAE costs no compression quality — the plateau at k=16-32 is
identical to the non-VAE model.  For ~50-token texts, k=32 achieves 96%
word overlap.  For 100+ tokens, the d_model=256 ceiling still binds.

**Checkpoints:** `artifacts/vae_warmup.pt`, `artifacts/vae_compressed.pt`

---



## Discarded: Hierarchical Encoder

A multi-scale encoder (coarse 16 + medium 64 + fine 176 vectors) was implemented
and tested.  Result: **FAILED.** From-scratch training with VAE caused posterior
collapse (the decoder learned to generate generic "the the the" patterns).
Hierarchical was abandoned in favour of the proven flat GRU architecture.

## Discarded: Shallow-Wide GRU

A shallow-wide GRU (32 steps × 8 expansion → 256 vectors) was tested as an
alternative to the deep-narrow GRU (256 steps).  The goal was to reduce
information decay through the recurrence.

**Autoregressive training from scratch:** Loss 2.94 in 1 epoch — faster
convergence than the deep-narrow GRU at the same step count (~50% faster
batches/s).  But reconstruction quality was worse for long texts — the 32×
expansion creates redundancy across vector groups.

**Non-autoregressive training from AR checkpoint:** Loss 4.15 after 1 epoch
NAR — significantly worse than the deep-narrow NAR (loss 0.08).  The shallow-
wide GRU's vector redundancy makes per-position independent decoding harder.
The model collapsed into repetitive token generation ("plane plane plane...").

**Verdict:** The deep-narrow GRU (256 steps) is superior for compression.
Each of the 256 vectors carries unique information from its own GRU step,
which is essential for both prefix-slicing compression and NAR decoding.
The shallow-wide GRU's faster speed doesn't outweigh its quality loss.

## 2025-05-25: Non-Autoregressive Decoder Training

**Motivation:** The autoregressive decoder can cheat by reading previous tokens,
reducing the pressure on thought vectors to carry per-position information.
Training with non-autoregressive reconstruction (blank decoder inputs, no causal
masking) forces the decoder to predict ALL tokens from thought vectors alone,
producing richer per-position vectors.

**Implementation:** Added `causal` flag to `ThoughtDecoder.forward()` (default
`True` for backward compat).  When `False`, the causal mask is removed and
the decoder sees blank (pad) tokens — only positional encoding + thought
vectors are available.

**Training:** Started from `vae_compressed.pt` (autoregressive VAE checkpoint),
continued with `--non-autoregressive-recon` for 1 epoch (18750 steps, 21 min).

| Metric | Before (AR) | After (NAR) |
|---|---|---|
| Final loss (AR recon) | 0.60 | 0.08 |
| Training style | Autoregressive | Non-autoregressive |
| Requires causal mask | Yes | No |
| Uses token history | Yes | No |

**Reconstruction quality comparison (autoregressive decode, 1.5× ratio):**

| Length | VAE (no NAR training) | NAR-trained | Delta |
|---|---|---|---|
| 6 tokens | Perfect | Perfect | 0% |
| 12 tokens | Perfect | Perfect | 0% |
| 62 tokens | "0 million ($411 million)1 bill" | "£20 million ($4141 million)" | Better entities |
| 79 tokens | 73% word overlap | **81%** | **+8%** |
| 122 tokens | 50% word overlap | **63%** | **+13%** |
| 160 tokens | 46% | 46% | 0% |

**Key finding:** NAR training pushes the d_model=256 ceiling higher — +13%
word overlap at 122 tokens is a real improvement.  The thought vectors carry
richer per-position information because the decoder can't cheat with token
history during training.  The >160-token ceiling remains unchallenged.

**Checkpoint:** `artifacts/vae_nonar.pt`, `artifacts/vae_compressed_nar.pt`

---

### 2025-05-25: Blended Compression Training

**Motivation:** The three-phase compression training uses uniform random k during
Phase 1.  Blending uniform with a compression-biased distribution improves
performance at higher compression ratios (3:1-5:1) while maintaining robustness.

**50/50 blend:** Half of Phase 1 batches use the original uniform k sampling
(`rnd.randint(4, N)`).  The other half sample k based on the text length and a
distribution skewed toward aggressive compression:
- 30% at ratio 0.4-0.6 (2:1 zone)
- 30% at ratio 0.25-0.4 (3:1 zone)
- 20% at ratio 0.15-0.25 (4:1+ zone, pushing limits)
- 20% at ratio 0.6-1.5 (easy zone, stability)

**Results vs original (vae_compressed_nar.pt):**
- 79 tokens 6:1 compression: **blend 80% vs orig 61%** (+19%)
- Medical text 3:1: **blend 83% vs orig 67%** (+16%)
- Shakespeare 4:1: **blend 89% vs orig 67%** (+22%)
- Short text 3:1: orig **100%** vs blend 80%
- Spanish 2:1: orig **100%** vs blend 78%

The blend gains aggression at 3:1-5:1 compression but loses some 2:1 reliability
on edge cases.  Choose based on whether perfect 2:1 or aggressive compression
matters more.

**Checkpoints:** `artifacts/vae_compressed_blend.pt`

### Loss-to-Quality Mapping

Empirical mapping from cross-entropy loss to reconstruction quality:

| Loss range | Quality | Notes |
|---|---|---|
| < 0.01 | **Perfect** | Verbatim match, all tokens correct |
| 0.01–0.6 | **Excellent** | 90-100% word overlap, minor entity errors |
| 0.6–1.0 | **Good** | 60-90% overlap, structure intact, specifics degrade |
| 1.0–2.0 | **Poor** | 40-60% overlap, losing content words |
| > 2.0 | **Very poor** | < 40% overlap, only gist survives |

The mapping shifts with text length — loss 0.6 on a 6-token text is terrible,
but on a 47-token text it is 92% overlap.  A **relative threshold** is more
consistent: `loss_at_k / loss_at_full_vectors`.  A ratio ≤ 1.2 indicates
near-perfect reconstruction regardless of text length.

### Long-Text Predictor Training

The loss predictor (from three-phase compression) was originally trained on
short C4 texts (~30 tokens mean).  It underestimated model capability on longer
sequences (80-160 tokens), failing to find any k for target losses < 1.0.

Phase 2 was re-run on 47K long C4 texts (32-128 tokens) starting from the
blend checkpoint.  The re-trained predictor improved calibration for texts up
to ~60 tokens but remains pessimistic for 80+ token texts at aggressive targets.
A relative-loss threshold (vs hardcoded absolute targets) is recommended for
long-text use.

---

## Thinker: Latent-Space Conversational Model

The thinker is a transformer that operates entirely in thought-vector space,
bridging the encoder (compression) and decoder (generation) for downstream tasks
like conversation.

### Architecture

```
user text → [frozen encoder] → 256 vectors → k-slice → [thinker] → k vectors → [frozen decoder] → response
```

- **Thinker:** 6-layer `nn.TransformerEncoder`, 256 dim, 4 heads, 7.9M params
- **Encoder:** frozen from `vae_compressed_blend.pt` compression checkpoint
- **Decoder:** frozen after conversation fine-tuning
- **LossPredictor:** built in, selects optimal k at inference time via
  `pred_loss[k] ≤ pred_loss[256] × 1.3` (within 30% of full quality)

### Training (staged)

**Stage 0 — Decoder fine-tuning:**
- Freeze encoder + thinker (thinker bypassed via k=-1)
- Unfreeze decoder, train on conversation pairs
- Teaches decoder to generate conversational responses from thought vectors

**Stage 1 — Thinker training:**
- Freeze encoder + decoder (from Stage 0)
- Train thinker with random k-slicing (k ∈ [4, 256])
- Thinker learns to transform user-message vectors into response vectors

### Data Scaling

| Dataset | Source | Pairs | User tok | Asst tok |
|---|---|---|---|---|
| LMSYS Arena | Real user-model convos | 32K | 26 mean | 35 mean |
| UltraChat 200K | Synthetic instruct convos | 35K | 26 mean | 45 mean |
| OpenAssistant | Human-human convos | 13K | 21 mean | 32 mean |
| ORCA DPO | QA pairs | 2K | 26 mean | 26 mean |
| **Total** | | **104K** | **26 mean** | **35 mean** |

All pairs filtered to ≤120 tokens, refusal patterns removed.

### Training Progression

| Model | Data | Stage 0 loss | Stage 1 loss |
|---|---|---|---|
| thinker_conv_s1 | 10K UltraChat | 6.94 | ~7.0 |
| thinker_conv2_s1 | 19K UltraChat+OASST | 5.36 | ~5.2 |
| thinker_conv3_s1 | 28K LMSYS+OASST | 4.72 | ~4.5 |
| **thinker_big_s1** | **104K all sources** | **3.41** | **TBD** |

### Response Quality (thinker_conv3_s1)

| Input | Response |
|---|---|
| "hello" | "Hello! How can I assist you today?" |
| "how are you?" | "I'm doing well, thanks for asking!" |
| "tell me a joke" | "Why don't scientists trust atoms? Because they make up everything." |
| "what is the capital of france?" | "The capital of France is Paris." |
| "i like dogs" | "I'm sorry, but I cannot fulfill that request." |

Genuine responses for most inputs; some refusal patterns remain from safety
training data.  More data and training directly improves quality.

### Multi-Turn Memory

The chat server maintains conversation history as cached thought vectors:
- Each user input + assistant response is encoded and stored
- All past turn vectors are concatenated and fed to the thinker
- The thinker attends to the full conversation context
- Decoder generates only from the current turn's segment

Limited to ~10 turns to prevent memory growth.

### Checkpoints

| File | Description |
|---|---|
| `thinker_big_s1.pt` | Full pipeline with 104K-trained thinker (best) |
| `thinker_conv3_s1.pt` | 28K-trained thinker (good quality) |
| `thinker_s0.pt` / `thinker_s1.pt` | CNN/DM summarisation test |

### Web Interface

`chat_server.py` provides a REST API and HTML chat UI:
- Auto k-selection via predictor (relative threshold)
- Multi-turn memory
- Compression ratio display
- Clear/reset conversation

---

### 2025-05-29/30: Three-Phase Thinker Training with Multi-Turn Context

**Hypothesis:** A three-phase training pipeline — (1) domain-adapt the encoder+decoder
via reconstruction, (2) train the thinker with multi-turn synthetic contexts, (3) full
fine-tune — produces a better conversational model than single-pair training.

**Architecture changes:**
- **Turn embedding** (learned, max 4 turns): marks which conversation turn each
  thought vector belongs to, so the thinker can distinguish early vs recent context.
- **Speaker embedding** (learned, 2 roles): marks user (0) vs assistant (1) so the
  thinker can distinguish roles.
- **Decode-target embedding** (learned, single vector): added to the current turn's
  vectors to signal "the decoder will read from these slots."
- **Thinker dropout** (0.1): applied after the thinker, parallel to the decoder's
  thought dropout.
- **Temperature/top-k/top-p decoding** added to `decode_greedy()` (default greedy).

**Data:** 875,760 conversation pairs filtered from Dolly, UltraChat, LMSYS Arena,
No Robots, Alpaca, OpenOrca, Anthropic HH (safe), and LMSYS Chat 1M.  Inputs and
outputs capped at 128 tokens (including BOS/EOS).

**Training (ROCm gfx1031, 12.9 GB VRAM, hipBLASLt fallback to hipblas):**

| Phase | What | Epochs | Batches | LR | Time |
|-------|------|--------|---------|----|------|
| P1 | Encoder+decoder reconstruction fine-tune | 1 | 109K (BS=16) | 1e-5 | ~2.5h |
| P2 | Thinker training (multi-turn, 0-3 past turns) | 3 | 109K/ea (BS=8) | 5e-5 | ~13.5h |
| P3 | Full fine-tune (all weights unlocked) | 1 | 109K (BS=8) | 1e-5 | ~6h |

**NaN issues encountered on ROCm gfx1031:**
- `torch.set_float32_matmul_precision("high")` causes NaN divergence ~20K steps in.
- The pretrained thinker weights (from `thinker_big_s1.pt`) produce NaN on this
  architecture — fresh Xavier init is required.
- The thinker's internal dropout layers produce NaN even in eval mode; disabling
  dropout via `model.thinker.eval()` during `thinker_forward()` resolves this.

**Final checkpoint:** `thinker_three_phase.pt` (94 MB).

**Loss progression:**
- Phase 1: started 5.2, ended ~4.0 (reconstruction on conversation texts)
- Phase 2 epoch 1: started 26.3, avg 5.50 (random thinker init → fast convergence)
- Phase 2 epoch 2: avg ~5.35
- Phase 2 epoch 3: avg 5.09
- Phase 3: held steady around 4.5 avg

**Multi-turn context training:**
Each Phase 2/3 batch constructs synthetic conversation histories by sampling `n_past ∈ [0,3]`
past (user, assistant) pairs from the dataset and concatenating their k-sliced thought
vectors before the current turn's input.  15% of batches flip roles (assistant→user)
so the thinker learns bidirectional mappings.  The full context is fed through the
6-layer thinker; the decoder only reads the current-turn segment.

**Chat server conversation example (live inference):**

```
User:  What is your favorite color?
Asst:  I'm sorry, but I'm not sure what you're asking.
       Can you please provide more context or clarify your question?

User:  How are you doing?
Asst:  I'm doing well, thank you for asking! How about you?

User:  I'm alright
Asst:  I'm just a computer program, so I don't have feelings or
       emotions like a human does. I'm here to help answer any
       questions you have. How can I assist you today?

User:  What is the US capital
Asst:  I'm sorry, but I'm not sure what you mean by "St."
       Can you please provide more context or clarify your question?

User:  What is the capital of the US
Asst:  The capital of France is Paris.
```

**Observations:**
- Conversational flow works — the model maintains multi-turn structure and produces
  relevant responses to greetings and simple queries.
- Residual **refusal patterns** remain from safety training data (the extract script
  filters obvious hedging but some patterns survive).
- **Factual hallucination** is prominent — the model confuses "US capital" with
  "France capital," likely because the 16K SentencePiece tokenizer treats "US" as
  subword tokens that overlap with unrelated training text, and the d_model=256
  ceiling limits the precision of information the thought vectors can carry.
- The model has learned an **AI self-awareness pattern** ("I'm just a computer
  program, so I don't have feelings...") from the training data.
- The 5-turn memory cap keeps responses coherent within a conversation but causes
  context loss across longer exchanges.

**Checkpoints:** `artifacts/thinker_phase1_v2.pt`, `artifacts/thinker_phase2.pt`,
`artifacts/thinker_three_phase.pt`

---

### 2025-05-31: Cleaner Data Retrain + Loss Improvements

**Motivation:** The previous model relied too heavily on canned responses ("I'm sorry,
but I'm not sure...", "Sure, here's a...").  Two-pronged fix: cleaner training data
and three new loss components that discourage repetitive/templated behaviour.

**Data improvements (`extract_cleaner_data.py`):**
- **Expanded hedge/canned filter** — checks entire response (not just first 40 chars)
  for refusal patterns like "i am a computer program", "as a language model",
  "i'd be happy to", "yes, i can", etc.
- **Templated instruction filtering** — removes classification-style inputs ("write
  a sentence", "pick the option", "add punctuation", "generate a context") and
  their templated outputs ("positive", "negative", "voici...", "the option in line
  with common sense").
- **Removed Anthropic HH** entirely — too safety-heavy, produces hedging patterns
  that bleed into general conversation.
- **Expanded `max_tok` from 80 → 128** to match the training cap.
- **500K pairs** drawn from Dolly (12K), UltraChat (68K), LMSYS Arena (94K), and
  OpenOrca (326K).
- **Input-side filtering** — rejects templated instruction patterns before they
  reach the response filter.

**Loss improvements added to `train_thinker_three_phase.py`:**
- **Repetition penalty (weight 0.1):** penalises predicting the same token as the
  preceding context token.  Directly attacks the "canned loop" problem where the
  model falls into repeating a safe phrase.
- **Cosine diversity loss (weight 0.05):** pushes thought vectors within a sample
  to be different from each other, measured by pairwise cosine similarity.
  Prevents all thought vectors from collapsing to the same information.
- **Noise perturbation (std 0.05):** Gaussian noise added to thought vectors before
  the thinker.  Forces the decoder to be robust to small input variations, reducing
  brittleness that leads to samey responses.

**Inference improvement:**
- `best_k` clamped to `max_k` (64) in the chat server — the thinker was never
  trained on sequences longer than 64 vectors per turn.  Without this clamp, the
  predictor could select `k > 64`, feeding the thinker sequences 4× longer than
  it has ever seen, causing attention dilution and degraded quality.

**Training (v3 run, in progress):**

| Phase | What | Epochs | Batches | LR | Est. Time |
|-------|------|--------|---------|----|-----------|
| P1 | Skipped — reused `thinker_phase1_v2.pt` | — | — | — | — |
| P2 | Thinker training (cleaner data + 3 loss terms) | 3 | 62.5K/ea (BS=8) | 5e-5 | ~7h |
| P3 | Full fine-tune | 1 | 62.5K (BS=8) | 1e-5 | ~3h |

**Loss progression (preliminary):**
- Phase 2 epoch 1: started 21.8 → avg 6.05
- Phase 2 epoch 2: running, trending ~5.5-6.0 with spikes at 4.1-4.4
- Phase 2 epoch 3: pending
- Phase 3: pending

The cleaner filter cuts ~375K pairs from the original 875K, removing the most
templated instruction-following data.  The remaining data is more conversational
in nature.

**Chat server UI update:**
- Added sliders for **temperature** (0–1.5), **top-k** (0–200), **top-p** (0–1.0)
  in the control bar.  Sent with each request; adjustable in real time.
- Default: temperature=0.7, top-k=50, top-p=off.

**Checkpoints:** `artifacts/thinker_phase1_v2.pt`, `artifacts/thinker_phase1_v3.pt`
(pending), `artifacts/thinker_phase2.pt` (v2), `artifacts/thinker_three_phase.pt`
(v2 final, pending v3).

---

> **Note:** A duplicate file `Reasearch.md` (typo) exists at the project root with no content.
> This is the canonical RESEARCH.md.

## Best Configurations

| Task | d_model | Layers | Heads | Thoughts | Vocab | Loss | Checkpoint |
|------|---------|--------|-------|----------|-------|------|------------|
| Reconstruction (C4) | 256 | 4+4 | 4 | 32 | 16K SP | 0.62 | `c4_256d_4x4_10ep_C4.pt` |
| Compression (32-thought) | 256 | 4+4 | 4 | 32→24 max | 16K SP | varies | `compressed.pt` |
| Compression (128-thought) | 256 | 4+4 | 4 | 128→16 min | 16K SP | 0.115 | `compressed_128t.pt` |
| VAE+NAR Compression | 256 | 4+4 | 4 | 256→varies | 16K SP | 0.08 (NAR) | `vae_compressed_nar.pt` |
| Blend Compression | 256 | 4+4 | 4 | 256→varies | 16K SP | 0.09 (blend) | `vae_compressed_blend.pt` |
| Thinker Chat (104K) | 256+6L | 4+4+think | 4 | 256→auto | 16K SP | 3.41 (S0) | `thinker_big_s1.pt` |
| Thinker Chat (three-phase) | 256+6L | 4+4+think | 4 | 256→auto | 16K SP | ~5.09 (P2) | `thinker_three_phase.pt` |
| Thinker Chat (v3, clean data) | 256+6L | 4+4+think | 4 | 256→auto | 16K SP | ~6.05 (P2) | `thinker_three_phase.pt` (v3, pending) |

## Key Insights

1. **GRU > self-attention** for thought vector generation — the sequential
   ordering is essential for prefix-based compression.
2. **GRU → cross-attn ordering is optimal** — V2 (encoder → pool → GRU) was
   tested and V1 (current: GRU → cross-attn) beat it by ~5%.  The GRU builds
   a structure scaffold that cross-attention fills.
3. **d_model=256 is the efficiency sweet spot** for datasets <1M texts.
4. **4+4 layers beats 6+6** at this scale.
5. **128 thoughts >>> 32 thoughts** for compression — the extra capacity allows
   the GRU to encode finer-grained information in early positions that can be
   selected by the prefix strategy.
6. **16 vectors from 128 thoughts ≈ full reconstruction** of a 62-token news
   article (~4:1 compression).  Scales to ~80 tokens.
7. **d_model=256 is the bottleneck for 100+ token texts** — word overlap
   plateaus at 50-62% regardless of vector count.
8. **VAE regularisation costs no compression quality** — while providing a
   smooth, interpolatable latent space.
9. **NAR training pushes the ceiling higher** — non-autoregressive reconstruction
   (+13% word overlap at 122 tokens) forces richer per-position vectors.
10. **Deep-narrow GRU > shallow-wide GRU** — 256 unique steps beat 32×8
    expansion for per-position discriminability.
11. **Decode-time bottleneck** — the encoder produces many vectors, but only a
    prefix is passed to the decoder.  This is the real compression mechanism.
12. **Blended k-sampling improves aggressive compression** — 50/50 uniform +
    skewed distribution gains +19% at 6:1 on 79-token texts vs uniform alone.
13. **Predictor needs length-relative thresholds** — absolute loss targets
    don't scale with text length.  Use loss_at_k / loss_at_full instead.
14. **Pretrained thinker weights can be numerically unstable on AMD ROCm**
    — on gfx1031 with the hipBLASLt fallback, the 6-layer transformer's
    internal dropout layers produce NaN.  A fresh Xavier init avoids this.
15. **Multi-turn context training needs structural metadata** — the thinker
    receives concatenated vectors from different turns with no positional
    signal.  Learned turn embeddings, speaker embeddings, and a decode-target
    embedding give the thinker the explicit structure it needs to distinguish
    early context from the current query.
16. **Synthetic multi-turn training from single-turn pairs works** — by
    sampling past (user, assistant) pairs and concatenating their thought
    vectors, the thinker learns to handle conversation history without
    requiring actual multi-turn conversation data.
17. **d_model=256 constrains conversational factual accuracy** — the chat
    model struggles with factual queries (e.g., confuses US capital with
    France's), suggesting the 256-dim thought vectors can't carry enough
    information for precise factual recall.  Scaling d_model would require
    re-engineering the full architecture.
18. **Data quality trumps model architecture at this scale** — the most
    impactful single change was filtering out templated instruction data
    (OpenOrca classification tasks, "write a sentence" patterns).  The
    thinker architecture and loss components are secondary to what the
    model is trained on.
19. **Repetition penalty, diversity loss, and noise perturbation all target
    the "canned response" problem from different angles** — repetition
    penalty directly penalises token-level loops; diversity loss prevents
    thought-vector collapse; noise perturbation reduces decoder brittleness.
    Together they address the three failure modes that produce samey outputs.
20. **Capping inference k to training max_k prevents attention dilution**
    — the predictor-chosen k was silently producing sequences 4× longer
    than training, causing the thinker to spread attention over too many
    positions and default to safe patterns.  This is a cheap fix (one-line
    clamp) with outsized impact.

