# BitThought Architecture & Training

## Core Idea

Encoder-decoder transformer that compresses text into dense latent vectors ("thought vectors"), then reconstructs text from them. The model learns a general **compression function** — not just memorizing text, but learning to represent semantic content in a fixed-dimensional latent space.

The latent space is the key output, not the reconstruction. Reconstruction is a training task that forces the encoder to produce useful representations. The real value is the thought vectors themselves, which can be manipulated, searched, composed, and reasoned over by downstream processors.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     SHARED EMBEDDING                         │
│              nn.Embedding(V, d) — 50K × 256                  │
│         Used by: encoder input, decoder input, LM head       │
│         (weight-tied: one matrix, three roles)               │
└─────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐      ┌───────────────────────┐
│    ENCODER      │      │       DECODER          │
│ Transformer × L │      │   Transformer × L      │
│  ┌───────────┐  │      │  ┌─────────────────┐   │
│  │ GRU +     │  │      │  │ Embed predictor │   │
│  │ CrossAttn │  │      │  │ Linear(d, d)    │   │
│  └───────────┘  │      │  └─────────────────┘   │
│       │         │      │         │              │
│  thoughts [N,d] │      │  pred_embeds [T,d]     │
│       │         │      │         │              │
│  loss_preds [N] │      │  embed_to_logits:      │
│  (0-1, sigmoid) │      │  pred_embeds @ W^T    │
└─────────────────┘      └───────────────────────┘
```

### Component Breakdown

#### Shared Embedding
A single `nn.Embedding(V, d)` serves triple duty:
- **Encoder input**: token IDs → embeddings
- **Decoder input**: target token IDs → embeddings
- **Decoder output projection**: predicted embeddings → vocabulary logits (at inference)

This saves 25.7M params (55% of the old model) by eliminating redundant copies. Source = target in autoencoding, so there's no representational loss.

#### Encoder (`BitThoughtEncoder`)
- Token embedding → sinusoidal positional encoding
- Stacked `nn.TransformerEncoderLayer` (L layers, full self-attention)
- Learned thought seed → GRU → cross-attention with encoder output
- LayerNorm → projection to thought dimension
- Optional loss predictor: 2-layer MLP outputting per-vector confidence (sigmoid-bounded [0, 1])

The thought seed is a learned `[1, N, d]` parameter. It's expanded to batch size and passed through a GRU to produce a sequence of N thought vectors. Each attends to the encoder's representation via cross-attention. The GRU ensures sequential coherence — thought vectors are ordered and context-dependent.

#### Decoder (`BitThoughtDecoder`)
- Thought vectors → hidden projection
- Token embedding + positional encoding for target sequence
- Stacked `nn.TransformerDecoderLayer` (L layers, causal-masked self-attention + cross-attention to thought vectors)
- Final LayerNorm → **embedding predictor** (`nn.Linear(d, d)`)

The key innovation: the decoder outputs a token **embedding** (not logits). At inference, this embedding is projected to vocabulary via `embed_to_logits`: `pred_embeds @ shared_embed.weight.T`. This gives the nearest token in embedding space.

#### Loss Predictor
A 2-layer MLP (`Linear → ReLU → Linear → Sigmoid`) that outputs per-vector confidence (0-1). Trained to predict reconstruction quality at each vector position: position i should output the confidence the model would have if only i+1 vectors were available.

Used at inference for adaptive stopping (stop generating vectors when confidence exceeds threshold). Used during training for **predictor-guided truncation**.

---

## Training

### Three-Stage Curriculum

| Stage | Dataset | Size | Purpose |
|-------|---------|------|---------|
| 1. STSB | Semantic Textual Similarity | 5.7K pairs | Learn basic sentence structure + contrastive alignment |
| 2. SNLI | Stanford Natural Language Inference | 700K pairs | Scale to diverse sentence patterns |
| 3. C4 | Common Crawl web text | 8M texts | Generalize to real-world, noisy text |

### Hybrid Reconstruction Loss

Two loss functions, randomly selected per batch with probability `exact_prob`:

```
With probability exact_prob:
  Cross-entropy on projected logits
  pred_embeds @ shared_embed.weight.T → logits [B, T, V]
  F.cross_entropy(logits, target_ids)
  Forces exact token decoding — penalizes wrong tokens directly.

With probability (1 - exact_prob):
  MSE in embedding space
  F.mse_loss(pred_embeds, target_embeds)
  Allows semantic similarity — "dog" ≈ "puppy" is a small error.
```

Typical `exact_prob` = 0.5. The MSE path provides semantic structure; the CE path sharpens exact token predictions. Without the CE path, the model produces embeddings that are close to the target but decode to periods via nearest-neighbor argmax.

### Embedding-Space Loss Details

The MSE path trains the decoder to produce embeddings that match the target token's embedding from the shared embedding matrix. Since `shared_embed.weight` is a learned 50K × 256 matrix where semantically similar tokens have similar vectors:

| Prediction | Target | MSE Loss | Implication |
|-----------|--------|----------|-------------|
| "dog" embed | "dog" embed | 0.0 | Perfect |
| "puppy" embed | "dog" embed | ~0.2 | Close — small penalty |
| "." embed | "dog" embed | ~1.5 | Far — large penalty |

This naturally eliminates the period-loop problem. The model learns to produce anything semantically reasonable rather than collapsing to the most frequent token.

### Force-Compression: Predictor-Guided Truncation

The decoder is trained on variable-length thought vector prefixes. During training:

```
Phase 1 (first 20% of steps):
  Keep all N vectors (warm start).

Phase 2 (remaining 80%):
  Predictor outputs confidence [0, 1] at each vector position.
  Threshold anneals from 0.8 → 0.4 over the phase.
  K = first position where confidence ≥ threshold.
  Decoder sees only K vectors.
```

This ties the training directly to inference behavior: the predictor learns where reconstruction quality is sufficient, and the decoder learns to work with any prefix length. At inference, the same predictor decides when to stop generating vectors.

### Scheduled Sampling

With probability `scheduled_sample_rate` (typically 0.1), the decoder's input is a mixture of ground-truth tokens and the model's own predictions:

```
decoder_input = input_ids[:, :-1]
if random.random() < scheduled_sample_rate:
    pred_embeds = model.decoder(thoughts, decoder_input)
    pred_tokens = model.embed_to_logits(pred_embeds).argmax(dim=-1)
    mask = torch.rand_like(decoder_input) < scheduled_sample_rate
    decoder_input = torch.where(mask, pred_tokens, decoder_input)
```

This bridges the train/test gap: during training the decoder sees ground-truth tokens, but at inference it sees its own predictions. Scheduled sampling exposes it to its own mistakes, teaching recovery.

### Repetition Penalty

A small penalty (typically 0.05) on adjacent repeated tokens:

```
decoded = logits.argmax(dim=-1)
repeats = (decoded[:, 1:] == decoded[:, :-1]).float()
valid = (target[:, 1:] != pad_token_id).float()
repeat_rate = (repeats * valid).sum() / valid.sum()
total_loss += repeat_penalty * repeat_rate
```

This directly penalizes the period-loop and repetition problems. Even a small weight (0.05) significantly reduces degenerate outputs.

### Summary of Loss Terms

```
L_total = L_recon + λ_len * K + λ_pred * L_predictor + λ_contrast * L_contrastive + λ_repeat * L_repeat
```

| Term | Weight | Function |
|------|--------|----------|
| L_recon | 1.0 | Hybrid CE/MSE reconstruction |
| λ_len | 0.005 | Length penalty (K vectors used) |
| λ_pred | 1.0 | Loss predictor calibration |
| λ_contrast | 0.0–0.5 | STSB contrastive alignment |
| λ_repeat | 0.05 | Anti-repetition penalty |

---

## Model Presets

| Preset | Params | d_model | Layers | Heads | Thoughts | Tokenizer |
|--------|--------|---------|--------|-------|----------|-----------|
| Tiny | 0.55M | 64 | 2 | 2 | 8 | Custom BPE (4K) |
| Small | 2.1M | 128 | 3 | 4 | 12 | Custom BPE (4K) |
| Medium | 9.3M | 256 | 4 | 8 | 16 | Custom BPE (4K) |
| Medium+GPT2 | 21.1M | 256 | 4 | 8 | 16 | GPT-2 (50K) |
| Large | 28.4M | 384 | 6 | 8 | 24 | GPT-2 (50K) |

Parameter counts include the tied embedding. Without weight tying, Medium+GPT2 would be 46.8M.

---

## Inference

### Encoding
```python
thoughts, loss_preds = model.encoder(input_ids)
# thoughts: [B, N, d] — all N thought vectors generated
# loss_preds: [B, N] — confidence at each position
```

### Adaptive Stopping
```python
# Find where confidence exceeds threshold
K = (loss_preds > threshold).nonzero()[0].item()
# Use only K vectors for decoding
thoughts = thoughts[:, :K, :]
```

### Decoding
```python
pred_embeds = model.decoder(thoughts, target_ids)  # [B, T, d]
logits = model.embed_to_logits(pred_embeds)        # [B, T, V]
tokens = logits.argmax(dim=-1)                      # [B, T]
```

---

## The Latent Space

The thought vectors are the actual output of the system — not the reconstructed text. Reconstruction is a training task. The latent space enables:

| Operation | Method | Why It's Cheaper |
|-----------|--------|-----------------|
| **Reasoning** | Processor transformer on thought vectors | O(N²) where N=16-64 thoughts vs O(S²) where S=512+ tokens |
| **Search** | Compare thought vectors via cosine similarity | 256-dim comparison vs full text matching |
| **Cross-modal** | Align text and image in same latent space | Shared dimensionality regardless of modality |
| **Composition** | Concatenate, interpolate, or mask thought vectors | Direct manipulation of semantic content |

A processor (transformer layers operating on thought vectors without token embedding/head) would compute self-attention over 16-64 positions instead of 512+ token positions — roughly 10-100× fewer operations for the attention mechanism.

---

## Training Recipe Summary

```
Stage 1: STSB (5.7K pairs, ~2 min)
  exact_prob=0.0, contrastive_weight=0.5
  Learn basic structure + align thought space

Stage 2: SNLI (700K, ~7 min/epoch)
  exact_prob=0.5, repeat_penalty=0.05
  Scale to diverse sentences + sharpen decoding

Stage 3: C4 (8M, ~8 hours/epoch)
  exact_prob=0.5, repeat_penalty=0.05, ss_rate=0.1
  Generalize to real-world text
```

Hardware: AMD RX 6700 XT (12GB), ROCm, batch sizes 16-1024 depending on loss path.
