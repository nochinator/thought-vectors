# Thought Vectors (Initial Implementation)

This repository now contains a concrete implementation scaffold for the Thought Vector System described in the specification.

## Included

- `ThoughtEncoder` with embeddings, positional encoding, transformer encoder, GRU thought generator, and cross-attention.
- `ThoughtDecoder` with transformer decoder, causal masking, and vocabulary projection.
- Group-based dataset utilities.
- Training-step loss computation with reconstruction + length penalty.
- Inference helpers for encode, greedy decode, and compression via reconstruction-loss threshold.

## Quick usage

```python
from thought_vectors import ThoughtEncoder, ThoughtDecoder, ThoughtVectorModel

encoder = ThoughtEncoder(vocab_size=50257)
decoder = ThoughtDecoder(vocab_size=50257)
model = ThoughtVectorModel(encoder, decoder)
```

## Notes

This is Phase-1/2 oriented code and ready to be integrated into a full training script and tokenizer setup.
