---
title: ThoughtVectors Chat
emoji: 💭
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: mit
short_description: 48M-param dialogue in latent space — runs in your browser
models:
- nochinator/thought-vectors
---

# ThoughtVectors chat — fully in-browser

Talk to a 48M-parameter model that converses entirely in a learned
thought-vector space — no token-level language modeling happens between
your message and its reply. Trained from scratch in ~25 GPU-hours on one
consumer GPU.

The whole model runs client-side via onnxruntime-web (~135 MB download,
cached after the first visit; nothing you type leaves the page). The ONNX
export is byte-exact with the released `FINAL_12H` checkpoint: 8/8 greedy
replies match the PyTorch reference verbatim.

It does small talk, follow-up questions, and multi-turn reference. It also
has a documented failure mode (cheerful replies to bad news) that the paper
traces to a training-data absence — probing for it is encouraged.

A model picker swaps in the paper's matched token-LM baseline (§6.5 — same
data, tokenizer, and compute) so you can compare both paradigms on the same
conversation, live. It downloads separately (~100 MB, fp16, byte-exact with
the released checkpoint) the first time you select it.

Paper, code, logs: https://github.com/nochinator/thought-vectors
