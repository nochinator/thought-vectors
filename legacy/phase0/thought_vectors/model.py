from __future__ import annotations

import math
import warnings

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_len = self.pe.size(1)
        seq_len = x.size(1)
        if seq_len > max_len:
            warnings.warn(
                f"Input sequence length ({seq_len}) exceeds maximum positional encoding length "
                f"({max_len}). Truncating to {max_len} tokens.",
                stacklevel=2,
            )
            x = x[:, :max_len]
            seq_len = max_len
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ThoughtEncoder(nn.Module):
    """Text -> fixed-count thought vectors."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_thoughts: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_thoughts = num_thoughts

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Thought vector generation: deep-narrow GRU — 256 sequential steps
        # create a natural ordering hierarchy where earlier vectors carry
        # primary information and later vectors refine.  The GRU's sequential
        # structure is essential for prefix-truncation compression.
        self.thought_seed = nn.Parameter(torch.randn(1, num_thoughts, d_model) * 0.02)
        self.thought_gru = nn.GRU(d_model, d_model, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # VAE projection layers: map normalised thought vectors to
        # a Gaussian posterior N(mu, sigma^2).  Dropout-free; the KL
        # regulariser prevents overfitting.
        self.mu_proj = nn.Linear(d_model, d_model)
        self.logvar_proj = nn.Linear(d_model, d_model)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return deterministic thought vectors (mu)."""
        h = self._encode(input_ids, padding_mask)
        return self.mu_proj(h)

    def encode_with_kl(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sampled_z, mu, logvar) for VAE training.

        The sampled z is what the decoder sees during training.  mu and
        logvar are used for the KL regulariser.
        """
        h = self._encode(input_ids, padding_mask)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def _encode(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        max_len = self.positional_encoding.pe.size(1)
        if input_ids.size(1) > max_len:
            input_ids = input_ids[:, :max_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :max_len]
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        batch_size = input_ids.size(0)

        # Deep-narrow GRU: 256 sequential steps create an importance hierarchy.
        # Earlier vectors carry primary information; later vectors refine.
        thoughts = self.thought_seed.expand(batch_size, -1, -1)
        thoughts, _ = self.thought_gru(thoughts)

        attended, _ = self.cross_attention(query=thoughts, key=encoded, value=encoded, key_padding_mask=padding_mask)
        return self.norm(thoughts + attended)


class ThoughtDecoder(nn.Module):
    """Thought vectors + shifted targets -> token logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.thought_dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Learned cross-attention position bias.
        # Decoder position i attending to thought vector j gets a bias of
        # self.position_attn_bias * (i - j), so earlier thought vectors (low j)
        # get positive bias for later decoder positions, forcing ordered decoding.
        self.position_attn_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        thought_vectors: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        seq_len = target_input_ids.size(1)
        tgt = self.token_embedding(target_input_ids) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)

        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=target_input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            causal_mask = None

        # Dropout on thought vectors prevents posterior collapse — the decoder
        # cannot rely on a stable prior and must actually read the thoughts.
        thought_vectors = self.thought_dropout(thought_vectors)

        # Position bias for cross-attention: decoder pos i gets bias proportional
        # to (i - j) when attending to thought vector j.  Positive bias means the
        # decoder prefers earlier thought vectors, forcing the GRU's slot ordering
        # to correspond to output token ordering.
        mem_len = thought_vectors.size(1)
        if seq_len > 1 and mem_len > 1:
            tgt_pos = torch.arange(seq_len, device=target_input_ids.device).float()
            mem_pos = torch.arange(mem_len, device=thought_vectors.device).float()
            # [seq_len, mem_len]
            position_bias = self.position_attn_bias * (tgt_pos[:, None] - mem_pos[None, :])
        else:
            position_bias = None

        decoded = self.decoder(
            tgt=tgt,
            memory=thought_vectors,
            tgt_mask=causal_mask,
            memory_mask=position_bias,
            tgt_key_padding_mask=target_padding_mask,
        )
        return self.lm_head(decoded)


class ThoughtVectorModel(nn.Module):
    """Encapsulates a ThoughtEncoder -> ThoughtDecoder pipeline.

    Encoder produces thought vectors from tokenized input; decoder
    reconstructs token logits from the thought vectors and shifted targets.
    Also includes a lightweight count predictor that estimates the
    optimal number of vectors from the pooled thought representation.
    """

    def __init__(self, encoder: ThoughtEncoder, decoder: ThoughtDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Tie encoder and decoder token embeddings to a single shared table
        self.encoder.token_embedding.weight = self.decoder.token_embedding.weight
        # Tie decoder LM head to the same shared embedding
        self.decoder.lm_head.weight = self.decoder.token_embedding.weight

        # Count prediction head: estimates optimal vector count from pooled thoughts
        d_model = encoder.d_model
        self.count_predictor = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.ReLU(),
            nn.Linear(max(1, d_model // 2), 1),
        )

    def reconstruct_logits(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        thoughts = self.encoder(input_ids, padding_mask)
        return self.decoder(thoughts, input_ids[:, :-1], None if padding_mask is None else padding_mask[:, :-1])


class LossPredictor(nn.Module):
    """Predicts reconstruction loss for each possible thought-vector prefix length.

    Input:  pooled thoughts [batch, d_model]  (mean over thought vectors)
    Output: [batch, num_thoughts] — predicted reconstruction loss at each k.
    """

    def __init__(self, d_model: int, num_thoughts: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.ReLU(),
            nn.Linear(max(1, d_model // 2), num_thoughts),
        )

    def forward(self, thoughts: torch.Tensor) -> torch.Tensor:
        return self.net(thoughts.mean(dim=1))  # [batch, num_thoughts]


class ThinkerModel(nn.Module):
    """Full pipeline: encoder → (k-slice) → thinker → decoder.

    The thinker is a transformer that operates entirely in latent space.
    It transforms encoded thought vectors (or a k-sliced subset) before
    the decoder reconstructs the output text.

    Supports multi-turn context via learned turn embeddings, speaker
    embeddings (user/assistant), and a decode-target embedding that
    marks which vectors the decoder will read from.

    A LossPredictor estimates reconstruction quality for each possible k,
    enabling automatic compression-level selection at inference time.

    Usage:
        model = ThinkerModel(encoder, decoder, thinker, predictor)
        logits = model(article_ids, summary_ids, k=32)
    """

    def __init__(
        self,
        encoder: ThoughtEncoder,
        decoder: ThoughtDecoder,
        thinker: nn.TransformerEncoder,
        predictor: LossPredictor,
        max_turns: int = 10,
        thinker_dropout: float = 0.1,
        thinker_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.thinker = thinker
        self.predictor = predictor
        d_model = encoder.d_model  # encoder/decoder dimension (256)
        t_dim = thinker_dim or d_model  # thinker internal dimension
        self.thinker_dim = t_dim

        # Tie embeddings (same as ThoughtVectorModel)
        self.encoder.token_embedding.weight = self.decoder.token_embedding.weight
        self.decoder.lm_head.weight = self.decoder.token_embedding.weight

        # Projections bridge encoder dimension (d_model) and thinker
        # dimension (t_dim) when they differ (e.g. 256 → 384).
        self.thinker_proj_in = nn.Linear(d_model, t_dim) if t_dim != d_model else nn.Identity()
        self.thinker_proj_out = nn.Linear(t_dim, d_model) if t_dim != d_model else nn.Identity()

        # Turn position embedding — marks which conversation turn a
        # vector belongs to so the thinker can distinguish early vs
        # recent context.  Zero-initialised so the pretrained thinker
        # doesn't receive shifted inputs on the first forward pass.
        self.turn_embedding = nn.Embedding(max_turns + 1, d_model)
        nn.init.zeros_(self.turn_embedding.weight)

        # Speaker embedding — marks each vector as user (0) or
        # assistant (1) so the thinker can distinguish roles.
        self.speaker_embedding = nn.Embedding(2, d_model)
        nn.init.zeros_(self.speaker_embedding.weight)

        # Decode-target embedding — a learned bias added to the
        # current-user's thought vectors to signal "the decoder will
        # read from these slots."  This is a single vector broadcast
        # to all positions where decode_mask=1.
        self.decode_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Dropout after the thinker prevents over-reliance on specific
        # latent features, parallel to the decoder's thought_dropout.
        self.thinker_dropout = nn.Dropout(thinker_dropout)

    def thinker_forward(
        self,
        thoughts: torch.Tensor,
        turn_ids: torch.Tensor | None = None,
        speaker_ids: torch.Tensor | None = None,
        decode_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the thinker with optional structural embeddings.

        Args:
            thoughts: [B, T, d_model] thought vectors
            turn_ids: [B, T] long — which conversation turn each vector belongs to
            speaker_ids: [B, T] long — 0=user, 1=assistant
            decode_mask: [B, T] bool — True for vectors the decoder will read

        Returns:
            [B, T, d_model] — transformed thought vectors
        """
        if turn_ids is not None:
            thoughts = thoughts + self.turn_embedding(turn_ids)
        if speaker_ids is not None:
            thoughts = thoughts + self.speaker_embedding(speaker_ids)
        if decode_mask is not None:
            thoughts = thoughts + self.decode_embedding * decode_mask.unsqueeze(-1).float()
        # Project into thinker dimension if different from d_model
        thoughts = self.thinker_proj_in(thoughts)
        # Run thinker in eval mode to disable internal dropout, which
        # produces NaN on this ROCm gfx1031 under the hipBLASLt fallback.
        self.thinker.eval()
        thoughts = self.thinker(thoughts)
        self.thinker.train()
        # Project back to d_model for the decoder
        thoughts = self.thinker_proj_out(thoughts)
        return self.thinker_dropout(thoughts)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        k: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional k-slicing.

        Args:
            input_ids: encoder input tokens [batch, seq_in]
            target_ids: decoder target tokens [batch, seq_out]
            padding_mask: padding mask for encoder input
            k: number of thought vectors to use.
                k=-1: use all 256 vectors, skip thinker (identity).
                k=0: use all 256 vectors, apply thinker.
                k>0: slice to k vectors, apply thinker.
                None: equivalent to k=0.

        Returns:
            logits: [batch, seq_out - 1, vocab]
        """
        thoughts = self.encoder(input_ids, padding_mask)  # [batch, 256, d_model]

        if k is not None and k >= 0:
            # Slice to k vectors
            thoughts = thoughts[:, :k, :]

        if k != -1:
            # Apply thinker transformation (no structural metadata in
            # the simple forward path — caller should use thinker_forward
            # for multi-turn context with embeddings).
            thoughts = self.thinker_forward(thoughts)

        pad_idx = getattr(self.decoder.token_embedding, 'padding_idx', None)
        target_padding = target_ids.eq(pad_idx) if pad_idx is not None else None
        return self.decoder(thoughts, target_ids[:, :-1], target_padding)

    def predicted_losses(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run predictor on encoder output, return [batch, num_thoughts] loss estimates."""
        with torch.no_grad():
            thoughts = self.encoder(input_ids, padding_mask)
            return self.predictor(thoughts)
