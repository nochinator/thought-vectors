"""
BitThought: Thought Vector Continuum — encoder-decoder transformer
that compresses text into dense latent vectors with adaptive stopping.

Architecture:
  - Standard PyTorch TransformerEncoder / TransformerDecoder layers
  - GRU + cross-attention for thought vector generation
  - Loss prediction head for adaptive inference-time compression
  - Multi-Token Prediction (MTP) auxiliary loss
  - Optional gradient checkpointing for memory-constrained training

4 model sizes: Tiny (1M), Small (3.1M), Medium (11M), Large (85M)
"""

from bitthought.config import BitThoughtConfig, ModelPresets

from bitthought.model import (
    BitThoughtEncoder,
    BitThoughtDecoder,
    BitThoughtModel,
    migrate_state_dict,
)
from bitthought.tokenization import ThoughtTokenizer
from bitthought.train import training_step, train_model
from bitthought.inference import (
    encode,
    decode_greedy,
    find_minimum_vectors,
    encode_with_compression,
)
from bitthought.data import load_groups, load_pairs, tokenize_dataset
from bitthought.data import ThoughtDataset, PairedThoughtDataset
from bitthought.data import collate_thought_batch, collate_pair_batch, collate_tokenized_batch
