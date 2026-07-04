"""
BitThought — Tokenizer with dual mode: custom BPE or pre-existing GPT-2.

Small preset: trains a fresh BPE on the dataset (full ASCII coverage).
Medium/Large: loads the pre-downloaded GPT-2 tokenizer.

Special tokens for custom BPE: <pad>=0, <bos>=1, <eos>=2
GPT-2 tokenizer uses its own native tokens.
"""

import string
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE as BPEModel
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

_ASCII_ALPHABET = list(string.printable.strip())
_GPT2_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "gpt2_tokenizer.json"
_LLAMA2_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "llama2_tokenizer"


class ThoughtTokenizer:
    """Dual-mode tokenizer: custom BPE or GPT-2.

    Mode 1 — custom BPE (``tokenizer_name=""``):
      Trains BPE with full ASCII alphabet guarantee. vocab_size 2K-5K.

    Mode 2 — GPT-2 (``tokenizer_name="gpt2"``):
      Loads pre-downloaded GPT-2 ByteLevel BPE (50K vocab).
    """

    def __init__(self, tokenizer_name: str = "", vocab_size: int = 4000,
                 min_frequency: int = 2):
        self._tokenizer_name = tokenizer_name
        self._vocab_target = vocab_size
        self._min_frequency = min_frequency
        self._tokenizer: Tokenizer | None = None

    def _build_custom(self):
        """Build a fresh BPE seeded with the ASCII alphabet."""
        self._tokenizer = Tokenizer(BPEModel(unk_token=None))
        self._tokenizer.pre_tokenizer = ByteLevelPreTokenizer(
            add_prefix_space=False
        )
        self._tokenizer.decoder = ByteLevelDecoder()
        self._tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> <bos> $B <eos>",
            special_tokens=[("<pad>", 0), ("<bos>", 1), ("<eos>", 2)],
        )

    def _load_gpt2(self):
        """Load pre-downloaded GPT-2 tokenizer."""
        if not _GPT2_PATH.exists():
            raise FileNotFoundError(
                f"GPT-2 tokenizer not found at {_GPT2_PATH}. "
                "Download it first."
            )
        self._tokenizer = Tokenizer.from_file(str(_GPT2_PATH))

    def _load_llama2(self):
        """Load Llama 2 SentencePiece Unigram tokenizer (32K vocab)."""
        tokenizer_file = _LLAMA2_PATH / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(
                f"Llama 2 tokenizer not found at {tokenizer_file}."
            )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        if self._tokenizer.token_to_id("<s>") is not None:
            self._tokenizer.add_special_tokens(["<s>", "</s>"])

    def fit(self, texts: Iterable[str] | None = None, *,
            min_frequency: int | None = None,
            max_vocab_size: int | None = None):
        """Train or load the tokenizer.

        For custom BPE: trains on *texts* (required).
        For GPT-2: loads pre-trained, *texts* is ignored.
        """
        if self._tokenizer_name == "gpt2":
            self._load_gpt2()
            return

        # Custom BPE — requires texts
        if texts is None:
            raise ValueError("Custom BPE requires training texts.")
        if min_frequency is not None:
            self._min_frequency = min_frequency
        if max_vocab_size is not None:
            self._vocab_target = max_vocab_size

        self._build_custom()
        trainer = BpeTrainer(
            vocab_size=self._vocab_target,
            min_frequency=self._min_frequency,
            special_tokens=["<pad>", "<bos>", "<eos>"],
            show_progress=False,
            initial_alphabet=_ASCII_ALPHABET,
        )
        self._tokenizer.train_from_iterator(texts, trainer=trainer)

    @classmethod
    def from_preset(cls, name: str) -> "ThoughtTokenizer":
        """Create a tokenizer for a preset name."""
        if name == "gpt2":
            tok = cls(tokenizer_name="gpt2")
            tok._load_gpt2()
            return tok
        if name in ("llama2", "llama", "llama-2"):
            tok = cls(tokenizer_name="llama2")
            tok._load_llama2()
            return tok
        return cls()

    @classmethod
    def from_file(cls, path: Path) -> "ThoughtTokenizer":
        """Load a previously saved tokenizer."""
        tok = cls.__new__(cls)
        tok._tokenizer_name = ""
        tok._vocab_target = 0
        tok._min_frequency = 0
        tok._tokenizer = Tokenizer.from_file(str(path))
        return tok

    @property
    def pad_token_id(self) -> int:
        if self._tokenizer_name == "gpt2":
            return 50256
        if self._tokenizer_name == "llama2":
            return 0
        return 0

    @property
    def bos_token_id(self) -> int:
        if self._tokenizer_name == "gpt2":
            return 50256
        if self._tokenizer_name == "llama2":
            return 1  # <s>
        return 1

    @property
    def eos_token_id(self) -> int:
        if self._tokenizer_name == "gpt2":
            return 50256
        if self._tokenizer_name == "llama2":
            return 2  # </s>
        return 2

    @property
    def vocab_size(self) -> int:
        if self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return self._vocab_target

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        if self._tokenizer is None:
            return [self.bos_token_id, self.eos_token_id]
        enc = self._tokenizer.encode(text)

        if self._tokenizer_name == "gpt2":
            if add_special_tokens:
                return [self.bos_token_id] + enc.ids + [self.eos_token_id]
            return enc.ids
        if self._tokenizer_name == "llama2":
            # Llama 2 adds BOS internally but NOT EOS — add it here
            if add_special_tokens:
                return enc.ids + [self.eos_token_id] if enc.ids[-1] != self.eos_token_id else enc.ids
            # Strip BOS, keep EOS if present
            ids = enc.ids[:]
            if ids and ids[0] == self.bos_token_id:
                ids = ids[1:]
            return ids
        else:
            if add_special_tokens:
                return enc.ids
            ids = enc.ids[1:-1] if len(enc.ids) > 2 else []
            return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if self._tokenizer is None:
            return ""
        if skip_special_tokens:
            ids = [i for i in ids
                   if i not in (self.pad_token_id, self.bos_token_id,
                                self.eos_token_id)]
        return self._tokenizer.decode(ids)

    def save(self, path: Path):
        """Save tokenizer to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._tokenizer is not None:
            self._tokenizer.save(str(path))
