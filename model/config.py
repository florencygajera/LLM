"""
Model configuration for from-scratch GPT-style Transformer.

Provides two sizes:
  - tiny  (~50M params):  good for smoke testing on CPU or single GPU
  - small (~150-300M params): needs a GPU with >=16 GB VRAM for training
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration for the decoder-only GPT model."""

    # --- vocabulary & sequence ---
    vocab_size: int = 32_000
    max_seq_len: int = 1024

    # --- transformer dimensions ---
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048          # feed-forward inner dim (usually 4 * d_model)

    # --- regularisation ---
    dropout: float = 0.1
    attn_dropout: float = 0.1

    # --- special token ids (set after tokenizer is loaded) ---
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # --- misc ---
    bias: bool = False        # bias in Linear layers (modern practice: no bias)
    rope: bool = True         # use Rotary Position Embeddings

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


# ---------------------------------------------------------------------------
# Pre-defined size presets
# ---------------------------------------------------------------------------

def tiny_config() -> GPTConfig:
    """~50 M parameter model.
    
    Hardware guidance:
      - CPU-only training is feasible for smoke-tests / very short runs.
      - A single consumer GPU (8 GB VRAM) is comfortable for full training
        with batch_size=8, seq_len=1024, fp16.
    """
    return GPTConfig(
        vocab_size=32_000,
        max_seq_len=1024,
        n_layers=8,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        attn_dropout=0.1,
        bias=False,
        rope=True,
    )


def small_config() -> GPTConfig:
    """~200 M parameter model.
    
    Hardware guidance:
      - Requires a GPU with >=16 GB VRAM (e.g. RTX 3090/4090, A100-40G).
      - fp16/bf16 strongly recommended.
      - Gradient accumulation of 4-8 steps helps fit larger effective batches.
      - CPU-only training is NOT practical beyond a few-step smoke test.
    """
    return GPTConfig(
        vocab_size=32_000,
        max_seq_len=2048,
        n_layers=16,
        n_heads=16,
        d_model=1024,
        d_ff=4096,
        dropout=0.1,
        attn_dropout=0.1,
        bias=False,
        rope=True,
    )


SIZE_REGISTRY = {
    "tiny": tiny_config,
    "small": small_config,
}


def get_config(size: str = "tiny") -> GPTConfig:
    """Return a GPTConfig by preset name."""
    if size not in SIZE_REGISTRY:
        raise ValueError(f"Unknown size '{size}'. Choose from {list(SIZE_REGISTRY)}")
    return SIZE_REGISTRY[size]()
