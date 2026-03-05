"""
Decoder-only GPT Transformer built entirely from scratch in PyTorch.

Features:
  - Rotary Position Embeddings (RoPE)
  - Pre-LayerNorm (RMSNorm)
  - SwiGLU feed-forward
  - Causal self-attention
  - KV-cache for fast autoregressive generation
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import GPTConfig


# ---------------------------------------------------------------------------
# RMSNorm  (simpler, faster alternative to LayerNorm used by LLaMA / Gemma)
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings
# ---------------------------------------------------------------------------
def _precompute_freqs(dim: int, max_seq_len: int, theta: float = 10_000.0) -> torch.Tensor:
    """Return complex-valued freqs_cis of shape (max_seq_len, dim//2)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x (B, n_heads, T, head_dim)."""
    # reshape to (..., head_dim//2, 2) then view as complex
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, T, D)
    return x_rotated.type_as(x)


# ---------------------------------------------------------------------------
# Multi-Head Causal Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model

        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # RoPE
        q = _apply_rope(q, freqs_cis)
        k = _apply_rope(k, freqs_cis)

        # KV cache for inference
        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
        new_kv = (k, v)

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # causal mask
        if mask is not None:
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.wo(out)
        out = self.resid_drop(out)
        return out, new_kv


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward (used by modern LLMs for better convergence)
# ---------------------------------------------------------------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.d_ff * 2 / 3)  # SwiGLU uses 2/3 of d_ff for gate
        # round to nearest multiple of 8 for efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)  # gate
        self.w2 = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.ln1(x)
        h, new_kv = self.attn(h, freqs_cis, mask=mask, kv_cache=kv_cache)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# Full GPT Model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight tying: share embedding weights with lm_head
        self.lm_head.weight = self.tok_emb.weight

        # precompute RoPE freqs
        self.register_buffer(
            "freqs_cis",
            _precompute_freqs(cfg.head_dim, cfg.max_seq_len * 2),
            persistent=False,
        )

        # initialise weights
        self.apply(self._init_weights)

        # report param count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # subtract the tied lm_head
        n_params -= self.lm_head.weight.numel()
        print(f"GPT model initialised -- {n_params / 1e6:.1f}M trainable parameters (excl. weight-tied lm_head)")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
    ) -> dict:
        """
        Args:
            input_ids:  (B, T)  token ids
            labels:     (B, T)  target ids for computing loss (shifted internally)
            kv_caches:  list of (k, v) tuples per layer, or None
        Returns:
            dict with keys 'logits' and optionally 'loss', 'kv_caches'
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)
        x = self.drop(x)

        # build causal mask
        if kv_caches is not None and kv_caches[0] is not None:
            # during generation with cache, T_query = 1 usually
            past_len = kv_caches[0][0].shape[2]
            freqs = self.freqs_cis[past_len: past_len + T]
            mask = None  # no mask needed for single-token generation
        else:
            freqs = self.freqs_cis[:T]
            mask = torch.full((T, T), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, new_kv = layer(x, freqs, mask=mask, kv_cache=cache_i)
            new_kv_caches.append(new_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        result = {"logits": logits, "kv_caches": new_kv_caches}

        if labels is not None:
            # shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k / top-p sampling and KV cache."""
        self.eval()
        if eos_token_id is None:
            eos_token_id = self.cfg.eos_token_id

        B = input_ids.shape[0]
        kv_caches = [None] * self.cfg.n_layers

        # prefill: process the whole prompt
        out = self.forward(input_ids, kv_caches=None)
        logits = out["logits"]
        kv_caches = out["kv_caches"]

        # sample first new token from last position
        next_logits = logits[:, -1, :]
        generated = [self._sample_token(next_logits, temperature, top_k, top_p)]

        for _ in range(max_new_tokens - 1):
            next_input = generated[-1].unsqueeze(1)  # (B, 1)
            out = self.forward(next_input, kv_caches=kv_caches)
            logits = out["logits"]
            kv_caches = out["kv_caches"]
            next_logits = logits[:, -1, :]
            tok = self._sample_token(next_logits, temperature, top_k, top_p)
            generated.append(tok)
            if (tok == eos_token_id).all():
                break

        generated = torch.stack(generated, dim=1)  # (B, gen_len)
        return torch.cat([input_ids, generated], dim=1)

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        # top-k
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        # top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
