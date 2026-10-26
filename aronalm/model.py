"""
AronaLM - Decoder-only Transformer model from scratch.

Architecture:
  - Token embedding + learned positional embedding
  - 2 decoder layers, each with multi-head self-attention (no bias) + FFN (with bias)
  - Final LayerNorm + output projection (with bias)
  - Causal masking via causal attention mask

Parameter count (d_model=64, heads=2, ffn_dim=128, vocab_size=1000, seq_len=64):
  Token Embedding:      1000 x 64    =  64,000
  Positional Embedding:  64 x 64     =   4,096
  Per layer (x2):
    Q/K/V/O projections: 4 x (64x64) = 16,384   (no bias)
    FFN fc1 (w+b):       64x128 + 128 =  8,320
    FFN fc2 (w+b):       128x64 +  64 =  8,256
    LayerNorm x2 (w+b):  2 x 128     =     256
    Layer subtotal:                       = 33,216
  2 layers total:                           = 66,432
  Final LayerNorm (w+b):                  =     128
  Output Projection (w+b): 64x1000+1000 =  65,000
  TOTAL:                                      199,656 (~200K)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the AronaLM decoder-only transformer."""
    vocab_size: int = 1000
    max_seq_len: int = 64
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    use_bias_in_proj: bool = False  # Q/K/V/O projections: no bias
    use_bias_in_ffn: bool = True    # FFN linear layers: with bias
    use_bias_in_output: bool = True # Output projection: with bias
    use_bias_in_layernorm: bool = True

    def total_parameters(self) -> int:
        """Compute total number of parameters."""
        dm, hd, fd, vs, sl = self.d_model, self.n_heads, self.ffn_dim, self.vocab_size, self.max_seq_len
        n = self.n_layers

        # Token embedding
        params = vs * dm
        # Positional embedding
        params += sl * dm
        # Per layer
        # Q/K/V/O projections (no bias)
        params += n * 4 * (dm * dm)
        # FFN fc1: weight(d_model, ffn_dim) + bias(ffn_dim)
        #          fc2: weight(ffn_dim, d_model) + bias(d_model)
        params += n * (2 * dm * fd + fd + dm)
        # LayerNorm x2 per decoder layer + 1 final (weight + bias each)
        # Total LayerNorm instances: n*2 + 1, each with weight+bias = 2*dm params
        params += (n * 2 + 1) * 2 * dm
        # Output projection (with bias)
        params += dm * vs + vs
        return params


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking. No bias on projections."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = config.d_model // config.n_heads
        self.n_heads = config.n_heads
        self.d_model = config.d_model

        # Q, K, V, O projections — NO bias (standard in modern transformers)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias_in_proj)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias_in_proj)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias_in_proj)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias_in_proj)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        scale = self.d_k ** 0.5

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with causal masking
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        if causal_mask is not None:
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        return out


class FeedForwardNet(nn.Module):
    """Position-wise FFN: Linear(d_model -> ffn_dim) -> ReLU -> Linear(ffn_dim -> d_model)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim, bias=config.use_bias_in_ffn)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(config.ffn_dim, config.d_model, bias=config.use_bias_in_ffn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.relu(self.fc1(x))))


class DecoderLayer(nn.Module):
    """Transformer decoder layer: Pre-LN (LayerNorm before attention and FFN)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, elementwise_affine=config.use_bias_in_layernorm)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model, elementwise_affine=config.use_bias_in_layernorm)
        self.ffn = FeedForwardNet(config)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class AronaLM(nn.Module):
    """
    AronaLM — Decoder-only Transformer from random initialization.

    Architecture summary:
      TokenEmbedding -> PositionalEmbedding -> [DecoderLayer x N] -> LayerNorm -> OutputProjection
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()
        self.config = config or TransformerConfig()
        cfg = self.config

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        # Build causal mask once: mask[i,j] = 0 means "masked out" (j > i)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1, 1, cfg.max_seq_len, cfg.max_seq_len)).bool(),
            persistent=False,
        )

        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_layernorm = nn.LayerNorm(cfg.d_model, elementwise_affine=cfg.use_bias_in_layernorm)
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=cfg.use_bias_in_output)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) tensor of token IDs
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # Token + positional embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_embedding(pos_ids)
        x = self.dropout(token_emb + pos_emb)

        # Decoder layers with causal masking
        for layer in self.layers:
            x = layer(x, self.causal_mask[:, :, :T, :T])

        # Final LN + output projection
        x = self.final_layernorm(x)
        logits = self.output_proj(x)
        return logits

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Causal language modeling loss.
        Labels are shifted: predict token at position t using tokens [0..t].
        The first token has no previous context, so its loss is ignored (label=-100).
        """
        logits = self.forward(input_ids)
        B, T, V = logits.shape

        # Shift: predict position t from positions 0..t-1
        shift_logits = logits[:, :-1, :].reshape(-1, V)
        shift_labels = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            prompt_ids: (B, T_prompt) token IDs to start from
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (>0 = stochastic, 0.0 = greedy)
            top_k: if set, only sample from top-k logits
        Returns:
            Generated token IDs (B, T_prompt + max_new_tokens)
        """
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r} ({type(temperature).__name__})")
        if math.isnan(temperature) or math.isinf(temperature):
            raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")
        if temperature < 0:
            raise ValueError(f"temperature must be a finite number >= 0, got {temperature!r}")

        greedy = (temperature == 0.0)

        generated = prompt_ids.clone()
        for _ in range(max_new_tokens):
            T = generated.shape[1]
            causal_mask = self.causal_mask[:, :, :T, :T]

            logits = self.forward(generated)
            logits = logits[:, -1, :]

            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = logits.topk(top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())
