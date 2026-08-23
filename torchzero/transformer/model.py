"""Decoder-only transformer built entirely from TorchZero primitives.

Architecture (pre-norm):
    token embedding (+ RoPE inside attention)
    N x [ RMSNorm -> causal multi-head attention -> residual
          RMSNorm -> MLP(GELU) -> residual ]
    final RMSNorm -> tied LM head

Execution path from tokens to logits is straight-line TorchZero code: every
step is inspectable in this file without framework internals.
"""

from __future__ import annotations

import numpy as np

from ..nn.module import Module, Parameter
from ..nn.layers import Linear, Embedding, RMSNorm, gelu
from ..tensor.tensor import Tensor


class TransformerConfig:
    def __init__(self, vocab_size, dim=128, n_layers=4, n_heads=4,
                 block_size=64, ffn_hidden=None, dropout=0.0,
                 tie_weights=True, norm_eps=1e-6):
        if dim % n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.block_size = block_size
        self.ffn_hidden = ffn_hidden or 4 * dim
        self.dropout = dropout
        self.tie_weights = tie_weights
        self.norm_eps = norm_eps

    @property
    def head_dim(self):
        return self.dim // self.n_heads

    def to_dict(self):
        return vars(self).copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def rope_cache(seq_len, head_dim, base=10000.0, dtype=np.float32):
    """Precomputed cos/sin tables for rotary embeddings, half-split form."""
    inv_freq = 1.0 / (base ** (
        np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    t = np.arange(seq_len, dtype=np.float64)
    freqs = np.outer(t, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(dtype), np.sin(emb).astype(dtype)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """x: (B, T, n_heads, head_dim). Returns rotated tensor."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    T = x.shape[1]
    x1, x2 = x.split(2, axis=-1)
    # cos/sin tables are (T, half); broadcast over batch and heads
    cos_t = Tensor(np.broadcast_to(cos[..., :half], (T, half)),
                   requires_grad=False).reshape(1, T, 1, half)
    sin_t = Tensor(np.broadcast_to(sin[..., :half], (T, half)),
                   requires_grad=False).reshape(1, T, 1, half)
    # broadcast cos/sin over batch and heads
    out1 = x1 * cos_t - x2 * sin_t
    out2 = x1 * sin_t + x2 * cos_t
    return out1.concat(out2, axis=-1)


class CausalSelfAttention(Module):
    def __init__(self, config: TransformerConfig, seed=None):
        super().__init__()
        h = config.head_dim
        self.wq = Linear(config.dim, config.dim, bias=False, seed=seed)
        self.wk = Linear(config.dim, config.dim, bias=False,
                         seed=None if seed is None else seed + 1)
        self.wv = Linear(config.dim, config.dim, bias=False,
                         seed=None if seed is None else seed + 2)
        self.proj = Linear(config.dim, config.dim, bias=False,
                           seed=None if seed is None else seed + 3)
        self.n_heads = config.n_heads
        self.head_dim = h
        self.scale = 1.0 / np.sqrt(h)

    def forward(self, x: Tensor, cos_sin=None, cache=None, trace=None):
        """x: (B, T, dim).

        ``cache`` is an optional dict {"k": Tensor, "v": Tensor} holding this
        layer's cached keys/values of shape (B, T_past, n_heads, head_dim);
        it is updated in place.
        """
        B, T, C = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.wq(x).reshape(B, T, H, Dh)
        k = self.wk(x).reshape(B, T, H, Dh)
        v = self.wv(x).reshape(B, T, H, Dh)

        if cos_sin is not None:
            cos, sin = cos_sin
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        q = q.transpose(0, 2, 1, 3)   # (B, H, T, Dh)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if cache is not None:
            if "k" in cache:
                k_full = cache["k"].concat(k, axis=2)
                v_full = cache["v"].concat(v, axis=2)
                cache["k"], cache["v"] = k_full.detach(), v_full.detach()
            else:
                k_full, v_full = k, v
                cache["k"], cache["v"] = k.detach(), v.detach()
        else:
            k_full, v_full = k, v

        T_kv = k_full.shape[2]
        scores = (q @ k_full.transpose(0, 1, 3, 2)) * float(self.scale)
        if cache is None:
            # causal mask: query i may see key j <= i (offset 0: no prefix)
            mask = np.tril(np.ones((T, T_kv), dtype=bool))
        else:
            offset = T_kv - T
            mask = np.tril(np.ones((T, T_kv), dtype=bool),
                           k=offset)
        scores = scores.masked_fill(~mask, -1e30)
        probs = scores.softmax(axis=-1)
        attn_out = probs @ v_full                            # (B,H,T,Dh)
        out = attn_out.transpose(0, 2, 1, 3).reshape(B, T, C)
        out = self.proj(out)

        if trace is not None:
            trace.append({
                "op": "attention",
                "q_shape": tuple(q.shape), "k_shape": tuple(k.shape),
                "v_shape": tuple(v.shape),
                "attn_probs_shape": tuple(probs.shape),
                "attn_probs": probs.detach().data,
            })
        return out


class FeedForward(Module):
    def __init__(self, config: TransformerConfig, seed=None):
        super().__init__()
        self.fc = Linear(config.dim, config.ffn_hidden, seed=seed)
        self.proj = Linear(config.ffn_hidden, config.dim,
                           seed=None if seed is None else seed + 1)

    def forward(self, x: Tensor, trace=None):
        h = gelu(self.fc(x))
        out = self.proj(h)
        if trace is not None:
            trace.append({
                "op": "mlp",
                "hidden_shape": tuple(h.shape),
                "hidden": h.detach().data,
                "out_shape": tuple(out.shape),
            })
        return out


class TransformerBlock(Module):
    def __init__(self, config: TransformerConfig, seed=None):
        super().__init__()
        self.norm_attn = RMSNorm(config.dim, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config,
                                        seed=None if seed is None else seed)
        self.norm_mlp = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlp = FeedForward(config,
                               seed=None if seed is None else seed + 10)

    def forward(self, x: Tensor, cos_sin=None, cache=None, trace=None):
        h = self.norm_attn(x)
        x = x + self.attn(h, cos_sin=cos_sin, cache=cache, trace=trace)
        x = x + self.mlp(self.norm_mlp(x), trace=trace)
        return x


class Transformer(Module):
    def __init__(self, config: TransformerConfig, seed=None):
        super().__init__()
        self.config = config
        rng = np.random.default_rng(seed)
        self.tok_emb = Embedding(config.vocab_size, config.dim,
                                 seed=None if seed is None else int(rng.integers(1 << 30)))
        self.blocks = [
            TransformerBlock(config, seed=None if seed is None else int(rng.integers(1 << 30)))
            for _ in range(config.n_layers)
        ]
        self.norm_f = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = Linear(config.dim, config.vocab_size, bias=False,
                              seed=None if seed is None else int(rng.integers(1 << 30)))
        self.kv_caches = None

    # ---------------------------------------------------------------- forward
    def forward(self, idx, targets=None, use_cache=False, reset_cache=True,
                trace=None):
        """idx: integer array/Tensor (B, T). Returns logits (B, T, V); also
        returns loss when ``targets`` given."""
        if isinstance(idx, Tensor):
            if idx.dtype.kind not in "iu":
                raise TypeError("input indices must be integers")
            B, T = idx.shape
        else:
            arr = np.asarray(idx)
            if arr.dtype.kind not in "iu":
                raise TypeError("input indices must be integers")
            if arr.ndim != 2:
                raise ValueError("expected (B, T) index array")
            B, T = arr.shape
            idx = Tensor(arr.astype(np.int64))
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block size "
                f"{self.config.block_size}")

        if use_cache:
            if reset_cache or self.kv_caches is None:
                self.kv_caches = [{} for _ in range(self.config.n_layers)]
        else:
            self.kv_caches = None

        # position offset for RoPE when extending a KV cache mid-generation
        pos_offset = 0
        if use_cache and self.kv_caches and "k" in self.kv_caches[0]:
            pos_offset = self.kv_caches[0]["k"].shape[2]
        cos_full, sin_full = rope_cache(pos_offset + T, self.config.head_dim)
        cos = cos_full[pos_offset:]
        sin = sin_full[pos_offset:]

        x = self.tok_emb(idx)                      # (B, T, dim)
        if trace is not None:
            trace.append({"op": "token_embedding",
                          "out_shape": tuple(x.shape),
                          "embedding": x.detach().data})
        for i, block in enumerate(self.blocks):
            cache = self.kv_caches[i] if self.kv_caches is not None else None
            x = block(x, cos_sin=(cos, sin), cache=cache, trace=trace)
            if trace is not None:
                trace.append({"op": f"block.{i}", "out_shape": tuple(x.shape),
                              "residual_stream": x.detach().data})
        x = self.norm_f(x)
        logits = self.lm_head(x)                   # (B, T, V)

        if trace is not None:
            trace.append({
                "op": "final_norm",
                "residual_stream": x.detach().data,
                "norm_mean": float(x.data.mean()),
                "norm_std": float(x.data.std()),
                "out_shape": tuple(x.shape),
            })
            trace.append({
                "op": "lm_head",
                "logits": logits.detach().data,
                "out_shape": tuple(logits.shape),
            })

        loss = None
        if targets is not None:
            flat_logits = logits.reshape(B * T, self.config.vocab_size)
            tgt = targets.data.astype(np.int64) if isinstance(targets, Tensor) \
                else np.asarray(targets, dtype=np.int64)
            loss = Tensor.cross_entropy(flat_logits, tgt.reshape(-1))
        if trace is not None and loss is not None:
            trace.append({"op": "cross_entropy", "loss": float(loss.item())})
        return logits, loss

    # ------------------------------------------------------------- inference
    def clear_cache(self):
        self.kv_caches = None

    @staticmethod
    def sample_next(logits_last, temperature=1.0, top_k=None,
                     seed=None):
        """Sample one next-token id from last-position logits (numpy array)."""
        rng = np.random.default_rng(seed)
        row = logits_last.astype(np.float64)
        if temperature <= 0 or top_k == 1:
            return int(row.argmax())
        row = row / temperature
        if top_k is not None and top_k < row.size:
            kth = np.sort(row)[-top_k]
            row = np.where(row < kth, -np.inf, row)
        row -= row.max()
        probs = np.exp(row)
        probs /= probs.sum()
        return int(rng.choice(row.size, p=probs))

    def generate(self, prompt_ids, max_new_tokens, temperature=1.0,
                 top_k=None, seed=None, use_kv_cache=True):
        """Autoregressive generation. Deterministic when temperature<=0."""
        ids = [int(i) for i in prompt_ids]
        if len(ids) > self.config.block_size:
            raise ValueError("prompt longer than block size")
        self.eval()
        if use_kv_cache:
            logits, _ = self.forward([ids], use_cache=True, reset_cache=True)
            for _ in range(max_new_tokens):
                nxt = self.sample_next(logits[0, -1].data, temperature,
                                       top_k, seed)
                ids.append(nxt)
                if len(ids) >= self.config.block_size:
                    break
                logits, _ = self.forward([[nxt]], use_cache=True,
                                         reset_cache=False)
            return ids
        # uncached: full forward over growing context each step
        for _ in range(max_new_tokens):
            ctx = ids[-self.config.block_size:]
            logits, _ = self.forward([ctx])
            nxt = self.sample_next(logits[0, -1].data, temperature, top_k,
                                   seed)
            ids.append(nxt)
            if len(ids) >= self.config.block_size:
                break
        return ids

    def num_parameters(self):
        return sum(int(p.data.size) for p in self.parameters())
