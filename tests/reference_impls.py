"""Independent reference implementations used ONLY for testing parity.

Written from textbook definitions in plain Python (math module, no numpy
vectorization, no ML framework) so they are maximally independent of both
numpy kernels and TorchZero code paths.
"""

import math


def ref_matmul(a, b):
    """Naive triple-loop matrix multiply."""
    n, k = len(a), len(a[0])
    k2, m = len(b), len(b[0])
    assert k == k2
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            aip = a[i][p]
            if aip == 0.0:
                continue
            for j in range(m):
                out[i][j] += aip * b[p][j]
    return out


def ref_transpose(a):
    return [list(row) for row in zip(*a)]


def ref_softmax_row(row):
    m = max(row)
    exps = [math.exp(v - m) for v in row]
    s = sum(exps)
    return [e / s for e in exps]


def ref_layernorm_row(x, weight, bias, eps=1e-5):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    denom = math.sqrt(var + eps)
    return [(v - mu) / denom * w + b for v, w, b in zip(x, weight, bias)]


def ref_rmsnorm_row(x, weight, eps=1e-6):
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [v * inv * w for v, w in zip(x, weight)]


def ref_gelu(v):
    beta = math.sqrt(2.0 / math.pi)
    kappa = 0.044715
    inner = beta * (v + kappa * v ** 3)
    return 0.5 * v * (1.0 + math.tanh(inner))


def ref_sigmoid(v):
    if v >= 0:
        z = math.exp(-v)
        return 1.0 / (1.0 + z)
    z = math.exp(v)
    return z / (1.0 + z)


def ref_causal_attention(q_rows, k_rows, v_rows, scale=None):
    """Single-head causal attention over sequences of vectors."""
    d = len(q_rows[0])
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    out = []
    for i, q in enumerate(q_rows):
        scores = []
        for j in range(i + 1):          # causal mask: j <= i
            s = sum(qc * kc for qc, kc in zip(q, k_rows[j])) * scale
            scores.append(s)
        probs = ref_softmax_row(scores)
        ctx = [sum(probs[j] * v_rows[j][c] for j in range(i + 1))
               for c in range(d)]
        out.append(ctx)
    return out


def ref_cross_entropy(logits_rows, targets):
    losses = []
    for row, t in zip(logits_rows, targets):
        m = max(row)
        lse = m + math.log(sum(math.exp(v - m) for v in row))
        losses.append(lse - row[t])
    return sum(losses) / len(losses)


def ref_adam_step(param, grad, m, v, t, lr=0.01, b1=0.9, b2=0.999,
                  eps=1e-8):
    """Single-element Adam update; returns (new_param, new_m, new_v)."""
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad * grad
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    return param - lr * mhat / (math.sqrt(vhat) + eps), m, v
