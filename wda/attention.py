"""Scaled dot-product self-attention from first principles (PRD §7).

This is the only attention implementation in the project. Both the toy model
and the warehouse model call `attention` (A-26). Every operation the PRD lists
is a separate, named statement, built from matrix products, an exponential, a
maximum and sums.

Shapes, for a batch of B sequences of n tokens:
    X (B, n, d_model)   Q, K (B, n, d_k)   V (B, n, d_v)
    S, S_scaled, A (B, n, n)               Y (B, n, d_v)
The batch dimension is optional.
"""

import math
from typing import NamedTuple

import torch
from torch import nn


class AttentionOutput(NamedTuple):
    """All intermediates. Callers keep what they need (the trace, H3's gradient of Q)."""

    Y: torch.Tensor
    A: torch.Tensor
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    S: torch.Tensor
    S_scaled: torch.Tensor


def stable_softmax(S: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis, computed as exp(s − max s) / Σ exp(s − max s).

    softmax(s − c) = softmax(s) for any constant c, so subtracting the row
    maximum changes nothing mathematically, but keeps every exponent ≤ 0, so
    exp cannot overflow (§11). The maximum is treated as a constant
    (detached): the result does not depend on it, so its gradient is zero.
    """
    shifted = S - S.max(dim=-1, keepdim=True).values.detach()
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=-1, keepdim=True)


def attention(
    X: torch.Tensor,
    W_Q: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    scaled: bool = True,
    uniform: bool = False,
) -> AttentionOutput:
    """Self-attention of the tokens X with itself.

    scaled=False drops the 1/√d_k factor (the §14 ablation, A-19).
    uniform=True replaces the attention weights with 1/n (the controls, A-18, A-27).
    The defaults compute exactly the PRD's scaled attention.
    """
    Q = X @ W_Q                                   # query projection
    K = X @ W_K                                   # key projection
    V = X @ W_V                                   # value projection
    S = Q @ K.transpose(-2, -1)                   # similarity scores QKᵀ
    d_k = Q.shape[-1]
    S_scaled = S / math.sqrt(d_k) if scaled else S    # scaling QKᵀ/√d_k
    if uniform:
        A = torch.full_like(S_scaled, 1.0 / S_scaled.shape[-1])
    else:
        A = stable_softmax(S_scaled)              # attention weights, rows sum to 1
    Y = A @ V                                     # output: weighted average of the values
    return AttentionOutput(Y, A, Q, K, V, S, S_scaled)


class AttentionWeights(nn.Module):
    """W_Q, W_K, W_V as trainable parameters, with entries drawn from N(0, init_std²).

    The draws come from the given generator in a fixed order and do not depend
    on `scaled` or `uniform`, so paired ablation arms start from identical
    weights (FAC-28). PHASE0 chooses init_std so that the components of Q and
    K have unit variance (A-19).
    """

    def __init__(self, d_model: int, d_k: int, d_v: int, init_std: float, generator: torch.Generator):
        super().__init__()
        self.W_Q = nn.Parameter(torch.randn(d_model, d_k, generator=generator) * init_std)
        self.W_K = nn.Parameter(torch.randn(d_model, d_k, generator=generator) * init_std)
        self.W_V = nn.Parameter(torch.randn(d_model, d_v, generator=generator) * init_std)

    def forward(self, X: torch.Tensor, scaled: bool = True, uniform: bool = False) -> AttentionOutput:
        return attention(X, self.W_Q, self.W_K, self.W_V, scaled=scaled, uniform=uniform)
