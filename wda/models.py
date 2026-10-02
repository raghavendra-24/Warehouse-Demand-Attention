"""The toy model and the warehouse model. Both call the shared attention core (A-26).

Every parameter is drawn from the model's own seeded generator, in a fixed
order that does not depend on the `scaled` or `uniform` switches. Two arms
built with the same seed therefore start bit-identical (FAC-28).
nn.Linear is used only outside the attention core (A-02).
"""

import math

import torch
from torch import nn

from wda.attention import AttentionOutput, AttentionWeights
from wda.config import ModelConfig


def _linear(fan_in: int, fan_out: int, generator: torch.Generator) -> nn.Linear:
    """nn.Linear with PyTorch's default U(−1/√fan_in, 1/√fan_in) initialisation, drawn from `generator`."""
    layer = nn.Linear(fan_in, fan_out)
    bound = 1.0 / math.sqrt(fan_in)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        layer.bias.uniform_(-bound, bound, generator=generator)
    return layer


class ToyModel(nn.Module):
    """Associative recall: tokens → attention → readout at the query (last) token → logits over values."""

    def __init__(self, cfg: ModelConfig, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.attn = AttentionWeights(cfg.d_model, cfg.d_k, cfg.d_v, cfg.qkv_init_std, g)
        self.head = _linear(cfg.d_v, cfg.n_out, g)

    def forward(self, tokens: torch.Tensor) -> tuple:
        out: AttentionOutput = self.attn(tokens, scaled=self.cfg.scaled, uniform=self.cfg.uniform)
        return self.head(out.Y[..., -1, :]), out


class WarehouseModel(nn.Module):
    """24 hourly tokens of 5 features → linear projection → attention → readout at hour t → ẑ(t+1)."""

    def __init__(self, cfg: ModelConfig, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.cfg = cfg
        self.inp = nn.Linear(cfg.d_in, cfg.d_model)
        with torch.no_grad():                       # PHASE0: W_in ~ N(0, 1/3), b_in = 0
            self.inp.weight.normal_(0.0, cfg.in_init_std, generator=g)
            self.inp.bias.zero_()
        self.attn = AttentionWeights(cfg.d_model, cfg.d_k, cfg.d_v, cfg.qkv_init_std, g)
        self.head = _linear(cfg.d_v, 1, g)

    def forward(self, features: torch.Tensor) -> tuple:
        out: AttentionOutput = self.attn(self.inp(features), scaled=self.cfg.scaled, uniform=self.cfg.uniform)
        return self.head(out.Y[..., -1, :]).squeeze(-1), out
