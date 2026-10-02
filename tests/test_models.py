"""Models: outputs per window or sequence, paired initialisation, switches pass through (FAC-28, FAC-54)."""

from dataclasses import replace

import torch

from wda import config
from wda.models import ToyModel, WarehouseModel


def test_toy_model_gives_logits_per_sequence_from_the_query_row():
    model = ToyModel(config.TOY_MODEL, seed=0)
    tokens = torch.rand(7, config.TOY.n_tokens, config.TOY.token_dim)
    logits, out = model(tokens)
    assert logits.shape == (7, config.TOY.n_values)
    assert out.A.shape == (7, config.TOY.n_tokens, config.TOY.n_tokens)


def test_warehouse_model_gives_one_prediction_per_window():
    model = WarehouseModel(config.WAREHOUSE_MODEL, seed=0)
    z, out = model(torch.randn(11, config.WINDOW, 5))
    assert z.shape == (11,)
    assert out.A.shape == (11, config.WINDOW, config.WINDOW)


def test_warehouse_model_has_the_phase0_parameter_count():
    model = WarehouseModel(config.WAREHOUSE_MODEL, seed=0)
    assert sum(p.numel() for p in model.parameters()) == 881


def test_paired_arms_start_bit_identical_and_seeds_differ():
    for build, cfg in ((ToyModel, config.TOY_MODEL), (WarehouseModel, config.WAREHOUSE_MODEL)):
        scaled = build(cfg, seed=1)
        torch.manual_seed(123)                          # the global RNG must not matter
        unscaled = build(replace(cfg, scaled=False), seed=1)
        uniform = build(replace(cfg, uniform=True), seed=1)
        for (name, a), (_, b), (_, c) in zip(scaled.state_dict().items(), unscaled.state_dict().items(),
                                             uniform.state_dict().items()):
            assert torch.equal(a, b) and torch.equal(a, c), name
        other = build(cfg, seed=2)
        assert not torch.equal(scaled.attn.W_Q, other.attn.W_Q)


def test_switches_reach_the_attention_core():
    tokens = torch.rand(3, config.TOY.n_tokens, config.TOY.token_dim)
    _, uniform = ToyModel(replace(config.TOY_MODEL, uniform=True), seed=0)(tokens)
    torch.testing.assert_close(uniform.A, torch.full_like(uniform.A, 1 / config.TOY.n_tokens))
    _, unscaled = ToyModel(replace(config.TOY_MODEL, scaled=False), seed=0)(tokens)
    torch.testing.assert_close(unscaled.S_scaled, unscaled.S)
