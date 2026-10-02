"""Associative recall, the toy learning task (PRD §12, A-18, PHASE0 section 1).

Each sequence holds n_pairs tokens, one per key–value pair, followed by a query
token. Keys are distinct within a sequence, and so are values, so guessing
among the values present gives accuracy 1/n_pairs. Token layout (length
n_keys + n_values + 1):

    pair token   [one-hot key | one-hot value | 0]
    query token  [one-hot key | zeros         | 1]

The target is the index of the value paired with the query key.
"""

import torch

from wda.config import ToyDataConfig


def make_recall(cfg: ToyDataConfig, n: int, seed: int) -> tuple:
    """(tokens of shape (n, n_pairs + 1, token_dim), targets of shape (n,))."""
    g = torch.Generator().manual_seed(seed)
    keys = torch.argsort(torch.rand(n, cfg.n_keys, generator=g), dim=1)[:, :cfg.n_pairs]       # distinct keys
    values = torch.argsort(torch.rand(n, cfg.n_values, generator=g), dim=1)[:, :cfg.n_pairs]   # distinct values
    which = torch.randint(0, cfg.n_pairs, (n,), generator=g)                                  # pair being asked about
    rows = torch.arange(n)

    tokens = torch.zeros(n, cfg.n_tokens, cfg.token_dim)
    pair = torch.arange(cfg.n_pairs)
    tokens[rows[:, None], pair, keys] = 1.0
    tokens[rows[:, None], pair, cfg.n_keys + values] = 1.0
    tokens[rows, cfg.n_pairs, keys[rows, which]] = 1.0                   # query: the asked key ...
    tokens[rows, cfg.n_pairs, cfg.token_dim - 1] = 1.0                    # ... and the query flag
    return tokens, values[rows, which]


def splits(cfg: ToyDataConfig) -> dict:
    """Train, validation and test sequences, each from its own seed."""
    return {
        "train": make_recall(cfg, cfg.n_train, cfg.seed_train),
        "val": make_recall(cfg, cfg.n_val, cfg.seed_val),
        "test": make_recall(cfg, cfg.n_test, cfg.seed_test),
    }
