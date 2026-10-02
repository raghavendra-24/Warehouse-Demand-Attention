"""Every tunable value and seed, as fixed in docs/PHASE0.md.

This module is the configuration file (A-24). Each result file records the
values it used, so a run can be traced back to this file.
"""

from dataclasses import dataclass, replace

# Test-set discipline (A-12, ARCHITECTURE D12): test metrics and test
# predictions can only be produced once this is True. Switching it on is a
# commit of its own, made after the configuration is frozen.
FINAL_TEST = False

# Fixed CPU thread count, so reruns on one machine are repeatable (A-24).
NUM_THREADS = 4


@dataclass(frozen=True)
class WarehouseDataConfig:
    """Generator parameters (PHASE0 section 2). Times are in hours."""

    n_hours: int = 52 * 168            # 8,736 hours
    start_day: int = 0                 # day of week of hour 0 (0 = Monday)
    base: float = 100.0                # B, orders per hour
    daily_a1: float = 60.0             # D(h) = a1 cos(2π(h−14)/24) + a2 cos(4π(h−14)/24)
    daily_a2: float = 15.0
    daily_peak_hour: int = 14
    weekday_offsets: tuple = (5.0, 5.0, 5.0, 5.0, 0.0, -15.0, -25.0)   # W(d), Monday..Sunday
    dispersion: float = 100.0          # negative-binomial r; Var = λ + λ²/r
    floor: float = 1.0                 # λ_min
    spike_onset_prob: float = 0.004    # per hour with no active event
    spike_durations: tuple = (3, 6)    # inclusive range, hours
    spike_multiplier: tuple = (2.0, 4.0)
    drop_onset_prob: float = 0.002
    drop_durations: tuple = (2, 4)
    drop_multiplier: tuple = (0.3, 0.6)
    spike_scale: float = 1.0           # shift condition: multiplies each drawn spike multiplier
    seed: int = 101


WAREHOUSE = WarehouseDataConfig()

# Splits by the week of the target hour, weeks numbered from 1 (A-12).
TRAIN_WEEKS = (1, 36)
VAL_WEEKS = (37, 44)
TEST_WEEKS = (45, 52)
WINDOW = 24                            # input hours y(t−23) … y(t) (A-04)

# Distribution-shift series (A-20, PHASE0 section 8): 8 weeks on the test
# calendar, preceded by a 24-hour warm-up, so the series starts on the Sunday
# before a Monday and yields 1,344 targets. All four share the shift seed.
_SHIFT_BASE = replace(WAREHOUSE, n_hours=WINDOW + 8 * 168, start_day=6, seed=202)
SHIFT_SERIES = {
    "control": _SHIFT_BASE,
    "higher_noise": replace(_SHIFT_BASE, dispersion=14.0),
    "larger_spikes": replace(_SHIFT_BASE, spike_scale=2.0),
    "both": replace(_SHIFT_BASE, dispersion=14.0, spike_scale=2.0),
}


@dataclass(frozen=True)
class ToyDataConfig:
    """Associative recall (A-18, PHASE0 section 1)."""

    n_pairs: int = 8
    n_keys: int = 16
    n_values: int = 16
    n_train: int = 50_000
    n_val: int = 2_000
    n_test: int = 5_000
    seed_train: int = 301
    seed_val: int = 302
    seed_test: int = 303

    @property
    def n_tokens(self) -> int:
        return self.n_pairs + 1

    @property
    def token_dim(self) -> int:
        return self.n_keys + self.n_values + 1


TOY = ToyDataConfig()


@dataclass(frozen=True)
class ModelConfig:
    """Sizes and initialisation of one model variant (PHASE0 design values)."""

    d_model: int
    d_k: int = 16
    d_v: int = 16
    qkv_init_std: float = 0.25         # entries of W_Q, W_K, W_V ~ N(0, std²)
    d_in: int = 0                      # > 0: linear input projection d_in → d_model
    in_init_std: float = 0.0
    n_out: int = 1                     # head outputs: 1 (forecast) or classes (toy)
    scaled: bool = True
    uniform: bool = False


# Toy: ‖x‖² = 2 for every token, so std² = 1/2 gives unit-variance q and k (A-19).
TOY_MODEL = ModelConfig(d_model=TOY.token_dim, qkv_init_std=0.5 ** 0.5, n_out=TOY.n_values)
# Warehouse: E‖f‖² = 3, so W_in ~ N(0, 1/3); then W ~ N(0, 1/16) for d_model = 16.
WAREHOUSE_MODEL = ModelConfig(d_model=16, qkv_init_std=0.25, d_in=5, in_init_std=(1 / 3) ** 0.5)
ABLATION_DK = (4, 64)
TOY_TARGET_ACCURACY = 0.95            # H4: steps to 95% validation accuracy
DIAG_BATCH_SIZE = 256                 # fixed diagnostic batch (PHASE0 section 6)
# Weights kept: best validation accuracy (toy), best validation MAE (warehouse).


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    batch_size: int = 256
    steps: int = 3_000
    eval_every: int = 50
    seeds: tuple = (0, 1, 2)           # initialisation and batch order; shared by paired arms


TOY_TRAIN = TrainConfig(steps=3_000)
WAREHOUSE_TRAIN = TrainConfig(steps=4_000)

# Gradient check (A-17): sizes all different, float64, central differences.
GRADCHECK = {"n": 5, "d_model": 3, "d_k": 4, "d_v": 2, "h": 1e-6, "rtol": 1e-6, "atol": 1e-8, "seed": 7}

# The fixed tiny example (FAC-14): used by experiments.attention_trace, by the
# hand-computed forward test, and by the worked example in docs/DERIVATION.md.
# n = 2 tokens, d_model = 3, d_k = 4, d_v = 1, so every size differs. Both S = QKᵀ
# and A are deliberately asymmetric, so KQᵀ and AᵀV bugs change the result.
TINY_EXAMPLE = {
    "X": [[1.0, 0.0, 1.0],
          [0.0, 2.0, 1.0]],
    "W_Q": [[1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0]],
    "W_K": [[1.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0]],
    "W_V": [[1.0],
            [2.0],
            [0.0]],
}
