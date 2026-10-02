"""One training loop for every run (ARCHITECTURE D6; PHASE0 design values).

Adam, a seeded batch order and a fixed step budget. The model is evaluated at
step 0, before any update (H2, H3), and every `eval_every` steps. The best
validation checkpoint is kept, which is early stopping on validation (A-12).
Every evaluation also logs attention diagnostics on a fixed batch (FAC-27).
"""

import numpy as np
import torch

from wda.config import TrainConfig
from wda.metrics import entropy_fraction


def diagnostics(model, X: torch.Tensor, y: torch.Tensor, loss_fn, full: bool = False) -> dict:
    """Attention statistics and gradient norms on one fixed batch, without changing the model.

    Only the readout (last) row of A reaches the loss, so the query gradient is
    taken for that row only: ∂L/∂q_i = 0 exactly for the other rows (PHASE0 §6).
    """
    pred, out = model(X)
    loss = loss_fn(pred, y)
    w = model.attn
    g_q, g_k, g_v, g_Q = torch.autograd.grad(loss, [w.W_Q, w.W_K, w.W_V, out.Q], allow_unused=True)
    readout_grad = torch.zeros(len(X)) if g_Q is None else g_Q[:, -1, :].norm(dim=-1)
    A = out.A.detach()
    record = {
        "entropy_all": entropy_fraction(A).mean().item(),
        "entropy_readout": entropy_fraction(A[:, -1, :]).mean().item(),
        "max_weight_readout": A[:, -1, :].max(dim=-1).values.mean().item(),
        "logit_sd": out.S_scaled.detach().std().item(),
        "grad_W_Q": 0.0 if g_q is None else g_q.norm().item(),
        "grad_W_K": 0.0 if g_k is None else g_k.norm().item(),
        "grad_W_V": 0.0 if g_v is None else g_v.norm().item(),
    }
    for p in (10, 50, 90):
        record[f"grad_q_readout_p{p}"] = float(np.percentile(readout_grad.numpy(), p))
    if full:
        record["grad_q_readout"] = readout_grad.numpy()
        record["entropy_readout_rows"] = entropy_fraction(A[:, -1, :]).numpy()
    return record


def fit(model, train: tuple, val: tuple, loss_fn, metric_fn, cfg: TrainConfig, seed: int,
        higher_is_better: bool, diag: tuple) -> dict:
    """Train `model` in place. train, val and diag are (inputs, targets) pairs.

    Returns the per-evaluation history, the best validation weights and their step.
    """
    X_train, y_train = train
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps)
    g = torch.Generator().manual_seed(seed)
    order, pos = torch.randperm(len(X_train), generator=g), 0
    history, best, running, n_running, last_finite = [], None, 0.0, 0, None

    for step in range(cfg.steps + 1):
        if step % cfg.eval_every == 0:
            with torch.no_grad():
                pred, _ = model(val[0])
                val_loss, val_metric = loss_fn(pred, val[1]).item(), metric_fn(pred, val[1])
            record = {"step": step, "train_loss": running / n_running if n_running else None,
                      "val_loss": val_loss, "val_metric": val_metric}
            record.update(diagnostics(model, *diag, loss_fn))
            history.append(record)
            running, n_running = 0.0, 0
            better = best is None or (val_metric > best[0] if higher_is_better else val_metric < best[0])
            if better:
                best = (val_metric, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if step == cfg.steps:
            break
        if pos + cfg.batch_size > len(X_train):
            order, pos = torch.randperm(len(X_train), generator=g), 0
        idx = order[pos:pos + cfg.batch_size]
        pos += cfg.batch_size
        pred, _ = model(X_train[idx])
        loss = loss_fn(pred, y_train[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}; last finite loss {last_finite}")
        last_finite = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        running, n_running = running + last_finite, n_running + 1

    return {"history": history, "best_metric": best[0], "best_step": best[1], "best_state": best[2]}
