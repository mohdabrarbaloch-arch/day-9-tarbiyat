"""Finite-difference gradient check for the hand-written backprop.

Compares analytic gradients against central differences on the loss. If these disagree,
every loss curve the app draws is fiction, so this runs before anything else.

Usage:  python scripts/gradcheck.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nn import MLP, TrainConfig, Trainer, output_loss_and_delta

TOLERANCE = 1e-6


def check(loss_name: str, output_dim: int, activation: str, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    cfg = TrainConfig(
        hidden_layers=[5, 4],
        activation=activation,  # type: ignore[arg-type]
        init="normal",
        optimizer="sgd",
        learning_rate=0.01,
        epochs=1,
        batch_size=8,
        seed=seed,
    )
    model = MLP(input_dim=3, output_dim=output_dim, config=cfg, rng=rng)
    trainer = Trainer(model, loss_name, rng)  # type: ignore[arg-type]

    x = rng.normal(size=(6, 3))
    if output_dim > 1:
        idx = rng.integers(0, output_dim, size=6)
        y = np.zeros((6, output_dim))
        y[np.arange(6), idx] = 1.0
    else:
        y = (rng.random((6, 1)) > 0.5).astype(np.float64)

    zs, acts = model._forward(x)
    _, gw, gb = trainer._backward(x, y, zs, acts)

    eps = 1e-6
    worst = 0.0
    for layer in range(len(model.weights)):
        for arr, grad in ((model.weights[layer], gw[layer]), (model.biases[layer], gb[layer])):
            flat = arr.reshape(-1)
            gflat = grad.reshape(-1)
            for pos in range(flat.size):
                original = flat[pos]
                flat[pos] = original + eps
                zs_p, _ = model._forward(x)
                loss_p, _, _ = output_loss_and_delta(loss_name, zs_p[-1], y, output_dim)  # type: ignore[arg-type]

                flat[pos] = original - eps
                zs_m, _ = model._forward(x)
                loss_m, _, _ = output_loss_and_delta(loss_name, zs_m[-1], y, output_dim)  # type: ignore[arg-type]

                flat[pos] = original
                numeric = (loss_p - loss_m) / (2 * eps)
                analytic = gflat[pos]
                denom = max(1.0, abs(numeric), abs(analytic))
                worst = max(worst, abs(numeric - analytic) / denom)
    return worst


def main() -> int:
    # Only combinations the model actually supports are exercised; the illegal ones are
    # asserted to raise in tests/test_nn.py.
    cases = [
        ("cce", 3, "relu"),
        ("cce", 4, "tanh"),
        ("cce", 3, "sigmoid"),
        ("bce", 1, "relu"),
        ("bce", 1, "sigmoid"),
        ("bce", 1, "tanh"),
        ("mse", 1, "tanh"),
        ("mse", 1, "relu"),
        ("mse", 3, "relu"),
        ("mse", 3, "tanh"),
    ]
    failures = 0
    for loss, out_dim, act in cases:
        err = check(loss, out_dim, act)
        ok = err < TOLERANCE
        if not ok:
            failures += 1
        print(
            f"{'PASS' if ok else 'FAIL'}  loss={loss:<4} out={out_dim} act={act:<7} max_rel_err={err:.3e}"
        )
    print(f"\n{len(cases) - failures}/{len(cases)} combinations passed (tol={TOLERANCE:g})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
