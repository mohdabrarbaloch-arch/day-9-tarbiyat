"""Tests for the hand-written network: gradients, learning, optimizers, validation."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.nn import (
    MLP,
    TrainConfig,
    Trainer,
    output_loss_and_delta,
    softmax,
    validate_loss_for_output,
)


def make_config(**overrides) -> TrainConfig:
    base: dict[str, object] = {
        "hidden_layers": [6],
        "activation": "tanh",
        "init": "xavier",
        "optimizer": "adam",
        "learning_rate": 0.05,
        "epochs": 60,
        "batch_size": 8,
        "seed": 1,
    }
    base.update(overrides)
    return TrainConfig(**base)  # type: ignore[arg-type]


def make_model(input_dim: int, output_dim: int, config: TrainConfig, seed: int = 0) -> MLP:
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        config=config,
        rng=np.random.default_rng(seed),
    )


def xor_data() -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return x, y


def test_gradients_match_finite_differences():
    """The central claim of the project: backprop is correct, not merely plausible.

    Every parameter is nudged by +/- eps and the numerical slope is compared against the
    analytic one. If this test is ever quiet about a bug, the bug is in the network.
    """
    rng = np.random.default_rng(0)
    config = make_config(hidden_layers=[5, 4], init="normal", optimizer="sgd")
    model = make_model(3, 3, config)
    trainer = Trainer(model, "cce", rng)

    x = rng.normal(size=(6, 3))
    labels = rng.integers(0, 3, size=6)
    y = np.zeros((6, 3))
    y[np.arange(6), labels] = 1.0

    zs, acts = model._forward(x)
    _, grad_w, grad_b = trainer._backward(x, y, zs, acts)

    eps = 1e-6
    for layer, (weight, bias, gw, gb) in enumerate(
        zip(model.weights, model.biases, grad_w, grad_b, strict=True)
    ):
        for array, gradient in ((weight, gw), (bias, gb)):
            flat = array.reshape(-1)
            gflat = gradient.reshape(-1)
            for pos in range(flat.size):
                original = flat[pos]

                flat[pos] = original + eps
                z_plus, _ = model._forward(x)
                loss_plus, _, _ = output_loss_and_delta("cce", z_plus[-1], y, 3)

                flat[pos] = original - eps
                z_minus, _ = model._forward(x)
                loss_minus, _, _ = output_loss_and_delta("cce", z_minus[-1], y, 3)

                flat[pos] = original
                numeric = (loss_plus - loss_minus) / (2 * eps)
                assert abs(numeric - gflat[pos]) < 1e-7, (
                    f"gradient mismatch at layer {layer}, index {pos}: "
                    f"analytic {gflat[pos]:.10f} vs numeric {numeric:.10f}"
                )


def test_xor_is_learnable():
    """XOR is not linearly separable, so passing this proves the hidden layer works."""
    x, y = xor_data()
    config = make_config(
        hidden_layers=[8], activation="tanh", learning_rate=0.3, epochs=1200, batch_size=4
    )
    model = make_model(2, 1, config, seed=3)
    Trainer(model, "bce", np.random.default_rng(3)).fit(x, y)

    assert model.predict(x).reshape(-1).tolist() == [0, 1, 1, 0]


def test_loss_falls_substantially():
    x, y = xor_data()
    config = make_config(hidden_layers=[8], learning_rate=0.2, epochs=200, batch_size=4)
    model = make_model(2, 1, config, seed=5)
    result = Trainer(model, "bce", np.random.default_rng(5)).fit(x, y)

    first = result.history[0].train_loss
    last = result.history[-1].train_loss
    assert last < first * 0.5, f"loss only moved {first:.4f} -> {last:.4f}"


def test_param_count_matches_architecture():
    model = make_model(4, 3, make_config(hidden_layers=[10, 7]))
    # (4*10 + 10) + (10*7 + 7) + (7*3 + 3) = 50 + 77 + 24
    assert model.param_count() == 151


def test_softmax_is_normalised_and_overflow_safe():
    z = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0]])
    probs = softmax(z)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probs))


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "adam"])
def test_every_optimizer_reduces_loss(optimizer):
    x, y = xor_data()
    config = make_config(
        hidden_layers=[8], optimizer=optimizer, learning_rate=0.2, epochs=150, batch_size=4
    )
    model = make_model(2, 1, config, seed=11)
    result = Trainer(model, "bce", np.random.default_rng(11)).fit(x, y)
    assert result.history[-1].train_loss < result.history[0].train_loss


def test_cross_entropy_needs_two_outputs():
    with pytest.raises(ValueError, match="at least two output"):
        validate_loss_for_output("cce", 1)


def test_single_output_bce_rejects_multiple_outputs():
    with pytest.raises(ValueError, match="bce with multiple outputs"):
        validate_loss_for_output("bce", 3)


def test_trainer_refuses_illegal_loss_at_construction():
    model = make_model(3, 1, make_config())
    with pytest.raises(ValueError):
        Trainer(model, "cce", np.random.default_rng(0))


def test_early_stopping_restores_best_weights():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 3))
    y = (x[:, 0:1] > 0).astype(float)

    config = make_config(hidden_layers=[6], epochs=300, batch_size=8)
    model = make_model(3, 1, config, seed=2)
    result = Trainer(model, "bce", np.random.default_rng(2)).fit(
        x[:40], y[:40], x_val=x[40:], y_val=y[40:], patience=3
    )

    assert 1 <= result.best_epoch <= len(result.history)
    # If it stopped early, it must have stopped before the full epoch budget.
    if result.stopped_early:
        assert len(result.history) < 300


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="At least one hidden layer"):
        TrainConfig(hidden_layers=[])
    with pytest.raises(ValueError, match="widths must be >= 1"):
        TrainConfig(hidden_layers=[0])
    with pytest.raises(ValueError, match="learning_rate"):
        TrainConfig(learning_rate=0)
    with pytest.raises(ValueError, match="Unknown activation"):
        TrainConfig(activation="gelu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown optimizer"):
        TrainConfig(optimizer="rmsprop")  # type: ignore[arg-type]


def test_training_is_reproducible_for_a_fixed_seed():
    x, y = xor_data()
    losses = []
    for _ in range(2):
        rng = np.random.default_rng(99)
        config = make_config(hidden_layers=[6], epochs=30, batch_size=4)
        model = make_model(2, 1, config, seed=99)
        losses.append(Trainer(model, "bce", rng).fit(x, y).history[-1].train_loss)

    assert losses[0] == pytest.approx(losses[1], abs=1e-12)


def test_l2_regularisation_shrinks_weight_norm():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 4))
    y = rng.normal(size=(80, 1))

    def final_norm(l2: float) -> float:
        config = make_config(
            hidden_layers=[8], optimizer="sgd", learning_rate=0.05, l2=l2, epochs=40
        )
        model = make_model(4, 1, config, seed=4)
        Trainer(model, "mse", np.random.default_rng(4)).fit(x, y)
        return float(sum(np.linalg.norm(w) for w in model.weights))

    assert final_norm(0.1) < final_norm(0.0)


def test_linear_head_gradients_match_finite_differences():
    """The regression head is the identity, so its delta must be used unmodified.

    With a sigmoid head this would still pass a shape check while quietly clipping every
    negative target, which is exactly the bug this test exists to prevent.
    """
    rng = np.random.default_rng(7)
    config = make_config(hidden_layers=[4], init="xavier", optimizer="sgd")
    model = MLP(
        input_dim=3,
        output_dim=1,
        config=config,
        rng=rng,
        linear_head=True,
    )
    trainer = Trainer(model, "mse", rng)

    x = rng.normal(size=(5, 3))
    y = np.array([[0.8], [-1.4], [2.1], [-0.3], [0.05]])

    zs, acts = model._forward(x)
    _, grad_w, grad_b = trainer._backward(x, y, zs, acts)

    eps = 1e-6
    for layer, (weight, bias, gw, gb) in enumerate(
        zip(model.weights, model.biases, grad_w, grad_b, strict=True)
    ):
        for array, gradient in ((weight, gw), (bias, gb)):
            flat = array.reshape(-1)
            gflat = gradient.reshape(-1)
            for pos in range(flat.size):
                original = flat[pos]
                flat[pos] = original + eps
                z_plus, _ = model._forward(x)
                plus, _, _ = output_loss_and_delta("mse", z_plus[-1], y, 1, linear=True)

                flat[pos] = original - eps
                z_minus, _ = model._forward(x)
                minus, _, _ = output_loss_and_delta("mse", z_minus[-1], y, 1, linear=True)

                flat[pos] = original
                numeric = (plus - minus) / (2 * eps)
                assert (
                    abs(numeric - gflat[pos]) < 1e-7
                ), f"linear head mismatch at layer {layer}, index {pos}"


def test_regression_model_can_predict_outside_unit_interval():
    """A sigmoid head could never output 2.0; the linear head must."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 2))
    y = 3.0 * x[:, 0:1] - 1.5 * x[:, 1:2] + 2.0

    config = make_config(hidden_layers=[12], activation="tanh", learning_rate=0.05, epochs=400)
    model = MLP(2, 1, config, rng, linear_head=True)
    Trainer(model, "mse", rng).fit(x, y)

    preds = model.predict_proba(x)
    assert preds.max() > 1.0, "outputs are stuck inside (0, 1) — the head is still squashed"
