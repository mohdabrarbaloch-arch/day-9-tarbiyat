"""A from-scratch multilayer perceptron with analytic backpropagation.

This is deliberately dependency-light: numpy plus a lot of noisy arithmetic. The point
of the project is that when the loss curve bends, something real happened — there is no
framework doing the work behind a curtain.

Everything is float64 so the numbers in the UI match the numbers in the tests.

A note on the output layer, because it is the easiest thing to get quietly wrong: the
backward pass needs the gradient with respect to the PRE-activation (z), not the
softmax/sigmoid probability. For softmax+CCE and sigmoid+BCE the activation derivative
cancels and the answer is just (p - y)/N. For MSE it does not, and for softmax+MSE you
need the full Jacobian. All of that is fused in `output_loss_and_delta` below, and
`scripts/gradcheck.py` proves it against finite differences before anything trusts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Activation = Literal["relu", "tanh", "sigmoid"]
Init = Literal["he", "xavier", "normal"]
OptimizerName = Literal["sgd", "momentum", "adam"]
LossName = Literal["mse", "bce", "cce"]


# --------------------------------------------------------------------------------------
# Activations
# --------------------------------------------------------------------------------------


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split by sign so exp() never overflows on large negative inputs.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1.0 - s)


_FORWARD = {"relu": _relu, "tanh": _tanh, "sigmoid": _sigmoid}
_BACKWARD = {"relu": _relu_grad, "tanh": _tanh_grad, "sigmoid": _sigmoid_grad}


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# Fused output activation + loss, returning the delta w.r.t. z
# --------------------------------------------------------------------------------------


def output_probabilities(z: np.ndarray, output_dim: int, *, linear: bool = False) -> np.ndarray:
    """Softmax for multi-class heads, sigmoid for a single binary output, identity for regression.

    `linear` matters more than it looks. A regression target is z-scored, so roughly a
    third of its values are negative, and a sigmoid head cannot represent those at all.
    Worse, MSE would then be measured against the squashed output rather than the value
    the model actually predicts, which understates the loss and quietly caps the best
    achievable R².
    """
    if linear:
        return z
    if output_dim > 1:
        return softmax(z)
    return _sigmoid(z)


def validate_loss_for_output(loss: LossName, output_dim: int) -> None:
    """Reject combinations whose gradient we do not implement."""
    if loss == "cce" and output_dim == 1:
        raise ValueError("cce needs at least two output units; use bce for a single output")
    if loss == "bce" and output_dim > 1:
        raise ValueError(
            "bce with multiple outputs would need independent sigmoids, which this model "
            "does not implement; use cce for a multi-class head"
        )
    if loss not in {"mse", "bce", "cce"}:
        raise ValueError(f"Unknown loss: {loss}")


def output_loss_and_delta(
    loss_name: LossName,
    z: np.ndarray,
    y_true: np.ndarray,
    output_dim: int,
    *,
    linear: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (loss, output, dL/dz) for the output layer.

    The delta is already averaged over the batch, so backprop can use it directly.
    """
    p = output_probabilities(z, output_dim, linear=linear)
    n = y_true.shape[0]
    eps = 1e-12

    if loss_name == "cce":
        p_safe = np.clip(p, eps, 1.0)
        loss = float(-np.mean(np.sum(y_true * np.log(p_safe), axis=1)))
        delta = (p - y_true) / n

    elif loss_name == "bce":
        p_safe = np.clip(p, eps, 1.0 - eps)
        loss = float(-np.mean(y_true * np.log(p_safe) + (1.0 - y_true) * np.log(1.0 - p_safe)))
        # d(sigmoid+BCE)/dz collapses to (p - y); dividing by p(1-p) here would be
        # numerically explosive and simply wrong.
        delta = (p - y_true) / n

    else:  # mse
        diff = p - y_true
        loss = float(np.mean(diff**2))
        g = 2.0 * diff / diff.size  # dL/dp
        if linear:
            delta = g  # dL/dz, the head is the identity
        elif output_dim > 1:
            # Full softmax Jacobian: dL/dz_i = p_i (g_i - sum_k g_k p_k)
            delta = p * (g - np.sum(g * p, axis=1, keepdims=True))
        else:
            delta = g * p * (1.0 - p)  # chain through sigmoid'(z)

    return loss, p, delta


# --------------------------------------------------------------------------------------
# Configuration and results
# --------------------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Everything the user can turn. Validated by the caller before it reaches here."""

    hidden_layers: list[int] = field(default_factory=lambda: [16])
    activation: Activation = "relu"
    init: Init = "he"
    optimizer: OptimizerName = "adam"
    learning_rate: float = 0.05
    epochs: int = 120
    batch_size: int = 16
    l2: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise ValueError("At least one hidden layer is required")
        if any(width < 1 for width in self.hidden_layers):
            raise ValueError("Hidden layer widths must be >= 1")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0:
            raise ValueError("l2 must be >= 0")
        if self.activation not in _FORWARD:
            raise ValueError(f"Unknown activation: {self.activation}")
        if self.optimizer not in {"sgd", "momentum", "adam"}:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.init not in {"he", "xavier", "normal"}:
            raise ValueError(f"Unknown init: {self.init}")

    def as_dict(self) -> dict[str, object]:
        return {
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
            "init": self.init,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "l2": self.l2,
            "seed": self.seed,
        }


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None


@dataclass
class TrainingResult:
    history: list[EpochMetrics]
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    input_dim: int
    output_dim: int
    config: TrainConfig
    final_train_loss: float
    final_val_loss: float | None
    best_epoch: int
    stopped_early: bool


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------


class MLP:
    """A plain multilayer perceptron. One hidden activation, softmax/sigmoid/mse head."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: TrainConfig,
        rng: np.random.Generator,
        linear_head: bool = False,
    ) -> None:
        if input_dim < 1 or output_dim < 1:
            raise ValueError("input_dim and output_dim must both be >= 1")
        self.input_dim = input_dim
        self.output_dim = output_dim
        # A regression head must be the identity; anything squashed would clip the target.
        self.linear_head = linear_head
        self.config = config
        self.rng = rng

        sizes = [input_dim, *config.hidden_layers, output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            self.weights.append(self._init_matrix(fan_in, fan_out))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))

        # Optimizer state, one entry per layer.
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t = 0

    def _init_matrix(self, fan_in: int, fan_out: int) -> np.ndarray:
        if self.config.init == "he":
            scale = math.sqrt(2.0 / fan_in)
        elif self.config.init == "xavier":
            scale = math.sqrt(1.0 / fan_in)
        else:
            scale = 0.05
        return self.rng.normal(0.0, scale, size=(fan_in, fan_out))

    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (pre-activations, activations). activations[0] is the input."""
        act = _FORWARD[self.config.activation]
        zs: list[np.ndarray] = []
        acts: list[np.ndarray] = [x]

        current = x
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = current @ w + b
            zs.append(z)
            current = z if i == last else act(z)
            acts.append(current)
        return zs, acts

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        zs, _ = self._forward(x)
        return output_probabilities(zs[-1], self.output_dim, linear=self.linear_head)

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        if self.output_dim > 1:
            return np.argmax(proba, axis=1)
        return (proba[:, 0] >= 0.5).astype(np.int64)

    def param_count(self) -> int:
        return int(sum(w.size for w in self.weights) + sum(b.size for b in self.biases))


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class Trainer:
    """Owns the optimisation loop so `MLP` stays a pure model."""

    def __init__(self, model: MLP, loss: LossName, rng: np.random.Generator) -> None:
        validate_loss_for_output(loss, model.output_dim)
        self.model = model
        self.loss_name: LossName = loss
        self.rng = rng

    def _loss(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        zs, _ = self.model._forward(x)
        loss, proba, _ = output_loss_and_delta(
            self.loss_name, zs[-1], y, self.model.output_dim, linear=self.model.linear_head
        )
        return loss, proba

    def _backward(
        self,
        x: np.ndarray,
        y: np.ndarray,
        zs: list[np.ndarray],
        acts: list[np.ndarray],
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """One backward pass. Returns loss and per-layer gradients."""
        model = self.model
        loss, _, delta = output_loss_and_delta(
            self.loss_name, zs[-1], y, model.output_dim, linear=model.linear_head
        )

        grad_w: list[np.ndarray] = [np.zeros_like(w) for w in model.weights]
        grad_b: list[np.ndarray] = [np.zeros_like(b) for b in model.biases]
        act_grad = _BACKWARD[model.config.activation]

        for layer in range(len(model.weights) - 1, -1, -1):
            grad_w[layer] = acts[layer].T @ delta
            grad_b[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ model.weights[layer].T) * act_grad(zs[layer - 1])

        return loss, grad_w, grad_b

    def _step(self, grad_w: list[np.ndarray], grad_b: list[np.ndarray]) -> None:
        model = self.model
        cfg = model.config
        model._t += 1
        t = model._t
        lr = cfg.learning_rate

        for i in range(len(model.weights)):
            gw = grad_w[i] + cfg.l2 * model.weights[i]
            gb = grad_b[i]

            if cfg.optimizer == "sgd":
                model.weights[i] -= lr * gw
                model.biases[i] -= lr * gb
                continue

            if cfg.optimizer == "momentum":
                beta = 0.9
                model._m_w[i] = beta * model._m_w[i] + gw
                model._m_b[i] = beta * model._m_b[i] + gb
                model.weights[i] -= lr * model._m_w[i]
                model.biases[i] -= lr * model._m_b[i]
                continue

            # Adam
            b1, b2, eps = 0.9, 0.999, 1e-8
            model._m_w[i] = b1 * model._m_w[i] + (1 - b1) * gw
            model._v_w[i] = b2 * model._v_w[i] + (1 - b2) * (gw**2)
            model._m_b[i] = b1 * model._m_b[i] + (1 - b1) * gb
            model._v_b[i] = b2 * model._v_b[i] + (1 - b2) * (gb**2)

            m_hat_w = model._m_w[i] / (1 - b1**t)
            v_hat_w = model._v_w[i] / (1 - b2**t)
            m_hat_b = model._m_b[i] / (1 - b1**t)
            v_hat_b = model._v_b[i] / (1 - b2**t)

            model.weights[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + eps)
            model.biases[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def _accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if x.shape[0] == 0:
            return 0.0
        pred = self.model.predict(x)
        if self.model.output_dim > 1:
            truth = np.argmax(y, axis=1)
        else:
            truth = (y[:, 0] >= 0.5).astype(np.int64)
        return float(np.mean(pred == truth))

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int | None = None,
        min_delta: float = 1e-5,
    ) -> TrainingResult:
        """Train in mini-batches. Optional early stopping on validation loss."""
        cfg = self.model.config
        n = x_train.shape[0]
        history: list[EpochMetrics] = []

        best_val = math.inf
        best_epoch = 0
        best_w = [w.copy() for w in self.model.weights]
        best_b = [b.copy() for b in self.model.biases]
        stale = 0
        stopped_early = False

        for epoch in range(1, cfg.epochs + 1):
            order = self.rng.permutation(n)
            epoch_losses: list[float] = []

            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                zs, acts = self.model._forward(xb)
                loss, gw, gb = self._backward(xb, yb, zs, acts)
                self._step(gw, gb)
                epoch_losses.append(loss)

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            train_acc = self._accuracy(x_train, y_train)

            val_loss = val_acc = None
            if x_val is not None and y_val is not None and x_val.shape[0] > 0:
                val_loss, _ = self._loss(x_val, y_val)
                val_acc = self._accuracy(x_val, y_val)

            history.append(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_loss,
                    train_accuracy=train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                )
            )

            if val_loss is not None and patience is not None:
                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    best_epoch = epoch
                    best_w = [w.copy() for w in self.model.weights]
                    best_b = [b.copy() for b in self.model.biases]
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stopped_early = True
                        break

        # Restore the best weights if early stopping was armed; otherwise keep the last.
        if patience is not None and best_epoch > 0:
            self.model.weights = best_w
            self.model.biases = best_b
        else:
            best_epoch = len(history)

        final_train = history[-1].train_loss if history else 0.0
        final_val = history[-1].val_loss if history else None

        return TrainingResult(
            history=history,
            weights=self.model.weights,
            biases=self.model.biases,
            input_dim=self.model.input_dim,
            output_dim=self.model.output_dim,
            config=cfg,
            final_train_loss=final_train,
            final_val_loss=final_val,
            best_epoch=best_epoch,
            stopped_early=stopped_early,
        )
