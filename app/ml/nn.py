"""A from-scratch multilayer perceptron with analytic backpropagation.

This is the whole point of the project: no PyTorch, no TensorFlow, no autograd. The
forward pass, the gradients and the optimizers are written out longhand in NumPy.

Design decision worth calling out: the output layer fuses its loss with its gradient, so
the delta it hands back is dL/dz (pre-activation), not dL/da (post-activation). Getting
that wrong is invisible in the common softmax+cross-entropy case, because the softmax
Jacobian and the CE derivative cancel to exactly `p - y` either way — which is precisely
why the bug survives casual testing and shows up only as "MSE never converges" or
"sigmoid regression is weirdly bad".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Activation = Literal["relu", "tanh", "sigmoid"]
Optimizer = Literal["sgd", "momentum", "adam"]
Init = Literal["he", "xavier", "normal"]
LossName = Literal["cce", "bce", "mse"]


@dataclass
class TrainConfig:
    hidden_layers: list[int] = field(default_factory=lambda: [16])
    activation: Activation = "relu"
    init: Init = "he"
    learning_rate: float = 0.05
    epochs: int = 120
    batch_size: int = 16
    l2: float = 0.0
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    early_stopping: bool = False
    patience: int = 12
    seed: int = 42


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_accuracy: float | None = None
    val_loss: float | None = None
    val_accuracy: float | None = None


@dataclass
class TrainResult:
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    history: list[EpochStats]
    stopped_early: bool
    best_epoch: int


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function (no overflow for large |z|)."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row max so exp() cannot overflow."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def output_probabilities(
    z: np.ndarray, output_dim: int, *, linear: bool = False
) -> np.ndarray:
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


class MLP:
    """A fully-connected network with a configurable depth."""

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

        dims = [input_dim, *config.hidden_layers, output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        for layer, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            is_output = layer == len(dims) - 2
            if config.init == "he":
                scale = np.sqrt(2.0 / fan_in)
            elif config.init == "xavier":
                scale = np.sqrt(1.0 / fan_in)
            else:
                scale = 0.05
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float64))
            if is_output and self.linear_head:
                # Start a regression head near zero so the initial prediction is the mean.
                self.weights[-1][:] *= 0.1

    @property
    def param_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.config.activation == "relu":
            return np.maximum(0.0, z)
        if self.config.activation == "tanh":
            return np.tanh(z)
        return _sigmoid(z)

    def _activation_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        if self.config.activation == "relu":
            return (z > 0).astype(np.float64)
        if self.config.activation == "tanh":
            return 1.0 - a**2
        return a * (1.0 - a)

    def _forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (pre-activations, activations). The last activation is the raw output z."""
        zs: list[np.ndarray] = []
        activations: list[np.ndarray] = [x]
        current = x
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = current @ w + b
            zs.append(z)
            current = z if i == last else self._activate(z)
            activations.append(current)
        return zs, activations

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        zs, _ = self._forward(x)
        return output_probabilities(zs[-1], self.output_dim, linear=self.linear_head)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x)

    def backward(self, x: np.ndarray, delta_out: np.ndarray) -> tuple[list, list]:
        """Backprop from the output delta. Returns (weight grads, bias grads)."""
        zs, activations = self._forward(x)
        weight_grads: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        bias_grads: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]

        delta = delta_out
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_grads[layer] = activations[layer].T @ delta
            bias_grads[layer] = np.sum(delta, axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * self._activation_grad(
                    zs[layer - 1], activations[layer]
                )
        return weight_grads, bias_grads

    def loss(self, x: np.ndarray, y: np.ndarray, loss_name: LossName) -> float:
        zs, _ = self._forward(x)
        value, _, _ = output_loss_and_delta(
            loss_name, zs[-1], y, self.output_dim, linear=self.linear_head
        )
        return value


class Trainer:
    """Mini-batch gradient descent with a choice of optimizer."""

    def __init__(self, model: MLP, loss: LossName, rng: np.random.Generator) -> None:
        self.model = model
        self.loss_name: LossName = loss
        self.rng = rng

    def _loss(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        zs, _ = self._forward_cached(x)
        loss, _, delta = output_loss_and_delta(
            self.loss_name, zs[-1], y, self.model.output_dim, linear=self.model.linear_head
        )
        return loss, delta

    def _forward_cached(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return self.model._forward(x)

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> TrainResult:
        cfg = self.model.config
        n = x_train.shape[0]
        batch_size = max(1, min(cfg.batch_size, n))

        velocities = [np.zeros_like(w) for w in self.model.weights]
        velocity_b = [np.zeros_like(b) for b in self.model.biases]
        m_w = [np.zeros_like(w) for w in self.model.weights]
        v_w = [np.zeros_like(w) for w in self.model.weights]
        m_b = [np.zeros_like(b) for b in self.model.biases]
        v_b = [np.zeros_like(b) for b in self.model.biases]

        history: list[EpochStats] = []
        best_val = float("inf")
        best_epoch = 0
        best_weights = [w.copy() for w in self.model.weights]
        best_biases = [b.copy() for b in self.model.biases]
        stale = 0
        stopped_early = False
        step = 0

        for epoch in range(1, cfg.epochs + 1):
            order = self.rng.permutation(n)
            x_shuffled = x_train[order]
            y_shuffled = y_train[order]

            for start in range(0, n, batch_size):
                xb = x_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                if xb.shape[0] == 0:
                    continue
                _, delta = self._loss(xb, yb)
                weight_grads, bias_grads = self.model.backward(xb, delta)
                step += 1
                self._apply(weight_grads, bias_grads, velocities, velocity_b,
                            m_w, v_w, m_b, v_b, step)

            train_loss = self.model.loss(x_train, y_train, self.loss_name)
            stats = EpochStats(
                epoch=epoch,
                train_loss=round(float(train_loss), 6),
                train_accuracy=(
                    round(float(self._accuracy(x_train, y_train)), 4)
                    if self.model.output_dim > 1 or self.loss_name != "mse"
                    else None
                ),
            )

            if x_val is not None and y_val is not None and y_val.shape[0]:
                val_loss = self.model.loss(x_val, y_val, self.loss_name)
                stats.val_loss = round(float(val_loss), 6)
                stats.val_accuracy = (
                    round(float(self._accuracy(x_val, y_val)), 4)
                    if self.model.output_dim > 1 or self.loss_name != "mse"
                    else None
                )

                if val_loss < best_val - 1e-6:
                    best_val = float(val_loss)
                    best_epoch = epoch
                    stale = 0
                    best_weights = [w.copy() for w in self.model.weights]
                    best_biases = [b.copy() for b in self.model.biases]
                else:
                    stale += 1
                    if cfg.early_stopping and stale >= cfg.patience:
                        stopped_early = True
                        history.append(stats)
                        break

            history.append(stats)

        if best_epoch > 0:
            self.model.weights = best_weights
            self.model.biases = best_biases

        return TrainResult(
            weights=self.model.weights,
            biases=self.model.biases,
            history=history,
            stopped_early=stopped_early,
            best_epoch=best_epoch or len(history),
        )

    def _accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        proba = self.model.predict_proba(x)
        if self.model.output_dim > 1:
            preds = np.argmax(proba, axis=1)
            truth = np.argmax(y, axis=1)
        else:
            preds = (proba[:, 0] >= 0.5).astype(int)
            truth = (y[:, 0] >= 0.5).astype(int)
        return float(np.mean(preds == truth))

    def _apply(
        self,
        weight_grads: list[np.ndarray],
        bias_grads: list[np.ndarray],
        velocities: list[np.ndarray],
        velocity_b: list[np.ndarray],
        m_w: list[np.ndarray],
        v_w: list[np.ndarray],
        m_b: list[np.ndarray],
        v_b: list[np.ndarray],
        step: int,
    ) -> None:
        cfg = self.model.config
        lr = cfg.learning_rate

        for i in range(len(self.model.weights)):
            gw = weight_grads[i]
            gb = bias_grads[i]

            # L2 is a weight decay applied to the gradient, not the bias.
            if cfg.l2 > 0:
                gw = gw + cfg.l2 * self.model.weights[i]

            if cfg.optimizer == "sgd":
                self.model.weights[i] -= lr * gw
                self.model.biases[i] -= lr * gb

            elif cfg.optimizer == "momentum":
                velocities[i] = cfg.momentum * velocities[i] - lr * gw
                velocity_b[i] = cfg.momentum * velocity_b[i] - lr * gb
                self.model.weights[i] += velocities[i]
                self.model.biases[i] += velocity_b[i]

            else:  # adam
                m_w[i] = cfg.beta1 * m_w[i] + (1 - cfg.beta1) * gw
                v_w[i] = cfg.beta2 * v_w[i] + (1 - cfg.beta2) * (gw**2)
                m_b[i] = cfg.beta1 * m_b[i] + (1 - cfg.beta1) * gb
                v_b[i] = cfg.beta2 * v_b[i] + (1 - cfg.beta2) * (gb**2)

                m_w_hat = m_w[i] / (1 - cfg.beta1**step)
                v_w_hat = v_w[i] / (1 - cfg.beta2**step)
                m_b_hat = m_b[i] / (1 - cfg.beta1**step)
                v_b_hat = v_b[i] / (1 - cfg.beta2**step)

                self.model.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + cfg.epsilon)
                self.model.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + cfg.epsilon)


def validate_loss_for_task(
    loss: LossName, task: str, output_dim: int
) -> None:
    """Refuse a loss that cannot mean anything for this target, instead of coercing it.

    Every rejection here corresponds to a silent-garbage outcome if allowed through.
    """
    if task == "regression" and loss != "mse":
        raise ValueError(
            f"{loss} cannot be used with a continuous target. Use mse, or change the target column."
        )
    if task == "classification" and loss == "mse":
        raise ValueError(
            "mse on a class label trains poorly and reports no useful confidence. Use cce or bce."
        )
    if loss == "cce" and output_dim < 2:
        raise ValueError(
            "cce needs at least two classes. This target has one, so use bce instead."
        )
    if loss == "bce" and output_dim > 1:
        raise ValueError(
            f"bce expects a single output, but this target has {output_dim} classes. Use cce."
        )
