"""Training orchestration: dataset → split → fit → evaluate → persist.

The important boundaries here:
  * a loss that cannot mean anything for the target is refused, not coerced;
  * the metrics always include a naive baseline, because 80% accuracy on a dataset
    that is 80% one class means the model learned nothing;
  * the encoding is persisted with the weights so prediction cannot drift.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.dataset import (
    DatasetError,
    EncodingSpec,
    confusion_matrix,
    encode,
    encode_rows,
    parse_csv,
    per_class_metrics,
    stratified_split,
)
from app.ml.nn import MLP, LossName, TrainConfig, Trainer, validate_loss_for_task
from app.models import Dataset, TrainingRun, User


class TrainingError(Exception):
    """A configuration or data problem the user can fix, phrased for them."""


@dataclass
class TrainOutcome:
    run: TrainingRun
    metrics: dict


def _choose_loss(task: str, output_dim: int, requested: str | None) -> LossName:
    """Pick the only sensible loss, then check the request against it."""
    if task == "regression":
        natural: LossName = "mse"
    elif output_dim == 2:
        natural = "bce"
    else:
        natural = "cce"

    if requested is None:
        return natural

    loss = requested  # type: ignore[assignment]
    validate_loss_for_task(loss, task, output_dim)
    return loss


def _build_config(raw: dict, settings: Settings) -> TrainConfig:
    """Clamp the request against the deployment's ceilings, then construct the config."""
    hidden = list(raw.get("hidden_layers") or [16])
    if len(hidden) > settings.max_hidden_layers:
        raise TrainingError(
            f"At most {settings.max_hidden_layers} hidden layers are allowed (you asked for {len(hidden)})."
        )
    if any(width > settings.max_layer_width or width < 1 for width in hidden):
        raise TrainingError(
            f"Every hidden layer must be between 1 and {settings.max_layer_width} units wide."
        )

    epochs = int(raw.get("epochs", 120))
    if epochs > settings.max_epochs:
        raise TrainingError(
            f"At most {settings.max_epochs} epochs are allowed on this deployment (you asked for {epochs})."
        )

    return TrainConfig(
        hidden_layers=hidden,
        activation=raw.get("activation", "relu"),
        init=raw.get("init", "he"),
        learning_rate=float(raw.get("learning_rate", 0.05)),
        epochs=epochs,
        batch_size=int(raw.get("batch_size", 16)),
        l2=float(raw.get("l2", 0.0)),
        early_stopping=bool(raw.get("early_stopping", False)),
        patience=int(raw.get("patience", 12)),
        seed=int(raw.get("seed", 42)),
    )


def _weights_to_json(weights: list[np.ndarray], biases: list[np.ndarray]) -> str:
    return json.dumps(
        {
            "weights": [np.round(w, 6).tolist() for w in weights],
            "biases": [np.round(b, 6).tolist() for b in biases],
        }
    )


def load_weights(blob: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    parsed = json.loads(blob)
    weights = [np.array(w, dtype=np.float64) for w in parsed["weights"]]
    biases = [np.array(b, dtype=np.float64) for b in parsed["biases"]]
    return weights, biases


def run_training(
    session: Session,
    user: User,
    dataset: Dataset,
    raw_config: dict,
    settings: Settings,
) -> TrainOutcome:
    """Train one model and persist everything needed to score with it later."""
    started = time.monotonic()

    parsed = parse_csv(
        dataset.csv_body.encode("utf-8"),
        dataset.target,
        max_rows=settings.max_dataset_rows,
        max_columns=settings.max_dataset_columns,
    )
    encoded = encode(parsed)

    loss = _choose_loss(parsed.task, encoded.output_dim, raw_config.get("loss"))
    config = _build_config(raw_config, settings)
    rng = np.random.default_rng(config.seed)

    x_train, y_train, x_test, y_test = stratified_split(
        encoded.x, encoded.y, float(raw_config.get("test_fraction", 0.2)), rng
    )

    output_dim = encoded.output_dim
    model = MLP(
        input_dim=encoded.x.shape[1],
        output_dim=output_dim,
        config=config,
        rng=rng,
        linear_head=encoded.task == "regression",
    )
    trainer = Trainer(model, loss, rng)

    run = TrainingRun(
        owner_id=user.id,
        dataset_id=dataset.id,
        label=(raw_config.get("label") or "").strip()[:200],
        status="running",
        config_json=json.dumps({**raw_config, "resolved_loss": loss}),
        loss=loss,
        test_fraction=float(raw_config.get("test_fraction", 0.2)),
        param_count=model.param_count,
        train_rows=int(x_train.shape[0]),
        test_rows=int(x_test.shape[0]),
    )
    session.add(run)
    session.flush()  # assign the id without committing a half-written row

    try:
        result = trainer.fit(x_train, y_train, x_test, y_test)
    except MemoryError as exc:  # pragma: no cover - depends on the host
        run.status = "failed"
        run.error = "Ran out of memory during training."
        session.commit()
        raise TrainingError(run.error) from exc

    metrics = _evaluate(model, x_test, y_test, encoded, loss)
    metrics["task"] = encoded.task
    metrics["epochs_run"] = len(result.history)
    metrics["stopped_early"] = result.stopped_early
    metrics["best_epoch"] = result.best_epoch
    metrics["duration_ms"] = int((time.monotonic() - started) * 1000)
    metrics["input_dim"] = int(encoded.x.shape[1])

    run.status = "completed"
    run.history_json = json.dumps(
        [
            {
                "epoch": s.epoch,
                "train_loss": s.train_loss,
                "train_accuracy": s.train_accuracy,
                "val_loss": s.val_loss,
                "val_accuracy": s.val_accuracy,
            }
            for s in result.history
        ]
    )
    run.metrics_json = json.dumps(metrics)
    run.weights_json = _weights_to_json(result.weights, result.biases)
    run.encoding_json = json.dumps(encoded.spec.as_dict()) if encoded.spec else None
    run.duration_ms = int((time.monotonic() - started) * 1000)
    from datetime import UTC, datetime

    run.finished_at = datetime.now(UTC)
    session.commit()
    session.refresh(run)
    return TrainOutcome(run=run, metrics=metrics)


def _evaluate(
    model: MLP, x_test: np.ndarray, y_test: np.ndarray, encoded, loss: LossName
) -> dict:
    """Score on the held-out split, always next to a naive baseline."""
    out: dict = {"loss": round(model.loss(x_test, y_test, loss), 6)}

    if encoded.task == "classification":
        proba = model.predict_proba(x_test)
        truth = np.argmax(y_test, axis=1)

        if proba.shape[1] == 1:
            preds = (proba[:, 0] >= 0.5).astype(int)
            truth_idx = (y_test[:, 0] >= 0.5).astype(int)
            accuracy = float(np.mean(preds == truth_idx))
            # Baseline: always predict the majority class.
            majority = float(np.bincount(truth_idx).max() / len(truth_idx))
            out.update(
                {
                    "accuracy": round(accuracy, 4),
                    "majority_baseline": round(majority, 4),
                    "labels": encoded.output_labels,
                    "confusion_matrix": confusion_matrix(
                        truth_idx, preds, 2
                    ),
                    "per_class": per_class_metrics(
                        truth_idx, preds, encoded.output_labels
                    ),
                }
            )
        else:
            preds = np.argmax(proba, axis=1)
            accuracy = float(np.mean(preds == truth))
            majority = float(np.bincount(truth, minlength=proba.shape[1]).max() / len(truth))
            out.update(
                {
                    "accuracy": round(accuracy, 4),
                    "majority_baseline": round(majority, 4),
                    "labels": encoded.output_labels,
                    "confusion_matrix": confusion_matrix(truth, preds, proba.shape[1]),
                    "per_class": per_class_metrics(truth, preds, encoded.output_labels),
                }
            )
        return out

    # Regression
    preds = model.predict_proba(x_test)[:, 0] * encoded.target_std + encoded.target_mean
    truth_raw = y_test[:, 0] * encoded.target_std + encoded.target_mean
    residual = truth_raw - preds
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((truth_raw - truth_raw.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residual**2)))
    # Baseline: always predict the training mean. An RMSE above this is worse than
    # refusing to model at all, which is worth seeing explicitly.
    baseline_rmse = float(np.sqrt(np.mean((truth_raw - truth_raw.mean()) ** 2)))
    out.update(
        {
            "r2": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(float(np.mean(np.abs(residual))), 4),
            "mean_baseline_rmse": round(baseline_rmse, 4),
            "target_mean": round(encoded.target_mean, 4),
            "target_std": round(encoded.target_std, 4),
        }
    )
    return out


def predict_rows(
    dataset: Dataset,
    run: TrainingRun,
    rows: list[dict[str, float | str]],
) -> list[dict]:
    """Score arbitrary rows against a finished run.

    The row is encoded with the spec captured at training time, not re-derived from the
    dataset as it stands now. Re-deriving is the tempting approach and it is wrong: the
    z-score means shift with the new data, an unseen category silently adds a column,
    and for classification the label set is recomputed from a batch whose target column
    is empty. All three produce a vector the model was never trained on.
    """
    if run.status != "completed" or not run.weights_json:
        raise TrainingError("This run has no trained weights to predict with.")

    if not run.encoding_json:
        raise TrainingError(
            "This run predates stored encodings and cannot be scored safely. Re-run training."
        )

    spec = EncodingSpec.from_dict(json.loads(run.encoding_json))
    expected_columns = sorted(set(spec.numeric_mean) | set(spec.categories))

    unknown = set(rows[0]) - set(expected_columns) if rows else set()
    if unknown:
        raise TrainingError(
            f"Unknown column(s): {', '.join(sorted(unknown))}. "
            f"Expected: {', '.join(expected_columns)}"
        )

    new_x = encode_rows(rows, spec)

    weights, biases = load_weights(run.weights_json)
    config = TrainConfig(
        **{k: v for k, v in run.config.items() if k != "hidden_layers"},
        hidden_layers=list(run.config["hidden_layers"]),
    )
    model = MLP(
        input_dim=new_x.shape[1],
        output_dim=len(weights[-1][0]),
        config=config,
        rng=np.random.default_rng(0),
        linear_head=spec.task == "regression",
    )
    if [w.shape for w in model.weights] != [w.shape for w in weights]:
        raise TrainingError(
            "The stored weights do not match the stored encoding. Re-run training."
        )
    model.weights = weights
    model.biases = biases

    proba = model.predict_proba(new_x)
    out: list[dict] = []

    if spec.task == "classification":
        preds = np.argmax(proba, axis=1)
        for i, p in enumerate(preds):
            label = spec.labels[int(p)]
            out.append(
                {
                    "index": i,
                    "predicted": label,
                    "confidence": round(float(proba[i, int(p)]), 4),
                    "probabilities": {
                        lbl: round(float(proba[i, j]), 4)
                        for j, lbl in enumerate(spec.labels)
                    },
                }
            )
    else:
        raw = proba[:, 0] * spec.target_std + spec.target_mean
        for i, value in enumerate(raw):
            out.append(
                {
                    "index": i,
                    "predicted": round(float(value), 4),
                    "confidence": None,
                    "probabilities": {},
                }
            )
    return out
