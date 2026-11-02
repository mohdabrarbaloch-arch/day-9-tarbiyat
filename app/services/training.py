"""Training orchestration: dataset → split → fit → evaluate → persist.

The important design decision in this file is the loss resolver. A classification head
cannot be scored with MSE and come out meaningful, and a regression target cannot be
trained with cross-entropy. Rather than quietly substituting one for the other and
reporting a number that looks fine, `resolve_loss` refuses the combination and says why.
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
    EncodedDataset,
    EncodingSpec,
    confusion_matrix,
    encode,
    encode_rows,
    parse_csv,
    per_class_metrics,
    stratified_split,
)
from app.ml.nn import MLP, LossName, TrainConfig, Trainer, validate_loss_for_output
from app.models import Dataset, TrainingRun, User


class TrainingError(Exception):
    """A run could not be completed for a reason the user can act on."""


@dataclass
class RunOutcome:
    run: TrainingRun
    test_predictions: np.ndarray
    test_truth: np.ndarray


def resolve_loss(task: str, output_dim: int, requested: LossName | None) -> LossName:
    """Pick a loss that is actually valid for this head, or refuse."""
    if requested is not None:
        validate_loss_for_output(requested, output_dim)
        if task == "regression" and requested != "mse":
            raise TrainingError(
                f"'{requested}' is a classification loss but this dataset has a continuous "
                "target. Use mse, or pick a target column with few distinct values."
            )
        if task == "classification" and requested == "mse":
            raise TrainingError(
                "mse on a classification head produces a number that looks like a loss but "
                "means nothing for accuracy. Use cce for multi-class, bce for two classes."
            )
        return requested

    if task == "regression":
        return "mse"
    return "bce" if output_dim == 1 else "cce"


def _weights_to_json(weights: list[np.ndarray], biases: list[np.ndarray]) -> str:
    return json.dumps(
        {
            "weights": [w.tolist() for w in weights],
            "biases": [b.tolist() for b in biases],
        }
    )


def load_weights(payload: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    blob = json.loads(payload)
    weights = [np.array(w, dtype=np.float64) for w in blob["weights"]]
    biases = [np.array(b, dtype=np.float64) for b in blob["biases"]]
    return weights, biases


def run_output_dim(run: TrainingRun) -> int:
    """Recover the output width a finished run was trained with."""
    weights, _ = load_weights(run.weights_json or '{"weights": [], "biases": []}')
    if not weights:
        raise TrainingError("This run has no stored weights to predict with.")
    return int(weights[-1].shape[1])


def _rebuild_encoded(dataset: Dataset) -> EncodedDataset:
    """Re-derive the encoded matrices from the stored CSV. Single source of truth."""
    parsed = parse_csv(
        dataset.csv_body.encode("utf-8"),
        target=dataset.target,
        max_rows=20_000,
        max_columns=60,
    )
    return encode(parsed)


def run_training(
    session: Session,
    user: User,
    dataset: Dataset,
    request_config: dict,
    settings: Settings,
) -> RunOutcome:
    """Train synchronously and persist the result. Raises TrainingError on bad input."""
    started = time.perf_counter()

    encoded = _rebuild_encoded(dataset)

    output_dim = encoded.output_dim
    loss = resolve_loss(encoded.task, output_dim, request_config.get("loss"))

    cfg = TrainConfig(
        hidden_layers=list(request_config["hidden_layers"]),
        activation=request_config["activation"],
        init=request_config["init"],
        optimizer=request_config["optimizer"],
        learning_rate=float(request_config["learning_rate"]),
        epochs=int(request_config["epochs"]),
        batch_size=int(request_config["batch_size"]),
        l2=float(request_config["l2"]),
        seed=int(request_config["seed"]),
    )

    if cfg.epochs > settings.max_epochs:
        raise TrainingError(f"epochs is capped at {settings.max_epochs} on this deployment.")
    if len(cfg.hidden_layers) > settings.max_hidden_layers:
        raise TrainingError(f"At most {settings.max_hidden_layers} hidden layers are allowed here.")

    rng = np.random.default_rng(cfg.seed)
    test_fraction = float(request_config.get("test_fraction", 0.2))

    try:
        train_idx, test_idx = stratified_split(encoded.y, test_fraction, rng)
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc

    if train_idx.size == 0:
        raise DatasetError("The split produced no training rows.")

    x_train, y_train = encoded.x[train_idx], encoded.y[train_idx]

    # Carve a validation slice out of training when early stopping is requested.
    early_stopping = bool(request_config.get("early_stopping"))
    x_val = y_val = None
    if early_stopping and train_idx.size >= 20:
        n_val = max(4, int(round(train_idx.size * 0.15)))
        val_idx = train_idx[-n_val:]
        fit_idx = train_idx[:-n_val]
        x_train, y_train = encoded.x[fit_idx], encoded.y[fit_idx]
        x_val, y_val = encoded.x[val_idx], encoded.y[val_idx]

    model = MLP(
        input_dim=encoded.x.shape[1],
        output_dim=output_dim,
        config=cfg,
        rng=rng,
        linear_head=encoded.task == "regression",
    )
    trainer = Trainer(model, loss, rng)

    result = trainer.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        patience=int(request_config.get("patience", 12)) if early_stopping else None,
    )

    x_test, y_test = encoded.x[test_idx], encoded.y[test_idx]

    if encoded.task == "classification":
        test_pred = model.predict(x_test)
        test_truth = np.argmax(y_test, axis=1)
        accuracy = float(np.mean(test_pred == test_truth)) if test_truth.size else 0.0
        metrics: dict = {
            "task": "classification",
            "accuracy": round(accuracy, 4),
            "test_rows": int(test_truth.size),
            "confusion_matrix": confusion_matrix(test_pred, test_truth, output_dim),
            "per_class": per_class_metrics(test_pred, test_truth, encoded.output_labels),
            "labels": encoded.output_labels,
            "loss": loss,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "final_train_loss": round(result.final_train_loss, 6),
            "final_val_loss": (
                round(result.final_val_loss, 6) if result.final_val_loss is not None else None
            ),
        }
        # A majority-class baseline makes the accuracy number interpretable.
        counts = np.bincount(test_truth, minlength=output_dim)
        metrics["majority_baseline"] = (
            round(float(counts.max() / test_truth.size), 4) if test_truth.size else 0.0
        )
    else:
        pred_std = model.predict_proba(x_test)[:, 0]
        pred_raw = pred_std * encoded.target_std + encoded.target_mean
        truth_raw = y_test[:, 0] * encoded.target_std + encoded.target_mean
        resid = pred_raw - truth_raw
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((truth_raw - truth_raw.mean()) ** 2)) or 1.0
        metrics = {
            "task": "regression",
            "rmse": round(float(np.sqrt(np.mean(resid**2))), 4),
            "mae": round(float(np.mean(np.abs(resid))), 4),
            "r2": round(1.0 - ss_res / ss_tot, 4),
            "test_rows": int(truth_raw.size),
            "target_mean": round(encoded.target_mean, 4),
            "target_std": round(encoded.target_std, 4),
            "loss": loss,
            "best_epoch": result.best_epoch,
            "stopped_early": result.stopped_early,
            "final_train_loss": round(result.final_train_loss, 6),
            "final_val_loss": (
                round(result.final_val_loss, 6) if result.final_val_loss is not None else None
            ),
            # A constant predictor's RMSE — if the model cannot beat this, it learned nothing.
            "baseline_rmse": round(float(np.std(truth_raw)), 4),
        }
        test_pred = pred_raw
        test_truth = truth_raw

    duration_ms = int((time.perf_counter() - started) * 1000)

    run = TrainingRun(
        owner_id=user.id,
        dataset_id=dataset.id,
        label=request_config.get("label") or f"{dataset.name} · {cfg.optimizer}",
        status="completed",
        config_json=json.dumps(cfg.as_dict()),
        loss=loss,
        test_fraction=test_fraction,
        history_json=json.dumps([h.__dict__ for h in result.history]),
        metrics_json=json.dumps(metrics),
        weights_json=_weights_to_json(result.weights, result.biases),
        encoding_json=json.dumps(encoded.spec.as_dict()) if encoded.spec else None,
        param_count=model.param_count(),
        train_rows=int(x_train.shape[0]),
        test_rows=int(test_idx.size),
        duration_ms=duration_ms,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    return RunOutcome(run=run, test_predictions=test_pred, test_truth=test_truth)


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
        raise TrainingError("The stored weights do not match the stored encoding. Re-run training.")
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
                        lbl: round(float(proba[i, j]), 4) for j, lbl in enumerate(spec.labels)
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
