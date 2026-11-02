"""CSV parsing, column profiling, encoding and the train/test split.

The rule here: everything the encoder learns about the data must be capturable and
reusable. If prediction has to re-derive it, the two paths can disagree.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

MAX_DISTINCT_CATEGORIES = 60
NUMERIC_LIKE = 0.95  # a column is numeric if this share of values parse as floats


class DatasetError(Exception):
    """Raised for anything wrong with the user's CSV that we can explain to them."""


@dataclass
class ColumnProfile:
    """What we learned about one column."""

    name: str
    kind: Literal["numeric", "categorical"]
    missing: int
    distinct: int
    categories: list[str] = field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "missing": self.missing,
            "distinct": self.distinct,
            "categories": self.categories,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
        }


@dataclass
class ParsedDataset:
    columns: list[str]
    rows: list[list[str]]
    target: str
    task: Literal["classification", "regression"]
    profiles: list[ColumnProfile]
    fingerprint: str


@dataclass
class EncodingSpec:
    """Everything `encode` learned, so a new row can be encoded the same way later.

    Without this, scoring a row means re-deriving the encoding from data that now
    includes that row — which shifts the z-score means, can add a one-hot column for a
    value training never saw, and (for classification) re-derives the label set from a
    batch whose target column is empty. The model then gets a vector it was never
    trained on and answers confidently and wrongly.
    """

    feature_names: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categories: dict[str, list[str]]
    labels: list[str]
    task: Literal["classification", "regression"]
    target_mean: float = 0.0
    target_std: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": self.feature_names,
            "numeric_mean": self.numeric_mean,
            "numeric_std": self.numeric_std,
            "categories": self.categories,
            "labels": self.labels,
            "task": self.task,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> EncodingSpec:
        return cls(
            feature_names=list(blob["feature_names"]),
            numeric_mean={k: float(v) for k, v in blob["numeric_mean"].items()},
            numeric_std={k: float(v) for k, v in blob["numeric_std"].items()},
            categories={k: list(v) for k, v in blob["categories"].items()},
            labels=list(blob["labels"]),
            task=blob["task"],
            target_mean=float(blob.get("target_mean", 0.0)),
            target_std=float(blob.get("target_std", 1.0)) or 1.0,
        )


@dataclass
class EncodedDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    output_dim: int
    output_labels: list[str]
    task: Literal["classification", "regression"]
    target_mean: float = 0.0
    target_std: float = 1.0
    fingerprint: str = ""
    spec: EncodingSpec | None = None


def _looks_numeric(values: list[str]) -> bool:
    present = [v for v in values if v != ""]
    if not present:
        return False
    ok = 0
    for v in present:
        try:
            float(v)
            ok += 1
        except ValueError:
            continue
    return ok / len(present) >= NUMERIC_LIKE


def parse_csv(
    raw: bytes,
    target: str,
    max_rows: int = 20_000,
    max_columns: int = 60,
) -> ParsedDataset:
    """Read a CSV, profile every column, and decide whether the target is a class or a number."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("The file is not valid UTF-8. Save it as UTF-8 CSV and retry.") from exc

    try:
        reader = csv.reader(io.StringIO(text))
        all_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error as exc:
        raise DatasetError(f"Could not parse the CSV: {exc}") from exc

    if len(all_rows) < 2:
        raise DatasetError("The CSV needs a header row and at least one data row.")

    header = [h.strip() for h in all_rows[0]]
    body = all_rows[1:]

    if len(header) > max_columns:
        raise DatasetError(f"That CSV has {len(header)} columns; the limit is {max_columns}.")
    if len(header) != len(set(header)):
        raise DatasetError("Column names must be unique; this CSV repeats one.")
    if not target:
        raise DatasetError("Choose a target column to predict.")
    if target not in header:
        raise DatasetError(
            f"Target column {target!r} is not in the file. Columns are: {', '.join(header)}"
        )

    if len(body) > max_rows:
        body = body[:max_rows]

    width = len(header)
    body = [row + [""] * (width - len(row)) if len(row) < width else row[:width] for row in body]
    body = [row for row in body if any(cell.strip() for cell in row)]
    if not body:
        raise DatasetError("Every data row was blank.")
    if len(body) < 10:
        raise DatasetError(
            f"Only {len(body)} data rows. Training needs at least 10 so the test split means something."
        )

    target_index = header.index(target)
    target_values = [row[target_index].strip() for row in body]
    target_present = [v for v in target_values if v]
    if len(target_present) < len(target_values):
        raise DatasetError(
            f"The target column {target!r} has {len(target_values) - len(target_present)} empty "
            "cells. Fill or drop those rows first."
        )

    # Classification vs regression: numeric-looking targets with many distinct values are
    # treated as continuous. A discrete target with few levels (even if numeric, like 0/1)
    # is a class label.
    distinct_targets = set(target_present)
    if _looks_numeric(target_present) and len(distinct_targets) > max(20, len(body) // 20):
        task: Literal["classification", "regression"] = "regression"
    else:
        task = "classification"
        if len(distinct_targets) < 2:
            raise DatasetError(
                f"The target column {target!r} has only one distinct value, so there is nothing to learn."
            )
        if len(distinct_targets) > 50:
            raise DatasetError(
                f"The target column {target!r} has {len(distinct_targets)} distinct values. "
                "For a class label that is too many — is this actually a number?"
            )

    profiles: list[ColumnProfile] = []
    for col_index, name in enumerate(header):
        values = [row[col_index].strip() for row in body]
        missing = sum(1 for v in values if v == "")
        present = [v for v in values if v != ""]
        numeric = _looks_numeric(present)

        if numeric:
            numbers = [float(v) for v in present if _is_float(v)]
            profiles.append(
                ColumnProfile(
                    name=name,
                    kind="numeric",
                    missing=missing,
                    distinct=len(set(present)),
                    minimum=min(numbers) if numbers else None,
                    maximum=max(numbers) if numbers else None,
                    mean=float(np.mean(numbers)) if numbers else None,
                )
            )
        else:
            categories = sorted(set(present))
            if len(categories) > MAX_DISTINCT_CATEGORIES:
                raise DatasetError(
                    f"Column {name!r} has {len(categories)} distinct text values. "
                    "That is an id or free text, not a feature — drop it or bin it."
                )
            profiles.append(
                ColumnProfile(
                    name=name,
                    kind="categorical",
                    missing=missing,
                    distinct=len(categories),
                    categories=categories,
                )
            )

    feature_profiles = [p for p in profiles if p.name != target]
    if not feature_profiles:
        raise DatasetError("After removing the target there are no feature columns left.")

    fingerprint = hashlib.sha256(raw).hexdigest()[:16]
    return ParsedDataset(
        columns=header,
        rows=body,
        target=target,
        task=task,
        profiles=profiles,
        fingerprint=fingerprint,
    )


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def encode(parsed: ParsedDataset, standardize: bool = True) -> EncodedDataset:
    """Turn rows into a numeric matrix, and record exactly how it did it."""
    feature_profiles = [p for p in parsed.profiles if p.name != parsed.target]
    feature_indices = [parsed.columns.index(p.name) for p in feature_profiles]
    target_index = parsed.columns.index(parsed.target)

    feature_names: list[str] = []
    for profile in feature_profiles:
        if profile.kind == "numeric":
            feature_names.append(profile.name)
        else:
            feature_names.extend(f"{profile.name}={value}" for value in profile.categories)

    n = len(parsed.rows)
    x = np.zeros((n, len(feature_names)), dtype=np.float64)
    numeric_positions: list[int] = []

    position = 0
    for profile, col_index in zip(feature_profiles, feature_indices, strict=True):
        if profile.kind == "numeric":
            numeric_positions.append(position)
            for row_i, row in enumerate(parsed.rows):
                raw = row[col_index].strip()
                x[row_i, position] = float(raw) if _is_float(raw) else 0.0
                if not _is_float(raw):
                    x[row_i, position] = math.nan
            position += 1
        else:
            lookup = {value: i for i, value in enumerate(profile.categories)}
            for row_i, row in enumerate(parsed.rows):
                offset = lookup.get(row[col_index].strip())
                if offset is not None:
                    x[row_i, position + offset] = 1.0
            position += len(profile.categories)

    # Missing numerics become the column mean, which is the least surprising imputation.
    for pos in numeric_positions:
        col = x[:, pos]
        nans = np.isnan(col)
        if nans.any():
            fill = float(np.nanmean(col)) if (~nans).any() else 0.0
            col[nans] = fill
            x[:, pos] = col

    if standardize:
        numeric_stats: dict[str, tuple[float, float]] = {}
        for pos in numeric_positions:
            col = x[:, pos]
            mean = float(col.mean())
            std = float(col.std())
            numeric_stats[feature_names[pos]] = (mean, std)
            if std > 0:
                x[:, pos] = (col - mean) / std
    else:
        numeric_stats = {}

    target_values = [row[target_index] for row in parsed.rows]

    if parsed.task == "classification":
        labels = sorted(set(target_values))
        label_to_index = {label: i for i, label in enumerate(labels)}
        y = np.zeros((n, len(labels)), dtype=np.float64)
        for row_i, value in enumerate(target_values):
            y[row_i, label_to_index[value]] = 1.0
        spec = EncodingSpec(
            feature_names=feature_names,
            numeric_mean={k: v[0] for k, v in numeric_stats.items()},
            numeric_std={k: (v[1] or 1.0) for k, v in numeric_stats.items()},
            categories={
                p.name: list(p.categories) for p in feature_profiles if p.kind != "numeric"
            },
            labels=labels,
            task="classification",
        )
        return EncodedDataset(
            x=x,
            y=y,
            feature_names=feature_names,
            output_dim=len(labels),
            output_labels=labels,
            task="classification",
            fingerprint=parsed.fingerprint,
            spec=spec,
        )

    raw_target = np.array([float(v) for v in target_values], dtype=np.float64)
    mean = float(raw_target.mean())
    std = float(raw_target.std()) or 1.0
    y = ((raw_target - mean) / std).reshape(-1, 1)
    spec = EncodingSpec(
        feature_names=feature_names,
        numeric_mean={k: v[0] for k, v in numeric_stats.items()},
        numeric_std={k: (v[1] or 1.0) for k, v in numeric_stats.items()},
        categories={p.name: list(p.categories) for p in feature_profiles if p.kind != "numeric"},
        labels=[],
        task="regression",
        target_mean=mean,
        target_std=std,
    )
    return EncodedDataset(
        x=x,
        y=y,
        feature_names=feature_names,
        output_dim=1,
        output_labels=[],
        task="regression",
        target_mean=mean,
        target_std=std,
        fingerprint=parsed.fingerprint,
        spec=spec,
    )


def encode_rows(rows: list[dict[str, float | str]], spec: EncodingSpec) -> np.ndarray:
    """Encode new rows using the exact encoding the model was trained with.

    Numeric columns use the training mean and standard deviation, not the new batch's.
    Categorical columns use the training category list, so an unseen value encodes as an
    all-zero one-hot group ("none of the above") instead of silently adding a column.
    """
    matrix = np.zeros((len(rows), len(spec.feature_names)), dtype=np.float64)
    position = {name: i for i, name in enumerate(spec.feature_names)}

    for row_i, row in enumerate(rows):
        for column, categories in spec.categories.items():
            value = str(row.get(column, ""))
            if value in categories:
                matrix[row_i, position[f"{column}={value}"]] = 1.0

        for column, mean in spec.numeric_mean.items():
            raw = row.get(column, "")
            try:
                numeric = float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                numeric = mean
            std = spec.numeric_std.get(column) or 1.0
            matrix[row_i, position[column]] = (numeric - mean) / std

    return matrix


def stratified_split(
    x: np.ndarray, y: np.ndarray, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split while keeping the class balance of the train set and the test set similar.

    A plain shuffle can drop a rare class out of the test set entirely, which makes the
    reported score meaningless. Stratifying per class avoids that.
    """
    n = x.shape[0]
    if y.ndim == 1 or y.shape[1] == 1:
        # Regression: a plain shuffle is fine.
        order = rng.permutation(n)
        cut = max(1, min(int(round(n * (1 - test_fraction))), n - 1))
        return x[order[:cut]], y[order[:cut]], x[order[cut:]], y[order[cut:]]

    labels = np.argmax(y, axis=1)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        n_test = max(1, int(round(len(members) * test_fraction))) if len(members) > 1 else 0
        test_idx.extend(members[:n_test].tolist())
        train_idx.extend(members[n_test:].tolist())

    if not train_idx or not test_idx:
        raise DatasetError("The dataset is too small or too imbalanced to split into train and test.")

    tr = np.array(sorted(train_idx))
    te = np.array(sorted(test_idx))
    return x[tr], y[tr], x[te], y[te]


def confusion_matrix(true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> list[list[int]]:
    """Rows are the truth, columns are what the model said."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(true_idx.tolist(), pred_idx.tolist(), strict=True):
        matrix[t][p] += 1
    return matrix


def per_class_metrics(
    true_idx: np.ndarray, pred_idx: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Precision/recall/F1 for each class, computed from the confusion counts directly."""
    n = len(labels)
    matrix = confusion_matrix(true_idx, pred_idx, n)
    out: dict[str, dict[str, float]] = {}

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(n)) - tp
        fn = sum(matrix[i][c] for c in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(sum(matrix[i])),
        }
    return out
