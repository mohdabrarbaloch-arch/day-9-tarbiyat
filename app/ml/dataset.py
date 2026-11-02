"""CSV parsing, column profiling, encoding and the train/test split.

The rule here is that nothing is silently guessed. Every column is profiled, encoded,
and reported back to the user, so a model that learns nothing can be traced to a
specific column decision rather than to mystery.
"""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ColumnKind = Literal["numeric", "categorical", "target"]


class DatasetError(ValueError):
    """Raised when a CSV cannot be safely turned into training data."""


@dataclass
class ColumnProfile:
    name: str
    kind: ColumnKind
    # numeric
    mean: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    missing: int = 0
    # categorical
    categories: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "missing": self.missing,
        }
        if self.kind == "numeric":
            out.update(
                {
                    "mean": self.mean,
                    "std": self.std,
                    "min": self.minimum,
                    "max": self.maximum,
                }
            )
        else:
            out["categories"] = self.categories
        return out


@dataclass
class ParsedDataset:
    columns: list[str]
    rows: list[list[str]]
    target: str
    task: Literal["classification", "regression"]
    profiles: list[ColumnProfile]
    fingerprint: str


def fingerprint_bytes(raw: bytes) -> str:
    """Stable content hash so a dataset can be cited without storing its body twice."""
    return hashlib.sha256(raw).hexdigest()[:16]


def _looks_numeric(values: list[str]) -> bool:
    seen = 0
    for v in values:
        s = v.strip()
        if not s:
            continue
        seen += 1
        try:
            float(s)
        except ValueError:
            return False
    return seen > 0


def parse_csv(
    raw: bytes,
    target: str,
    max_rows: int,
    max_columns: int,
) -> ParsedDataset:
    """Parse and profile a CSV. Raises DatasetError with an actionable message."""
    if not raw.strip():
        raise DatasetError("The uploaded file is empty.")

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("The file is not valid UTF-8. Re-export it as UTF-8 CSV.") from exc

    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel  # type: ignore[assignment]

    reader = csv.reader(io.StringIO(text), dialect)
    try:
        header = next(reader)
    except StopIteration as exc:
        raise DatasetError("The CSV has no header row.") from exc

    header = [h.strip() for h in header]
    if not header or all(not h for h in header):
        raise DatasetError("The header row is empty.")

    if len(header) < 2:
        raise DatasetError("A dataset needs at least two columns (one feature and a target).")

    if len(header) > max_columns:
        raise DatasetError(f"Too many columns ({len(header)}). The limit is {max_columns}.")

    if len(set(header)) != len(header):
        dupes = sorted({h for h in header if header.count(h) > 1})
        raise DatasetError(f"Duplicate column names: {', '.join(dupes)}")

    if target not in header:
        raise DatasetError(
            f"Target column '{target}' is not in the file. Available: {', '.join(header)}"
        )

    rows: list[list[str]] = []
    for row in reader:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise DatasetError(
                f"Row {len(rows) + 2} has {len(row)} fields but the header has {len(header)}."
            )
        rows.append([cell.strip() for cell in row])
        if len(rows) > max_rows:
            raise DatasetError(
                f"Too many rows. The limit is {max_rows:,} for interactive training."
            )

    if len(rows) < 10:
        raise DatasetError(
            f"Only {len(rows)} usable rows. At least 10 are needed to split and train."
        )

    target_index = header.index(target)
    target_values = [r[target_index] for r in rows]
    target_numeric = _looks_numeric(target_values)

    if target_numeric:
        uniques = len({v for v in target_values if v})
        # Low-cardinality integers are classes, not a regression target.
        task: Literal["classification", "regression"] = (
            "classification" if uniques <= 10 else "regression"
        )
    else:
        task = "classification"
        uniques = len({v for v in target_values if v})
        if uniques > 50:
            raise DatasetError(
                f"The target has {uniques} distinct values. Classification targets are "
                "capped at 50 classes."
            )
        if uniques < 2:
            raise DatasetError("The target column has only one distinct value.")

    profiles: list[ColumnProfile] = []
    for index, name in enumerate(header):
        values = [r[index] for r in rows]
        missing = sum(1 for v in values if not v)
        present = [v for v in values if v]

        if name == target:
            kind: ColumnKind = "target"
        elif _looks_numeric(present):
            kind = "numeric"
        else:
            kind = "categorical"

        profile = ColumnProfile(name=name, kind=kind, missing=missing)

        if kind == "numeric":
            floats = np.array([float(v) for v in present], dtype=np.float64)
            profile.mean = float(floats.mean()) if floats.size else None
            profile.std = float(floats.std()) if floats.size else None
            profile.minimum = float(floats.min()) if floats.size else None
            profile.maximum = float(floats.max()) if floats.size else None
        elif kind == "categorical":
            cats = sorted(set(present))
            if len(cats) > 60:
                raise DatasetError(
                    f"Column '{name}' has {len(cats)} categories, which would explode the "
                    "input width. Encode it numerically or drop it before uploading."
                )
            profile.categories = cats

        profiles.append(profile)

    feature_columns = [p for p in profiles if p.name != target]
    if not feature_columns:
        raise DatasetError("There are no feature columns left after removing the target.")
    if all(p.kind == "categorical" and len(p.categories) < 2 for p in feature_columns):
        raise DatasetError("Every feature column is constant; there is nothing to learn.")

    return ParsedDataset(
        columns=header,
        rows=rows,
        target=target,
        task=task,
        profiles=profiles,
        fingerprint=fingerprint_bytes(raw),
    )


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


def encode(parsed: ParsedDataset, standardize: bool = True) -> EncodedDataset:
    """One-hot encode categoricals and z-score numeric features."""
    target_index = parsed.columns.index(parsed.target)
    feature_profiles = [p for p in parsed.profiles if p.name != parsed.target]

    feature_names: list[str] = []
    for profile in feature_profiles:
        if profile.kind == "numeric":
            feature_names.append(profile.name)
        else:
            feature_names.extend(f"{profile.name}={cat}" for cat in profile.categories)

    n = len(parsed.rows)
    x = np.zeros((n, len(feature_names)), dtype=np.float64)

    column_cursor = 0
    numeric_positions: list[int] = []

    for profile in feature_profiles:
        source_index = parsed.columns.index(profile.name)
        if profile.kind == "numeric":
            numeric_positions.append(column_cursor)
            for row_i, row in enumerate(parsed.rows):
                raw = row[source_index]
                if raw:
                    x[row_i, column_cursor] = float(raw)
            column_cursor += 1
        else:
            cat_to_pos = {cat: column_cursor + i for i, cat in enumerate(profile.categories)}
            for row_i, row in enumerate(parsed.rows):
                pos = cat_to_pos.get(row[source_index])
                if pos is not None:
                    x[row_i, pos] = 1.0
            column_cursor += len(profile.categories)

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

    raw_target = np.array([float(v) if v else np.nan for v in target_values], dtype=np.float64)
    if np.isnan(raw_target).any():
        fill = float(np.nanmean(raw_target))
        raw_target = np.where(np.isnan(raw_target), fill, raw_target)
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


def encode_rows(
    rows: list[dict[str, float | str]],
    spec: EncodingSpec,
) -> np.ndarray:
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
    y: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx), stratified for classification one-hot targets."""
    n = y.shape[0]
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")

    if y.shape[1] == 1:
        order = rng.permutation(n)
        cut = max(1, int(round(n * (1 - test_fraction))))
        cut = min(cut, n - 1)
        return np.sort(order[:cut]), np.sort(order[cut:])

    labels = np.argmax(y, axis=1)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        if idx.size < 2:
            # A class with one member cannot be split; keep it in train.
            train_parts.append(idx)
            continue
        n_test = max(1, int(round(idx.size * test_fraction)))
        n_test = min(n_test, idx.size - 1)
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)
    return train_idx, test_idx


def confusion_matrix(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> list[list[int]]:
    """Plain integer confusion matrix, rows = truth, columns = predicted."""
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for p, t in zip(pred, truth, strict=True):
        matrix[int(t)][int(p)] += 1
    return matrix


def per_class_metrics(
    pred: np.ndarray, truth: np.ndarray, labels: list[str]
) -> list[dict[str, float | str]]:
    """Precision, recall and F1 per class, computed without sklearn."""
    out: list[dict[str, float | str]] = []
    for index, label in enumerate(labels):
        tp = int(np.sum((pred == index) & (truth == index)))
        fp = int(np.sum((pred == index) & (truth != index)))
        fn = int(np.sum((pred != index) & (truth == index)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append(
            {
                "label": label,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": float(tp + fn),
            }
        )
    return out
