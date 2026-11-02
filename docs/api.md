# API reference

Base URL: `http://127.0.0.1:8000`

All endpoints return JSON. Authenticated endpoints need `Authorization: Bearer <token>`.

Errors are uniform:

```json
{ "detail": "Human-readable message." }
```

| Status | When |
|---|---|
| `400` | malformed request (bad CSV, unparseable row) |
| `401` | missing, expired or invalid token |
| `404` | not found **or not yours** — cross-tenant access never leaks existence |
| `409` | conflict (email already registered) |
| `413` | upload above `MAX_UPLOAD_BYTES` |
| `422` | validation failed (Pydantic, or an illegal training config) |
| `429` | rate limited; `Retry-After` carries the seconds |

---

## Health

### `GET /api/health`

```json
{ "status": "ok", "version": "1.0.0", "environment": "development" }
```

---

## Auth

### `POST /api/auth/register`

```json
{ "email": "you@example.com", "password": "at-least-8-chars" }
```

→ `201`

```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "user": { "id": 1, "email": "you@example.com", "created_at": "2026-11-03T09:12:44" }
}
```

`409` if the email exists. Password minimum is 8 characters.

### `POST /api/auth/login`

```json
{ "email": "demo@tarbiyat.pk", "password": "tarbiyat-demo-2026" }
```

→ `200` with the same body as register. `401` on bad credentials. Rate limited by
`RATE_LIMIT_AUTH`.

### `GET /api/auth/me`

→ `200` `{ "id": 1, "email": "...", "created_at": "..." }`

---

## Datasets

### `GET /api/datasets`

```json
[
  {
    "id": 2,
    "name": "Customer churn (binary)",
    "target": "churned",
    "task": "classification",
    "row_count": 400,
    "column_count": 6,
    "description": "Binary churn label...",
    "created_at": "2026-11-03T09:10:02"
  }
]
```

### `POST /api/datasets`

`multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | `.csv`, UTF-8 |
| `name` | string | no | defaults to the filename |
| `target` | string | yes | must be a column in the CSV |

→ `201` with the dataset object above.

Rejections: `400` if the target column is missing or the CSV has fewer than 2 columns;
`413` if the file exceeds the cap.

### `GET /api/datasets/{id}`

```json
{
  "id": 2,
  "name": "Customer churn (binary)",
  "target": "churned",
  "task": "classification",
  "row_count": 400,
  "column_count": 6,
  "description": "...",
  "created_at": "...",
  "profile": [
    { "name": "tenure_months", "kind": "numeric", "missing": 0,
      "min": 1.0, "max": 72.0, "mean": 32.4 },
    { "name": "plan", "kind": "categorical", "missing": 0,
      "categories": ["basic", "premium", "standard"] }
  ],
  "preview": [ { "tenure_months": "2", "plan": "basic", "churned": "1" } ]
}
```

### `DELETE /api/datasets/{id}`

→ `204`, no body. Also deletes the runs trained on it.

---

## Runs

### `POST /api/runs`

Trains synchronously and returns the finished run. Expect the response to take as long
as training takes — hundreds of milliseconds for the seeded datasets.

```json
{
  "dataset_id": 1,
  "label": "iris try 1",
  "hidden_layers": [12, 8],
  "activation": "relu",
  "optimizer": "adam",
  "learning_rate": 0.01,
  "epochs": 120,
  "batch_size": 16,
  "test_fraction": 0.25,
  "l2": 0.0,
  "seed": 42,
  "loss": null
}
```

| Field | Type | Default | Constraints |
|---|---|---|---|
| `dataset_id` | int | — | must exist and be yours |
| `label` | string | auto | ≤ 120 chars |
| `hidden_layers` | int[] | `[16]` | 1–4 entries, each 1–256 |
| `activation` | enum | `relu` | `relu` \| `tanh` \| `sigmoid` |
| `init` | enum | `he` | `he` \| `xavier` \| `normal` |
| `optimizer` | enum | `adam` | `sgd` \| `momentum` \| `adam` |
| `learning_rate` | float | `0.05` | > 0, ≤ 2.0 |
| `epochs` | int | `120` | 1–400 |
| `batch_size` | int | `16` | 1–512 |
| `test_fraction` | float | `0.2` | 0.05–0.5 |
| `l2` | float | `0.0` | ≥ 0, ≤ 0.5 |
| `early_stopping` | bool | `false` | stop when validation stops improving |
| `patience` | int | `12` | 2–100 epochs without improvement |
| `seed` | int | `42` | 0 – 2³¹−1 |
| `loss` | enum \| null | null | `cce` \| `bce` \| `mse`; null = chosen from the task |

→ `201`

```json
{
  "id": 7,
  "dataset_id": 1,
  "label": "iris try 1",
  "status": "completed",
  "loss": "cce",
  "param_count": 191,
  "train_rows": 112,
  "test_rows": 38,
  "duration_ms": 136,
  "config": { "hidden_layers": [12, 8], "activation": "relu", "optimizer": "adam",
              "learning_rate": 0.01, "epochs": 120, "batch_size": 16,
              "l2": 0.0, "seed": 42, "init": "he" },
  "metrics": {
    "task": "classification",
    "accuracy": 0.9722,
    "majority_baseline": 0.3333,
    "labels": ["setosa", "versicolor", "virginica"],
    "confusion_matrix": [[12, 0, 0], [0, 11, 1], [0, 0, 12]],
    "per_class": [
      { "label": "setosa",     "precision": 1.0,   "recall": 1.0,   "f1": 1.0,   "support": 12.0 },
      { "label": "versicolor", "precision": 1.0,   "recall": 0.917, "f1": 0.957, "support": 12.0 },
      { "label": "virginica",  "precision": 0.923, "recall": 1.0,   "f1": 0.96,  "support": 12.0 }
    ],
    "test_rows": 38,
    "loss": "cce",
    "best_epoch": 120,
    "stopped_early": false,
    "final_train_loss": 0.0612,
    "final_val_loss": null
  },
  "history": [
    { "epoch": 1, "train_loss": 1.0986, "train_accuracy": 0.3333,
      "val_loss": null, "val_accuracy": null }
  ],
  "created_at": "2026-11-03T09:14:02",
  "finished_at": "2026-11-03T09:14:02"
}
```

For regression, `metrics` instead carries:

```json
{
  "task": "regression",
  "rmse": 2.1914,
  "mae": 1.6832,
  "r2": 0.8174,
  "baseline_rmse": 5.1279,
  "target_mean": 96.288,
  "target_std": 5.1,
  "test_rows": 75
}
```

`422` cases worth knowing: `cce` on a regression dataset, `bce` on a 3-class target,
`cce` on a single-output head, a hidden layer wider than 256, more than 4 layers, or
`epochs` above the deployment's `MAX_EPOCHS`.

### `GET /api/runs`

A list of run summaries — same shape as above minus `history`, `weights` and `metrics`.

### `GET /api/runs/{id}`

The full run including `history`. Weights are never returned.

### `POST /api/runs/{id}/predict`

```json
{
  "rows": [
    { "sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2 }
  ]
}
```

Every feature column is required; unknown columns are rejected with `422` listing what
was expected. An empty `rows` array is a `400`, and a run with no stored weights is a
`409`.

→ `200`

```json
{
  "task": "classification",
  "predictions": [
    {
      "index": 0,
      "predicted": "setosa",
      "confidence": 1.0,
      "probabilities": { "setosa": 1.0, "versicolor": 0.0, "virginica": 0.0 }
    }
  ]
}
```

For regression, `predicted` is a float in the target's original units and
`confidence` is `null`.

`409` if the run has no weights (not completed).

### `DELETE /api/runs/{id}`

→ `204`, no body.

---

## Rate limits

Per IP, fixed windows, configured in `.env`:

| Endpoint | Default |
|---|---|
| `POST /api/auth/login`, `/register` | `RATE_LIMIT_AUTH` (10/minute) |
| `POST /api/runs` | `RATE_LIMIT_WRITE` (60/minute) |
| `POST /api/datasets` | `RATE_LIMIT_WRITE` (60/minute) |

Over the limit returns `429` with a `Retry-After` header.
