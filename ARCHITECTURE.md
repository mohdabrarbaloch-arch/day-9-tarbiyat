# Architecture

## The one-sentence version

A FastAPI service wraps a hand-written NumPy neural network; a vanilla-JS SPA talks to
it over plain REST; everything the model learned about encoding a row is stored next to
its weights so prediction can't drift from training.

## System diagram

```
                        ┌──────────────────────────────┐
                        │   Browser (no build step)    │
                        │   index.html + app.js + css  │
                        │  - auth, upload, builder UI  │
                        │  - canvas loss curve         │
                        └───────────────┬──────────────┘
                                        │ JSON over HTTPS
                                        │ Bearer JWT
                        ┌───────────────▼──────────────┐
                        │        FastAPI app           │
                        │  middleware: security headers│
                        │              CORS            │
                        │              body size cap   │
                        │  routers: auth / datasets /  │
                        │           runs               │
                        └───┬──────────┬───────────┬───┘
                            │          │           │
              ┌─────────────▼──┐  ┌────▼─────┐  ┌──▼──────────────┐
              │ services/      │  │ ml/      │  │ api/deps.py     │
              │ training.py    │  │ dataset  │  │ JWT → User      │
              │ - config sanity│  │ - profile│  └──┬──────────────┘
              │ - loss legality│  │ - encode │     │
              │ - orchestrate  │  │ - split  │     │
              └───────┬────────┘  └────┬─────┘     │
                      │                │           │
              ┌───────▼────────┐       │           │
              │ ml/nn.py       │       │           │
              │ MLP + Trainer  │       │           │
              │ backprop, Adam │       │           │
              └───────┬────────┘       │           │
                      │                │           │
              ┌───────▼────────────────▼───────────▼──────┐
              │  SQLAlchemy 2.0  →  SQLite / PostgreSQL   │
              │  users · datasets · runs                  │
              └───────────────────────────────────────────┘
```

## Tech stack and why

| Choice | Reason |
|---|---|
| **FastAPI 0.115** | Pydantic-native validation means the API refuses a bad training config before any NumPy runs. |
| **SQLAlchemy 2.0 (typed)** | `Mapped[...]` annotations catch schema mistakes at import time. SQLite for zero-setup, Postgres via `DATABASE_URL`. |
| **Plain NumPy** | The gradient is the product. Adding PyTorch would reduce this project to a form over someone else's `model.fit()`. |
| **Vanilla JS SPA** | No build step means `git clone && uvicorn` gives you a working UI. A React app would add a toolchain for three screens. |
| **bcrypt + JWT (HS256)** | Passwords never stored reversibly; tokens stateless so no session table to prune. |

## Data flow: from CSV to a prediction

**1. Upload** — `POST /api/datasets` with a multipart CSV. The service:
- enforces `MAX_UPLOAD_BYTES` before parsing,
- parses and profiles up to `MAX_DATASET_ROWS` rows,
- decides `classification` vs `regression` from the target column's cardinality and
  whether its values parse as numbers,
- stores the raw bytes plus a JSON column profile.

**2. Train** — `POST /api/runs`. In order:
1. **Legality check.** The requested loss must fit the detected task. `cce` on a
   single regression output is rejected with 422, not silently coerced.
2. **Encode.** Numeric columns z-scored; categorical columns one-hot with categories
   sorted for determinism. The learned encoding is captured in an `EncodingSpec`.
3. **Split.** Stratified for classification so a rare class can't vanish from the test
   set; random for regression.
4. **Fit.** `Trainer` loops epochs in shuffled mini-batches, computing analytic
   gradients and applying the chosen optimizer.
5. **Evaluate** on the held-out split, always alongside a naive baseline (majority
   class, or the mean predictor).
6. **Persist** weights, per-epoch history, metrics, and the `EncodingSpec` as JSON.

**3. Predict** — `POST /api/runs/{id}/predict`. Rows are encoded using the *stored*
`EncodingSpec`:
- numeric values use the training mean and standard deviation,
- an unseen category encodes as an all-zero one-hot group ("none of the above"),
- the model's weight shapes are checked against the encoding before the forward pass.

That last point is the whole reason `EncodingSpec` exists. The obvious implementation —
append the new rows to the CSV and re-run the encoder — is subtly broken three ways:
the z-score means shift, a new category adds a column and the weights stop lining up,
and for classification the label set gets recomputed from a batch whose target column
is empty. All three produce a confident, wrong answer instead of an error.

## Backprop details worth knowing

The output layer fuses loss and gradient so the delta is `dL/dz`, not `dL/da`:

- **softmax + cross-entropy** → `(p - y) / n`
- **sigmoid + BCE** → `(p - y) / n` (dividing by `p(1-p)` would re-introduce the same
  singularities BCE was chosen to avoid)
- **softmax + MSE** → `p * (g - Σ g·p)` where `g = dL/dp`. This full Jacobian is the
  step most from-scratch implementations skip.
- **linear head + MSE** → `g` directly, since the head is the identity.

`scripts/gradcheck.py` verifies all of these against central finite differences.

## Scaling notes

What breaks first, in order:

1. **Synchronous training.** A run holds its request open. First fix: push `run_training`
   onto a worker queue (arq or Celery), return 202 with a run id, let the SPA poll
   `GET /api/runs/{id}`. The status field and history already support that shape.
2. **Weights in a TEXT column.** JSON weights are readable but large — a 3-layer net
   with 200 units is megabytes. Move to object storage with a pointer in the row.
3. **In-process rate limiting.** Resets on restart, doesn't coordinate across workers.
   Swap the counter dict for Redis `INCR` + `EXPIRE`.
4. **Encoding at request time.** Every predict re-reads and re-parses the stored spec.
   Cache it per run id; it's immutable after training.
5. **No model registry.** Weights live on the run row. A versioned registry would let
   you promote a model without pinning a training run.

## Security posture

- Passwords: bcrypt, cost 12.
- JWT: HS256, expiry from `ACCESS_TOKEN_TTL_MINUTES`. In production the app refuses
  to start with a default `SECRET_KEY`.
- Every dataset and run is scoped to `owner_id`; a cross-tenant read returns 404, not a
  403 that would confirm the resource exists.
- Upload size capped both by middleware and again at parse time.
- Security headers (nosniff, frame-deny, referrer-policy) added in middleware.
- CSV content is never executed or interpolated — it's parsed, profiled and stored.
